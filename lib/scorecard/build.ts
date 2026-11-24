import { createHash } from 'node:crypto';
import { readCatalog, type SimulatorCatalogEntry } from '../api/catalog-server';
import { buildPredictRequestBody, type EstimateAdapterInput } from '../api/estimate-adapter';
import {
  BENCHMARK_DATASET_ID,
  INFERENCEX_COMPARISON_DATASET_ID,
  aggregateBenchmarkRows,
  benchmarkFilterValue,
  configurationForRow,
  configurationKey,
  getBenchmarkDataset,
  getInferenceXDataset,
} from '../validation/benchmark-dataset';
import {
  comparisonRequestsPerLoadUnit,
  measuredPoolTopology,
  perPoolBatch,
  predictionFabric,
  type ServingMode,
} from '../validation/benchmark-predictions';
import { backendVersionsComparable, normalizeBackendVersion, UNREPORTED_VERSION } from '../validation/benchmark-versions';
import type { DatasetConfiguration, DatasetRow } from '../validation/benchmark-types';
import { buildCells, buildCoverage, buildHeadToHead, buildSourceOverlap, buildTotals } from './aggregate';
import {
  eligibility,
  type Eligible,
  type ScorecardBackend,
  type ScorecardSimulatorSupport,
  type ScorecardSystem,
} from './eligibility';
import { isModelConfigRejection, predictPoint, type PredictOutcome, type PredictStatus } from './predict';
import { percentile } from './metrics';
import {
  SCORECARD_SCHEMA_VERSION,
  type ScorecardChannel,
  type ScorecardMeasurement,
  type ScorecardPoint,
  type ScorecardPrediction,
  type ScorecardSource,
  type ScorecardSummary,
} from './types';

/**
 * The scorecard sweep.
 *
 * Loads both measured sources, enumerates every simulator the catalogue
 * exposes, predicts each measured concurrency level where that simulator can
 * model the run, and folds the results into the published artifact.
 *
 * Costs are dominated by prediction calls, so the sweep takes a prediction
 * cache: a daily run only pays for points that are new or whose simulator
 * version changed.
 */

export interface PredictionCacheEntry {
  status: PredictStatus;
  reason: string | null;
  prediction: ScorecardPrediction | null;
  engineVersion: string | null;
}

export interface PredictionCache {
  entries: Record<string, PredictionCacheEntry>;
}

export interface BuildOptions {
  gatewayUrl: string;
  channel: ScorecardChannel;
  timeoutSeconds?: number;
  /** In-flight prediction calls. Kept low so the job stays a measurement. */
  concurrency?: number;
  sources?: ScorecardSource[];
  /** Cap work units after ordering, for a local or smoke run. */
  limit?: number;
  /** Restrict to these models, for a local or smoke run. */
  models?: string[];
  cache?: PredictionCache;
  log?: (message: string) => void;
  /**
   * Pause between measured-dataset load attempts. The Hugging Face dataset
   * server answers "busier than usual" under load, and a build job can afford
   * to wait where a page cannot.
   */
  datasetRetryDelayMs?: number;
  /** Shards another runner owns; recorded so a partial snapshot is explicit. */
  incomplete?: { leg: string; reason: string }[];
}

export interface BuildResult {
  summary: ScorecardSummary;
  points: ScorecardPoint[];
  cache: PredictionCache;
  /** Cached entry count, for the provenance line. */
  cacheHits: number;
  cacheMisses: number;
  /** Work units by outcome, for a job summary line. */
  outcomes: Record<PredictStatus | 'cached', number>;
  /** Per-call latency, so a runner's cost is visible before it is scheduled. */
  timings: Record<string, TimingSummary>;
  totalTiming: TimingSummary;
}

export interface TimingSummary {
  count: number;
  meanMs: number;
  p50Ms: number;
  p90Ms: number;
  maxMs: number;
}

const DEFAULT_TIMEOUT_SECONDS = 90;
const DEFAULT_CONCURRENCY = 4;
const DEFAULT_DATASET_RETRY_DELAY_MS = 15_000;
const DATASET_RETRY_ATTEMPTS = 6;
const MAX_ATTEMPTS = 2;
const RETRY_DELAY_MS = 2000;

function hash(parts: readonly string[]): string {
  return createHash('sha256').update(parts.join('\u0000')).digest('hex').slice(0, 24);
}

function asNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'bigint') {
    const converted = Number(value);
    return Number.isSafeInteger(converted) ? converted : null;
  }
  return null;
}

function asPositiveInteger(value: unknown): number | null {
  const parsed = asNumber(value);
  return parsed !== null && Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null;
}

function mapSystem(raw: unknown): ScorecardSystem | null {
  if (!raw || typeof raw !== 'object') return null;
  const value = raw as Record<string, unknown>;
  if (typeof value.id !== 'string' || value.id.trim().length === 0) return null;
  const label = typeof value.name === 'string' && value.name.trim().length > 0 ? value.name.trim() : value.id;
  return { id: value.id, label };
}

function mapBackend(raw: unknown): ScorecardBackend | null {
  if (!raw || typeof raw !== 'object') return null;
  const value = raw as Record<string, unknown>;
  if (typeof value.id !== 'string' || !Array.isArray(value.versions)) return null;

  const systems: Record<string, string[]> = {};
  if (value.systems && typeof value.systems === 'object') {
    for (const [system, versions] of Object.entries(value.systems as Record<string, unknown>)) {
      if (Array.isArray(versions)) {
        systems[system] = versions.filter((entry): entry is string => typeof entry === 'string');
      }
    }
  }

  const versionSupport: Record<string, { disaggregated: boolean }> = {};
  if (value.version_support && typeof value.version_support === 'object') {
    for (const [version, entry] of Object.entries(value.version_support as Record<string, unknown>)) {
      if (!entry || typeof entry !== 'object') continue;
      versionSupport[version] = {
        disaggregated: (entry as Record<string, unknown>).disaggregated === true,
      };
    }
  }

  return {
    id: value.id,
    versions: value.versions.filter((entry): entry is string => typeof entry === 'string'),
    defaultVersion: typeof value.default_version === 'string' ? value.default_version : null,
    systems,
    versionSupport,
    disaggregationRequiresFabric: value.disaggregation_requires_fabric === true,
  };
}

function toSupport(entry: SimulatorCatalogEntry): ScorecardSimulatorSupport {
  return {
    id: entry.id,
    label: entry.label,
    available: entry.available,
    version: entry.version,
    revision: entry.revision,
    models: entry.models.flatMap((model): string[] => {
      if (typeof model === 'string' && model.trim().length > 0) return [model];
      if (model && typeof model === 'object') {
        const id = (model as Record<string, unknown>).id;
        return typeof id === 'string' && id.trim().length > 0 ? [id] : [];
      }
      return [];
    }),
    modelsExhaustive: entry.modelsExhaustive,
    systems: entry.systems.flatMap((system): ScorecardSystem[] => {
      const mapped = mapSystem(system);
      return mapped ? [mapped] : [];
    }),
    backends: entry.backends.flatMap((backend): ScorecardBackend[] => {
      const mapped = mapBackend(backend);
      return mapped ? [mapped] : [];
    }),
    networks: entry.networks,
    defaultNetwork: entry.defaultNetwork,
  };
}

/** The source a measured row came from. */
function rowSource(row: DatasetRow): ScorecardSource {
  return row.benchmark_source === 'inferencex' ? 'inferencex' : 'redhat';
}

/**
 * The measured serving mode. A disaggregated run records its pools in the
 * topology string; everything else is treated as aggregated.
 */
function servingModeOf(configuration: DatasetConfiguration): ServingMode {
  const topology = configuration.inferencex_topology?.toLocaleLowerCase() ?? '';
  return topology.startsWith('disaggregated') ? 'disagg' : 'agg';
}

interface WorkUnit {
  identity: string;
  source: ScorecardSource;
  model: string;
  accelerator: string;
  isl: number;
  osl: number;
  tp: number;
  workload: string;
  configurationKey: string;
  measuredRelease: string | null;
  releaseMatch: boolean;
  servingMode: ServingMode;
  measured: ScorecardMeasurement;
  concurrency: number;
  plan: Eligible;
  /** Release of the simulator itself; part of the prediction cache key. */
  simulatorVersion: string | null;
  /** Revision of the simulator build; a new build is a new answer. */
  simulatorRevision: string | null;
  body: Record<string, unknown>;
}

interface SkippedUnit {
  base: Omit<WorkUnit, 'plan' | 'body' | 'simulatorVersion' | 'simulatorRevision' | 'releaseMatch'>;
  simulator: string;
  status: 'ineligible' | 'unsupported';
  reason: string;
}

/**
 * One entry in the ordered plan.
 *
 * Capability rejections are part of the plan rather than a side list so that a
 * shard boundary, a `--limit` sample, and the coverage denominator all count
 * the same points. A sample cannot report more attempts than it made.
 */
type PlannedUnit =
  | { kind: 'work'; unit: WorkUnit }
  | { kind: 'skipped'; unit: SkippedUnit };

function simulatorOf(entry: PlannedUnit): string {
  return entry.kind === 'work' ? entry.unit.plan.simulator : entry.unit.simulator;
}

function identityOf(entry: PlannedUnit): string {
  return entry.kind === 'work' ? entry.unit.identity : entry.unit.base.identity;
}

/**
 * A measured series: one runtime configuration at one workload shape.
 *
 * `configurationKey` alone is not enough to group rows for a prediction. The
 * same runtime configuration is measured at several input/output lengths and
 * tensor-parallel sizes, and a prediction is made for one of each. Folding
 * those rows together would take the shape from whichever row came first and
 * average measurements from several workloads into one series.
 */
interface ConfigurationGroup {
  values: DatasetConfiguration;
  configurationId: string;
  rows: DatasetRow[];
}

function groupConfigurations(rows: readonly DatasetRow[]): Map<string, ConfigurationGroup> {
  const groups = new Map<string, ConfigurationGroup>();
  for (const row of rows) {
    const values = configurationForRow(row);
    const configurationId = configurationKey(values);
    // Normalized filter values keep `8192` and `'8192'` in one series.
    const seriesKey = [
      configurationId,
      benchmarkFilterValue(row, 'isl') ?? '',
      benchmarkFilterValue(row, 'osl') ?? '',
      benchmarkFilterValue(row, 'tp') ?? '',
    ].join('\u0000');
    const existing = groups.get(seriesKey) ?? { values, configurationId, rows: [] };
    existing.rows.push(row);
    groups.set(seriesKey, existing);
  }
  return groups;
}

/**
 * Build the request body for one measured level.
 *
 * Mirrors the Benchmark comparison page's plan: PP=1 and DP=1 because the
 * dataset does not record them, MoE widths left to the service, and a fabric
 * only when the backend requires one.
 */
function requestBody(
  plan: Eligible,
  run: {
    model: string;
    systemId: string;
    isl: number;
    osl: number;
    tp: number;
    servingMode: ServingMode;
    concurrency: number;
    topology: ReturnType<typeof measuredPoolTopology>;
    backend: string;
  },
  modelConfig: Record<string, unknown> | null,
): Record<string, unknown> {
  const input: EstimateAdapterInput = {
    model_path: run.model,
    system: run.systemId,
    isl: run.isl,
    osl: run.osl,
    batch_size: run.concurrency,
    requests_per_load_unit: comparisonRequestsPerLoadUnit(run.concurrency),
    tp_size: run.tp,
    pp_size: 1,
    attention_dp_size: 1,
    backend: run.backend,
    backend_version: plan.version ?? undefined,
    simulator: plan.simulator,
    hf_model_config: modelConfig,
    mode: run.servingMode,
    fabric: predictionFabric(run.servingMode, plan.requiresFabric, plan.fabric ?? ''),
    prefill: run.servingMode === 'disagg'
      ? {
        tp: run.topology.prefillTp,
        pp: 1,
        workers: run.topology.prefillWorkers,
        batch: perPoolBatch(run.concurrency, run.topology.prefillWorkers),
      }
      : undefined,
    decode: run.servingMode === 'disagg'
      ? {
        tp: run.topology.decodeTp,
        pp: 1,
        workers: run.topology.decodeWorkers,
        batch: perPoolBatch(run.concurrency, run.topology.decodeWorkers),
      }
      : undefined,
  };
  return buildPredictRequestBody(input);
}

function measurementOf(points: ReturnType<typeof aggregateBenchmarkRows>): ScorecardMeasurement[] {
  return points.map(point => ({
    // The dataset reports TTFT/TPOT in milliseconds and the aggregation helper
    // converts them to seconds; the scorecard speaks milliseconds throughout.
    ttftMs: point.ttftSeconds === null ? null : point.ttftSeconds * 1000,
    tpotMs: point.tpotSeconds === null ? null : point.tpotSeconds * 1000,
    outputTokensPerSecond: point.outputTokensPerSecond,
  }));
}

async function mapWithLimit<T, R>(
  values: readonly T[],
  limit: number,
  mapper: (value: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let nextIndex = 0;
  const workers = Array.from({ length: Math.max(1, Math.min(limit, values.length)) }, async () => {
    for (;;) {
      const index = nextIndex;
      nextIndex += 1;
      if (index >= values.length) return;
      results[index] = await mapper(values[index], index);
    }
  });
  await Promise.all(workers);
  return results;
}

/**
 * The outcome of reading a model's public config.
 *
 * "The host does not have this file" and "the host failed to answer" are
 * different facts: the first is stable, the second must be retried rather than
 * cached as if the model had no config at all.
 */
type ModelConfigLookup =
  | { status: 'found'; config: Record<string, unknown> }
  | { status: 'absent' }
  | { status: 'failed' };

class ModelConfigStore {
  private readonly cache = new Map<string, Promise<ModelConfigLookup>>();

  get(model: string): Promise<ModelConfigLookup> {
    const cached = this.cache.get(model);
    if (cached) return cached;
    const pending = this.fetch(model).then(lookup => {
      // A transient failure must not become a permanent verdict: drop it so
      // the next point asks again.
      if (lookup.status === 'failed') this.cache.delete(model);
      return lookup;
    });
    this.cache.set(model, pending);
    return pending;
  }

  private async fetch(model: string): Promise<ModelConfigLookup> {
    try {
      const response = await fetch(`https://huggingface.co/${model}/raw/main/config.json`, {
        headers: { Accept: 'application/json' },
        signal: AbortSignal.timeout(15_000),
      });
      if (response.status === 404 || response.status === 410) return { status: 'absent' };
      if (!response.ok) return { status: 'failed' };
      const data: unknown = await response.json();
      if (!data || typeof data !== 'object' || Array.isArray(data)) return { status: 'failed' };
      return { status: 'found', config: data as Record<string, unknown> };
    } catch {
      return { status: 'failed' };
    }
  }
}

function cacheKey(
  simulatorId: string,
  simulatorVersion: string | null,
  simulatorRevision: string | null,
  body: Record<string, unknown>,
): string {
  // A revision can change while the release label stays the same, so the
  // revision is part of the identity too. Otherwise a rebuilt simulator would
  // be published with predictions from the build it replaced.
  return hash([simulatorId, simulatorVersion ?? '', simulatorRevision ?? '', JSON.stringify(body)]);
}

function retriable(outcome: PredictOutcome): boolean {
  if (outcome.status !== 'inconclusive') return false;
  // A 422 is a verdict; anything else is us failing to get an answer. A 2xx
  // that did not carry a usable body is a transport failure too.
  return outcome.httpStatus === null
    || outcome.httpStatus === 429
    || outcome.httpStatus >= 500
    || (outcome.httpStatus >= 200 && outcome.httpStatus < 300);
}

async function runPrediction(
  unit: WorkUnit,
  options: Required<Pick<BuildOptions, 'gatewayUrl' | 'timeoutSeconds' | 'concurrency'>>,
  cache: PredictionCache,
  modelConfigs: ModelConfigStore,
): Promise<{ outcome: PredictOutcome; cached: boolean }> {
  const key = cacheKey(unit.plan.simulator, unit.simulatorVersion, unit.simulatorRevision, unit.body);
  const hit = cache.entries[key];
  if (hit && hit.status !== 'inconclusive') {
    return {
      outcome: {
        status: hit.status,
        reason: hit.reason,
        prediction: hit.prediction,
        engineVersion: hit.engineVersion,
        simulator: unit.plan.simulator,
        httpStatus: null,
        durationMs: 0,
      },
      cached: true,
    };
  }
  // An inconclusive entry is a record of our own outage. Reuse it and the next
  // sweep would report the same failure without asking again, so drop it.
  if (hit) delete cache.entries[key];

  const call = (body: Record<string, unknown>) => predictPoint(options.gatewayUrl, body, {
    timeoutSeconds: options.timeoutSeconds,
    expectedSimulator: unit.plan.simulator,
  });

  let outcome = await call(unit.body);
  let durationMs = outcome.durationMs;
  for (let attempt = 1; attempt < MAX_ATTEMPTS && retriable(outcome); attempt += 1) {
    await new Promise(resolve => setTimeout(resolve, RETRY_DELAY_MS * attempt));
    outcome = await call(unit.body);
    durationMs += outcome.durationMs;
  }

  // A 422 that asks for the model configuration is our omission, not the
  // simulator's verdict. Fetch the public config once and try again.
  if (outcome.status === 'unsupported' && isModelConfigRejection(outcome.reason)) {
    const lookup = await modelConfigs.get(unit.model);
    if (lookup.status === 'found') {
      outcome = await call({ ...unit.body, model_config: lookup.config });
      durationMs += outcome.durationMs;
    } else if (lookup.status === 'failed') {
      // The simulator asked for a config we could not read. That is an
      // incomplete request, not a capability verdict, so the point stays
      // inconclusive and is asked again on the next sweep.
      outcome = {
        status: 'inconclusive',
        reason: `Could not read the model configuration for ${unit.model}.`,
        prediction: null,
        engineVersion: null,
        simulator: unit.plan.simulator,
        httpStatus: null,
        durationMs: 0,
      };
    }
  }

  const reported: PredictOutcome = { ...outcome, durationMs };
  // Only a settled answer is reusable. Publishing an outage as a cached
  // verdict would keep it in the artifact long after the service recovered.
  if (reported.status !== 'inconclusive') {
    cache.entries[key] = {
      status: reported.status,
      reason: reported.reason,
      prediction: reported.prediction,
      engineVersion: reported.engineVersion,
    };
  }
  return { outcome: reported, cached: false };
}

/**
 * Predict every point in `work` and return the scored rows.
 *
 * Exposed separately from `buildScorecard` so a shard can do the network work
 * and a merge step can reuse the same point shape.
 */
export async function scoreWorkUnits(
  work: readonly WorkUnit[],
  options: BuildOptions,
  cache: PredictionCache,
): Promise<{
  points: ScorecardPoint[];
  outcomes: Record<PredictStatus | 'cached', number>;
  cacheHits: number;
  cacheMisses: number;
  durations: { simulator: string; durationMs: number }[];
}> {
  const resolved = {
    gatewayUrl: options.gatewayUrl,
    timeoutSeconds: options.timeoutSeconds ?? DEFAULT_TIMEOUT_SECONDS,
    concurrency: options.concurrency ?? DEFAULT_CONCURRENCY,
  };
  const modelConfigs = new ModelConfigStore();
  const outcomes: Record<PredictStatus | 'cached', number> = {
    supported: 0,
    unsupported: 0,
    inconclusive: 0,
    cached: 0,
  };
  let cacheHits = 0;
  let cacheMisses = 0;
  const durations: { simulator: string; durationMs: number }[] = [];

  const points = await mapWithLimit(work, resolved.concurrency, async unit => {
    const { outcome, cached } = await runPrediction(unit, resolved, cache, modelConfigs);
    if (cached) {
      cacheHits += 1;
      outcomes.cached += 1;
    } else {
      cacheMisses += 1;
      outcomes[outcome.status] += 1;
      durations.push({ simulator: unit.plan.simulator, durationMs: outcome.durationMs });
    }

    return {
      identity: unit.identity,
      source: unit.source,
      model: unit.model,
      accelerator: unit.accelerator,
      isl: unit.isl,
      osl: unit.osl,
      tp: unit.tp,
      concurrency: unit.concurrency,
      workload: unit.workload,
      configurationKey: unit.configurationKey,
      measuredRelease: unit.measuredRelease,
      // The request planned the measured release, but the answer says which
      // release actually ran. A simulator that resolved a different one did
      // not predict the measured run and must stay out of the matched slice.
      releaseMatch: unit.releaseMatch
        && outcome.status === 'supported'
        && unit.measuredRelease !== null
        && outcome.engineVersion !== null
        && backendVersionsComparable(outcome.engineVersion, unit.measuredRelease),
      servingMode: unit.servingMode,
      simulator: unit.plan.simulator,
      status: outcome.status,
      reason: outcome.reason,
      measured: unit.measured,
      predicted: outcome.prediction,
      engineVersion: outcome.engineVersion,
    } satisfies ScorecardPoint;
  });

  return { points, outcomes, cacheHits, cacheMisses, durations };
}

/**
 * Load a measured dataset, retrying while the upstream dataset server is busy.
 *
 * The dataset loader already retries a handful of times; this adds a longer,
 * build-only budget on top. A page cannot afford to hold a request for minutes,
 * a scheduled sweep can, and a snapshot with a missing source must say so
 * rather than silently score a narrower population.
 */
async function loadDatasetWithRetry<T>(
  leg: ScorecardSource,
  load: () => Promise<T>,
  log: (message: string) => void,
  delayMs: number = DEFAULT_DATASET_RETRY_DELAY_MS,
): Promise<T> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < DATASET_RETRY_ATTEMPTS; attempt += 1) {
    try {
      return await load();
    } catch (error: unknown) {
      lastError = error;
      if (attempt === DATASET_RETRY_ATTEMPTS - 1) break;
      const detail = error instanceof Error ? error.message : String(error);
      log(`${leg} dataset attempt ${attempt + 1}/${DATASET_RETRY_ATTEMPTS} failed: ${detail}`);
      await new Promise(resolve => setTimeout(resolve, delayMs));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(`${leg} dataset unavailable`);
}

export async function buildScorecard(options: BuildOptions): Promise<BuildResult> {
  const log = options.log ?? (() => {});
  const startedAt = Date.now();
  const sources = options.sources ?? ['redhat', 'inferencex'];
  const incomplete = [...(options.incomplete ?? [])];

  const catalog = await readCatalog(
    options.gatewayUrl,
    options.timeoutSeconds ?? DEFAULT_TIMEOUT_SECONDS,
  );
  if (!catalog.ok) {
    throw new Error(`The simulator catalogue could not be read: ${catalog.error.message}`);
  }
  if (catalog.legacy) {
    throw new Error('The gateway exposes no simulator list, so a scorecard cannot be produced.');
  }
  const supports = catalog.simulators.map(toSupport);
  const available = supports.filter(support => support.available);
  if (available.length === 0) {
    throw new Error('The catalogue reported no available simulator.');
  }
  log(`catalogue: ${available.map(support => `${support.label}${support.version ? ` ${support.version}` : ''}`).join(', ')}`);

  const rows: DatasetRow[] = [];
  const datasets: ScorecardSummary['datasets'] = [];

  if (sources.includes('redhat')) {
    try {
      const dataset = await loadDatasetWithRetry(
        'redhat',
        () => getBenchmarkDataset(),
        log,
        options.datasetRetryDelayMs,
      );
      const redhatRows = dataset.rows.filter(row => rowSource(row) === 'redhat');
      rows.push(...redhatRows);
      datasets.push({
        source: 'redhat',
        id: BENCHMARK_DATASET_ID,
        revision: dataset.source.revision,
        rowCount: redhatRows.length,
      });
      log(`redhat dataset: ${redhatRows.length} rows`);
    } catch (error: unknown) {
      const reason = error instanceof Error ? error.message : 'dataset unavailable';
      incomplete.push({ leg: 'redhat', reason });
      log(`redhat dataset unavailable: ${reason}`);
    }
  }

  if (sources.includes('inferencex')) {
    try {
      const dataset = await loadDatasetWithRetry(
        'inferencex',
        () => getInferenceXDataset(),
        log,
        options.datasetRetryDelayMs,
      );
      rows.push(...dataset.rows);
      datasets.push({
        source: 'inferencex',
        id: INFERENCEX_COMPARISON_DATASET_ID,
        // The snapshot is a file, not a revisioned tree; the retrieval time is
        // the closest honest provenance.
        revision: dataset.source.retrievedAt,
        rowCount: dataset.rows.length,
      });
      log(`inferencex dataset: ${dataset.rows.length} rows`);
    } catch (error: unknown) {
      const reason = error instanceof Error ? error.message : 'snapshot unavailable';
      incomplete.push({ leg: 'inferencex', reason });
      log(`inferencex dataset unavailable: ${reason}`);
    }
  }

  if (rows.length === 0) {
    throw new Error('No measured rows were available; refusing to publish an empty scorecard.');
  }

  const plan: PlannedUnit[] = [];

  for (const group of groupConfigurations(rows).values()) {
    const first = group.rows[0];
    const configurationId = group.configurationId;
    const source = rowSource(first);
    const model = benchmarkFilterValue(first, 'model');
    const accelerator = benchmarkFilterValue(first, 'accelerator');
    const backend = benchmarkFilterValue(first, 'backend');
    const islValue = asNumber(first.isl);
    const oslValue = asNumber(first.osl);
    const tp = asPositiveInteger(first.tp);
    if (!model || !accelerator || !backend || islValue === null || oslValue === null || tp === null) continue;
    if (options.models && options.models.length > 0 && !options.models.includes(model)) continue;

    // Some sweep rows record a variable-length traffic mix (mean prompt and
    // output token counts) rather than a fixed pair. A prediction needs one
    // length, so those runs are outside the comparison instead of being
    // predicted against a shape the measurement never had.
    const shapeReason = islValue > 0 && oslValue > 0
      ? null
      : 'The measured row runs a variable-length traffic mix, so a fixed input/output prediction is not comparable.';
    const isl = islValue;
    const osl = oslValue;

    const normalizedRelease = normalizeBackendVersion(first.backend, first.backend_version);
    // The normalizer uses a sentinel label when a row names no release. That is
    // "unknown", not a release, so it must not enter the matched slice.
    const measuredRelease = normalizedRelease === UNREPORTED_VERSION ? null : normalizedRelease;
    const servingMode = servingModeOf(group.values);
    const topology = measuredPoolTopology(group.values, tp);
    const levels = aggregateBenchmarkRows(group.rows);
    const measurements = measurementOf(levels);

    const run = {
      model,
      accelerator,
      isl,
      osl,
      tp,
      backend,
      release: measuredRelease,
      servingMode,
      topology,
    };

    for (const support of available) {
      const decision = shapeReason
        ? { status: 'ineligible' as const, reason: shapeReason }
        : eligibility(support, run);
      for (let index = 0; index < levels.length; index += 1) {
        const level = levels[index];
        const measured = measurements[index];
        const workload = `${isl}/${osl}`;
        const base = {
          identity: hash([
            source,
            model,
            accelerator,
            configurationId,
            String(isl),
            String(osl),
            String(tp),
            String(level.concurrency),
          ]),
          source,
          model,
          accelerator,
          isl,
          osl,
          tp,
          workload,
          configurationKey: configurationId,
          measuredRelease,
          servingMode,
          measured,
          concurrency: level.concurrency,
        };

        if (decision.status !== 'eligible') {
          plan.push({
            kind: 'skipped',
            unit: { base, simulator: support.id, status: decision.status, reason: decision.reason },
          });
          continue;
        }

        plan.push({
          kind: 'work',
          unit: {
            ...base,
            releaseMatch: decision.plan.releaseMatch,
            plan: decision.plan,
            simulatorVersion: support.version,
            simulatorRevision: support.revision,
            body: requestBody(decision.plan, {
              model,
              systemId: decision.plan.systemId,
              isl,
              osl,
              tp,
              servingMode,
              concurrency: level.concurrency,
              topology,
              backend,
            }, null),
          },
        });
      }
    }
  }

  // Deterministic order makes `limit` reproducible and keeps a shard boundary
  // stable between runs. Rejected points sort with the rest so a shard and a
  // sample both cover the same thing.
  plan.sort((left, right) =>
    identityOf(left).localeCompare(identityOf(right))
    || simulatorOf(left).localeCompare(simulatorOf(right)));

  const limited = options.limit && options.limit > 0 ? plan.slice(0, options.limit) : plan;
  const work = limited.flatMap(entry => (entry.kind === 'work' ? [entry.unit] : []));
  const skipped = limited.flatMap(entry => (entry.kind === 'skipped' ? [entry.unit] : []));
  log(`planned points: ${limited.length}${limited.length < plan.length ? ` of ${plan.length}` : ''}`
    + ` (${work.length} to predict, ${skipped.length} rejected by the catalogue)`);

  const cache = options.cache ?? { entries: {} };
  const { points, outcomes, cacheHits, cacheMisses, durations } = await scoreWorkUnits(work, options, cache);

  const skippedPoints: ScorecardPoint[] = skipped.map(item => ({
    identity: item.base.identity,
    source: item.base.source,
    model: item.base.model,
    accelerator: item.base.accelerator,
    isl: item.base.isl,
    osl: item.base.osl,
    tp: item.base.tp,
    concurrency: item.base.concurrency,
    workload: item.base.workload,
    configurationKey: item.base.configurationKey,
    measuredRelease: item.base.measuredRelease,
    // A catalogue rejection is not a release comparison: the point was never
    // predicted. Keep it out of the matched slice so it cannot inflate coverage
    // for a release the simulator never ran.
    releaseMatch: false,
    servingMode: item.base.servingMode,
    simulator: item.simulator,
    status: item.status,
    reason: item.reason,
    measured: item.base.measured,
    predicted: null,
    engineVersion: null,
  }));

  const allPoints = [...points, ...skippedPoints];
  const answered = points.filter(point => point.status === 'supported').length;
  // A shared count is only meaningful when both sources loaded. Emitting zero
  // for a source that failed would read as "they have nothing in common",
  // which is a different claim from "we could not read one of them".
  const loadedSources = new Set(datasets.map(dataset => dataset.source));
  const sourceOverlap = loadedSources.has('redhat') && loadedSources.has('inferencex')
    ? buildSourceOverlap(rows)
    : undefined;

  const summary: ScorecardSummary = {
    schemaVersion: SCORECARD_SCHEMA_VERSION,
    channel: options.channel,
    generatedAt: new Date().toISOString(),
    gatewayUrl: options.gatewayUrl,
    durationSeconds: Math.round((Date.now() - startedAt) / 1000),
    attemptedPoints: limited.length,
    answeredPoints: answered,
    cachedPoints: cacheHits,
    datasets,
    ...(sourceOverlap ? { sourceOverlap } : {}),
    simulators: available.map(support => ({
      id: support.id,
      label: support.label,
      version: support.version,
      revision: support.revision,
      available: support.available,
    })),
    assumptions: {
      ppSize: 1,
      dpSize: 1,
      moeWidths: 'simulator-resolved',
      fromMeasuredRow: ['model', 'accelerator', 'isl', 'osl', 'tp', 'concurrency', 'backend release'],
    },
    incomplete,
    cells: buildCells(allPoints),
    coverage: buildCoverage(allPoints),
    totals: buildTotals(allPoints),
    headToHead: buildHeadToHead(allPoints),
  };

  const bySimulator = new Map<string, number[]>();
  for (const entry of durations) {
    const values = bySimulator.get(entry.simulator) ?? [];
    values.push(entry.durationMs);
    bySimulator.set(entry.simulator, values);
  }
  const timings = Object.fromEntries(
    [...bySimulator.entries()].map(([simulator, values]) => [simulator, summarizeTimings(values)]),
  );
  const totalTiming = summarizeTimings(durations.map(entry => entry.durationMs));

  return { summary, points: allPoints, cache, cacheHits, cacheMisses, outcomes, timings, totalTiming };
}

function summarizeTimings(values: readonly number[]): TimingSummary {
  if (values.length === 0) return { count: 0, meanMs: 0, p50Ms: 0, p90Ms: 0, maxMs: 0 };
  const sorted = [...values].sort((left, right) => left - right);
  return {
    count: sorted.length,
    meanMs: Math.round(sorted.reduce((sum, value) => sum + value, 0) / sorted.length),
    p50Ms: Math.round(percentile(sorted, 0.5)),
    p90Ms: Math.round(percentile(sorted, 0.9)),
    maxMs: Math.round(sorted[sorted.length - 1]),
  };
}

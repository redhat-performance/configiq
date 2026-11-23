import { z } from 'zod';

/**
 * The simulator scorecard artifact.
 *
 * The scorecard compares measured benchmark points against the predictions a
 * simulator returns for the same configuration. It is computed by a scheduled
 * job and published, never derived in the browser: a full sweep is thousands of
 * simulator calls, and a page cannot run one per visitor.
 *
 * Two things this contract deliberately keeps apart:
 *
 * - `status` separates three different "no number" outcomes. `ineligible` means
 *   the point is outside the comparison at all (the catalogue lists no such
 *   accelerator, model, or backend). `unsupported` means the catalogue lists
 *   the part but the run is beyond what it can model. `inconclusive` means we
 *   asked and never got a verdict — a timeout, a rate limit, a service error.
 *   Folding the last into either of the others would report our own outage as
 *   either a capability gap or a hardware gap.
 * - `releaseMatch` is a dimension, not a filter applied when scoring. A
 *   simulator that models one engine release is still scored against runs on
 *   other releases; those points are simply tagged so the page can default to
 *   the like-for-release comparison without pretending the rest do not exist.
 */

export const SCORECARD_SCHEMA_VERSION = 1;

export const SCORECARD_SOURCES = ['redhat', 'inferencex'] as const;
export type ScorecardSource = (typeof SCORECARD_SOURCES)[number];

/** The source filter the page offers: one measured source, or both. */
export const SCORECARD_SOURCE_FILTERS = ['all', ...SCORECARD_SOURCES] as const;
export type ScorecardSourceFilter = (typeof SCORECARD_SOURCE_FILTERS)[number];

export const SCORECARD_METRICS = ['ttft', 'tpot'] as const;
export type ScorecardMetric = (typeof SCORECARD_METRICS)[number];

/** Headline, robust, tail, and signed error. Bias is a direction, not a size. */
export const SCORECARD_STATISTICS = ['mape', 'mdape', 'p90', 'bias'] as const;
export type ScorecardStatistic = (typeof SCORECARD_STATISTICS)[number];

/**
 * Which points an aggregate covers. `matched` keeps only points whose measured
 * engine release is the one the simulator predicted with; `any` keeps every
 * point that produced both a measurement and a prediction.
 */
export const SCORECARD_RELEASE_MATCH = ['matched', 'any'] as const;
export type ScorecardReleaseMatch = (typeof SCORECARD_RELEASE_MATCH)[number];

export const SCORECARD_CHANNELS = ['dev', 'prod'] as const;
export type ScorecardChannel = (typeof SCORECARD_CHANNELS)[number];

export const POINT_STATUSES = ['supported', 'unsupported', 'inconclusive', 'ineligible'] as const;
export type PointStatus = (typeof POINT_STATUSES)[number];

export const SERVING_MODES = ['agg', 'disagg'] as const;
export type ScorecardServingMode = (typeof SERVING_MODES)[number];

/** One simulator as the catalogue described it when the sweep ran. */
export interface ScorecardSimulator {
  id: string;
  label: string;
  /** Release of the simulator itself, not the engine it models. */
  version: string | null;
  revision: string | null;
  available: boolean;
}

/** One measured dataset the sweep read. */
export interface ScorecardDataset {
  source: ScorecardSource;
  id: string;
  revision: string | null;
  rowCount: number;
}

export interface ScorecardProvenance {
  schemaVersion: typeof SCORECARD_SCHEMA_VERSION;
  channel: ScorecardChannel;
  generatedAt: string;
  /** Gateway the predictions came from. Never inferred from the page's host. */
  gatewayUrl: string;
  durationSeconds: number;
  /** Points attempted (one measured configuration level, per simulator). */
  attemptedPoints: number;
  /** Points the simulator answered. */
  answeredPoints: number;
  /** Points served from the incremental prediction cache. */
  cachedPoints: number;
  datasets: ScorecardDataset[];
  simulators: ScorecardSimulator[];
  /**
   * Assumptions the sweep could not read from the measured row. They make the
   * error a statement about prediction agreement under these inputs, not about
   * the simulator's absolute accuracy.
   */
  assumptions: {
    ppSize: number;
    dpSize: number;
    moeWidths: 'simulator-resolved';
    /** Fixed measured fields the sweep does send, named for the reader. */
    fromMeasuredRow: string[];
  };
  /** Shards that did not finish, so a partial snapshot is never silent. */
  incomplete: { leg: string; reason: string }[];
}

/** A measured concurrency level, with only the fields the scorecard reads. */
export interface ScorecardMeasurement {
  ttftMs: number | null;
  tpotMs: number | null;
  outputTokensPerSecond: number | null;
}

export interface ScorecardPrediction {
  ttftMs: number;
  tpotMs: number;
  requestLatencyMs: number | null;
  outputTokensPerSecond: number | null;
}

/**
 * One (measured level, simulator) attempt.
 *
 * `identity` is stable across simulators: it names the measured point and the
 * concurrency level, so two simulator rows can be paired for head-to-head.
 */
export interface ScorecardPoint {
  identity: string;
  source: ScorecardSource;
  model: string;
  accelerator: string;
  isl: number;
  osl: number;
  tp: number;
  concurrency: number;
  /** Display identity for the workload, `isl/osl`. */
  workload: string;
  configurationKey: string;
  /** Measured engine release, normalized for display. */
  measuredRelease: string | null;
  /** Whether the simulator predicted with the measured release. */
  releaseMatch: boolean;
  servingMode: ScorecardServingMode;
  simulator: string;
  status: PointStatus;
  /** Why the point is not supported or inconclusive. Null when supported. */
  reason: string | null;
  measured: ScorecardMeasurement | null;
  predicted: ScorecardPrediction | null;
  /** Engine release the response reported, when it named one. */
  engineVersion: string | null;
}

/** An aggregate over configurations, not over points. */
export interface ScorecardCell {
  simulator: string;
  source: ScorecardSource;
  model: string;
  accelerator: string;
  workload: string;
  releaseMatch: ScorecardReleaseMatch;
  metric: ScorecardMetric;
  statistic: ScorecardStatistic;
  value: number;
  /** Number of configurations the value covers. */
  configurations: number;
  /** Number of measured levels behind those configurations. */
  points: number;
}

/** How many attempts of each kind a slice produced. */
export interface ScorecardCoverage {
  simulator: string;
  source: ScorecardSource;
  model: string;
  accelerator: string;
  releaseMatch: ScorecardReleaseMatch;
  supported: number;
  unsupported: number;
  inconclusive: number;
  /** Points the catalogue puts outside the comparison entirely. */
  ineligible: number;
  total: number;
}

/**
 * One statistic rolled up over every measured configuration for a simulator.
 *
 * The page cannot produce these from `cells`: an average of per-slice MAPEs is
 * not the MAPE over all configurations. So the sweep publishes the rollup it
 * actually computed, and the page renders it.
 */
export interface ScorecardTotal {
  simulator: string;
  /** `all` combines the measured sources; a source entry is that source only. */
  source: ScorecardSourceFilter;
  metric: ScorecardMetric;
  releaseMatch: ScorecardReleaseMatch;
  statistic: ScorecardStatistic;
  value: number;
  configurations: number;
  points: number;
}

/** A paired comparison between two simulators on the same measured levels. */
export interface ScorecardHeadToHead {
  source: ScorecardSource;
  model: string;
  accelerator: string;
  workload: string;
  metric: ScorecardMetric;
  /** Simulator ids, sorted so a row is stable across snapshots. */
  pair: [string, string];
  paired: number;
  firstWins: number;
  secondWins: number;
  ties: number;
  /** Median of (first APE − second APE) in percentage points. */
  medianDeltaPct: number;
}

/** Everything the page needs for its first render. */
export interface ScorecardSummary extends ScorecardProvenance {
  cells: ScorecardCell[];
  coverage: ScorecardCoverage[];
  totals: ScorecardTotal[];
  headToHead: ScorecardHeadToHead[];
}

/** The per-point detail, fetched only when a view needs it. */
export interface ScorecardPoints {
  schemaVersion: typeof SCORECARD_SCHEMA_VERSION;
  channel: ScorecardChannel;
  generatedAt: string;
  points: ScorecardPoint[];
}

// ── Schemas ────────────────────────────────────────────────────────────────
//
// The published artifact is external input to every reader. Validate it rather
// than casting: a stale or hand-edited file must fail closed, not render a
// half-empty scorecard that looks like a result.

const simulatorSchema = z.object({
  id: z.string().min(1),
  label: z.string().min(1),
  version: z.string().nullable(),
  revision: z.string().nullable(),
  available: z.boolean(),
});

const datasetSchema = z.object({
  source: z.enum(SCORECARD_SOURCES),
  id: z.string().min(1),
  revision: z.string().nullable(),
  rowCount: z.number().int().nonnegative(),
});

const provenanceSchema = z.object({
  schemaVersion: z.literal(SCORECARD_SCHEMA_VERSION),
  channel: z.enum(SCORECARD_CHANNELS),
  generatedAt: z.string().min(1),
  gatewayUrl: z.string().min(1),
  durationSeconds: z.number().nonnegative(),
  attemptedPoints: z.number().int().nonnegative(),
  answeredPoints: z.number().int().nonnegative(),
  cachedPoints: z.number().int().nonnegative(),
  datasets: z.array(datasetSchema),
  simulators: z.array(simulatorSchema),
  assumptions: z.object({
    ppSize: z.number().int().positive(),
    dpSize: z.number().int().positive(),
    moeWidths: z.literal('simulator-resolved'),
    fromMeasuredRow: z.array(z.string()),
  }),
  incomplete: z.array(z.object({ leg: z.string(), reason: z.string() })),
});

const measurementSchema = z.object({
  ttftMs: z.number().nullable(),
  tpotMs: z.number().nullable(),
  outputTokensPerSecond: z.number().nullable(),
});

const predictionSchema = z.object({
  ttftMs: z.number(),
  tpotMs: z.number(),
  requestLatencyMs: z.number().nullable(),
  outputTokensPerSecond: z.number().nullable(),
});

export const scorecardPointSchema = z.object({
  identity: z.string().min(1),
  source: z.enum(SCORECARD_SOURCES),
  model: z.string().min(1),
  accelerator: z.string().min(1),
  isl: z.number().int().nonnegative(),
  osl: z.number().int().nonnegative(),
  tp: z.number().int().positive(),
  concurrency: z.number().int().positive(),
  workload: z.string().min(1),
  configurationKey: z.string().min(1),
  measuredRelease: z.string().nullable(),
  releaseMatch: z.boolean(),
  servingMode: z.enum(SERVING_MODES),
  simulator: z.string().min(1),
  status: z.enum(POINT_STATUSES),
  reason: z.string().nullable(),
  measured: measurementSchema.nullable(),
  predicted: predictionSchema.nullable(),
  engineVersion: z.string().nullable(),
});

export const scorecardCellSchema = z.object({
  simulator: z.string().min(1),
  source: z.enum(SCORECARD_SOURCES),
  model: z.string().min(1),
  accelerator: z.string().min(1),
  workload: z.string().min(1),
  releaseMatch: z.enum(SCORECARD_RELEASE_MATCH),
  metric: z.enum(SCORECARD_METRICS),
  statistic: z.enum(SCORECARD_STATISTICS),
  value: z.number(),
  configurations: z.number().int().nonnegative(),
  points: z.number().int().nonnegative(),
});

export const scorecardCoverageSchema = z.object({
  simulator: z.string().min(1),
  source: z.enum(SCORECARD_SOURCES),
  model: z.string().min(1),
  accelerator: z.string().min(1),
  releaseMatch: z.enum(SCORECARD_RELEASE_MATCH),
  supported: z.number().int().nonnegative(),
  unsupported: z.number().int().nonnegative(),
  inconclusive: z.number().int().nonnegative(),
  ineligible: z.number().int().nonnegative(),
  total: z.number().int().nonnegative(),
});

export const scorecardTotalSchema = z.object({
  simulator: z.string().min(1),
  source: z.enum(SCORECARD_SOURCE_FILTERS),
  metric: z.enum(SCORECARD_METRICS),
  releaseMatch: z.enum(SCORECARD_RELEASE_MATCH),
  statistic: z.enum(SCORECARD_STATISTICS),
  value: z.number(),
  configurations: z.number().int().nonnegative(),
  points: z.number().int().nonnegative(),
});

export const scorecardHeadToHeadSchema = z.object({
  source: z.enum(SCORECARD_SOURCES),
  model: z.string().min(1),
  accelerator: z.string().min(1),
  workload: z.string().min(1),
  metric: z.enum(SCORECARD_METRICS),
  pair: z.tuple([z.string().min(1), z.string().min(1)]),
  paired: z.number().int().nonnegative(),
  firstWins: z.number().int().nonnegative(),
  secondWins: z.number().int().nonnegative(),
  ties: z.number().int().nonnegative(),
  medianDeltaPct: z.number(),
});

export const scorecardSummarySchema = provenanceSchema.extend({
  cells: z.array(scorecardCellSchema),
  coverage: z.array(scorecardCoverageSchema),
  totals: z.array(scorecardTotalSchema),
  headToHead: z.array(scorecardHeadToHeadSchema),
});

export const scorecardPointsSchema = z.object({
  schemaVersion: z.literal(SCORECARD_SCHEMA_VERSION),
  channel: z.enum(SCORECARD_CHANNELS),
  generatedAt: z.string().min(1),
  points: z.array(scorecardPointSchema),
});

export type ScorecardSummaryInput = z.input<typeof scorecardSummarySchema>;

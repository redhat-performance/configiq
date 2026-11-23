import { median } from '@/lib/scorecard/metrics';
import type {
  ScorecardCell,
  ScorecardCoverage,
  ScorecardHeadToHead,
  ScorecardMetric,
  ScorecardReleaseMatch,
  ScorecardSourceFilter,
  ScorecardStatistic,
  ScorecardSummary,
} from '@/lib/scorecard/types';

/**
 * Pure selectors for the scorecard page.
 *
 * The page never recomputes a statistic. It selects the slice the sweep already
 * published, so a reader on any client sees the number the job produced.
 */

export interface CoverageCounts {
  supported: number;
  unsupported: number;
  inconclusive: number;
  ineligible: number;
  total: number;
}

export interface StatisticSet {
  mape: number;
  mdape: number;
  p90: number;
  bias: number;
}

export interface TotalsSlice {
  values: StatisticSet;
  configurations: number;
  points: number;
}

export interface SelectTotalsOptions {
  simulator: string;
  source: ScorecardSourceFilter;
  metric: ScorecardMetric;
  releaseMatch: ScorecardReleaseMatch;
}

/** The four statistics for one simulator slice, or null when it has no evidence. */
export function selectTotals(summary: ScorecardSummary, options: SelectTotalsOptions): TotalsSlice | null {
  const rows = summary.totals.filter(total =>
    total.simulator === options.simulator
    && total.source === options.source
    && total.metric === options.metric
    && total.releaseMatch === options.releaseMatch);
  if (rows.length === 0) return null;

  const values: Partial<StatisticSet> = {};
  let configurations = 0;
  let points = 0;
  for (const row of rows) {
    values[row.statistic] = row.value;
    configurations = Math.max(configurations, row.configurations);
    points = Math.max(points, row.points);
  }
  if (values.mape === undefined) return null;
  return {
    values: {
      mape: values.mape,
      mdape: values.mdape ?? Number.NaN,
      p90: values.p90 ?? Number.NaN,
      bias: values.bias ?? Number.NaN,
    },
    configurations,
    points,
  };
}

/** Coverage for one simulator slice. Counts add, so this sum is exact. */
export function sumCoverage(
  coverage: readonly ScorecardCoverage[],
  options: { simulator: string; source: ScorecardSourceFilter; releaseMatch: ScorecardReleaseMatch },
): CoverageCounts {
  const counts: CoverageCounts = { supported: 0, unsupported: 0, inconclusive: 0, ineligible: 0, total: 0 };
  for (const row of coverage) {
    if (row.simulator !== options.simulator) continue;
    if (row.releaseMatch !== options.releaseMatch) continue;
    if (options.source !== 'all' && row.source !== options.source) continue;
    counts.supported += row.supported;
    counts.unsupported += row.unsupported;
    counts.inconclusive += row.inconclusive;
    counts.ineligible += row.ineligible;
  }
  recomputeTotal(counts);
  return counts;
}

function recomputeTotal(counts: CoverageCounts): void {
  counts.total = counts.supported + counts.unsupported + counts.inconclusive + counts.ineligible;
}

export interface MatrixRow {
  key: string;
  source: string;
  model: string;
  accelerator: string;
  workload: string;
  /** One entry per simulator, absent when that simulator has no cell here. */
  values: Map<string, ScorecardCell>;
}

export interface MatrixOptions {
  metric: ScorecardMetric;
  statistic: ScorecardStatistic;
  releaseMatch: ScorecardReleaseMatch;
  source: ScorecardSourceFilter;
  /** An exact model name, or an empty string for every model. */
  model: string;
  /** An exact accelerator name, or an empty string for every GPU. */
  gpu: string;
  /** An exact `input/output` workload, or an empty string for all. */
  workload: string;
}

/** The values a reader can choose, taken from the slice they are looking at. */
export interface FilterOptions {
  models: string[];
  accelerators: string[];
  workloads: string[];
}

export function buildFilterOptions(
  summary: ScorecardSummary,
  options: Omit<MatrixOptions, 'model' | 'gpu' | 'workload'>,
): FilterOptions {
  const models = new Set<string>();
  const accelerators = new Set<string>();
  const workloads = new Set<string>();

  for (const cell of summary.cells) {
    if (cell.metric !== options.metric) continue;
    if (cell.statistic !== options.statistic) continue;
    if (cell.releaseMatch !== options.releaseMatch) continue;
    if (options.source !== 'all' && cell.source !== options.source) continue;
    models.add(cell.model);
    accelerators.add(cell.accelerator);
    workloads.add(cell.workload);
  }

  return {
    models: [...models].sort((left, right) => left.localeCompare(right)),
    accelerators: [...accelerators].sort((left, right) => left.localeCompare(right)),
    workloads: [...workloads].sort(compareWorkload),
  };
}

function sourceLabel(source: string): string {
  return source === 'inferencex' ? 'InferenceX' : 'Red Hat';
}

/**
 * One row per measured configuration slice, with a column per simulator.
 *
 * A slice is (source, model, accelerator, workload), which is what a reader
 * compares: the same measured run, predicted by each simulator. A simulator
 * that produced no cell for a slice leaves a gap rather than a zero.
 */
export function buildMatrix(summary: ScorecardSummary, options: MatrixOptions): MatrixRow[] {
  const rows = new Map<string, MatrixRow>();

  for (const cell of summary.cells) {
    if (cell.metric !== options.metric) continue;
    if (cell.statistic !== options.statistic) continue;
    if (cell.releaseMatch !== options.releaseMatch) continue;
    if (options.source !== 'all' && cell.source !== options.source) continue;
    if (options.model && cell.model !== options.model) continue;
    if (options.gpu && cell.accelerator !== options.gpu) continue;
    if (options.workload && cell.workload !== options.workload) continue;

    const key = [cell.source, cell.model, cell.accelerator, cell.workload].join('\u0000');
    const row = rows.get(key) ?? {
      key,
      source: sourceLabel(cell.source),
      model: cell.model,
      accelerator: cell.accelerator,
      workload: cell.workload,
      values: new Map<string, ScorecardCell>(),
    };
    row.values.set(cell.simulator, cell);
    rows.set(key, row);
  }

  return [...rows.values()].sort((left, right) =>
    left.source.localeCompare(right.source)
    || left.model.localeCompare(right.model)
    || left.accelerator.localeCompare(right.accelerator)
    || compareWorkload(left.workload, right.workload));
}

export interface ChartRow {
  key: string;
  /** Axis label: model, accelerator, and workload, shortened for a chart. */
  label: string;
  /** One value per simulator, absent when that simulator has no cell here. */
  values: Map<string, number>;
}

export interface ChartRows {
  /** The worst configurations, largest absolute value first. */
  shown: ChartRow[];
  /** How many configurations had at least one value. */
  scored: number;
}

const MODEL_LABEL_CHARS = 26;

/**
 * Turn matrix rows into the chart's categories.
 *
 * The chart is a view of the same cells the table lists, not a second
 * aggregation: one bar per published cell. It ranks by absolute value so the
 * configurations that dominate the mean appear first, and caps the count so
 * labels stay legible. The table behind the flip still lists every row.
 */
export function buildChartRows(
  rows: readonly MatrixRow[],
  simulatorIds: readonly string[],
  limit: number,
): ChartRows {
  const scored = rows
    .map(row => {
      const values = new Map<string, number>();
      for (const id of simulatorIds) {
        const cell = row.values.get(id);
        if (cell) values.set(id, cell.value);
      }
      return { row, values };
    })
    .filter(entry => entry.values.size > 0);

  // Category labels double as chart keys. Only the rare label shared by several
  // rows gains its source, so the common case stays short.
  const labelCounts = new Map<string, number>();
  for (const { row } of scored) {
    const label = chartLabel(row);
    labelCounts.set(label, (labelCounts.get(label) ?? 0) + 1);
  }
  const labelFor = (row: MatrixRow) => {
    const label = chartLabel(row);
    return (labelCounts.get(label) ?? 0) > 1
      ? `${label} · ${row.source}`
      : label;
  };

  const ranked = [...scored].sort((left, right) =>
    magnitude(right.values) - magnitude(left.values)
    || labelFor(left.row).localeCompare(labelFor(right.row)));

  return {
    scored: scored.length,
    shown: ranked.slice(0, Math.max(1, limit)).map(({ row, values }) => ({
      key: row.key,
      label: labelFor(row),
      values,
    })),
  };
}

function magnitude(values: Map<string, number>): number {
  let largest = 0;
  for (const value of values.values()) {
    if (Number.isFinite(value)) largest = Math.max(largest, Math.abs(value));
  }
  return largest;
}

function chartLabel(row: MatrixRow): string {
  const model = row.model.length > MODEL_LABEL_CHARS
    ? `${row.model.slice(0, MODEL_LABEL_CHARS - 1)}…`
    : row.model;
  return `${model} · ${row.accelerator} · ${row.workload}`;
}

/** `8000/1000` sorts above `128/128`, unlike a plain string compare. */
function compareWorkload(left: string, right: string): number {
  const [leftIn, leftOut] = left.split('/').map(Number);
  const [rightIn, rightOut] = right.split('/').map(Number);
  if ([leftIn, leftOut, rightIn, rightOut].every(Number.isFinite)) {
    return (leftIn - rightIn) || (leftOut - rightOut);
  }
  return left.localeCompare(right);
}

/** The head-to-head slice every view reads: one metric, one measured source. */
export interface HeadToHeadOptions {
  metric: ScorecardMetric;
  source: ScorecardSourceFilter;
}

export interface HeadToHeadPairOption {
  key: string;
  pair: [string, string];
  /** Measured configurations that compared this pair. */
  configurations: number;
  paired: number;
}

export type HeadToHeadDimension = 'model' | 'accelerator';

export interface HeadToHeadCategory {
  key: string;
  label: string;
  configurations: number;
  paired: number;
  ties: number;
  /** Mean per-configuration win share, so one long sweep cannot outvote one run. */
  firstShare: number;
  secondShare: number;
  /** Per-configuration first-share range; null when one configuration is the range. */
  minShare: number | null;
  maxShare: number | null;
  /** Median gap between the pair, in percentage points of APE. */
  medianGapPct: number;
}

export interface HeadToHeadSummary {
  categories: HeadToHeadCategory[];
  configurations: number;
  paired: number;
  ties: number;
  firstShare: number;
  secondShare: number;
  medianGapPct: number;
}

function pairKey(pair: readonly [string, string]): string {
  // Readable and printable, because it doubles as the pair selector's value.
  return pair.join('|');
}

export interface HeadToHeadRowFilter extends HeadToHeadOptions {
  /** One pair to select, or null/omitted for every pair in the slice. */
  pair?: readonly [string, string] | null;
}

/**
 * The head-to-head rows for a metric, source, and pair.
 *
 * Exported so the table behind the flip and the chart on its front select the
 * same rows. Two filters that drifted apart would let the chart and table
 * disagree about what the section is showing.
 */
export function selectHeadToHeadRows(
  rows: readonly ScorecardHeadToHead[],
  options: HeadToHeadRowFilter,
): ScorecardHeadToHead[] {
  const key = options.pair ? pairKey(options.pair) : null;
  return rows.filter(row =>
    row.metric === options.metric
    && (options.source === 'all' || row.source === options.source)
    && row.paired > 0
    && (key === null || pairKey(row.pair) === key));
}

/**
 * Every simulator pair the selected snapshot slice compared.
 *
 * The published artifact decides which pairs exist. The page shows a pair
 * selector only when there is more than one, so a deployment with two
 * simulators reads as a single head-to-head rather than a one-option control.
 */
export function buildHeadToHeadPairs(
  rows: readonly ScorecardHeadToHead[],
  options: HeadToHeadOptions,
): HeadToHeadPairOption[] {
  const pairs = new Map<string, HeadToHeadPairOption>();
  for (const row of selectHeadToHeadRows(rows, options)) {
    const key = pairKey(row.pair);
    const entry = pairs.get(key) ?? { key, pair: row.pair, configurations: 0, paired: 0 };
    entry.configurations += 1;
    entry.paired += row.paired;
    pairs.set(key, entry);
  }
  return [...pairs.values()].sort((left, right) => left.key.localeCompare(right.key));
}

export interface HeadToHeadSummaryOptions extends HeadToHeadOptions {
  pair: readonly [string, string] | null;
  dimension: HeadToHeadDimension;
}

/**
 * Roll one pair up by model or by GPU.
 *
 * A row in the artifact is one measured configuration. Its win share is the
 * share of paired concurrency levels that configuration won. A category's
 * share is the mean of those configuration shares — equal weight per
 * configuration — not the pooled level count: a run swept at twelve levels
 * must not outvote one swept at three. The candle is the range across those
 * configuration shares, and it is omitted when one configuration is the whole
 * category, because a range over one value would imply evidence that is not
 * there.
 */
export function buildHeadToHeadSummary(
  rows: readonly ScorecardHeadToHead[],
  options: HeadToHeadSummaryOptions,
): HeadToHeadSummary {
  if (!options.pair) return emptyHeadToHeadSummary();
  const selected = selectHeadToHeadRows(rows, options);
  if (selected.length === 0) return emptyHeadToHeadSummary();

  const groups = new Map<string, ScorecardHeadToHead[]>();
  for (const row of selected) {
    const label = row[options.dimension];
    const group = groups.get(label) ?? [];
    group.push(row);
    groups.set(label, group);
  }

  const categories = [...groups.entries()]
    .map(([label, group]) => summarizeHeadToHeadCategory(label, group))
    // Most evidence first: a category with one configuration sits last, where
    // its "1 configuration" note reads as the caveat it is.
    .sort((left, right) =>
      right.configurations - left.configurations
      || left.label.localeCompare(right.label));

  return {
    categories,
    configurations: selected.length,
    paired: sum(selected.map(row => row.paired)),
    ties: sum(selected.map(row => row.ties)),
    firstShare: mean(selected.map(row => winShare(row.firstWins, row.paired))),
    secondShare: mean(selected.map(row => winShare(row.secondWins, row.paired))),
    medianGapPct: median(selected.map(row => row.medianDeltaPct).sort((a, b) => a - b)),
  };
}

function summarizeHeadToHeadCategory(
  label: string,
  rows: readonly ScorecardHeadToHead[],
): HeadToHeadCategory {
  const shares = rows.map(row => winShare(row.firstWins, row.paired));
  const hasRange = rows.length > 1;
  return {
    key: label,
    label,
    configurations: rows.length,
    paired: sum(rows.map(row => row.paired)),
    ties: sum(rows.map(row => row.ties)),
    firstShare: mean(shares),
    secondShare: mean(rows.map(row => winShare(row.secondWins, row.paired))),
    minShare: hasRange ? Math.min(...shares) : null,
    maxShare: hasRange ? Math.max(...shares) : null,
    medianGapPct: median(rows.map(row => row.medianDeltaPct).sort((a, b) => a - b)),
  };
}

function emptyHeadToHeadSummary(): HeadToHeadSummary {
  return {
    categories: [],
    configurations: 0,
    paired: 0,
    ties: 0,
    firstShare: 0,
    secondShare: 0,
    medianGapPct: 0,
  };
}

function winShare(wins: number, paired: number): number {
  return paired > 0 ? wins / paired : 0;
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function mean(values: readonly number[]): number {
  return values.length > 0 ? sum(values) / values.length : 0;
}

/** A percentage with one decimal, or an em dash when there is no evidence. */
export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const percent = value * 100;
  if (Math.abs(percent) >= 1000) return `${Math.round(percent).toLocaleString('en-US')}%`;
  return `${percent.toFixed(percent >= 100 || percent <= -100 ? 0 : 1)}%`;
}

/** A signed percentage, for bias: the sign is the point. */
export function formatSignedPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const percent = value * 100;
  const magnitude = Math.abs(percent) >= 100 ? Math.round(Math.abs(percent)) : Number(Math.abs(percent).toFixed(1));
  return `${percent > 0 ? '+' : percent < 0 ? '−' : ''}${magnitude}%`;
}

/** How the page headline should describe a slice with no evidence. */
export function emptyReason(
  coverage: CoverageCounts,
): string | null {
  if (coverage.supported > 0) return null;
  if (coverage.unsupported > 0) return 'The catalogue lists these parts but no measured run was within what this simulator models.';
  if (coverage.inconclusive > 0) return 'Every attempt failed to return a prediction.';
  if (coverage.ineligible > 0) return 'These measured runs are outside this simulator\'s catalogue.';
  return 'No measured runs matched this view.';
}

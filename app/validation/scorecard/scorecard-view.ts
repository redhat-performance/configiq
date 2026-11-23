import type {
  ScorecardCell,
  ScorecardCoverage,
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

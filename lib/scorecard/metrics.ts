import type {
  ScorecardMetric,
  ScorecardPoint,
  ScorecardSource,
} from './types';

/**
 * Error arithmetic for the scorecard.
 *
 * Everything here is pure so the same functions run in the build job and in
 * the page's head-to-head view. Two rules matter more than the formulas:
 *
 * 1. Aggregate over configurations, then across configurations. A run swept at
 *    twelve concurrency levels must not outvote one swept at three.
 * 2. Never score a point whose measured value is too small to divide by. A
 *    0.3 ms TPOT is measurement noise, and dividing by it produces a 900%
 *    error that dominates every mean.
 */

/**
 * Smallest measured value an error is computed from, in milliseconds.
 *
 * These are floors for the denominator, not quality thresholds: they only
 * exclude values where a sub-millisecond difference would swamp the ratio.
 */
export const MIN_MEASURED_MS: Record<ScorecardMetric, number> = {
  ttft: 1,
  tpot: 0.05,
};

/** Signed relative error, or null when the measured value cannot be divided by. */
export function signedRelativeError(
  predictedMs: number,
  measuredMs: number,
  metric: ScorecardMetric,
): number | null {
  if (!Number.isFinite(predictedMs) || !Number.isFinite(measuredMs)) return null;
  if (measuredMs <= 0 || measuredMs < MIN_MEASURED_MS[metric]) return null;
  if (predictedMs < 0) return null;
  return (predictedMs - measuredMs) / measuredMs;
}

/** Absolute percentage error, or null when the pair cannot be scored. */
export function percentageError(
  predictedMs: number,
  measuredMs: number,
  metric: ScorecardMetric,
): number | null {
  const signed = signedRelativeError(predictedMs, measuredMs, metric);
  return signed === null ? null : Math.abs(signed);
}

/** One scored (measured level, simulator) pair, already validated. */
export interface Observation {
  configurationKey: string;
  metric: ScorecardMetric;
  predictedMs: number;
  measuredMs: number;
}

export interface AggregateStats {
  /** Macro mean of per-configuration mean APE. */
  mape: number;
  /** Macro median of per-configuration mean APE. */
  mdape: number;
  /** 90th percentile of per-configuration mean APE. */
  p90: number;
  /** Macro mean of per-configuration mean signed error. Negative = under-prediction. */
  bias: number;
  configurations: number;
  points: number;
}

/** Nearest-rank percentile on an ascending array. */
export function percentile(sorted: readonly number[], fraction: number): number {
  if (sorted.length === 0) return Number.NaN;
  const rank = Math.ceil(fraction * sorted.length);
  return sorted[Math.min(sorted.length, Math.max(1, rank)) - 1];
}

/** Median of an ascending array. */
export function median(sorted: readonly number[]): number {
  if (sorted.length === 0) return Number.NaN;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/**
 * Aggregate observations into the four published statistics.
 *
 * Returns null when no configuration produced a scorable point, so a cell with
 * no evidence is absent rather than reported as zero error.
 */
export function aggregateObservations(observations: readonly Observation[]): AggregateStats | null {
  const byConfiguration = new Map<string, { apes: number[]; signed: number[] }>();

  for (const observation of observations) {
    const signed = signedRelativeError(observation.predictedMs, observation.measuredMs, observation.metric);
    if (signed === null) continue;
    const group = byConfiguration.get(observation.configurationKey) ?? { apes: [], signed: [] };
    group.apes.push(Math.abs(signed));
    group.signed.push(signed);
    byConfiguration.set(observation.configurationKey, group);
  }

  if (byConfiguration.size === 0) return null;

  const configurationApe = [...byConfiguration.values()].map(group => mean(group.apes)).sort((a, b) => a - b);
  const configurationBias = [...byConfiguration.values()].map(group => mean(group.signed));
  const points = [...byConfiguration.values()].reduce((sum, group) => sum + group.apes.length, 0);

  return {
    mape: mean(configurationApe),
    mdape: median(configurationApe),
    p90: percentile(configurationApe, 0.9),
    bias: mean(configurationBias),
    configurations: configurationApe.length,
    points,
  };
}

/** The measured and predicted value one point contributes for a metric. */
export function pointValues(
  point: ScorecardPoint,
  metric: ScorecardMetric,
): { predictedMs: number; measuredMs: number } | null {
  if (point.status !== 'supported' || !point.predicted || !point.measured) return null;
  const measuredMs = metric === 'ttft' ? point.measured.ttftMs : point.measured.tpotMs;
  const predictedMs = metric === 'ttft' ? point.predicted.ttftMs : point.predicted.tpotMs;
  if (measuredMs === null || !Number.isFinite(predictedMs)) return null;
  if (signedRelativeError(predictedMs, measuredMs, metric) === null) return null;
  return { predictedMs, measuredMs };
}

export interface HeadToHeadRow {
  source: ScorecardSource;
  model: string;
  accelerator: string;
  workload: string;
  metric: ScorecardMetric;
  /** Simulator ids, in sorted order so a row is stable across snapshots. */
  pair: [string, string];
  /** Measured levels both simulators scored. */
  paired: number;
  /** Levels where the first simulator's error was smaller. */
  firstWins: number;
  secondWins: number;
  ties: number;
  /**
   * Median of (first APE − second APE) in percentage points. Negative means
   * the first simulator was closer on a typical paired level.
   */
  medianDeltaPct: number;
}

function roundedPct(value: number): number {
  return Math.round(value * 10) / 10;
}

/**
 * Compare simulators only where they predicted the same measured level.
 *
 * Comparing two independent MAPEs over different coverage is not a comparison:
 * the simulator that skipped the hard points would look better. This runs over
 * the paired subset instead, and reports how large that subset is.
 */
export function headToHead(points: readonly ScorecardPoint[]): HeadToHeadRow[] {
  interface Entry {
    source: ScorecardSource;
    model: string;
    accelerator: string;
    workload: string;
    metric: ScorecardMetric;
    pair: [string, string];
    deltas: number[];
    firstWins: number;
    secondWins: number;
    ties: number;
  }

  const byIdentity = new Map<string, ScorecardPoint[]>();
  for (const point of points) {
    if (point.status !== 'supported') continue;
    const group = byIdentity.get(point.identity) ?? [];
    group.push(point);
    byIdentity.set(point.identity, group);
  }

  const rows = new Map<string, Entry>();

  for (const group of byIdentity.values()) {
    if (group.length < 2) continue;
    const bySimulator = new Map(group.map(point => [point.simulator, point]));
    const simulators = [...bySimulator.keys()].sort((a, b) => a.localeCompare(b));

    for (let first = 0; first < simulators.length; first += 1) {
      for (let second = first + 1; second < simulators.length; second += 1) {
        const firstPoint = bySimulator.get(simulators[first])!;
        const secondPoint = bySimulator.get(simulators[second])!;
        for (const metric of ['ttft', 'tpot'] as const) {
          const firstValues = pointValues(firstPoint, metric);
          const secondValues = pointValues(secondPoint, metric);
          if (!firstValues || !secondValues) continue;

          const firstApe = percentageError(firstValues.predictedMs, firstValues.measuredMs, metric)!;
          const secondApe = percentageError(secondValues.predictedMs, secondValues.measuredMs, metric)!;
          // One row is one measured configuration, so configurations that
          // share a model, GPU, and workload must not be pooled here: the
          // page weights configurations equally and draws their range.
          const key = [
            firstPoint.source,
            firstPoint.model,
            firstPoint.accelerator,
            firstPoint.workload,
            firstPoint.configurationKey,
            String(firstPoint.tp),
            metric,
            simulators[first],
            simulators[second],
          ].join('\u0000');

          const entry = rows.get(key) ?? {
            source: firstPoint.source,
            model: firstPoint.model,
            accelerator: firstPoint.accelerator,
            workload: firstPoint.workload,
            metric,
            pair: [simulators[first], simulators[second]] as [string, string],
            deltas: [],
            firstWins: 0,
            secondWins: 0,
            ties: 0,
          };
          entry.deltas.push((firstApe - secondApe) * 100);
          // A tie is equality to six decimal places of relative error: any
          // coarser threshold would call two distinct predictions equal.
          if (Math.abs(firstApe - secondApe) < 1e-6) entry.ties += 1;
          else if (firstApe < secondApe) entry.firstWins += 1;
          else entry.secondWins += 1;
          rows.set(key, entry);
        }
      }
    }
  }

  return [...rows.values()]
    .map(entry => ({
      source: entry.source,
      model: entry.model,
      accelerator: entry.accelerator,
      workload: entry.workload,
      metric: entry.metric,
      pair: entry.pair,
      paired: entry.deltas.length,
      firstWins: entry.firstWins,
      secondWins: entry.secondWins,
      ties: entry.ties,
      medianDeltaPct: roundedPct(median([...entry.deltas].sort((a, b) => a - b))),
    }))
    .sort((left, right) =>
      left.source.localeCompare(right.source)
      || left.model.localeCompare(right.model)
      || left.accelerator.localeCompare(right.accelerator)
      || left.workload.localeCompare(right.workload)
      || left.metric.localeCompare(right.metric)
      || left.pair[0].localeCompare(right.pair[0])
      || left.pair[1].localeCompare(right.pair[1]));
}

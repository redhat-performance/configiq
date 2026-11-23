import { aggregateObservations, headToHead, type Observation } from './metrics';
import {
  SCORECARD_METRICS,
  SCORECARD_RELEASE_MATCH,
  type ScorecardCell,
  type ScorecardCoverage,
  type ScorecardHeadToHead,
  type ScorecardMetric,
  type ScorecardPoint,
  type ScorecardReleaseMatch,
  type ScorecardSource,
  type ScorecardTotal,
} from './types';

/**
 * Fold point attempts into the two published views.
 *
 * Coverage is computed for the same slices as the error cells so a reader can
 * always ask "out of how many?" beside a number. Nothing is dropped: a point
 * the catalogue puts outside the comparison contributes to `ineligible`, a run
 * the catalogue says the simulator cannot model contributes to `unsupported`,
 * and one we failed to reach contributes to `inconclusive`.
 */

function included(point: ScorecardPoint, releaseMatch: ScorecardReleaseMatch): boolean {
  return releaseMatch === 'any' || point.releaseMatch;
}

export function buildCoverage(points: readonly ScorecardPoint[]): ScorecardCoverage[] {
  const groups = new Map<string, ScorecardCoverage>();

  for (const point of points) {
    for (const releaseMatch of SCORECARD_RELEASE_MATCH) {
      if (!included(point, releaseMatch)) continue;
      const key = [point.simulator, point.source, point.model, point.accelerator, releaseMatch].join('\u0000');
      const group = groups.get(key) ?? {
        simulator: point.simulator,
        source: point.source,
        model: point.model,
        accelerator: point.accelerator,
        releaseMatch,
        supported: 0,
        unsupported: 0,
        inconclusive: 0,
        ineligible: 0,
        total: 0,
      };
      group[point.status] += 1;
      group.total += 1;
      groups.set(key, group);
    }
  }

  return [...groups.values()].sort(compareSlice);
}

export function buildCells(points: readonly ScorecardPoint[]): ScorecardCell[] {
  const groups = new Map<string, { sample: ScorecardPoint; releaseMatch: ScorecardReleaseMatch; observations: Observation[] }>();

  for (const point of points) {
    if (point.status !== 'supported') continue;
    for (const releaseMatch of SCORECARD_RELEASE_MATCH) {
      if (!included(point, releaseMatch)) continue;
      for (const metric of SCORECARD_METRICS) {
        const key = [
          point.simulator,
          point.source,
          point.model,
          point.accelerator,
          point.workload,
          releaseMatch,
          metric,
        ].join('\u0000');
        const group = groups.get(key) ?? { sample: point, releaseMatch, observations: [] };
        if (point.measured && point.predicted) {
          const measuredMs = metric === 'ttft' ? point.measured.ttftMs : point.measured.tpotMs;
          const predictedMs = metric === 'ttft' ? point.predicted.ttftMs : point.predicted.tpotMs;
          if (measuredMs !== null) {
            group.observations.push({
              configurationKey: point.configurationKey,
              metric,
              measuredMs,
              predictedMs,
            });
          }
        }
        groups.set(key, group);
      }
    }
  }

  const cells: ScorecardCell[] = [];
  for (const group of groups.values()) {
    const stats = aggregateObservations(group.observations);
    if (!stats) continue;
    const sample = group.sample;
    const metric = group.observations[0].metric;
    const values: Record<'mape' | 'mdape' | 'p90' | 'bias', number> = {
      mape: stats.mape,
      mdape: stats.mdape,
      p90: stats.p90,
      bias: stats.bias,
    };
    for (const statistic of ['mape', 'mdape', 'p90', 'bias'] as const) {
      cells.push({
        simulator: sample.simulator,
        source: sample.source,
        model: sample.model,
        accelerator: sample.accelerator,
        workload: sample.workload,
        releaseMatch: group.releaseMatch,
        metric,
        statistic,
        value: values[statistic],
        configurations: stats.configurations,
        points: stats.points,
      });
    }
  }

  return cells.sort(compareSlice);
}

/**
 * Roll every measured configuration up to one number per simulator.
 *
 * A source-agnostic entry is published beside each per-source entry because the
 * page shows both: `all` is the mixed population, and a source entry is what a
 * reader sees after filtering. Neither can be derived from the other by
 * averaging, so both are computed here from the same observations.
 */
export function buildTotals(points: readonly ScorecardPoint[]): ScorecardTotal[] {
  interface Bucket {
    simulator: string;
    source: ScorecardSource | 'all';
    metric: ScorecardMetric;
    releaseMatch: ScorecardReleaseMatch;
    observations: Observation[];
  }

  const buckets = new Map<string, Bucket>();
  for (const point of points) {
    if (point.status !== 'supported' || !point.measured || !point.predicted) continue;
    for (const releaseMatch of SCORECARD_RELEASE_MATCH) {
      if (!included(point, releaseMatch)) continue;
      for (const metric of SCORECARD_METRICS) {
        const measuredMs = metric === 'ttft' ? point.measured.ttftMs : point.measured.tpotMs;
        const predictedMs = metric === 'ttft' ? point.predicted.ttftMs : point.predicted.tpotMs;
        if (measuredMs === null) continue;
        for (const source of ['all', point.source] as const) {
          const key = [point.simulator, source, metric, releaseMatch].join('\u0000');
          const bucket = buckets.get(key) ?? {
            simulator: point.simulator,
            source,
            metric,
            releaseMatch,
            observations: [],
          };
          bucket.observations.push({
            configurationKey: point.configurationKey,
            metric,
            measuredMs,
            predictedMs,
          });
          buckets.set(key, bucket);
        }
      }
    }
  }

  const totals: ScorecardTotal[] = [];
  for (const bucket of buckets.values()) {
    const stats = aggregateObservations(bucket.observations);
    if (!stats) continue;
    const values: Record<'mape' | 'mdape' | 'p90' | 'bias', number> = {
      mape: stats.mape,
      mdape: stats.mdape,
      p90: stats.p90,
      bias: stats.bias,
    };
    for (const statistic of ['mape', 'mdape', 'p90', 'bias'] as const) {
      totals.push({
        simulator: bucket.simulator,
        source: bucket.source,
        metric: bucket.metric,
        releaseMatch: bucket.releaseMatch,
        statistic,
        value: values[statistic],
        configurations: stats.configurations,
        points: stats.points,
      });
    }
  }

  return totals.sort((left, right) =>
    left.simulator.localeCompare(right.simulator)
    || left.source.localeCompare(right.source)
    || left.metric.localeCompare(right.metric)
    || left.releaseMatch.localeCompare(right.releaseMatch)
    || left.statistic.localeCompare(right.statistic));
}

/** The paired view, in the artifact's own shape rather than the metrics module's. */
export function buildHeadToHead(points: readonly ScorecardPoint[]): ScorecardHeadToHead[] {
  return headToHead(points);
}

function compareSlice(left: ScorecardCell | ScorecardCoverage, right: ScorecardCell | ScorecardCoverage): number {
  return left.source.localeCompare(right.source)
    || left.accelerator.localeCompare(right.accelerator)
    || left.model.localeCompare(right.model)
    || left.simulator.localeCompare(right.simulator)
    || left.releaseMatch.localeCompare(right.releaseMatch)
    || ('workload' in left && 'workload' in right ? left.workload.localeCompare(right.workload) : 0)
    || ('metric' in left && 'metric' in right ? left.metric.localeCompare(right.metric) : 0)
    || ('statistic' in left && 'statistic' in right ? left.statistic.localeCompare(right.statistic) : 0);
}

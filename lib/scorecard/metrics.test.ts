import { describe, expect, it } from 'vitest';
import {
  aggregateObservations,
  headToHead,
  median,
  percentageError,
  percentile,
  pointValues,
  signedRelativeError,
  type Observation,
} from './metrics';
import { point } from './__fixtures__/points';

describe('signedRelativeError', () => {
  it('is signed relative to the measurement', () => {
    expect(signedRelativeError(110, 100, 'ttft')).toBeCloseTo(0.1);
    expect(signedRelativeError(90, 100, 'ttft')).toBeCloseTo(-0.1);
    expect(signedRelativeError(100, 100, 'ttft')).toBe(0);
  });

  it('refuses to divide by a value below the metric floor', () => {
    expect(signedRelativeError(2, 0.5, 'ttft')).toBeNull();
    expect(signedRelativeError(2, 0.01, 'tpot')).toBeNull();
    expect(signedRelativeError(2, 0, 'tpot')).toBeNull();
    expect(signedRelativeError(2, Number.NaN, 'tpot')).toBeNull();
  });

  it('refuses a non-finite or negative prediction', () => {
    expect(signedRelativeError(Number.POSITIVE_INFINITY, 100, 'ttft')).toBeNull();
    expect(signedRelativeError(-1, 100, 'ttft')).toBeNull();
  });
});

describe('percentageError', () => {
  it('drops the sign', () => {
    expect(percentageError(90, 100, 'ttft')).toBeCloseTo(0.1);
    expect(percentageError(110, 100, 'ttft')).toBeCloseTo(0.1);
  });

  it('matches the signed helper on unscorable pairs', () => {
    expect(percentageError(2, 0.01, 'tpot')).toBeNull();
  });
});

describe('percentile and median', () => {
  it('use nearest rank on an ascending array', () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(percentile(values, 0.9)).toBe(9);
    expect(percentile(values, 0.5)).toBe(5);
    expect(percentile([7], 0.9)).toBe(7);
  });

  it('average the middle pair of an even-length array', () => {
    expect(median([1, 2, 3, 4])).toBe(2.5);
    expect(median([1, 2, 3])).toBe(2);
  });

  it('report NaN rather than zero for no values', () => {
    expect(percentile([], 0.9)).toBeNaN();
    expect(median([])).toBeNaN();
  });
});

describe('aggregateObservations', () => {
  it('weights configurations, not points', () => {
    const observations: Observation[] = [
      // Configuration A is swept at three levels with a mean APE of 13.33%.
      { configurationKey: 'A', metric: 'ttft', measuredMs: 100, predictedMs: 110 },
      { configurationKey: 'A', metric: 'ttft', measuredMs: 100, predictedMs: 90 },
      { configurationKey: 'A', metric: 'ttft', measuredMs: 100, predictedMs: 120 },
      // Configuration B is one perfect level. A point-weighted mean would be
      // 10%, so this pins the macro rule.
      { configurationKey: 'B', metric: 'ttft', measuredMs: 100, predictedMs: 100 },
    ];

    const stats = aggregateObservations(observations);
    expect(stats).not.toBeNull();
    expect(stats!.mape).toBeCloseTo(0.0666667);
    expect(stats!.mdape).toBeCloseTo(0.0666667);
    expect(stats!.p90).toBeCloseTo(0.1333333);
    expect(stats!.bias).toBeCloseTo(0.0333333);
    expect(stats!.configurations).toBe(2);
    expect(stats!.points).toBe(4);
  });

  it('skips a point whose measurement is below the floor', () => {
    const stats = aggregateObservations([
      { configurationKey: 'A', metric: 'tpot', measuredMs: 0.01, predictedMs: 5 },
      { configurationKey: 'B', metric: 'tpot', measuredMs: 10, predictedMs: 11 },
    ]);
    expect(stats).not.toBeNull();
    expect(stats!.configurations).toBe(1);
    expect(stats!.points).toBe(1);
    expect(stats!.mape).toBeCloseTo(0.1);
  });

  it('returns null when no configuration is scorable', () => {
    expect(aggregateObservations([])).toBeNull();
    expect(aggregateObservations([
      { configurationKey: 'A', metric: 'ttft', measuredMs: 0, predictedMs: 10 },
    ])).toBeNull();
  });
});

describe('pointValues', () => {
  it('returns the pair for a supported point', () => {
    expect(pointValues(point(), 'ttft')).toEqual({ predictedMs: 110, measuredMs: 100 });
  });

  it('returns null for anything but a supported point with values', () => {
    expect(pointValues(point({ status: 'inconclusive', predicted: null }), 'ttft')).toBeNull();
    expect(pointValues(point({ predicted: null }), 'tpot')).toBeNull();
    expect(pointValues(point({ measured: { ttftMs: null, tpotMs: null, outputTokensPerSecond: null } }), 'ttft')).toBeNull();
  });
});

describe('headToHead', () => {
  it('compares only levels both simulators predicted', () => {
    const shared = point({ identity: 'shared', configurationKey: 'A' });
    const rows = headToHead([
      { ...shared, simulator: 'aisim', predicted: { ttftMs: 110, tpotMs: 10, requestLatencyMs: null, outputTokensPerSecond: null } },
      { ...shared, simulator: 'blis', predicted: { ttftMs: 120, tpotMs: 10, requestLatencyMs: null, outputTokensPerSecond: null } },
    ]);

    const ttft = rows.find(row => row.metric === 'ttft');
    expect(ttft).toBeDefined();
    expect(ttft!.pair).toEqual(['aisim', 'blis']);
    expect(ttft!.paired).toBe(1);
    expect(ttft!.firstWins).toBe(1);
    expect(ttft!.secondWins).toBe(0);
    expect(ttft!.ties).toBe(0);
    // 10% − 20% = −10 percentage points.
    expect(ttft!.medianDeltaPct).toBeCloseTo(-10);
  });

  it('keeps configurations with the same workload in separate rows', () => {
    // Two runtime configurations measured at the same model, GPU, and
    // workload. Pooling them would let one long sweep decide the row and
    // would deny the page the per-configuration range it draws.
    const first = point({ identity: 'config-a-level', configurationKey: 'A', concurrency: 8 });
    const second = point({ identity: 'config-b-level', configurationKey: 'B', concurrency: 8 });
    const rows = headToHead([
      { ...first, simulator: 'aisim', predicted: { ttftMs: 110, tpotMs: 10, requestLatencyMs: null, outputTokensPerSecond: null } },
      { ...first, simulator: 'blis', predicted: { ttftMs: 120, tpotMs: 10, requestLatencyMs: null, outputTokensPerSecond: null } },
      { ...second, simulator: 'aisim', predicted: { ttftMs: 120, tpotMs: 10, requestLatencyMs: null, outputTokensPerSecond: null } },
      { ...second, simulator: 'blis', predicted: { ttftMs: 110, tpotMs: 10, requestLatencyMs: null, outputTokensPerSecond: null } },
    ]);

    const ttft = rows.filter(row => row.metric === 'ttft');
    expect(ttft).toHaveLength(2);
    expect(ttft.map(row => row.paired)).toEqual([1, 1]);
    expect(ttft.map(row => row.firstWins).sort()).toEqual([0, 1]);
    // The row carries the fields that make it a configuration, so a consumer
    // can key and group it without guessing.
    expect(ttft.map(row => row.configurationKey).sort()).toEqual(['A', 'B']);
    expect(ttft.every(row => row.tp === 4)).toBe(true);
  });

  it('does not compare across different measured levels', () => {
    const rows = headToHead([
      point({ identity: 'level-1', simulator: 'aisim' }),
      point({ identity: 'level-2', simulator: 'blis' }),
    ]);
    expect(rows).toEqual([]);
  });

  it('ignores a simulator that did not answer', () => {
    const shared = point({ identity: 'shared' });
    const rows = headToHead([
      { ...shared, simulator: 'aisim' },
      { ...shared, simulator: 'blis', status: 'inconclusive', predicted: null, reason: 'timed out' },
    ]);
    expect(rows).toEqual([]);
  });

  it('counts an equal error as a tie', () => {
    const shared = point({ identity: 'shared' });
    const rows = headToHead([
      { ...shared, simulator: 'aisim' },
      { ...shared, simulator: 'blis' },
    ]);
    const ttft = rows.find(row => row.metric === 'ttft');
    expect(ttft!.ties).toBe(1);
    expect(ttft!.firstWins).toBe(0);
    expect(ttft!.secondWins).toBe(0);
    expect(ttft!.medianDeltaPct).toBe(0);
  });
});

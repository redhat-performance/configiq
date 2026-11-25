import { describe, expect, it } from 'vitest';
import { buildCells, buildCoverage, buildHeadToHead, buildSourceOverlap, buildTotals } from './aggregate';
import { point } from './__fixtures__/points';

describe('buildCoverage', () => {
  it('counts every outcome in its own bucket', () => {
    const coverage = buildCoverage([
      point({ status: 'supported' }),
      point({ status: 'unsupported', predicted: null, reason: 'beyond the catalogue' }),
      point({ status: 'inconclusive', predicted: null, reason: 'timed out' }),
      point({ status: 'ineligible', predicted: null, reason: 'no such GPU' }),
    ]);

    const any = coverage.find(row => row.releaseMatch === 'any');
    expect(any).toMatchObject({
      supported: 1,
      unsupported: 1,
      inconclusive: 1,
      ineligible: 1,
      total: 4,
    });
  });

  it('publishes the matched slice only when a point matched', () => {
    const unmatched = buildCoverage([point({ releaseMatch: false })]);
    expect(unmatched.map(row => row.releaseMatch)).toEqual(['any']);

    const matched = buildCoverage([point({ releaseMatch: true })]);
    expect(matched.map(row => row.releaseMatch)).toEqual(['any', 'matched']);
  });

  it('keeps sources and simulators apart', () => {
    // Every simulator/source combination must survive grouping. Varying both
    // at once would still pass if either field were dropped from the key.
    const coverage = buildCoverage([
      point({ simulator: 'aisim', source: 'redhat', identity: 'a-r' }),
      point({ simulator: 'aisim', source: 'inferencex', identity: 'a-i' }),
      point({ simulator: 'blis', source: 'redhat', identity: 'b-r' }),
      point({ simulator: 'blis', source: 'inferencex', identity: 'b-i' }),
    ]);
    expect(coverage).toHaveLength(4);
    expect(coverage.map(row => `${row.simulator}:${row.source}`).sort())
      .toEqual(['aisim:inferencex', 'aisim:redhat', 'blis:inferencex', 'blis:redhat']);
  });
});

describe('buildSourceOverlap', () => {
  function row(
    benchmarkSource: string,
    model: string,
    accelerator: string,
    isl: number,
    osl: number,
  ): Record<string, unknown> {
    return {
      benchmark_source: benchmarkSource,
      model,
      accelerator,
      isl,
      osl,
    };
  }

  it('counts a pair once even when both sources measure it several times', () => {
    const overlap = buildSourceOverlap([
      row('redhat', 'org/model', 'H200', 1024, 128),
      row('redhat', 'org/model', 'H200', 2048, 128),
      row('inferencex', 'org/model', 'H200', 1024, 128),
      row('inferencex', 'org/model', 'H200', 8192, 1024),
      row('redhat', 'org/other', 'B200', 1024, 128),
    ]);

    expect(overlap).toEqual({ sharedPairs: 1, sharedWorkloads: 1 });
  });

  it('does not call the same model on another GPU a shared pair', () => {
    const overlap = buildSourceOverlap([
      row('redhat', 'org/model', 'H200', 1024, 128),
      row('inferencex', 'org/model', 'B200', 1024, 128),
    ]);

    expect(overlap).toEqual({ sharedPairs: 0, sharedWorkloads: 0 });
  });

  it('needs the same input and output lengths, not just the same pair', () => {
    const overlap = buildSourceOverlap([
      row('redhat', 'org/model', 'H200', 8000, 1000),
      row('inferencex', 'org/model', 'H200', 8192, 1024),
    ]);

    expect(overlap).toEqual({ sharedPairs: 1, sharedWorkloads: 0 });
  });

  it('reads numeric fields and ignores a row without a complete shape', () => {
    const overlap = buildSourceOverlap([
      { benchmark_source: 'redhat', model: 'org/model', accelerator: 'H200', isl: 1024, osl: 128 },
      { benchmark_source: 'inferencex', model: 'org/model', accelerator: 'H200', isl: '1024', osl: '128' },
      { benchmark_source: 'redhat', model: 'org/model', accelerator: 'H200' },
      { benchmark_source: 'inferencex', model: '', accelerator: 'H200', isl: 1024, osl: 128 },
    ]);

    expect(overlap).toEqual({ sharedPairs: 1, sharedWorkloads: 1 });
  });

  it('counts a shared pair even when one source records no lengths', () => {
    const overlap = buildSourceOverlap([
      { benchmark_source: 'redhat', model: 'org/model', accelerator: 'H200' },
      { benchmark_source: 'inferencex', model: 'org/model', accelerator: 'H200', isl: 1024, osl: 128 },
    ]);

    expect(overlap).toEqual({ sharedPairs: 1, sharedWorkloads: 0 });
  });

  it('does not treat two variable-length mixes as one fixed workload', () => {
    // A nonpositive length is the dataset's marker for mean prompt/output
    // lengths, not a shape both sources measured.
    const overlap = buildSourceOverlap([
      row('redhat', 'org/model', 'H200', 0, 0),
      row('inferencex', 'org/model', 'H200', 0, 0),
    ]);

    expect(overlap).toEqual({ sharedPairs: 1, sharedWorkloads: 0 });
  });
});

describe('buildCells', () => {
  it('aggregates supported configurations and skips the rest', () => {
    const cells = buildCells([
      point({ configurationKey: 'A', measured: { ttftMs: 100, tpotMs: 10, outputTokensPerSecond: 100 }, predicted: { ttftMs: 110, tpotMs: 10, requestLatencyMs: null, outputTokensPerSecond: null } }),
      point({ configurationKey: 'B', measured: { ttftMs: 200, tpotMs: 20, outputTokensPerSecond: 100 }, predicted: { ttftMs: 200, tpotMs: 20, requestLatencyMs: null, outputTokensPerSecond: null } }),
      point({ configurationKey: 'C', status: 'unsupported', predicted: null }),
      point({ configurationKey: 'D', status: 'inconclusive', predicted: null }),
      point({ configurationKey: 'E', status: 'ineligible', predicted: null }),
    ]);

    const ttft = cells.find(cell =>
      cell.releaseMatch === 'any' && cell.metric === 'ttft' && cell.statistic === 'mape');
    expect(ttft).toBeDefined();
    expect(ttft!.value).toBeCloseTo(0.05);
    expect(ttft!.configurations).toBe(2);
    expect(ttft!.points).toBe(2);

    // Two configurations cleared the arithmetic, not five.
    const p90 = cells.find(cell =>
      cell.releaseMatch === 'any' && cell.metric === 'ttft' && cell.statistic === 'p90');
    expect(p90!.value).toBeCloseTo(0.1);
  });

  it('emits one cell per statistic for each metric', () => {
    const cells = buildCells([point({ releaseMatch: true })]);
    expect(cells).toHaveLength(2 * 2 * 4);
    expect(new Set(cells.map(cell => cell.metric))).toEqual(new Set(['ttft', 'tpot']));
    expect(new Set(cells.map(cell => cell.statistic))).toEqual(new Set(['mape', 'mdape', 'p90', 'bias']));
  });
});

describe('buildTotals', () => {
  it('publishes a combined entry beside each per-source entry', () => {
    const totals = buildTotals([
      point({
        source: 'redhat',
        configurationKey: 'redhat-1',
        measured: { ttftMs: 100, tpotMs: 10, outputTokensPerSecond: 100 },
        predicted: { ttftMs: 110, tpotMs: 10, requestLatencyMs: null, outputTokensPerSecond: null },
      }),
      point({
        source: 'inferencex',
        configurationKey: 'inferencex-1',
        measured: { ttftMs: 200, tpotMs: 20, outputTokensPerSecond: 100 },
        predicted: { ttftMs: 160, tpotMs: 20, requestLatencyMs: null, outputTokensPerSecond: null },
      }),
    ]);

    const mape = (source: string) => totals.find(total =>
      total.source === source && total.metric === 'ttft' && total.statistic === 'mape' && total.releaseMatch === 'any');

    expect(mape('all')?.value).toBeCloseTo(0.15);
    expect(mape('all')?.configurations).toBe(2);
    expect(mape('redhat')?.value).toBeCloseTo(0.1);
    expect(mape('inferencex')?.value).toBeCloseTo(0.2);
  });

  it('counts one vote per workload shape and parallel size', () => {
    // The same runtime configuration measured at two shapes is two measured
    // configurations for "one configuration, one vote".
    const totals = buildTotals([
      point({ configurationKey: 'A', isl: 1024, osl: 128, tp: 4, identity: 'shape-1' }),
      point({ configurationKey: 'A', isl: 8192, osl: 1024, tp: 8, identity: 'shape-2' }),
    ]);
    const mape = totals.find(total =>
      total.source === 'redhat' && total.metric === 'ttft' && total.statistic === 'mape' && total.releaseMatch === 'any');
    expect(mape?.configurations).toBe(2);
  });

  it('does not pool different models or GPUs that share a runtime configuration', () => {
    // `configurationKey` covers runtime settings only, so two models on one
    // GPU (or one model on two GPUs) can carry the same key.
    const totals = buildTotals([
      point({ configurationKey: 'A', model: 'org/first', accelerator: 'H200', identity: 'model-1' }),
      point({ configurationKey: 'A', model: 'org/second', accelerator: 'H200', identity: 'model-2' }),
      point({ configurationKey: 'A', model: 'org/first', accelerator: 'B200', identity: 'gpu-1' }),
      point({ configurationKey: 'A', model: 'org/first', accelerator: 'H200', identity: 'model-1' }),
    ]);
    const mape = totals.find(total =>
      total.source === 'redhat' && total.metric === 'ttft' && total.statistic === 'mape' && total.releaseMatch === 'any');
    expect(mape?.configurations).toBe(3);
  });

  it('publishes all four statistics from one aggregation', () => {
    const totals = buildTotals([point({ releaseMatch: true })]);
    // Two release slices × two metrics × two sources × four statistics.
    expect(totals).toHaveLength(32);
    const anyTtftBias = totals.find(total =>
      total.releaseMatch === 'any' && total.metric === 'ttft' && total.statistic === 'bias');
    expect(anyTtftBias!.value).toBeCloseTo(0.1);
  });

  it('omits a simulator with no supported point', () => {
    expect(buildTotals([point({ status: 'inconclusive', predicted: null })])).toEqual([]);
  });
});

describe('buildHeadToHead', () => {
  it('uses the metrics pairing rather than comparing aggregates', () => {
    const rows = buildHeadToHead([
      point({ identity: 'shared', simulator: 'aisim' }),
      point({ identity: 'shared', simulator: 'blis' }),
    ]);
    expect(rows).toHaveLength(2);
    expect(rows.every(row => row.paired === 1)).toBe(true);
  });
});

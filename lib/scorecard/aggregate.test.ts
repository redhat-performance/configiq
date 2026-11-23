import { describe, expect, it } from 'vitest';
import { buildCells, buildCoverage, buildHeadToHead, buildTotals } from './aggregate';
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
    const coverage = buildCoverage([
      point({ simulator: 'aisim', source: 'redhat' }),
      point({ simulator: 'blis', source: 'inferencex' }),
    ]);
    expect(coverage).toHaveLength(2);
    expect(coverage.map(row => `${row.simulator}:${row.source}`).sort())
      .toEqual(['aisim:redhat', 'blis:inferencex']);
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

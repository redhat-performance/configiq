import { describe, expect, it } from 'vitest';
import { normalizeScorecardPoints, normalizeScorecardSummary } from './normalize';
import { point, summary } from './__fixtures__/points';

const legacyCell = {
  simulator: 'aisimulate',
  source: 'redhat' as const,
  model: 'org/model',
  accelerator: 'H200',
  workload: '1024/128',
  releaseMatch: 'any' as const,
  metric: 'tpot' as const,
  statistic: 'mdape' as const,
  value: 0.2,
  configurations: 1,
  points: 2,
};

describe('normalizeScorecardSummary', () => {
  it('rewrites a retired simulator id to the current one everywhere', () => {
    const legacy = summary({
      simulators: [
        { id: 'aisimulate', label: 'AISimulate', version: '0.14.0', revision: null, available: true },
        { id: 'blis', label: 'BLIS', version: '0.15.0', revision: null, available: true },
      ],
      cells: [legacyCell],
      coverage: [{
        simulator: 'aisimulate',
        source: 'redhat',
        model: 'org/model',
        accelerator: 'H200',
        releaseMatch: 'any',
        supported: 1,
        unsupported: 0,
        inconclusive: 0,
        ineligible: 0,
        total: 1,
      }],
      totals: [{
        simulator: 'aisimulate',
        source: 'all',
        metric: 'tpot',
        releaseMatch: 'any',
        statistic: 'mdape',
        value: 0.2,
        configurations: 1,
        points: 2,
      }],
      headToHead: [{
        source: 'redhat',
        model: 'org/model',
        accelerator: 'H200',
        workload: '1024/128',
        metric: 'tpot',
        pair: ['aisimulate', 'blis'],
        paired: 2,
        firstWins: 1,
        secondWins: 1,
        ties: 0,
        medianDeltaPct: 0,
      }],
    });

    const normalized = normalizeScorecardSummary(legacy);

    expect(normalized.simulators.map(simulator => simulator.id)).toEqual(['aisim', 'blis']);
    expect(normalized.simulators[0].label).toBe('AISim');
    expect(normalized.cells[0].simulator).toBe('aisim');
    expect(normalized.coverage[0].simulator).toBe('aisim');
    expect(normalized.totals[0].simulator).toBe('aisim');
    expect(normalized.headToHead[0].pair).toEqual(['aisim', 'blis']);
  });

  it('leaves current ids and their labels untouched', () => {
    const current = summary();
    expect(normalizeScorecardSummary(current)).toEqual(current);
  });
});

describe('normalizeScorecardPoints', () => {
  it('rewrites the simulator on each point', () => {
    const [normalized] = normalizeScorecardPoints([point({ simulator: 'aisimulate' })]);
    expect(normalized.simulator).toBe('aisim');
  });
});

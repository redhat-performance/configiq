import { describe, expect, it } from 'vitest';
import type {
  ScorecardCell,
  ScorecardCoverage,
  ScorecardHeadToHead,
  ScorecardTotal,
} from '@/lib/scorecard/types';
import { summary } from '@/lib/scorecard/__fixtures__/points';
import {
  buildChartRows,
  buildFilterOptions,
  buildHeadToHeadPairs,
  buildHeadToHeadSummary,
  buildMatrix,
  emptyReason,
  formatPercent,
  formatSignedPercent,
  selectTotals,
  sumCoverage,
  type CoverageCounts,
} from './scorecard-view';

function total(overrides: Partial<ScorecardTotal> = {}): ScorecardTotal {
  return {
    simulator: 'aisim',
    source: 'all',
    metric: 'ttft',
    releaseMatch: 'any',
    statistic: 'mape',
    value: 0.1,
    configurations: 3,
    points: 5,
    ...overrides,
  };
}

function cell(overrides: Partial<ScorecardCell> = {}): ScorecardCell {
  return {
    simulator: 'aisim',
    source: 'redhat',
    model: 'org/model',
    accelerator: 'H200',
    workload: '1024/128',
    releaseMatch: 'any',
    metric: 'ttft',
    statistic: 'mape',
    value: 0.1,
    configurations: 1,
    points: 1,
    ...overrides,
  };
}

function coverage(overrides: Partial<ScorecardCoverage> = {}): ScorecardCoverage {
  return {
    simulator: 'aisim',
    source: 'redhat',
    model: 'org/model',
    accelerator: 'H200',
    releaseMatch: 'any',
    supported: 2,
    unsupported: 1,
    inconclusive: 0,
    ineligible: 0,
    total: 3,
    ...overrides,
  };
}

function headToHead(overrides: Partial<ScorecardHeadToHead> = {}): ScorecardHeadToHead {
  return {
    source: 'redhat',
    model: 'org/model',
    accelerator: 'H200',
    workload: '1024/128',
    metric: 'ttft',
    pair: ['aisim', 'blis'],
    paired: 2,
    firstWins: 1,
    secondWins: 1,
    ties: 0,
    medianDeltaPct: 0,
    ...overrides,
  };
}

describe('selectTotals', () => {
  it('collects the four statistics of one slice', () => {
    const artifact = summary({
      totals: [
        total({ statistic: 'mape', value: 0.1 }),
        total({ statistic: 'mdape', value: 0.08 }),
        total({ statistic: 'p90', value: 0.2 }),
        total({ statistic: 'bias', value: -0.03 }),
      ],
    });

    expect(selectTotals(artifact, {
      simulator: 'aisim', source: 'all', metric: 'ttft', releaseMatch: 'any',
    })).toEqual({
      values: { mape: 0.1, mdape: 0.08, p90: 0.2, bias: -0.03 },
      configurations: 3,
      points: 5,
    });
  });

  it('returns null when the slice has no evidence', () => {
    expect(selectTotals(summary(), {
      simulator: 'aisim', source: 'all', metric: 'ttft', releaseMatch: 'any',
    })).toBeNull();
  });

  it('does not mix sources or release slices', () => {
    const artifact = summary({
      totals: [
        total({ source: 'all', releaseMatch: 'any' }),
        total({ source: 'redhat', releaseMatch: 'matched', statistic: 'mape', value: 0.5 }),
      ],
    });
    expect(selectTotals(artifact, {
      simulator: 'aisim', source: 'redhat', metric: 'ttft', releaseMatch: 'any',
    })).toBeNull();
  });

  it('fills a missing statistic with NaN so the page renders a dash, not zero', () => {
    const artifact = summary({ totals: [total({ statistic: 'mape' })] });
    const slice = selectTotals(artifact, {
      simulator: 'aisim', source: 'all', metric: 'ttft', releaseMatch: 'any',
    });
    expect(slice!.values.mdape).toBeNaN();
  });
});

describe('sumCoverage', () => {
  it('adds only the requested slice', () => {
    const counts = sumCoverage([
      coverage({ supported: 2, unsupported: 1 }),
      coverage({ source: 'inferencex', supported: 4, unsupported: 0, total: 4 }),
      coverage({ releaseMatch: 'matched', supported: 5, unsupported: 0, total: 5 }),
      coverage({ simulator: 'blis', supported: 6, unsupported: 0, total: 6 }),
    ], { simulator: 'aisim', source: 'all', releaseMatch: 'any' });

    expect(counts).toEqual<CoverageCounts>({
      supported: 6,
      unsupported: 1,
      inconclusive: 0,
      ineligible: 0,
      total: 7,
    });
  });

  it('can narrow to one measured source', () => {
    const counts = sumCoverage([
      coverage({ source: 'redhat', supported: 2, unsupported: 0, total: 2 }),
      coverage({ source: 'inferencex', supported: 3, unsupported: 0, total: 3 }),
    ], { simulator: 'aisim', source: 'inferencex', releaseMatch: 'any' });
    expect(counts).toMatchObject({ supported: 3, total: 3 });
  });
});

describe('buildMatrix', () => {
  const options = {
    metric: 'ttft' as const,
    statistic: 'mape' as const,
    releaseMatch: 'any' as const,
    source: 'all' as const,
    model: '',
    gpu: '',
    workload: '',
  };

  it('puts one row per measured configuration with a column per simulator', () => {
    const artifact = summary({
      cells: [
        cell({ simulator: 'aisim', value: 0.1 }),
        cell({
          simulator: 'blis',
          value: 0.2,
          model: 'org/other',
          accelerator: 'B200',
          workload: '1000/1000',
        }),
      ],
    });

    const rows = buildMatrix(artifact, options);
    expect(rows).toHaveLength(2);
    expect(rows[0].model).toBe('org/model');
    expect(rows[0].values.get('aisim')?.value).toBe(0.1);
    expect(rows[1].values.get('blis')?.value).toBe(0.2);
  });

  it('leaves a gap rather than a zero when a simulator has no cell', () => {
    const rows = buildMatrix(summary({ cells: [cell()] }), options);
    expect(rows[0].values.has('blis')).toBe(false);
  });

  it('filters metric, statistic, release, source, model, GPU, and workload', () => {
    const artifact = summary({
      cells: [
        cell(),
        cell({ metric: 'tpot' }),
        cell({ statistic: 'p90' }),
        cell({ releaseMatch: 'matched' }),
        cell({ source: 'inferencex', model: 'org/other' }),
      ],
    });

    expect(buildMatrix(artifact, options)).toHaveLength(2);
    expect(buildMatrix(artifact, { ...options, model: 'org/other' })).toHaveLength(1);
    expect(buildMatrix(artifact, { ...options, model: 'org/missing' })).toHaveLength(0);
    expect(buildMatrix(artifact, { ...options, gpu: 'H200' })).toHaveLength(2);
    expect(buildMatrix(artifact, { ...options, gpu: 'B200' })).toHaveLength(0);
    expect(buildMatrix(artifact, { ...options, workload: '1024/128' })).toHaveLength(2);
    expect(buildMatrix(artifact, { ...options, workload: '8192/1024' })).toHaveLength(0);
    expect(buildMatrix(artifact, { ...options, source: 'inferencex' })).toHaveLength(1);
    expect(buildMatrix(artifact, { ...options, releaseMatch: 'matched' })).toHaveLength(1);
  });

  it('sorts workloads numerically, not as strings', () => {
    const artifact = summary({
      cells: [
        cell({ workload: '1000/1000' }),
        cell({ workload: '128/128' }),
        cell({ workload: '8192/1024' }),
      ],
    });
    expect(buildMatrix(artifact, options).map(row => row.workload))
      .toEqual(['128/128', '1000/1000', '8192/1024']);
  });
});

describe('buildFilterOptions', () => {
  it('lists the distinct values a reader can choose from the slice', () => {
    const artifact = summary({
      cells: [
        cell({ model: 'org/b', accelerator: 'H200', workload: '1000/1000' }),
        cell({ model: 'org/a', accelerator: 'B200', workload: '128/128' }),
        cell({ model: 'org/a', accelerator: 'H200', workload: '8192/1024' }),
        cell({ metric: 'tpot', model: 'org/other' }),
      ],
    });

    expect(buildFilterOptions(artifact, {
      metric: 'ttft', statistic: 'mape', releaseMatch: 'any', source: 'all',
    })).toEqual({
      models: ['org/a', 'org/b'],
      accelerators: ['B200', 'H200'],
      workloads: ['128/128', '1000/1000', '8192/1024'],
    });
  });

  it('offers only the sources the current slice contains', () => {
    const artifact = summary({
      cells: [
        cell({ source: 'redhat' }),
        cell({ source: 'inferencex', model: 'org/other' }),
      ],
    });
    expect(buildFilterOptions(artifact, {
      metric: 'ttft', statistic: 'mape', releaseMatch: 'any', source: 'inferencex',
    }).models).toEqual(['org/other']);
  });
});

describe('buildChartRows', () => {
  function matrix(...cells: ScorecardCell[]) {
    return buildMatrix(summary({ cells }), {
      metric: 'ttft',
      statistic: 'mape',
      releaseMatch: 'any',
      source: 'all',
      model: '',
      gpu: '',
      workload: '',
    });
  }

  it('keeps one bar per scored configuration, with one value per simulator', () => {
    const rows = matrix(
      cell({ simulator: 'aisim', value: 0.1 }),
      cell({ simulator: 'blis', value: 0.2 }),
    );
    const chart = buildChartRows(rows, ['aisim', 'blis']);

    expect(chart).toHaveLength(1);
    expect(chart[0].values.get('aisim')).toBe(0.1);
    expect(chart[0].values.get('blis')).toBe(0.2);
  });

  it('ranks by the largest absolute value so the worst configuration leads', () => {
    const rows = matrix(
      cell({ model: 'org/small', value: 0.05 }),
      cell({ model: 'org/big', value: -0.8 }),
      cell({ model: 'org/mid', value: 0.4 }),
    );
    expect(buildChartRows(rows, ['aisim']).map(row => row.label.split(' · ')[0]))
      .toEqual(['org/big', 'org/mid', 'org/small']);
  });

  it('leaves a configuration with no prediction for a simulator out of that series', () => {
    const rows = matrix(cell({ simulator: 'aisim', value: 0.1 }));
    const chart = buildChartRows(rows, ['aisim', 'blis']);
    expect(chart[0].values.has('blis')).toBe(false);
  });

  it('keeps every scored configuration in the chart', () => {
    const rows = matrix(
      ...[0.1, 0.2, 0.3, 0.4].map((value, index) => cell({ model: `org/m${index}`, value })),
    );
    const chart = buildChartRows(rows, ['aisim']);
    expect(chart).toHaveLength(4);
  });

  it('ignores rows with no value for any requested simulator', () => {
    const rows = matrix(cell({ simulator: 'other', value: 0.1 }));
    const chart = buildChartRows(rows, ['aisim']);
    expect(chart).toEqual([]);
  });

  it('names the source only when two rows would share a label', () => {
    const rows = matrix(
      cell({ source: 'redhat', value: 0.1 }),
      cell({ source: 'inferencex', value: 0.2 }),
    );
    const labels = buildChartRows(rows, ['aisim']).map(row => row.label);
    expect(labels).toEqual([
      'org/model · H200 · 1024/128 · InferenceX',
      'org/model · H200 · 1024/128 · Red Hat',
    ]);
  });
});

describe('buildHeadToHeadPairs', () => {
  it('lists each pair once with its configurations and paired levels', () => {
    const pairs = buildHeadToHeadPairs([
      headToHead({ paired: 3 }),
      headToHead({ workload: '2048/128', paired: 5 }),
      headToHead({ pair: ['aisim', 'other'], paired: 2 }),
    ], { metric: 'ttft', source: 'all' });

    expect(pairs).toEqual([
      { key: 'aisim|blis', pair: ['aisim', 'blis'], configurations: 2, paired: 8 },
      { key: 'aisim|other', pair: ['aisim', 'other'], configurations: 1, paired: 2 },
    ]);
  });

  it('follows the metric and source selectors', () => {
    const rows = [
      headToHead({ metric: 'tpot' }),
      headToHead({ pair: ['aisim', 'other'], source: 'inferencex' }),
    ];

    expect(buildHeadToHeadPairs(rows, { metric: 'ttft', source: 'all' }).map(entry => entry.key))
      .toEqual(['aisim|other']);
    expect(buildHeadToHeadPairs(rows, { metric: 'tpot', source: 'redhat' }).map(entry => entry.key))
      .toEqual(['aisim|blis']);
    expect(buildHeadToHeadPairs(rows, { metric: 'ttft', source: 'inferencex' }).map(entry => entry.key))
      .toEqual(['aisim|other']);
  });
});

describe('buildHeadToHeadSummary', () => {
  const all = { metric: 'ttft', source: 'all' } as const;
  const pair: [string, string] = ['aisim', 'blis'];
  const byModel = { ...all, pair, dimension: 'model' } as const;
  const byGpu = { ...all, pair, dimension: 'accelerator' } as const;

  it('weights configurations equally rather than pooling their levels', () => {
    // Best on one level, worse on ten: a pooled share would be 1/11 (9%), but
    // the short sweep and the long sweep are two configurations, so the mean
    // configuration share is 50%.
    const rows = [
      headToHead({ paired: 1, firstWins: 1, secondWins: 0 }),
      headToHead({ workload: '2048/128', paired: 10, firstWins: 0, secondWins: 10 }),
    ];

    const result = buildHeadToHeadSummary(rows, byModel);

    expect(result.configurations).toBe(2);
    expect(result.paired).toBe(11);
    expect(result.firstShare).toBeCloseTo(0.5);
    expect(result.secondShare).toBeCloseTo(0.5);
    expect(result.categories[0].firstShare).toBeCloseTo(0.5);
  });

  it('draws the candle from the range of configuration shares', () => {
    const rows = [
      headToHead({ workload: '128/128', paired: 2, firstWins: 2, secondWins: 0 }),
      headToHead({ workload: '1024/128', paired: 2, firstWins: 1, secondWins: 1 }),
      headToHead({ workload: '2048/128', paired: 2, firstWins: 0, secondWins: 2 }),
    ];

    const [category] = buildHeadToHeadSummary(rows, byModel).categories;

    expect(category.configurations).toBe(3);
    expect(category.minShare).toBe(0);
    expect(category.maxShare).toBe(1);
    expect(category.firstShare).toBeCloseTo(0.5);
  });

  it('omits the candle when one configuration is the whole category', () => {
    const [category] = buildHeadToHeadSummary([headToHead()], byModel).categories;

    expect(category.configurations).toBe(1);
    expect(category.minShare).toBeNull();
    expect(category.maxShare).toBeNull();
  });

  it('keeps ties out of both best shares', () => {
    const [category] = buildHeadToHeadSummary([
      headToHead({ paired: 4, firstWins: 1, secondWins: 1, ties: 2 }),
    ], byModel).categories;

    expect(category.ties).toBe(2);
    expect(category.firstShare).toBe(0.25);
    expect(category.secondShare).toBe(0.25);
  });

  it('groups by GPU when asked, using the same configuration-first rule', () => {
    const rows = [
      headToHead({ accelerator: 'H200', paired: 1, firstWins: 1, secondWins: 0 }),
      headToHead({ accelerator: 'H200', workload: '2048/128', paired: 1, firstWins: 0, secondWins: 1 }),
      headToHead({ accelerator: 'B200', paired: 1, firstWins: 0, secondWins: 1 }),
    ];

    const result = buildHeadToHeadSummary(rows, byGpu);

    expect(result.categories.map(category => category.label)).toEqual(['H200', 'B200']);
    expect(result.categories[0].firstShare).toBeCloseTo(0.5);
    expect(result.categories[1].firstShare).toBe(0);
  });

  it('puts the categories with the least evidence last', () => {
    const rows = [
      headToHead({ model: 'org/thin' }),
      headToHead({ model: 'org/busy-a' }),
      headToHead({ model: 'org/busy-a', workload: '2048/128' }),
      headToHead({ model: 'org/busy-b' }),
      headToHead({ model: 'org/busy-b', workload: '2048/128' }),
    ];

    expect(buildHeadToHeadSummary(rows, byModel).categories.map(category => category.label))
      .toEqual(['org/busy-a', 'org/busy-b', 'org/thin']);
  });

  it('reports the median gap across configurations', () => {
    const rows = [
      headToHead({ workload: '128/128', medianDeltaPct: -30 }),
      headToHead({ workload: '1024/128', medianDeltaPct: -10 }),
      headToHead({ workload: '2048/128', medianDeltaPct: -20 }),
    ];

    expect(buildHeadToHeadSummary(rows, byModel).medianGapPct).toBe(-20);
  });

  it('filters a source, a metric, and a pair rather than mixing them', () => {
    const rows = [
      headToHead(),
      headToHead({ source: 'inferencex', paired: 2, firstWins: 0, secondWins: 2 }),
      headToHead({ metric: 'tpot', paired: 2, firstWins: 0, secondWins: 2 }),
      headToHead({ pair: ['aisim', 'other'], paired: 2, firstWins: 0, secondWins: 2 }),
    ];

    expect(buildHeadToHeadSummary(rows, { ...byModel, source: 'redhat' }).firstShare).toBe(0.5);
    expect(buildHeadToHeadSummary(rows, { ...byModel, pair: ['aisim', 'other'] }).firstShare).toBe(0);
    expect(buildHeadToHeadSummary(rows, { ...byModel, pair: null })).toEqual({
      categories: [],
      configurations: 0,
      paired: 0,
      ties: 0,
      firstShare: 0,
      secondShare: 0,
      medianGapPct: 0,
    });
  });

  it('returns no categories when the pair is absent from the slice', () => {
    expect(buildHeadToHeadSummary([], byModel).categories).toEqual([]);
  });
});

describe('formatPercent', () => {
  it('renders one decimal below 100%', () => {
    expect(formatPercent(0.1234)).toBe('12.3%');
    expect(formatPercent(0)).toBe('0.0%');
  });

  it('drops decimals at and above 100%', () => {
    expect(formatPercent(1.234)).toBe('123%');
  });

  it('separates thousands rather than printing six digits', () => {
    expect(formatPercent(12)).toBe('1,200%');
  });

  it('renders a dash for missing evidence', () => {
    expect(formatPercent(null)).toBe('—');
    expect(formatPercent(undefined)).toBe('—');
    expect(formatPercent(Number.NaN)).toBe('—');
  });
});

describe('formatSignedPercent', () => {
  it('keeps the sign, using a minus sign rather than a hyphen', () => {
    expect(formatSignedPercent(0.1)).toBe('+10%');
    expect(formatSignedPercent(-0.1)).toBe('−10%');
    expect(formatSignedPercent(0)).toBe('0%');
  });

  it('drops decimals once the magnitude reaches 100%', () => {
    expect(formatSignedPercent(1.5)).toBe('+150%');
  });

  it('renders a dash for missing evidence', () => {
    expect(formatSignedPercent(null)).toBe('—');
    expect(formatSignedPercent(Number.NaN)).toBe('—');
  });
});

describe('emptyReason', () => {
  const base: CoverageCounts = { supported: 0, unsupported: 0, inconclusive: 0, ineligible: 0, total: 0 };

  it('is silent when something was predicted', () => {
    expect(emptyReason({ ...base, supported: 1, total: 1 })).toBeNull();
  });

  it('names the reason the reader can act on', () => {
    expect(emptyReason({ ...base, unsupported: 1, total: 1 })).toContain('catalogue lists these parts');
    expect(emptyReason({ ...base, inconclusive: 1, total: 1 })).toContain('failed');
    expect(emptyReason({ ...base, ineligible: 1, total: 1 })).toContain('outside');
    expect(emptyReason(base)).toContain('No measured runs');
  });

  it('prefers the earliest bucket when several are present', () => {
    expect(emptyReason({ supported: 0, unsupported: 1, inconclusive: 1, ineligible: 1, total: 3 }))
      .toContain('catalogue lists these parts');
  });
});

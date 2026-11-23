// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type {
  ScorecardCell,
  ScorecardCoverage,
  ScorecardHeadToHead,
  ScorecardSummary,
  ScorecardTotal,
} from '@/lib/scorecard/types';
import { summary } from '@/lib/scorecard/__fixtures__/points';

// The page reads its shared view through `useSearchParams`. Tests mount it
// directly, so supply the current window search rather than the app router.
const navigation = vi.hoisted(() => ({ search: '' }));

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(navigation.search),
}));

const scorecardState = vi.hoisted(() => {
  type State = {
    summary: import('@/lib/scorecard/types').ScorecardSummary | null;
    notPublished: boolean;
    error: string | null;
    isLoading: boolean;
    reload: () => void;
  };
  let current: State = {
    summary: null,
    notPublished: false,
    error: null,
    isLoading: true,
    reload: () => {},
  };
  return {
    set(next: State) {
      current = next;
    },
    get: () => current,
  };
});

vi.mock('@/lib/hooks/useScorecard', () => ({
  useScorecard: () => scorecardState.get(),
}));

// Victory measures its container and draws SVG; the page tests only need to
// know the chart mounted with one series per simulator.
vi.mock('@patternfly/react-charts/victory', async () => {
  const React = await import('react');
  type ChartProps = { children?: React.ReactNode; name?: string };
  return {
    Chart: ({ children }: ChartProps) => React.createElement('div', { 'data-chart': 'chart' }, children),
    ChartAxis: () => React.createElement('span', { 'data-chart': 'axis' }),
    ChartGroup: ({ children }: ChartProps) => React.createElement('div', { 'data-chart': 'group' }, children),
    ChartBar: ({ name }: ChartProps) => React.createElement('span', { 'data-chart': 'bar', 'data-series': name }),
  };
});

const clipboard = vi.hoisted(() => ({ text: '' }));

const { default: SimulatorScorecard } = await import('./SimulatorScorecard');

function totalsFor(
  simulator: string,
  values: { mape: number; mdape: number; p90: number; bias: number },
  options: {
    source?: 'all' | 'redhat';
    releaseMatch?: 'any' | 'matched';
    metric?: 'ttft' | 'tpot';
  } = {},
): ScorecardTotal[] {
  return (['mape', 'mdape', 'p90', 'bias'] as const).map(statistic => ({
    simulator,
    source: options.source ?? 'all',
    metric: options.metric ?? 'ttft',
    releaseMatch: options.releaseMatch ?? 'any',
    statistic,
    value: values[statistic],
    configurations: 3,
    points: 5,
  }));
}

function coverageFor(
  simulator: string,
  counts: Partial<ScorecardCoverage> = {},
): ScorecardCoverage {
  const row: ScorecardCoverage = {
    simulator,
    source: 'redhat',
    model: 'org/model',
    accelerator: 'H200',
    releaseMatch: 'any',
    supported: 3,
    unsupported: 1,
    inconclusive: 0,
    ineligible: 0,
    total: 4,
    ...counts,
  };
  row.total = row.supported + row.unsupported + row.inconclusive + row.ineligible;
  return row;
}

function cellFor(
  simulator: string,
  overrides: Partial<ScorecardCell> = {},
): ScorecardCell {
  return {
    simulator,
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

const headToHeadRow: ScorecardHeadToHead = {
  source: 'redhat',
  model: 'org/model',
  accelerator: 'H200',
  workload: '1024/128',
  metric: 'ttft',
  pair: ['aisimulate', 'blis'],
  paired: 2,
  firstWins: 1,
  secondWins: 1,
  ties: 0,
  medianDeltaPct: -10,
};

function published(overrides: Partial<ScorecardSummary> = {}): ScorecardSummary {
  return summary({
    simulators: [
      {
        id: 'aisimulate',
        label: 'AISimulate',
        version: '0.14.0',
        revision: 'deadbeef',
        available: true,
      },
      {
        id: 'blis',
        label: 'BLIS',
        version: '0.15.0',
        revision: 'cafebabe',
        available: true,
      },
    ],
    totals: [
      ...totalsFor('aisimulate', { mape: 0.123, mdape: 0.1, p90: 0.2, bias: -0.05 }),
      ...totalsFor('blis', { mape: 0.2, mdape: 0.18, p90: 0.3, bias: 0.02 }),
      // TPOT is the page default, so the fixture carries its totals too.
      ...totalsFor('aisimulate', { mape: 0.15, mdape: 0.11, p90: 0.25, bias: -0.04 }, { metric: 'tpot' }),
      ...totalsFor('blis', { mape: 0.22, mdape: 0.19, p90: 0.31, bias: 0.03 }, { metric: 'tpot' }),
      // A distinct ttft/mape/redhat slice the link test can select; ttft and
      // mape are not the page defaults, so the link has to carry them.
      {
        simulator: 'aisimulate',
        source: 'redhat',
        metric: 'ttft',
        releaseMatch: 'any',
        statistic: 'mape',
        value: 0.44,
        configurations: 1,
        points: 1,
      },
    ],
    coverage: [coverageFor('aisimulate'), coverageFor('blis', { supported: 1 })],
    datasets: [
      {
        source: 'redhat',
        id: 'redhat-performance/configiq-performance-upstream-data',
        revision: 'abc1234',
        rowCount: 10,
      },
      {
        source: 'inferencex',
        id: 'redhat-performance/configiq-inferencex-comparison',
        revision: '2026-11-11T19:22:57.864Z',
        rowCount: 786,
      },
    ],
    cells: [
      cellFor('aisimulate', { value: 0.1 }),
      cellFor('blis', { value: 0.2 }),
      cellFor('aisimulate', { metric: 'tpot', statistic: 'mdape', value: 0.33 }),
    ],
    headToHead: [
      headToHeadRow,
      // TPOT is the page default, so head to head must cover it.
      { ...headToHeadRow, metric: 'tpot' },
    ],
    ...overrides,
  });
}

let root: Root;
let container: HTMLDivElement;

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true;
  navigation.search = '';
  clipboard.text = '';
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText: vi.fn(async (text: string) => { clipboard.text = text; }) },
  });
  scorecardState.set({
    summary: null,
    notPublished: false,
    error: null,
    isLoading: true,
    reload: vi.fn(),
  });
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  window.history.replaceState({}, '', '/validation/scorecard');
});

async function mount(preloaded: ScorecardSummary | null, search = ''): Promise<void> {
  window.history.replaceState({}, '', `/validation/scorecard${search}`);
  navigation.search = window.location.search;
  scorecardState.set({
    summary: preloaded,
    notPublished: preloaded === null,
    error: null,
    isLoading: false,
    reload: vi.fn(),
  });
  await act(async () => {
    root.render(<SimulatorScorecard />);
    await Promise.resolve();
  });
}

function button(label: string): HTMLButtonElement | undefined {
  return Array.from(container.querySelectorAll('button'))
    .find(candidate => candidate.textContent?.trim() === label);
}

describe('Simulator scorecard page', () => {
  it('says plainly when no snapshot is published', async () => {
    await mount(null);
    expect(container.textContent).toContain('No scorecard snapshot is published yet');
    expect(container.textContent).not.toContain('Agreement by simulator');
  });

  it('leads with one agreement tile per simulator', async () => {
    await mount(published());

    expect(container.querySelector('h1')?.textContent).toBe('Simulator scorecard');
    expect(container.textContent).toContain('AISimulate');
    expect(container.textContent).toContain('BLIS');
    // The default slice is TPOT, and the headline is MdAPE: 0.11 and 0.19,
    // rendered as percentages rather than as fractions.
    expect(container.textContent).toContain('11.0%');
    expect(container.textContent).toContain('19.0%');
    expect(container.textContent).toContain('MdAPE');
    // MAPE stays visible beside it, as 0.15 and 0.22.
    expect(container.textContent).toContain('15.0%');
    expect(container.textContent).toContain('22.0%');
    expect(container.textContent).toContain('p90');
    expect(container.textContent).toContain('Bias');
  });

  it('shows what each simulator refused or was never asked', async () => {
    await mount(published());
    expect(container.textContent).toContain('Predicted');
    expect(container.textContent).toContain('Unsupported');
    expect(container.textContent).toContain('Ineligible');
    // 3 of 4 predicted for AISimulate.
    expect(container.textContent).toContain('3 (75%)');
  });

  it('puts the measured configurations in one table with a column per simulator', async () => {
    await mount(published());
    expect(container.textContent).toContain('Where the error sits');
    expect(container.textContent).toContain('org/model');
    expect(container.textContent).toContain('1024/128');
  });

  it('names both measured sources and drops the gateway and release chrome', async () => {
    await mount(published());
    expect(container.textContent).toContain('Benchmark data from Red Hat');
    expect(container.textContent).toContain('SemiAnalysis InferenceX');
    expect(container.textContent).toContain('786 rows');
    expect(container.textContent).not.toContain('Gateway');
    expect(container.textContent).not.toContain('Simulator releases');
    expect(container.textContent).not.toContain('https://aisimulators.example.test');
  });

  it('draws one chart series per simulator on the flip tile front', async () => {
    await mount(published());
    const card = container.querySelector('[data-flip-card="scorecard-error"]');
    expect(card).not.toBeNull();
    expect(card?.querySelectorAll('[data-chart="bar"]')).toHaveLength(2);
    expect(card?.querySelector('[class*="flipFront"] table')).toBeNull();
    const backTable = card?.querySelector('[class*="flipBack"] table');
    expect(backTable).not.toBeNull();
    expect(backTable?.textContent).toContain('org/model');
  });

  it('flips the error tile to the table and back', async () => {
    await mount(published());
    const card = container.querySelector('[data-flip-card="scorecard-error"]') as HTMLElement;
    const buttons = [...card.querySelectorAll<HTMLButtonElement>('button[aria-controls="scorecard-error-table"]')];
    expect(buttons).toHaveLength(2);

    await act(async () => buttons[0].click());
    expect(card.className).toContain('flipped');
    expect(buttons[0].getAttribute('aria-expanded')).toBe('true');

    await act(async () => buttons[1].click());
    expect(card.className).not.toContain('flipped');
  });

  it('compares simulators only on levels both predicted', async () => {
    await mount(published());
    expect(container.textContent).toContain('Head to head');
    expect(container.textContent).toContain('AISimulate vs BLIS');
    // One win each: an even outcome bar, described for assistive tech. The
    // default TPOT slice carries the same result as TTFT in this fixture.
    expect(container.querySelector('[aria-label*="won 1 of 2 paired levels"]')).not.toBeNull();
  });

  it('disables the like-for-release view until a point matched a release', async () => {
    await mount(published());
    expect(button('Like for release')?.disabled).toBe(true);
    expect(container.querySelector('[aria-label="Help for Releases"]')).not.toBeNull();
  });

  it('enables the like-for-release view once matched totals exist', async () => {
    await mount(published({
      totals: [
        ...totalsFor('aisimulate', { mape: 0.123, mdape: 0.1, p90: 0.2, bias: -0.05 }, { releaseMatch: 'matched' }),
      ],
    }));
    expect(button('Like for release')?.disabled).toBe(false);
  });

  it('offers the measured models, GPUs, and workloads as choices, not a free-text box', async () => {
    await mount(published());

    const gpu = container.querySelector<HTMLSelectElement>('select[aria-label="GPU"]');
    const model = container.querySelector<HTMLSelectElement>('select[aria-label="Model"]');
    const workload = container.querySelector<HTMLSelectElement>('select[aria-label="Workload"]');
    expect(model).not.toBeNull();
    expect(gpu).not.toBeNull();
    expect(workload).not.toBeNull();
    expect([...gpu!.options].map(option => option.value)).toEqual(['', 'H200']);
    expect([...model!.options].map(option => option.value)).toEqual(['', 'org/model']);
    expect(container.textContent).not.toContain('Filter by model, GPU, or workload');

    await act(async () => {
      gpu!.value = 'H200';
      gpu!.dispatchEvent(new Event('change', { bubbles: true }));
    });

    expect(button('Clear filters')).toBeDefined();
    expect(container.textContent).toContain('filtered');
    // The tiles cannot follow a row filter: the snapshot holds no per-model or
    // per-GPU totals, so the page says so instead of looking broken.
    expect(container.textContent).toContain('These totals cover every configuration in the snapshot');
  });

  it('does not read the reader’s settings to decide what to show', async () => {
    // The page imports no settings hook. If it did, this mount would fail on
    // the missing provider rather than render every published simulator.
    await mount(published());
    expect(container.textContent).toContain('AISimulate');
    expect(container.textContent).toContain('BLIS');
  });
});

describe('Simulator scorecard shared view', () => {
  it('applies metric, statistic, source, and row filters from a link', async () => {
    await mount(published(), '?v=1&metric=ttft&statistic=mape&source=redhat&workload=1024%2F128');

    // The selected slice is the redhat ttft/MAPE total, not the default
    // tpot/MdAPE one.
    expect(container.textContent).toContain('44.0%');
    expect(container.textContent).toContain('filtered');
  });

  it('keeps a link’s row filters while the snapshot is still loading', async () => {
    window.history.replaceState({}, '', '/validation/scorecard?v=1&gpu=H200');
    navigation.search = window.location.search;
    scorecardState.set({
      summary: null,
      notPublished: false,
      error: null,
      isLoading: true,
      reload: vi.fn(),
    });

    await act(async () => {
      root.render(<SimulatorScorecard />);
      await Promise.resolve();
    });

    scorecardState.set({
      summary: published(),
      notPublished: false,
      error: null,
      isLoading: false,
      reload: vi.fn(),
    });
    await act(async () => {
      root.render(<SimulatorScorecard />);
      await Promise.resolve();
    });

    const gpu = container.querySelector<HTMLSelectElement>('select[aria-label="GPU"]');
    expect(gpu?.value).toBe('H200');
  });

  it('round-trips a customised view through the copy-link button', async () => {
    await mount(published(), '?v=1&metric=ttft&statistic=mape&source=redhat&workload=1024%2F128');

    const copy = button('Copy link');
    expect(copy).toBeDefined();
    await act(async () => {
      copy?.click();
      await Promise.resolve();
    });

    expect(clipboard.text).not.toBe('');
    const url = new URL(clipboard.text);
    expect(url.pathname).toBe('/validation/scorecard');
    expect(url.searchParams.get('v')).toBe('1');
    expect(url.searchParams.get('metric')).toBe('ttft');
    expect(url.searchParams.get('statistic')).toBe('mape');
    expect(url.searchParams.get('source')).toBe('redhat');
    expect(url.searchParams.get('workload')).toBe('1024/128');
    expect(url.searchParams.get('model')).toBeNull();
    expect(url.searchParams.get('gpu')).toBeNull();
    // `releaseMatch=any` is the page default, so the link leaves it out.
    expect(url.searchParams.get('releaseMatch')).toBeNull();
  });

  it('warns about and drops an unknown parameter rather than failing the view', async () => {
    await mount(published(), '?v=1&unexpected=1');
    expect(container.textContent).toContain('Part of this link was ignored');
    expect(container.textContent).toContain('11.0%');
  });
});

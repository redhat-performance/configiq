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

// The page reads the GPU catalog only for presentation (vendor/architecture
// grouping and the memory line). The measured labels themselves come from the
// snapshot, so the fixture catalog only needs the parts the snapshot names.
const catalogState = vi.hoisted(() => ({
  gpuOptions: [
    {
      systemId: 'h200_sxm',
      label: 'NVIDIA H200',
      vendor: 'nvidia',
      architecture: 'hopper',
      vramGb: 141,
      bandwidthTbps: 4.8,
      tflopsBf16: 989,
      tdpWatts: 700,
      gpusPerNode: 8,
    },
  ],
}));

vi.mock('@/lib/hooks/useCatalog', () => ({
  useCatalog: () => catalogState,
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
  pair: ['aisim', 'blis'],
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
        id: 'aisim',
        label: 'AISim',
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
      ...totalsFor('aisim', { mape: 0.123, mdape: 0.1, p90: 0.2, bias: -0.05 }),
      ...totalsFor('blis', { mape: 0.2, mdape: 0.18, p90: 0.3, bias: 0.02 }),
      // TPOT is the page default, so the fixture carries its totals too.
      ...totalsFor('aisim', { mape: 0.15, mdape: 0.11, p90: 0.25, bias: -0.04 }, { metric: 'tpot' }),
      ...totalsFor('blis', { mape: 0.22, mdape: 0.19, p90: 0.31, bias: 0.03 }, { metric: 'tpot' }),
      // A distinct ttft/mape/redhat slice the link test can select; ttft and
      // mape are not the page defaults, so the link has to carry them.
      {
        simulator: 'aisim',
        source: 'redhat',
        metric: 'ttft',
        releaseMatch: 'any',
        statistic: 'mape',
        value: 0.44,
        configurations: 1,
        points: 1,
      },
    ],
    coverage: [coverageFor('aisim'), coverageFor('blis', { supported: 1 })],
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
      cellFor('aisim', { value: 0.1 }),
      cellFor('blis', { value: 0.2 }),
      cellFor('aisim', { metric: 'tpot', statistic: 'mdape', value: 0.33 }),
    ],
    headToHead: [
      headToHeadRow,
      // TPOT is the page default, so head to head must cover it. The second
      // org/model configuration gives that category a range; org/other has a
      // single configuration, and org/inference belongs to the other source.
      { ...headToHeadRow, metric: 'tpot' },
      {
        ...headToHeadRow,
        metric: 'tpot',
        workload: '2048/128',
        firstWins: 2,
        secondWins: 0,
      },
      {
        ...headToHeadRow,
        metric: 'tpot',
        model: 'org/other',
        accelerator: 'B200',
        paired: 1,
        firstWins: 0,
        secondWins: 1,
        ties: 0,
      },
      {
        ...headToHeadRow,
        metric: 'tpot',
        source: 'inferencex',
        model: 'org/inference',
        paired: 4,
        firstWins: 4,
        secondWins: 0,
      },
      // A second pair makes the pair selector a real choice.
      {
        ...headToHeadRow,
        metric: 'tpot',
        pair: ['aisim', 'other'],
        paired: 1,
        firstWins: 0,
        secondWins: 1,
      },
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

function gpuToggle(): HTMLButtonElement {
  const wrapper = container.querySelector('label[for="scorecard-gpu"]')?.parentElement;
  const toggle = wrapper?.querySelector<HTMLButtonElement>('.pf-v6-c-menu-toggle');
  if (!toggle) throw new Error('GPU control not rendered');
  return toggle;
}

async function openGpuMenu(): Promise<void> {
  await act(async () => {
    gpuToggle().click();
    await Promise.resolve();
  });
}

function menuOption(text: string): HTMLElement {
  const option = [...document.body.querySelectorAll<HTMLElement>('.pf-v6-c-menu__item')]
    .find(element => element.textContent?.includes(text));
  if (!option) throw new Error(`option not found: ${text}`);
  return option;
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
    expect(container.textContent).toContain('AISim');
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
    // 3 of 4 predicted for AISim.
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
    expect(container.textContent).toContain('AISim vs BLIS');
    // One win each: an even outcome bar, described for assistive tech. The
    // default TPOT slice carries the same result as TTFT in this fixture.
    expect(container.querySelector('[aria-label*="won 1 of 2 paired levels"]')).not.toBeNull();
  });

  it('puts win share by model and GPU on the head-to-head tile front', async () => {
    await mount(published());
    const card = container.querySelector('[data-flip-card="scorecard-paired"]');

    expect(card).not.toBeNull();
    expect(card?.textContent).toContain('Win share by model');
    expect(card?.textContent).toContain('Win share by GPU');
    expect(card?.querySelector('[class*="flipFront"] table')).toBeNull();

    const back = card?.querySelector('[class*="flipBack"] table');
    expect(back).not.toBeNull();
    expect(back?.textContent).toContain('Source');
    expect(back?.textContent).toContain('Ties');
    expect(back?.textContent).toContain('org/model');
  });

  it('marks a single-configuration category instead of drawing a range', async () => {
    await mount(published());
    const card = container.querySelector('[data-flip-card="scorecard-paired"]');

    // org/other is one configuration; org/model is two, so only the second
    // can carry a candle.
    expect(card?.textContent).toContain('1 configuration');
    expect(card?.querySelectorAll('[class*="h2hCandle"]').length).toBeGreaterThan(0);
  });

  it('applies the benchmark data selector to head to head', async () => {
    await mount(published());
    const source = container.querySelector<HTMLSelectElement>('select[aria-label="Benchmark data"]');

    await act(async () => {
      source!.value = 'inferencex';
      source!.dispatchEvent(new Event('change', { bubbles: true }));
    });

    const card = container.querySelector('[data-flip-card="scorecard-paired"]');
    expect(card?.textContent).toContain('org/inference');
    expect(card?.textContent).not.toContain('org/model');
  });

  it('offers a pair selector only when the snapshot compared more than one pair', async () => {
    await mount(published());
    const select = container.querySelector<HTMLSelectElement>('select[aria-label="Simulator pair"]');
    expect(select).not.toBeNull();
    expect([...select!.options].map(option => option.value)).toEqual(['aisim|blis', 'aisim|other']);

    await act(async () => {
      select!.value = 'aisim|other';
      select!.dispatchEvent(new Event('change', { bubbles: true }));
    });

    const card = container.querySelector('[data-flip-card="scorecard-paired"]');
    expect(card?.textContent).toContain('AISim vs other');
  });

  it('hides the pair selector when there is one pair', async () => {
    await mount(published({
      headToHead: [headToHeadRow, { ...headToHeadRow, metric: 'tpot' }],
    }));

    expect(container.querySelector('select[aria-label="Simulator pair"]')).toBeNull();
    expect(container.textContent).toContain('AISim vs BLIS');
  });

  it('disables the like-for-release view until a point matched a release', async () => {
    await mount(published());
    expect(button('Like for release')?.disabled).toBe(true);
    expect(container.querySelector('[aria-label="Help for Releases"]')).not.toBeNull();
  });

  it('enables the like-for-release view once matched totals exist', async () => {
    await mount(published({
      totals: [
        ...totalsFor('aisim', { mape: 0.123, mdape: 0.1, p90: 0.2, bias: -0.05 }, { releaseMatch: 'matched' }),
      ],
    }));
    expect(button('Like for release')?.disabled).toBe(false);
  });

  it('offers the measured models and GPUs as searchable choices, not a free-text box', async () => {
    await mount(published());

    const model = container.querySelector<HTMLInputElement>('input[aria-label="Model"]');
    const workload = container.querySelector<HTMLSelectElement>('select[aria-label="Workload"]');
    expect(model?.value).toBe('All models');
    expect(model?.getAttribute('role')).toBe('combobox');
    expect(workload).not.toBeNull();
    expect([...workload!.options].map(option => option.value)).toEqual(['', '1024/128']);
    expect(container.textContent).toContain('All GPUs');
    expect(container.textContent).not.toContain('Filter by model, GPU, or workload');

    // The GPU control groups the measured part by the catalog's vendor and
    // architecture, and selects the measured label the matrix filters on.
    await openGpuMenu();
    expect(document.body.textContent).toContain('Nvidia Hopper');
    await act(async () => {
      menuOption('H200').click();
      await Promise.resolve();
    });

    expect(container.textContent).toContain('H200 — 141 GB');
    expect(container.textContent).toContain('4.8 TB/s · 989 TFLOPS');
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
    expect(container.textContent).toContain('AISim');
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

    expect(gpuToggle().textContent).toContain('H200 — 141 GB');
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

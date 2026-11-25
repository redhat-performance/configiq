// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type {
  ScorecardCell,
  ScorecardCoverage,
  ScorecardHeadToHead,
  ScorecardPoint,
  ScorecardSummary,
  ScorecardTotal,
} from '@/lib/scorecard/types';
import { point, summary } from '@/lib/scorecard/__fixtures__/points';

// The page reads its shared view through `useSearchParams`. Tests mount it
// directly, so supply the current window search rather than the app router.
const navigation = vi.hoisted(() => ({ search: '' }));

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(navigation.search),
}));

const scorecardState = vi.hoisted(() => {
  type State = {
    summary: import('@/lib/scorecard/types').ScorecardSummary | null;
    points: import('@/lib/scorecard/types').ScorecardPoint[] | null;
    pointsLoading: boolean;
    pointsError: string | null;
    notPublished: boolean;
    error: string | null;
    isLoading: boolean;
    reload: () => void;
    loadPoints: () => void;
  };
  const base: State = {
    summary: null,
    points: null,
    pointsLoading: false,
    pointsError: null,
    notPublished: false,
    error: null,
    isLoading: true,
    reload: () => {},
    loadPoints: () => {},
  };
  let current: State = base;
  return {
    set(next: Partial<State>) {
      current = { ...base, ...next };
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
    ChartLabel: () => React.createElement('span', { 'data-chart': 'label' }),
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
  configurationKey: 'configuration-1',
  tp: 4,
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

/**
 * Point detail for a row filter.
 *
 * Every point is one configuration at workload `1024/128`, so a workload
 * filter selects all four. The errors are distinct per simulator and GPU so a
 * filtered tile cannot be mistaken for the pooled one: the first simulator's
 * H200 TPOT MdAPE is 25.0% here and 11.0% in the pooled totals.
 */
function publishedPoints(): ScorecardPoint[] {
  return [
    point({
      identity: 'point-aisim-h200',
      simulator: 'aisim',
      model: 'org/model',
      accelerator: 'H200',
      workload: '1024/128',
      configurationKey: 'configuration-aisim-h200',
      measured: { ttftMs: 100, tpotMs: 10, outputTokensPerSecond: 100 },
      predicted: {
        ttftMs: 144,
        tpotMs: 12.5,
        requestLatencyMs: null,
        outputTokensPerSecond: 80,
      },
    }),
    point({
      identity: 'point-blis-h200',
      simulator: 'blis',
      model: 'org/model',
      accelerator: 'H200',
      workload: '1024/128',
      configurationKey: 'configuration-blis-h200',
      measured: { ttftMs: 100, tpotMs: 10, outputTokensPerSecond: 100 },
      predicted: {
        ttftMs: 150,
        tpotMs: 13.5,
        requestLatencyMs: null,
        outputTokensPerSecond: 74,
      },
    }),
    point({
      identity: 'point-aisim-b200',
      simulator: 'aisim',
      model: 'org/other',
      accelerator: 'B200',
      workload: '1024/128',
      configurationKey: 'configuration-aisim-b200',
      measured: { ttftMs: 100, tpotMs: 10, outputTokensPerSecond: 100 },
      predicted: {
        ttftMs: 160,
        tpotMs: 18,
        requestLatencyMs: null,
        outputTokensPerSecond: 56,
      },
    }),
    point({
      identity: 'point-blis-b200',
      simulator: 'blis',
      model: 'org/other',
      accelerator: 'B200',
      workload: '1024/128',
      configurationKey: 'configuration-blis-b200',
      measured: { ttftMs: 100, tpotMs: 10, outputTokensPerSecond: 100 },
      predicted: {
        ttftMs: 170,
        tpotMs: 19,
        requestLatencyMs: null,
        outputTokensPerSecond: 53,
      },
    }),
  ];
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

interface MountOptions {
  /** `null` leaves the point detail unfetched; `undefined` uses the fixture. */
  points?: ScorecardPoint[] | null;
  pointsLoading?: boolean;
  pointsError?: string | null;
  loadPoints?: () => void;
  reload?: () => void;
}

async function mount(
  preloaded: ScorecardSummary | null,
  search = '',
  options: MountOptions = {},
): Promise<void> {
  window.history.replaceState({}, '', `/validation/scorecard${search}`);
  navigation.search = window.location.search;
  scorecardState.set({
    summary: preloaded,
    points: options.points === undefined
      ? (preloaded ? publishedPoints() : null)
      : options.points,
    pointsLoading: options.pointsLoading ?? false,
    pointsError: options.pointsError ?? null,
    notPublished: preloaded === null,
    error: null,
    isLoading: false,
    reload: options.reload ?? vi.fn(),
    loadPoints: options.loadPoints ?? vi.fn(),
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

  it('puts the coverage map above the filter panel', async () => {
    await mount(published());

    const coverage = container.querySelector('[data-tour="coverage"]');
    const filters = container.querySelector('[data-tour="filters"]');

    expect(coverage).not.toBeNull();
    expect(filters).not.toBeNull();
    expect(
      coverage!.compareDocumentPosition(filters!) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
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
    expect(container.textContent).toContain('P90');
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

  it('puts the measured sources above the page actions', async () => {
    // Benchmark comparison stacks provenance first and the copy and tour
    // controls under it; the two validation pages must read the same way.
    await mount(published());
    const badges = container.querySelector('[data-tour="sources"]');
    const copy = Array.from(container.querySelectorAll('button'))
      .find(candidate => candidate.textContent?.includes('Copy link'));
    expect(badges).not.toBeNull();
    expect(copy).toBeDefined();
    expect(badges!.compareDocumentPosition(copy!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('names both measured sources and drops the gateway and release chrome', async () => {
    await mount(published());
    expect(container.textContent).toContain('Benchmark data from Red Hat');
    expect(container.textContent).toContain('SemiAnalysis InferenceX');
    expect(container.textContent).toContain('786 rows');
    // The snapshot timestamp is 19:22 UTC: the next day east of UTC, so the
    // badge names the UTC date rather than the reader's local one.
    expect(container.textContent).toContain('snapshot 11 Nov 2026');
    expect(container.textContent).not.toContain('Gateway');
    expect(container.textContent).not.toContain('Simulator releases');
    expect(container.textContent).not.toContain('https://aisimulators.example.test');
  });

  it('names a measured source the sweep could not load, and why', async () => {
    // A snapshot that loaded one source must not look like a complete one:
    // the reader sees which data is missing before reading any comparison.
    await mount(published({
      datasets: [{
        source: 'redhat',
        id: 'redhat-performance/configiq-performance-upstream-data',
        revision: 'abc1234',
        rowCount: 10,
      }],
      incomplete: [{
        leg: 'inferencex',
        reason: 'Private InferenceX dataset access is not configured',
      }],
    }));

    expect(container.textContent).toContain('Benchmark data from SemiAnalysis InferenceX unavailable');
    expect(container.textContent).toContain('Private InferenceX dataset access is not configured');
  });

  it('does not claim a source is unavailable when every source loaded', async () => {
    await mount(published());
    expect(container.textContent).not.toContain('unavailable');
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

  it('sizes the error tile to its rows instead of reserving a blank box', async () => {
    await mount(published());
    const card = container.querySelector<HTMLElement>('[data-flip-card="scorecard-error"]');
    const front = card?.querySelector<HTMLElement>('[class*="flipFront"]');

    // The front face is in flow and caps its own content; the card has no
    // fixed height, so a filtered view ends where the chart ends.
    expect(card?.style.height).toBe('');
    expect(card?.className).toContain('flipCardFit');
    expect(front?.style.maxHeight).toBe('560px');
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
    // One best each: an even outcome bar, described for assistive tech. The
    // default TPOT slice carries the same result as TTFT in this fixture.
    expect(container.querySelector('[aria-label*="best on 1 of 2 paired levels"]')).not.toBeNull();
  });

  it('puts best share by model and GPU on the head-to-head tile front', async () => {
    await mount(published());
    const card = container.querySelector('[data-flip-card="scorecard-paired"]');

    expect(card).not.toBeNull();
    expect(card?.textContent).toContain('Best share by model');
    expect(card?.textContent).toContain('Best share by GPU');
    expect(card?.querySelector('[class*="flipFront"] table')).toBeNull();

    const back = card?.querySelector('[class*="flipBack"] table');
    expect(back).not.toBeNull();
    expect(back?.textContent).toContain('Source');
    expect(back?.textContent).toContain('Ties');
    expect(back?.textContent).not.toContain('Paired levels');
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

  it('explains agreement and coverage from a single help icon', async () => {
    await mount(published());
    const heading = container.querySelector('#scorecard-headline-title');
    const head = heading?.parentElement;

    expect(head?.querySelectorAll('button[aria-label^="Help for"]').length).toBe(1);
    expect(head?.querySelector('[aria-label="Help for Agreement and coverage"]')).not.toBeNull();
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
    await mount(published({
      coverage: [
        coverageFor('aisim'),
        coverageFor('blis', { supported: 1 }),
        coverageFor('aisim', { model: 'org/other', accelerator: 'B200', supported: 1, unsupported: 0 }),
        coverageFor('blis', { model: 'org/other', accelerator: 'B200', supported: 1, unsupported: 0 }),
      ],
    }));

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

    // The row filter reaches the agreement tiles: the point detail folds to
    // the H200 slice rather than leaving pooled totals above a filtered table.
    const agreement = container.querySelector('[data-tour="agreement"]');
    expect(agreement?.textContent).toContain('25.0%');
    expect(agreement?.textContent).toContain('35.0%');
    expect(agreement?.textContent).not.toContain('11.0%');

    // The coverage map stays a whole-snapshot view: a row filter must not
    // shrink the Venn diagrams, or their overlap stops describing the data.
    const coverage = container.querySelector('[data-tour="coverage"]');
    expect(coverage?.textContent).toContain('org/model');
    expect(coverage?.textContent).toContain('org/other');
    expect(coverage?.textContent).not.toContain('Applying the row filters');
  });

  it('asks for the point detail when a row filter is active and it is not loaded', async () => {
    const loadPoints = vi.fn();
    await mount(published(), '?v=1&gpu=H200', { points: null, loadPoints });

    expect(loadPoints).toHaveBeenCalled();
    expect(container.textContent).toContain('Applying the row filters');
    // The status replaces the tiles while the detail loads, so it has to be
    // announced rather than only rendered.
    const status = container.querySelector('[data-tour="agreement"] [role="status"]');
    expect(status?.textContent).toContain('Applying the row filters');
  });

  it('keeps the whole-snapshot numbers under a warning when the point detail fails', async () => {
    await mount(published(), '?v=1&gpu=H200', {
      points: null,
      pointsError: 'The point detail could not be loaded.',
    });

    expect(container.textContent).toContain('The row filter could not be applied');
    // The pooled totals are still visible, but under an explicit warning
    // rather than silently presented as the filtered slice.
    expect(container.querySelector('[data-tour="agreement"]')?.textContent).toContain('11.0%');
  });

  it('selects a coverage point into the model and GPU filters and shares it', async () => {
    await mount(published({
      coverage: [
        coverageFor('aisim'),
        coverageFor('blis', { supported: 1 }),
        coverageFor('aisim', {
          model: 'org/other',
          accelerator: 'B200',
          supported: 1,
          unsupported: 0,
          total: 1,
        }),
        coverageFor('blis', {
          model: 'org/other',
          accelerator: 'B200',
          supported: 1,
          unsupported: 0,
          total: 1,
        }),
      ],
      cells: [
        cellFor('aisim', { metric: 'tpot', statistic: 'mdape', value: 0.11 }),
        cellFor('blis', { metric: 'tpot', statistic: 'mdape', value: 0.19 }),
        cellFor('aisim', {
          model: 'org/other',
          accelerator: 'B200',
          metric: 'tpot',
          statistic: 'mdape',
          value: 0.3,
        }),
        cellFor('blis', {
          model: 'org/other',
          accelerator: 'B200',
          metric: 'tpot',
          statistic: 'mdape',
          value: 0.4,
        }),
      ],
    }), '?workload=1024%2F128');

    const dot = container.querySelector<HTMLButtonElement>('button[aria-label="org/other, B200"]');
    expect(dot).not.toBeNull();

    // Start with a workload selected: the click has to clear it, and a test
    // that began with nothing selected could not tell that it did.
    const workloadBefore = container.querySelector<HTMLSelectElement>('select[aria-label="Workload"]');
    expect(workloadBefore?.value).toBe('1024/128');

    await act(async () => {
      dot!.click();
      await Promise.resolve();
    });

    const model = container.querySelector<HTMLInputElement>('input[aria-label="Model"]');
    const workload = container.querySelector<HTMLSelectElement>('select[aria-label="Workload"]');
    expect(model?.value).toBe('org/other');
    expect(gpuToggle().textContent).toContain('B200');
    // A dot names a pair, not a shape: the workload is not part of the click.
    expect(workload?.value).toBe('');

    await act(async () => {
      button('Copy link')?.click();
      await Promise.resolve();
    });

    const params = new URL(clipboard.text).searchParams;
    expect(params.get('model')).toBe('org/other');
    expect(params.get('gpu')).toBe('B200');
    expect(params.get('workload')).toBeNull();
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

    // The selected slice is redhat TTFT MAPE over the two AISim configurations
    // the workload filter keeps (44% and 60%), not the default TPOT MdAPE
    // pooled total (11.0%).
    expect(container.textContent).toContain('52.0%');
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

  it('applies the comparison pair from a link and copies it back', async () => {
    await mount(published(), '?v=1&pair=aisim%7Cother');

    const select = container.querySelector<HTMLSelectElement>('select[aria-label="Simulator pair"]');
    expect(select?.value).toBe('aisim|other');

    await act(async () => {
      button('Copy link')?.click();
      await Promise.resolve();
    });

    expect(new URL(clipboard.text).searchParams.get('pair')).toBe('aisim|other');
  });

  it('shares the pair the reader picks from the selector', async () => {
    await mount(published());
    const select = container.querySelector<HTMLSelectElement>('select[aria-label="Simulator pair"]');

    await act(async () => {
      select!.value = 'aisim|other';
      select!.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await act(async () => {
      button('Copy link')?.click();
      await Promise.resolve();
    });

    expect(new URL(clipboard.text).searchParams.get('pair')).toBe('aisim|other');
  });

  it('does not share a pair the reader never chose', async () => {
    await mount(published());

    await act(async () => {
      button('Copy link')?.click();
      await Promise.resolve();
    });

    expect(new URL(clipboard.text).searchParams.get('pair')).toBeNull();
  });

  it('falls back to the first pair and shares that fallback when a link names one the snapshot lacks', async () => {
    await mount(published(), '?v=1&pair=aisim%7Cghost');

    const select = container.querySelector<HTMLSelectElement>('select[aria-label="Simulator pair"]');
    expect(select?.value).toBe('aisim|blis');

    // The link named a pair this snapshot does not contain, so the view is the
    // first pair, and the copied link must describe that view rather than the
    // stale choice it arrived with.
    await act(async () => {
      button('Copy link')?.click();
      await Promise.resolve();
    });
    expect(new URL(clipboard.text).searchParams.get('pair')).toBeNull();
  });

  it('warns about and drops an unknown parameter rather than failing the view', async () => {
    await mount(published(), '?v=1&unexpected=1');
    expect(container.textContent).toContain('Part of this link was ignored');
    expect(container.textContent).toContain('11.0%');
  });
});

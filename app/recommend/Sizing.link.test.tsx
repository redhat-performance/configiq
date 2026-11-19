// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { RecommendParams } from '@/contexts/RecommendContext';

// The page reads its shared view through `useSearchParams`. Tests mount it
// directly, so supply the current window search rather than the app router.
const navigation = vi.hoisted(() => ({ search: '' }));

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(navigation.search),
}));

const startSizing = vi.hoisted(() => vi.fn());
const startPareto = vi.hoisted(() => vi.fn());
const recommendState = vi.hoisted(() => ({ isLoading: false }));

vi.mock('@/contexts/RecommendContext', () => ({
  useRecommend: () => ({
    params: null,
    runMode: null,
    isLoading: recommendState.isLoading,
    result: null,
    error: null,
    errorCode: null,
    elapsed: 0,
    debugRequest: null,
    debugResponse: null,
    debugStatus: null,
    debugDuration: null,
    frontier: { status: 'idle' },
    startSizing,
    startPareto,
    reset: vi.fn(),
  }),
}));

const settings = vi.hoisted(() => ({
  simulator: 'aisimulate',
  applyViewForSession: vi.fn((view: { simulator?: string }) => {
    // Model the production replacement: an override without a simulator falls
    // back to the saved profile.
    settings.simulator = view.simulator ?? 'aisimulate';
  }),
}));

vi.mock('@/contexts/SettingsContext', () => ({
  useSettings: () => ({
    hydrated: true,
    hfToken: '',
    defaultModel: 'Qwen/Qwen2.5-7B-Instruct',
    defaultSystem: 'h200_sxm',
    inferenceBackend: 'vllm',
    backendVersion: '0.9.0',
    simulator: settings.simulator,
    costingsEnabled: false,
    pricingSource: 'merged',
    preferredCloudProvider: null,
    setDefaultSystem: vi.fn(),
    setDefaultModel: vi.fn(),
    applyViewForSession: settings.applyViewForSession,
  }),
}));

const versionSupport = vi.hoisted(() => ({
  aisimulate: {} as Record<string, { disaggregated: boolean }>,
  blis: {} as Record<string, { disaggregated: boolean }>,
}));

// A simulator's fabric capability is catalog data, so the fixture keeps it in
// one place: the default shape claims no fabric, and the fabric-specific tests
// opt into the capability they exercise.
const aisimulateFabric = vi.hoisted(() => ({
  required: false,
  networks: [] as string[],
  defaultNetwork: null as string | null,
}));

// Which comparison options a capacity run can return, as the catalog reports it.
const capacityCapability = vi.hoisted(() => ({ disaggregated: false }));

vi.mock('@/lib/hooks/useCatalog', () => {
  const buildCatalog = () => {
  const backend = {
    id: 'vllm',
    aisimulateId: 'vllm',
    versions: ['0.9.0'],
    defaultVersion: '0.9.0',
    memoryFraction: null,
    memoryFractionKind: 'of_total',
    runtimeField: 'gpu_memory_utilization',
    systems: {},
    systemDefaults: {},
    disaggregationRequiresFabric: true,
    capacityOptions: { disaggregated: capacityCapability.disaggregated },
  };
  const aisimulateBackend = {
    ...backend,
    disaggregationRequiresFabric: aisimulateFabric.required,
    versionSupport: versionSupport.aisimulate,
  };
  const blisBackend = { ...backend, versionSupport: versionSupport.blis };
  const catalog = {
    modelOptions: ['Qwen/Qwen2.5-7B-Instruct'],
    gpuOptions: [
      { systemId: 'h200_sxm', label: 'NVIDIA H200', vendor: 'nvidia', vramGb: 141 },
      { systemId: 'h100_sxm', label: 'NVIDIA H100', vendor: 'nvidia', vramGb: 80 },
    ],
    modelSpecs: new Map(),
    networkOptions: ['ib-400g'],
    backendOptions: [aisimulateBackend],
    simulators: [
      { id: 'aisimulate', label: 'AISimulate', version: null, default: true },
      { id: 'blis', label: 'BLIS', version: 'test-blis', default: false },
    ],
    simulatorSupport: {
      aisimulate: {
        id: 'aisimulate',
        label: 'AISimulate',
        version: null,
        recommendSearch: 'single',
        modelsExhaustive: true,
        modelSpecs: new Map(),
        networks: aisimulateFabric.networks,
        defaultNetwork: aisimulateFabric.defaultNetwork,
        models: ['Qwen/Qwen2.5-7B-Instruct'],
        systems: ['h200_sxm'],
        backends: [aisimulateBackend],
        objectives: ['min_gpus', 'throughput_demand', 'concurrency_capacity', 'pareto'],
      },
      // The second catalog prices the linked system that the default one does
      // not offer, so accepting the shared simulator makes it selectable again.
      blis: {
        id: 'blis',
        label: 'BLIS',
        version: 'test-blis',
        recommendSearch: 'single',
        modelsExhaustive: true,
        modelSpecs: new Map(),
        networks: ['ib-400g'],
        defaultNetwork: 'ib-400g',
        models: ['Qwen/Qwen2.5-7B-Instruct'],
        systems: ['h200_sxm', 'h100_sxm'],
        backends: [blisBackend],
        objectives: ['min_gpus', 'throughput_demand', 'concurrency_capacity', 'pareto'],
      },
    },
    objectives: ['min_gpus', 'throughput_demand', 'concurrency_capacity', 'pareto'],
    timeoutSeconds: 60,
    isLoading: false,
  };
  return catalog;
  };
  return {
    useCatalog: () => buildCatalog(),
    engineVersionSupport: (
      engine: { versionSupport?: Record<string, { disaggregated: boolean }> } | null,
      version: string | null,
    ) => (engine && version ? engine.versionSupport?.[version] ?? null : null),
  };
});

vi.mock('@/lib/hooks/useTestedModels', () => ({
  useTestedModels: () => ({ pairs: [], validatedModelIds: [], isAvailable: false }),
}));

vi.mock('@/lib/hooks/useTestedClassifier', () => ({
  useTestedClassifier: () => ({ prediction: null, isLoading: false }),
}));

vi.mock('@/lib/hooks/useCostings', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/hooks/useCostings')>();
  return {
    ...actual,
    useCostings: () => ({
      models: [],
      gpuCloudRates: new Map(),
      gpuHardwareCosts: new Map(),
      isLoading: false,
    }),
  };
});

vi.mock('@/lib/huggingface/fetch-config', () => ({
  fetchModelConfig: vi.fn(async () => ({ success: false, error: 'not used' })),
}));

vi.mock('@/lib/app-config', () => ({
  getAppConfig: () => ({
    defaultModel: 'Qwen/Qwen2.5-7B-Instruct',
    defaultSystem: 'h200_sxm',
    defaultOpenModel: 'Qwen/Qwen2.5-7B-Instruct',
    defaultFrontierModel: 'claude-fable-5',
    defaultBackend: 'vllm',
    approvedModels: [],
    huggingFaceModels: [],
    suggestedModelNames: [],
    modelRequestUrl: '',
    workloadPresets: [
      { key: 'rag', label: 'RAG / Search', isl: 16384, osl: 512 },
      { key: 'voice', label: 'Voice / Real-time', isl: 512, osl: 128 },
    ],
    concurrencyPresets: [],
  }),
}));

// Capture the link builder so a test can read the URL the page would copy.
const shareUrl = vi.hoisted(() => ({
  getUrl: null as null | (() => string | null),
}));
vi.mock('@/components/help/CopyLinkButton', () => ({
  CopyLinkButton: (props: { getUrl: () => string | null }) => {
    shareUrl.getUrl = props.getUrl;
    return null;
  },
}));
vi.mock('@/components/help/TourButton', () => ({ TourButton: () => null }));

const { default: Sizing } = await import('./Sizing');

let root: ReturnType<typeof createRoot>;
let container: HTMLDivElement;

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true;
  startSizing.mockClear();
  startPareto.mockClear();
  for (const support of [versionSupport.aisimulate, versionSupport.blis]) {
    for (const version of Object.keys(support)) delete support[version];
  }
  recommendState.isLoading = false;
  aisimulateFabric.required = false;
  aisimulateFabric.networks = [];
  aisimulateFabric.defaultNetwork = null;
  capacityCapability.disaggregated = false;
  settings.simulator = 'aisimulate';
  settings.applyViewForSession.mockClear();
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => ({ data: [] }),
  })));
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  window.history.replaceState({}, '', '/recommend');
  vi.unstubAllGlobals();
});

async function mountAt(search: string) {
  window.history.replaceState({}, '', `/recommend${search}`);
  navigation.search = window.location.search;
  await act(async () => {
    root.render(<Sizing />);
    await Promise.resolve();
  });
}

async function calculate(): Promise<RecommendParams> {
  const button = Array.from(container.querySelectorAll('button'))
    .find(candidate => candidate.textContent === 'Calculate');
  expect(button).toBeDefined();
  await act(async () => {
    button?.click();
    await Promise.resolve();
  });
  expect(startSizing).toHaveBeenCalled();
  const calls = startSizing.mock.calls;
  return calls[calls.length - 1][0] as RecommendParams;
}

async function pareto(): Promise<RecommendParams> {
  const button = Array.from(container.querySelectorAll('button'))
    .find(candidate => candidate.textContent === 'Pareto frontier');
  expect(button).toBeDefined();
  await act(async () => {
    button?.click();
    await Promise.resolve();
  });
  expect(startPareto).toHaveBeenCalled();
  const calls = startPareto.mock.calls;
  return calls[calls.length - 1][0] as RecommendParams;
}

/**
 * Mount a view for a simulator whose catalog reports that it models P/D through
 * an inter-node fabric. The fixture that declares the capability is selected
 * here, so the fabric tests exercise the control rather than assume it.
 */
async function mountFabricView(search: string) {
  settings.simulator = 'blis';
  await mountAt(search);
}

/**
 * Press a serving mode in the expanded panel.
 *
 * Pressing the pressed view is the only way back to the unpinned default, so
 * every clear-control test goes through this.
 */
async function pressView(label: 'Aggregated' | 'Disaggregated') {
  await act(async () => {
    Array.from(container.querySelectorAll('button'))
      .find(button => button.textContent?.includes('Adjust?'))
      ?.click();
    await Promise.resolve();
  });
  const view = container.querySelector('[role="group"][aria-label="Serving mode"]');
  const button = Array.from(view?.querySelectorAll('button') ?? [])
    .find(candidate => candidate.textContent === label);
  expect(button).toBeDefined();
  await act(async () => {
    button?.click();
    await Promise.resolve();
  });
}

async function selectDisaggregated() {
  await act(async () => {
    Array.from(container.querySelectorAll('button'))
      .find(button => button.textContent?.includes('Adjust?'))
      ?.click();
    await Promise.resolve();
  });
  const disaggregated = Array.from(container.querySelectorAll('button'))
    .find(button => button.textContent === 'Disaggregated');
  expect(disaggregated).toBeDefined();
  expect(disaggregated?.hasAttribute('disabled')).toBe(false);
  // Pressing a pressed view clears it, so only press when it is not pinned.
  if (disaggregated?.getAttribute('aria-pressed') === 'true') return;
  await act(async () => {
    disaggregated?.click();
    await Promise.resolve();
  });
}

async function acceptSharedSimulator() {
  const button = Array.from(container.querySelectorAll('button'))
    .find(candidate => /Use BLIS for this view/.test(candidate.textContent ?? ''));
  expect(button).toBeDefined();
  await act(async () => {
    button?.click();
    await Promise.resolve();
  });
}

describe('recommend sizing shared views', () => {
  it('forwards the per-pool context limits a legacy link carries', async () => {
    // A link written before the serving mode existed names pool lengths only.
    // The page keeps the pre-mode search, so those lengths apply to a
    // disaggregated request even though the link never says so.
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&prefillMaxSeqLen=8192&decodeMaxSeqLen=4096',
    );

    const sent = await calculate();
    expect(sent.prefill_max_seq_len).toBe(8192);
    expect(sent.decode_max_seq_len).toBe(4096);
    expect(sent.mode).toBe('disagg');
  });

  it('restores disaggregated serving and its fabric from a shared link', async () => {
    await mountFabricView(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&servingMode=disagg&fabric=ib-400g',
    );

    expect(container.querySelector('#recommend-network-fabric')).not.toBeNull();
    const sent = await calculate();
    expect(sent.fabric).toBe('ib-400g');
    expect(sent.mode).toBe('disagg');
  });

  it('forwards the named fabric for disaggregated serving', async () => {
    await mountFabricView('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency&fabric=ib-400g');

    await selectDisaggregated();
    expect(container.querySelector('#recommend-network-fabric')).not.toBeNull();
    const sent = await calculate();
    expect(sent.fabric).toBe('ib-400g');
    expect(sent.mode).toBe('disagg');
  });

  it('searches prefill/decode by default for a latency link that names a fabric', async () => {
    // The link predates the serving-mode field. Its named fabric described the
    // P/D search the page made before the control existed, so the default must
    // keep searching that topology instead of silently narrowing to aggregated.
    await mountFabricView('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency&fabric=ib-400g');

    expect(container.querySelector('#recommend-network-fabric')).not.toBeNull();
    const sizing = await calculate();
    expect(sizing.fabric).toBe('ib-400g');
    expect(sizing.mode).toBe('disagg');
    const sweep = await pareto();
    expect(sweep.fabric).toBe('ib-400g');
    expect(sweep.mode).toBeUndefined();
  });

  it('honours an explicitly aggregated link even when it names a fabric', async () => {
    await mountFabricView(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&servingMode=agg&fabric=ib-400g',
    );

    const sizing = await calculate();
    // Aggregated is an explicit null fabric, not an omission the simulator
    // would fill with its default.
    expect(sizing.fabric).toBeNull();
    expect(sizing.mode).toBe('agg');

    // The frontier reads the same view: an aggregated view asks for aggregate
    // candidates only, and the mode itself is omitted (the sweep rejects it).
    const sweep = await pareto();
    expect(sweep.fabric).toBeNull();
    expect(sweep.mode).toBeUndefined();
  });

  it('shares the fabric the request submits, not one it merely remembers', async () => {
    // The remembered fabric name stays selected when the view is aggregated,
    // but the request sends an explicit null. A link that exported the name
    // would describe a P/D topology the recipient would inherit.
    await mountFabricView(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&servingMode=agg&fabric=ib-400g',
    );

    const url = shareUrl.getUrl?.() ?? '';
    expect(url).toContain('servingMode=agg');
    expect(url).toContain('fabric=none');
    expect(url).not.toContain('fabric=ib-400g');
  });

  it('reads the serving mode in the collapsed summary', async () => {
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&servingMode=agg',
    );

    // The summary is the read-only view of the form; a mode the user cannot
    // see without expanding the panel is not a choice they made.
    expect(container.textContent).toContain('Serving mode: aggregated.');
  });

  it('leaves the deployment to the service until a view is pinned', async () => {
    // The page's default is no narrowing at all: no mode, and no fabric that
    // would settle the topology the service is free to choose.
    await mountFabricView('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency');

    expect(container.textContent).toContain(
      'Serving mode: service default (aggregated or disaggregated).',
    );
    const sizing = await calculate();
    expect(sizing.mode).toBeUndefined();
    expect(sizing.fabric).toBeUndefined();
    const sweep = await pareto();
    expect(sweep.mode).toBeUndefined();
    expect(sweep.fabric).toBeUndefined();
  });

  it('leaves the serving mode out of a link while no view is pinned', async () => {
    // A link describes the request it would reproduce. An unpinned view sends
    // no mode and no fabric, so a link that named either would make the
    // recipient answer a different question.
    await mountFabricView('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency');

    const url = shareUrl.getUrl?.() ?? '';
    // Assert a real link first: every check below is an omission, which an
    // empty string would satisfy whether or not sharing works at all.
    expect(url.startsWith('/recommend?')).toBe(true);
    expect(url).not.toContain('servingMode=');
    expect(url).not.toContain('fabric=');
    expect(url).not.toContain('prefillMaxSeqLen=');
    expect(url).not.toContain('decodeMaxSeqLen=');
  });

  it('keeps Disaggregated reachable after clearing an aggregate-only fabric', async () => {
    // Choosing "None" remembers an explicit aggregate-only fabric. Clearing the
    // pin must not strand that value: the default hides the selector, so a
    // remembered "None" would disable Disaggregated with no way back to it.
    await mountFabricView('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency&fabric=none');
    expect((await calculate()).mode).toBe('agg');

    await pressView('Aggregated');

    const view = container.querySelector('[role="group"][aria-label="Serving mode"]');
    const disaggregated = Array.from(view?.querySelectorAll('button') ?? [])
      .find(button => button.textContent === 'Disaggregated');
    expect(disaggregated?.hasAttribute('disabled')).toBe(false);

    const sent = await calculate();
    expect(sent.mode).toBeUndefined();
    expect(sent.fabric).toBeUndefined();
  });

  it('returns to the service default when the pinned view is pressed again', async () => {
    await mountFabricView(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&servingMode=disagg&fabric=ib-400g',
    );
    expect((await calculate()).mode).toBe('disagg');

    // The unpinned default must stay reachable: pressing the pressed view is
    // the only way back, so it has to clear rather than re-pin.
    await pressView('Disaggregated');

    const sent = await calculate();
    expect(sent.mode).toBeUndefined();
    expect(sent.fabric).toBeUndefined();
  });

  it('clears a pinned view the form cannot submit', async () => {
    versionSupport.aisimulate['0.9.0'] = { disaggregated: false };
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&servingMode=disagg',
    );

    await act(async () => {
      Array.from(container.querySelectorAll('button'))
        .find(button => button.textContent?.includes('Adjust?'))
        ?.click();
      await Promise.resolve();
    });
    const view = container.querySelector('[role="group"][aria-label="Serving mode"]');
    const disaggregated = Array.from(view?.querySelectorAll('button') ?? [])
      .find(button => button.textContent === 'Disaggregated');
    // The held view cannot be submitted, but the press that clears it must
    // stay available: dropping your own choice cannot depend on making
    // another one first.
    expect(disaggregated?.hasAttribute('disabled')).toBe(false);
    await act(async () => {
      disaggregated?.click();
      await Promise.resolve();
    });

    const sent = await calculate();
    expect(sent.mode).toBeUndefined();
    expect(sent.fabric).toBeUndefined();
  });

  it('omits the serving mode from a summary for a goal without one', async () => {
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=capacity');

    expect(container.textContent).not.toContain('Serving mode:');
  });

  it('does not offer a serving mode for a demand link', async () => {
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=demand');

    await act(async () => {
      Array.from(container.querySelectorAll('button'))
        .find(button => button.textContent?.includes('Adjust?'))
        ?.click();
      await Promise.resolve();
    });

    // The simulator reports the deployment that sustains the rate, so the page
    // must not claim one, and the request must not narrow the search.
    expect(Array.from(container.querySelectorAll('button'))
      .some(button => button.textContent === 'Disaggregated')).toBe(false);
    const sent = await calculate();
    expect(sent.mode).toBeUndefined();
    expect(sent.fabric).toBeUndefined();
  });

  it('restores the pre-mode topology a legacy link describes', async () => {
    // A link applied to an already-open page replaces the view it names. A
    // pre-mode link carries its topology only through the fabric and pool
    // lengths, so those set the mode instead of inheriting the reader's.
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&servingMode=agg&fabric=ib-400g',
    );
    expect((await calculate()).mode).toBe('agg');

    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency&fabric=ib-400g');
    expect((await calculate()).mode).toBe('disagg');
  });

  it('omits the serving mode for capacity, which reports both topologies', async () => {
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=capacity');

    const sent = await calculate();
    // Capacity sizing enumerates aggregated shapes and returns disaggregated
    // comparison options separately, so it ignores mode and rejects a fabric.
    expect(sent.mode).toBeUndefined();
    expect(sent.fabric).toBeUndefined();
  });

  it('asks for aggregate-only candidates when the link says "no fabric"', async () => {
    // A "no fabric" choice is not an omission: it must reach the API as an
    // explicit null rather than falling back to the simulator's default fabric.
    await mountFabricView('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency&fabric=none');

    const sizing = await calculate();
    expect(sizing.fabric).toBeNull();
    expect(sizing.mode).toBe('agg');
    expect((await pareto()).fabric).toBeNull();
  });

  it('leaves the fabric to the simulator when the request omits one', async () => {
    await mountFabricView('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency');

    await selectDisaggregated();
    const sent = await calculate();
    expect(sent.fabric).toBe('ib-400g');
    expect(sent.mode).toBe('disagg');
  });

  it('sends no fabric when the simulator models P/D without one', async () => {
    // With no fabric capability in the catalog, the control stays hidden and
    // the request must not claim a topology the simulator cannot model.
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&servingMode=disagg&prefillMaxSeqLen=8192&decodeMaxSeqLen=4096',
    );

    expect(container.querySelector('#recommend-network-fabric')).toBeNull();
    const sent = await calculate();
    expect(sent.fabric).toBeUndefined();
    expect(sent.mode).toBe('disagg');
    expect(sent.prefill_max_seq_len).toBe(8192);
    expect(sent.decode_max_seq_len).toBe(4096);
  });

  it('keeps a linked disaggregated mode aggregated until the accepted simulator can model it', async () => {
    versionSupport.aisimulate['0.9.0'] = { disaggregated: false };
    versionSupport.blis['0.9.0'] = { disaggregated: true };
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&simulator=blis&backend=vllm&backendVersion=0.9.0&servingMode=disagg&fabric=ib-400g',
    );

    // The reader's simulator cannot model prefill/decode, so the link's mode
    // stays held and the form stays aggregated until they accept. The held
    // view's backend reports no fabric, so the aggregate request names none.
    expect(container.querySelector('#recommend-network-fabric')).toBeNull();
    const heldSizing = await calculate();
    expect(heldSizing.fabric).toBeUndefined();
    expect(heldSizing.mode).toBe('agg');

    await acceptSharedSimulator();

    expect(container.querySelector('#recommend-network-fabric')).not.toBeNull();
    expect((await calculate()).fabric).toBe('ib-400g');
  });

  it('stays aggregated when the accepted simulator cannot model prefill/decode', async () => {
    versionSupport.aisimulate['0.9.0'] = { disaggregated: false };
    versionSupport.blis['0.9.0'] = { disaggregated: false };
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&simulator=blis&backend=vllm&backendVersion=0.9.0&servingMode=disagg&fabric=ib-400g',
    );

    await acceptSharedSimulator();

    expect(container.querySelector('#recommend-network-fabric')).toBeNull();
    expect((await calculate()).fabric).toBeNull();
  });

  it('offers the shared simulator instead of switching silently', async () => {
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&simulator=blis&backend=vllm&backendVersion=0.9.0',
    );

    expect(container.textContent).toContain('This view was shared using');
    expect(container.textContent).toContain('BLIS');
    // Nothing is switched and nothing is persisted until the reader accepts.
    expect(settings.applyViewForSession).not.toHaveBeenCalled();

    await act(async () => {
      Array.from(container.querySelectorAll('button'))
        .find(button => /Use BLIS for this view/.test(button.textContent ?? ''))
        ?.click();
    });

    expect(settings.applyViewForSession).toHaveBeenCalledWith({
      simulator: 'blis',
      backend: 'vllm',
      backendVersion: '0.9.0',
    });
  });

  it('restores the linked system once the reader accepts its simulator', async () => {
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm&goal=latency' +
      '&simulator=blis&backend=vllm&backendVersion=0.9.0',
    );

    // The reader's simulator does not price the linked system, so the form
    // falls back while the banner offers the sender's simulator.
    expect((await calculate()).system).toBe('h200_sxm');

    await acceptSharedSimulator();

    // The sender's simulator prices it, so the next calculation must use the
    // shared system rather than the fallback selected while the banner was up.
    expect((await calculate()).system).toBe('h100_sxm');
  });

  it('marks output stale when the reader accepts another simulator', async () => {
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&simulator=blis&backend=vllm&backendVersion=0.9.0',
    );
    // Simulate output the reader can see (or a run in flight) while deciding.
    recommendState.isLoading = true;
    await act(async () => {
      root.render(<Sizing />);
      await Promise.resolve();
    });
    expect(container.textContent).not.toContain('Inputs have changed');

    await acceptSharedSimulator();

    // The visible result was produced by the previous simulator.
    expect(container.textContent).toContain('Inputs have changed');
  });

  it('adopts the shared backend and version when the simulator already matches', async () => {
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&simulator=aisimulate&backend=vllm&backendVersion=0.9.0',
    );

    expect(settings.applyViewForSession).toHaveBeenCalledWith({
      backend: 'vllm',
      backendVersion: '0.9.0',
    });
    expect(container.textContent).not.toContain('This view was shared using');
  });

  it('keeps whatever the link omits at the recipient value', async () => {
    // A partial link changes only what it names. Filling in the rest would
    // silently rewrite the view the recipient already had open.
    await mountAt('?v=1&isl=4096');

    const sent = await calculate();
    expect(sent.isl).toBe(4096);
    expect(sent.model_path).toBe('Qwen/Qwen2.5-7B-Instruct');
    expect(sent.system).toBe('h200_sxm');
    expect(sent.backend).toBe('vllm');
  });

  it('carries the concurrency a latency link was sized for', async () => {
    // Latency sizing submits a target concurrency, so a link that dropped it
    // would describe a one-user view instead of the shared one.
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency&concurrency=32&ttft=900&tpot=20',
    );

    const sent = await calculate();
    expect(sent.target_concurrency).toBe(32);
    expect(sent.target_request_rate).toBeUndefined();
  });

  it('keeps a fractional TTFT the form accepts', async () => {
    // The TTFT input takes decimals and the API takes a float. An
    // integer-only link field would drop the sender's target and keep the
    // recipient's instead.
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency&ttft=1.4');

    const sent = await calculate();
    expect(sent.ttft).toBe(1.4);
    expect(container.textContent).not.toContain('Ignored "ttft"');
  });

  it('selects the workload the linked input shape names', async () => {
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&isl=16384&osl=512');

    const pressed = Array.from(container.querySelectorAll('button[aria-pressed="true"]'))
      .map(button => button.textContent);
    expect(pressed).toContain('RAG / Search');
    expect(pressed).not.toContain('Default');
  });

  it('drops the workload selection when a sequence length is cleared', async () => {
    // The numeric state keeps the last accepted length while the visible field
    // is empty. Selecting from it would leave a preset pressed for a workload
    // the form no longer states.
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm');

    const adjust = Array.from(container.querySelectorAll('button'))
      .find(button => (button.textContent ?? '').startsWith('Adjust?'));
    expect(adjust).toBeDefined();
    await act(async () => {
      adjust?.click();
      await Promise.resolve();
    });

    const label = Array.from(container.querySelectorAll('label'))
      .find(candidate => (candidate.textContent ?? '').includes('Avg input tokens (ISL)'));
    const isl = label?.nextElementSibling as HTMLInputElement | null;
    expect(isl).not.toBeNull();
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setter?.call(isl, '');
      isl?.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });

    const row = container.querySelector('[data-tour="workload-presets"]');
    expect(row).not.toBeNull();
    const pressed = Array.from(row?.querySelectorAll('button[aria-pressed="true"]') ?? []);
    expect(pressed).toHaveLength(0);
  });

  it('describes the capacity options the catalog reports', async () => {
    const search = '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=capacity';
    // The deployment-view block sits in the expandable input panel, so open it
    // once; it stays open across the remount below.
    const expand = async () => {
      const adjust = Array.from(container.querySelectorAll('button'))
        .find(button => (button.textContent ?? '').startsWith('Adjust?'));
      if (adjust == null) return;
      await act(async () => {
        adjust.click();
        await Promise.resolve();
      });
    };

    await mountAt(search);
    await expand();
    expect(container.textContent).toContain(
      'reports the aggregated deployments that hold the target',
    );

    // The catalog now reports a P/D comparison option, so the control explains
    // both kinds of result the service can return. The input panel stays open
    // across the re-render, so the changed copy is visible without reopening it.
    capacityCapability.disaggregated = true;
    await mountAt(search);
    expect(container.textContent).toContain(
      'reports aggregated and disaggregated options',
    );
  });
});

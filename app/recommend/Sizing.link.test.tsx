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
    startPareto: vi.fn(),
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

vi.mock('@/lib/hooks/useCatalog', () => {
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
    versionSupport: {},
    disaggregationRequiresFabric: true,
    capacityOptions: { disaggregated: false },
  };
  const catalog = {
    modelOptions: ['Qwen/Qwen2.5-7B-Instruct'],
    gpuOptions: [
      { systemId: 'h200_sxm', label: 'NVIDIA H200', vendor: 'nvidia', vramGb: 141 },
      { systemId: 'h100_sxm', label: 'NVIDIA H100', vendor: 'nvidia', vramGb: 80 },
    ],
    modelSpecs: new Map(),
    networkOptions: ['ib-400g'],
    backendOptions: [backend],
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
        networks: ['ib-400g'],
        defaultNetwork: 'ib-400g',
        models: ['Qwen/Qwen2.5-7B-Instruct'],
        systems: ['h200_sxm'],
        backends: [backend],
        objectives: ['min_gpus', 'throughput_demand', 'concurrency_capacity', 'pareto'],
      },
      // BLIS prices the linked system that AISimulate does not offer, so
      // accepting the shared simulator makes it selectable again.
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
        backends: [backend],
        objectives: ['min_gpus', 'throughput_demand', 'concurrency_capacity', 'pareto'],
      },
    },
    objectives: ['min_gpus', 'throughput_demand', 'concurrency_capacity', 'pareto'],
    timeoutSeconds: 60,
    isLoading: false,
  };
  return {
    useCatalog: () => catalog,
    engineVersionSupport: () => null,
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

vi.mock('@/components/help/CopyLinkButton', () => ({ CopyLinkButton: () => null }));
vi.mock('@/components/help/TourButton', () => ({ TourButton: () => null }));

const { default: Sizing } = await import('./Sizing');

let root: ReturnType<typeof createRoot>;
let container: HTMLDivElement;

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true;
  startSizing.mockClear();
  recommendState.isLoading = false;
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
  it('forwards the per-pool context limits the link carries', async () => {
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency' +
      '&prefillMaxSeqLen=8192&decodeMaxSeqLen=4096',
    );

    const sent = await calculate();
    expect(sent.prefill_max_seq_len).toBe(8192);
    expect(sent.decode_max_seq_len).toBe(4096);
  });

  it('forwards the named fabric', async () => {
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency&fabric=ib-400g');

    expect((await calculate()).fabric).toBe('ib-400g');
  });

  it('asks for aggregate-only candidates when the link says "no fabric"', async () => {
    // A "no fabric" choice is not an omission: it must reach the API as an
    // explicit null rather than falling back to the simulator's default fabric.
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency&fabric=none');

    expect((await calculate()).fabric).toBeNull();
  });

  it('leaves the fabric to the simulator when the link omits it', async () => {
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&goal=latency');

    expect((await calculate()).fabric).toBe('ib-400g');
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

    // AISimulate does not price the linked system, so the form falls back while
    // the banner offers the sender's simulator.
    expect((await calculate()).system).toBe('h200_sxm');

    await acceptSharedSimulator();

    // BLIS prices it, so the next calculation must use the shared system rather
    // than the fallback that was selected while the banner was up.
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
});

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

vi.mock('@/contexts/RecommendContext', () => ({
  useRecommend: () => ({
    params: null,
    runMode: null,
    isLoading: false,
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

vi.mock('@/contexts/SettingsContext', () => ({
  useSettings: () => ({
    hydrated: true,
    hfToken: '',
    defaultModel: 'Qwen/Qwen2.5-7B-Instruct',
    defaultSystem: 'h200_sxm',
    inferenceBackend: 'vllm',
    backendVersion: '0.9.0',
    simulator: 'aisimulate',
    costingsEnabled: false,
    pricingSource: 'merged',
    preferredCloudProvider: null,
    setDefaultSystem: vi.fn(),
    setDefaultModel: vi.fn(),
    applyViewForSession: vi.fn(),
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
    ],
    modelSpecs: new Map(),
    networkOptions: ['ib-400g'],
    backendOptions: [backend],
    simulators: [{ id: 'aisimulate', label: 'AISimulate', version: null, default: true }],
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

vi.mock('@/components/help/CopyLinkButton', () => ({ CopyLinkButton: () => null }));
vi.mock('@/components/help/TourButton', () => ({ TourButton: () => null }));

const { default: Sizing } = await import('./Sizing');

let root: ReturnType<typeof createRoot>;
let container: HTMLDivElement;

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true;
  startSizing.mockClear();
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
  expect(startSizing).toHaveBeenCalledTimes(1);
  return startSizing.mock.calls[0][0] as RecommendParams;
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
});

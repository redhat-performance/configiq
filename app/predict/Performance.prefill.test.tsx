// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { InferenceConfigResult } from '@/lib/gpu-math/inference-config';

// Deliberately not a SemVer release: the UI renders whatever version the
// catalog reports, and a plausible pin here would look like a real one.
const BLIS_FIXTURE_VERSION = vi.hoisted(() => 'test-blis-version');

// The page reads its shared-view link through `useSearchParams`. Tests mount it
// directly, so supply the current window search rather than the app router.
const navigation = vi.hoisted(() => ({ search: '' }));

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(navigation.search),
}));

vi.mock('@/contexts/SettingsContext', () => ({
  useSettings: vi.fn(),
}));

// The tour button needs the app shell's provider; these tests render the page
// in isolation, so the button is not part of what they exercise.
vi.mock('@/components/help/TourButton', () => ({
  TourButton: () => null,
}));

vi.mock('@/lib/app-config', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/app-config')>();
  return {
    ...actual,
    getAppConfig: () => ({
      ...actual.getAppConfig(),
      defaultSystem: 'retired_gpu',
      workloadPresets: [
        { key: 'interactive-chat', label: 'Interactive chat', isl: 4000, osl: 300, prefix: 3950 },
      ],
      concurrencyPresets: [
        { key: 'small', label: 'Small (100)', concurrency: 100 },
      ],
    }),
  };
});

vi.mock('@/lib/hooks/useCatalog', () => {
  // One stable object per module: a fresh array on every render would retrigger
  // effects that depend on the catalog's identity.
  const catalog = {
    gpuOptions: [
      { systemId: 'h200_sxm', displayName: 'NVIDIA H200', vramGb: 141 },
      { systemId: 'h100_sxm', displayName: 'NVIDIA H100', vramGb: 80 },
    ],
    modelOptions: ['Qwen/Qwen2.5-7B-Instruct', 'settings/default-model'],
    modelSpecs: new Map(),
    backendOptions: [],
    networkOptions: ['ib-400g'],
    simulators: [
      { id: 'aisimulate', label: 'AISimulate', version: null, default: true },
      { id: 'blis', label: 'BLIS', version: BLIS_FIXTURE_VERSION, default: false },
    ],
    simulatorSupport: {
      blis: {
        id: 'blis',
        label: 'BLIS',
        version: BLIS_FIXTURE_VERSION,
        recommendSearch: 'single',
        modelsExhaustive: true,
        modelSpecs: new Map(),
        networks: ['ib-400g'],
        defaultNetwork: 'ib-400g',
        models: ['Qwen/Qwen2.5-7B-Instruct', 'settings/default-model'],
        // This catalog deliberately lacks h100_sxm: it exercises a shared link
        // whose system the recipient's current simulator does not offer.
        systems: ['h200_sxm'],
        backends: [{
          id: 'vllm',
          aisimulateId: 'vllm',
          versions: ['0.31.0'],
          defaultVersion: '0.31.0',
          memoryFraction: null,
          memoryFractionKind: 'of_total',
          runtimeField: 'gpu_memory_utilization',
          systems: {},
          systemDefaults: {},
          versionSupport: {},
          disaggregationRequiresFabric: true,
          capacityOptions: { disaggregated: false },
        }],
        objectives: [],
      },
    },
    isLoading: false,
  };
  return {
    systemMemoryFraction: () => null,
    engineVersionSupport: () => null,
    useCatalog: () => catalog,
  };
});

vi.mock('@/lib/hooks/useCostings', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/hooks/useCostings')>();
  return {
    ...actual,
    useCostings: vi.fn(() => ({
      models: [],
      gpuCloudRates: new Map([
        ['h100_sxm', {
          'aws.us-east-1': {
            on_demand: 2,
            reserved_1yr: null,
            reserved_3yr: null,
            spot_median: null,
            rate_basis: 'gpu_hour',
          },
        }],
      ]),
      gpuHardwareCosts: new Map([
        ['h100_sxm', {
          new_usd: 30000,
          new_usd_low: null,
          new_usd_high: null,
          indicative: true,
          source_label: 'test',
          source_url: null,
          source_date: null,
        }],
      ]),
      health: null,
      modelsUpdatedAt: null,
      modelsStale: false,
      isLoading: false,
      error: null,
    })),
  };
});
vi.mock('@/lib/api/estimate-adapter', () => ({
  fetchEstimateAsInferenceResult: vi.fn(),
  buildPredictRequestBody: (input: Record<string, unknown>) => input,
  isMoeConfig: () => false,
  EstimateError: class EstimateError extends Error {},
}));

vi.mock('@/lib/huggingface/fetch-config', () => ({
  fetchModelConfig: vi.fn(),
}));

vi.mock('@/components/ui/ModelInput', () => ({
  ModelInput: ({ id, model, onChange }: { id: string; model: string; onChange: (value: string) => void }) => (
    <label>Model<input id={id} aria-label="Model" value={model} onChange={(event) => onChange(event.target.value)} /></label>
  ),
}));

vi.mock('@/components/ModelComboBox/ModelComboBox', () => ({
  ComboBox: ({ id, value, onChange }: { id: string; value: string; onChange: (value: string, source: 'user' | 'auto') => void }) => (
    <label>Model<input id={id} aria-label="Model" value={value} onChange={(event) => onChange(event.target.value, 'user')} /></label>
  ),
}));

vi.mock('@/components/ui/GpuSystemInput', () => ({
  GpuSystemInput: ({ id, value, onChange, gpuOptions }: {
    id: string;
    value: string;
    onChange: (value: string) => void;
    gpuOptions: Array<{ systemId: string; displayName: string }>;
  }) => (
    <label>GPU<select id={id} aria-label="GPU system" value={value} onChange={(event) => onChange(event.target.value)}>
      {gpuOptions.map((gpu) => <option key={gpu.systemId} value={gpu.systemId}>{gpu.displayName}</option>)}
    </select></label>
  ),
}));

vi.mock('./performanceHelpers', () => ({
  Term: ({ k }: { k: string }) => <span data-term={k}>?</span>,
  FlipTile: ({ children }: React.PropsWithChildren) => <>{children}</>,
  Sparkline: () => null,
  useCountUp: (value: number) => value,
}));
vi.mock('@/components/ProductTour', () => ({ ProductTour: () => null }));
vi.mock('./SaveEstimateModal', () => ({ SaveEstimateModal: () => null }));
vi.mock('@/components/GpuChipLoader/GpuChipLoader', () => ({ GpuChipLoader: () => null }));
vi.mock('@/components/ui/InfoStrip', () => ({
  InfoStrip: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  InfoStripAction: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

import Performance from './Performance';
import { fetchEstimateAsInferenceResult } from '@/lib/api/estimate-adapter';
import { fetchModelConfig } from '@/lib/huggingface/fetch-config';
import { useSettings } from '@/contexts/SettingsContext';

let container: HTMLDivElement;
let root: ReturnType<typeof createRoot>;
const writeText = vi.fn(async (_url: string) => undefined);

const makeSettings = (
  costingsEnabled: boolean,
  simulator: 'aisimulate' | 'blis' = 'aisimulate',
): ReturnType<typeof useSettings> => ({
  hydrated: true,
  defaultModel: 'settings/default-model',
  defaultSystem: 'h200_sxm',
  hfToken: '',
  inferenceBackend: 'vllm',
  simulator,
  backendVersion: 'latest',
  costingsEnabled,
  preferredCloudProvider: 'aws.us-east-1',
  pricingSource: 'aicostings',
  setDefaultModel: vi.fn(),
  setDefaultSystem: vi.fn(),
  setHfToken: vi.fn(),
  setInferenceBackend: vi.fn(),
  setSimulator: vi.fn(),
  setBackendVersion: vi.fn(),
  setCostingsEnabled: vi.fn(),
  setPreferredCloudProvider: vi.fn(),
  setPricingSource: vi.fn(),
  applyViewForSession: vi.fn(),
});
const estimateResult: InferenceConfigResult = {
  simulator: 'aisimulate',
  memory_available: true,
  memory_analysis: {
    weight_gb: 14,
    weight_gb_per_gpu: 14,
    total_vram_gb: 80,
    usable_hbm_per_gpu: 72,
    tp_size: 1,
    replicas: 1,
    kv_cache_budget_gb: 40,
    kv_cache_used_gb: 1,
    max_sequences_from_memory: 32,
    kv_category: 'KV-1',
    kv_category_label: 'Standard dense',
  },
  vllm_config: {
    tensor_parallel_size: 1,
    max_model_len: 4096,
    max_num_seqs: 32,
    gpu_memory_utilization: 0.9,
    max_num_batched_tokens: 4096,
    enable_chunked_prefill: true,
    enable_prefix_caching: false,
    quantization: 'none',
  },
  parallelism_strategy: {
    strategy: 'TP_ONLY',
    pp_size: 1,
    topology_note: 'single GPU',
  },
  bottleneck_analysis: {
    primary: 'TTFT',
    risk: 'low',
    fix_suggestions: [],
  },
  performance: {
    ttft_ms: 120,
    tpot_ms: 24,
    request_latency_ms: 3192,
    throughput_tokens_per_sec: 41.7,
    concurrency: 8,
  },
  diagnostics: {
    nvidia_smi_watch: '',
    dcgm_metrics: [],
    vllm_metrics: [],
  },
  warnings: [],
};

beforeEach(() => {
  vi.mocked(useSettings).mockReturnValue(makeSettings(false));
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true;
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText },
  });
  writeText.mockClear();
  vi.stubGlobal('localStorage', {
    getItem: vi.fn((key: string) => {
      if (key.includes('saved-estimates')) return '[]';
      return 'seen';
    }),
    setItem: vi.fn(),
    removeItem: vi.fn(),
    clear: vi.fn(),
  });
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => ({ data: [] }),
  })));
  vi.mocked(fetchEstimateAsInferenceResult).mockClear();
  vi.mocked(fetchEstimateAsInferenceResult).mockResolvedValue(estimateResult);
  vi.mocked(fetchModelConfig).mockReset();
  vi.mocked(fetchModelConfig).mockResolvedValue({
    success: true,
    config: { hidden_size: 4096 },
    source: 'huggingface',
  });
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  window.history.replaceState({}, '', '/predict');
  vi.unstubAllGlobals();
});

async function mountAt(search: string) {
  window.history.replaceState({}, '', `/predict${search}`);
  navigation.search = window.location.search;
  await act(async () => {
    root.render(<Performance />);
    await Promise.resolve();
  });
  return {
    model: container.querySelector<HTMLInputElement>('#predict-performance-model'),
    gpu: container.querySelector<HTMLSelectElement>('#predict-performance-gpu'),
  };
}

describe('performance page widget handoff', () => {
  it('offers the shared link in the header, not beside Calculate', async () => {
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    const copy = Array.from(container.querySelectorAll('button'))
      .find(button => button.textContent === 'Copy link');
    expect(copy).toBeDefined();
    // Every other tool page puts the link beside the tour button in the header.
    // Beside Calculate it read as part of the request rather than the view.
    expect(copy!.closest('[data-tour="calculate"]')).toBeNull();
    const header = container.querySelector('h1')?.parentElement?.parentElement;
    expect(header?.contains(copy!)).toBe(true);
  });

  it('drops the workload selection when a sequence length is cleared', async () => {
    // Clearing either field leaves its numeric state at the last accepted
    // value. Selecting from that value would keep a preset pressed for a
    // workload the form no longer states.
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    const isl = container.querySelector<HTMLInputElement>(
      'input[aria-label="Input sequence length (ISL)"]',
    );
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

  it('hydrates both destination controls without settings overwriting the model', async () => {
    const { model, gpu } = await mountAt(
      '?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm',
    );
    expect(model?.value).toBe('Qwen/Qwen2.5-7B-Instruct');
    expect(gpu?.value).toBe('h100_sxm');
  });

  it('ignores invalid handoff values and uses safe page defaults', async () => {
    const { model, gpu } = await mountAt('?model=javascript%3Aalert(1)&system=h200%20sxm');
    expect(model?.value).toBe('settings/default-model');
    expect(gpu?.value).toBe('h200_sxm');
  });
  it('renders all serving performance metrics with glossary help', async () => {
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    await act(async () => {
      const calculate = Array.from(container.querySelectorAll('button'))
        .find((button) => button.textContent === 'Calculate');
      calculate?.click();
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });

    expect(container.textContent).toContain('120.0 ms');
    expect(container.textContent).toContain('24.0 ms');
    expect(container.querySelector('[data-term="requestLatency"]')).not.toBeNull();
    expect(container.querySelector('[data-term="ttft"]')).not.toBeNull();
    expect(container.querySelector('[data-term="concurrent"]')).not.toBeNull();
    expect(container.querySelector('[data-term="tpot"]')).not.toBeNull();
  });

  it('forwards the selected network fabric for BLIS disaggregated predictions', async () => {
    vi.mocked(useSettings).mockReturnValue(makeSettings(false, 'blis'));
    expect(useSettings().simulator).toBe('blis');
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    await act(async () => {
      Array.from(container.querySelectorAll('button'))
        .find(button => button.textContent === 'Disaggregated')?.click();
    });
    await act(async () => {
      const fabric = container.querySelector<HTMLSelectElement>('#predict-network-fabric');
      expect(fabric).not.toBeNull();
      if (fabric) {
        fabric.value = 'ib-400g';
        fabric.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
    expect(container.querySelector<HTMLSelectElement>('#predict-network-fabric')?.value).toBe('ib-400g');
    await act(async () => {
      const calculate = Array.from(container.querySelectorAll<HTMLButtonElement>('button'))
        .find(button => button.textContent === 'Calculate');
      expect(calculate?.disabled).toBe(false);
      calculate?.click();
    });
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50));
    });

    expect(fetchEstimateAsInferenceResult).toHaveBeenCalledOnce();
    const input = vi.mocked(fetchEstimateAsInferenceResult).mock.calls[0][0];
    expect(input).toMatchObject({
      simulator: 'blis',
      mode: 'disagg',
      fabric: 'ib-400g',
      prefill: { workers: 1 },
      decode: { workers: 1 },
    });
  });

  it('renders separate cloud and on-prem cost tiles', async () => {
    vi.mocked(useSettings).mockReturnValue(makeSettings(true));
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    expect(container.querySelector('[data-testid="cloud-cost-tile"]')).toBeNull();
    expect(container.querySelector('[data-testid="self-hosted-cost-tile"]')).toBeNull();
    await act(async () => {
      const calculate = Array.from(container.querySelectorAll('button'))
        .find((button) => button.textContent === 'Calculate');
      calculate?.click();
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });

    expect(container.querySelector('[data-testid="cloud-cost-tile"]')).not.toBeNull();
    expect(container.querySelector('[data-testid="self-hosted-cost-tile"]')).not.toBeNull();
  });
});


describe('performance page disagg link fields', () => {
  it('restores both pool configurations from a shared link', async () => {
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm&servingMode=disagg' +
      '&prefillTp=2&prefillPp=1&prefillWorkers=3&prefillBatch=8' +
      '&decodeTp=4&decodePp=2&decodeWorkers=5&decodeBatch=128',
    );

    await act(async () => {
      Array.from(container.querySelectorAll('button'))
        .find(button => button.textContent === 'Calculate')
        ?.click();
    });
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50));
    });

    expect(fetchEstimateAsInferenceResult).toHaveBeenCalledOnce();
    const input = vi.mocked(fetchEstimateAsInferenceResult).mock.calls[0][0];
    expect(input).toMatchObject({
      mode: 'disagg',
      prefill: { tp: 2, pp: 1, workers: 3, batch: 8 },
      decode: { tp: 4, pp: 2, workers: 5, batch: 128 },
    });
  });

  it('enables the manual override for a linked max_num_seqs', async () => {
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm&maxNumSeqs=64');

    await act(async () => {
      Array.from(container.querySelectorAll('button'))
        .find(button => button.textContent === 'Calculate')
        ?.click();
    });
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50));
    });

    const input = vi.mocked(fetchEstimateAsInferenceResult).mock.calls[0][0];
    // Without the override the value is restored into the form but never sent.
    expect(input.max_num_seqs).toBe(64);
  });

  it('applies a linked max_num_seqs even when the link changes no model or GPU', async () => {
    // The parked-value effect has to fire on the park itself: the model/GPU
    // reset never runs for a link that leaves both alone.
    await mountAt('?v=1&maxNumSeqs=96');

    await act(async () => {
      Array.from(container.querySelectorAll('button'))
        .find(button => button.textContent === 'Calculate')
        ?.click();
    });
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50));
    });

    const input = vi.mocked(fetchEstimateAsInferenceResult).mock.calls[0][0];
    expect(input.max_num_seqs).toBe(96);
  });

  it('carries a linked memory fraction into the request', async () => {
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm&gpuMemUtil=0.75');

    await act(async () => {
      Array.from(container.querySelectorAll('button'))
        .find(button => button.textContent === 'Calculate')
        ?.click();
    });
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50));
    });

    const input = vi.mocked(fetchEstimateAsInferenceResult).mock.calls[0][0];
    expect(input.gpu_memory_utilization).toBe(0.75);
  });

  it('shows a linked memory fraction as a manual override', async () => {
    // The value reaches the request either way, but without the flag the
    // section labels it "Auto-computed" and leaves the control read-only.
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm&gpuMemUtil=0.75');

    expect(container.textContent).toContain('Manual override');
    expect(container.textContent).toContain('Reset to auto');
    expect(container.querySelector('input[type="range"]')).not.toBeNull();
  });

  it('exports the serving-policy context limit the request uses', async () => {
    // The override wins over the (empty) max-sequence field, so a link that
    // read the inactive field would omit the context the sender actually used.
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    const toggle = container.querySelector<HTMLButtonElement>('#acc-serving');
    expect(toggle).not.toBeNull();
    await act(async () => {
      toggle?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      await Promise.resolve();
    });

    const item = toggle?.closest('.pf-v6-c-accordion__item') ?? toggle?.parentElement?.parentElement;
    const override = Array.from(item?.querySelectorAll('button') ?? [])
      .find(button => /Override/.test(button.textContent ?? ''));
    expect(override).toBeDefined();
    await act(async () => {
      override?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      await Promise.resolve();
    });

    const limit = item?.querySelector<HTMLInputElement>('input[aria-label="Serving context limit"]');
    expect(limit).not.toBeNull();
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setter?.call(limit, '4096');
      limit?.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });

    await act(async () => {
      Array.from(container.querySelectorAll('button'))
        .find(button => button.textContent === 'Copy link')
        ?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      await Promise.resolve();
    });

    const copied = String(writeText.mock.calls[0]?.[0] ?? '');
    expect(new URL(copied).searchParams.get('maxModelLen')).toBe('4096');
  });

  it('carries a pinned chunked-prefill choice into the request', async () => {
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm&chunkedPrefill=off');

    await act(async () => {
      Array.from(container.querySelectorAll('button'))
        .find(button => button.textContent === 'Calculate')
        ?.click();
    });
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50));
    });

    const input = vi.mocked(fetchEstimateAsInferenceResult).mock.calls[0][0];
    expect(input.enable_chunked_prefill).toBe(false);
  });

  it('restores the linked system once the shared simulator is accepted', async () => {
    const blisSettings = makeSettings(false, 'blis');
    vi.mocked(useSettings).mockReturnValue(blisSettings);

    const { gpu } = await mountAt(
      `?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm&simulator=aisimulate&backend=vllm&backendVersion=${BLIS_FIXTURE_VERSION}`,
    );
    // BLIS's catalog has no h100, so the page settles on the system it offers.
    expect(gpu?.value).toBe('h200_sxm');

    const useShared = Array.from(container.querySelectorAll('button')).find(button =>
      // The banner uses the display label, which shortens AISimulate.
      /Use AISim for this view/.test(button.textContent ?? ''),
    );
    expect(useShared).toBeDefined();

    await act(async () => {
      useShared?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(blisSettings.applyViewForSession).toHaveBeenCalledWith({
      simulator: 'aisimulate',
      backend: 'vllm',
      backendVersion: BLIS_FIXTURE_VERSION,
    });

    // Accepting widens the catalog. The linked system must come back rather
    // than staying on the fallback the previous simulator forced.
    vi.mocked(useSettings).mockReturnValue(makeSettings(false, 'aisimulate'));
    await act(async () => {
      root.render(<Performance />);
      await Promise.resolve();
    });

    expect(container.querySelector<HTMLSelectElement>('#predict-performance-gpu')?.value).toBe('h100_sxm');
  });

  it('keeps a linked override when an accepted simulator restores the system', async () => {
    // Accepting the shared simulator restores a system the page had replaced,
    // which runs the model/GPU override reset a second time. The parked
    // max_num_seqs must survive that reset and still reach the request.
    const blisSettings = makeSettings(false, 'blis');
    vi.mocked(useSettings).mockReturnValue(blisSettings);

    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm&maxNumSeqs=64' +
      `&simulator=aisimulate&backend=vllm&backendVersion=${BLIS_FIXTURE_VERSION}`,
    );

    const useShared = Array.from(container.querySelectorAll('button')).find(button =>
      /Use AISim for this view/.test(button.textContent ?? ''),
    );
    expect(useShared).toBeDefined();
    await act(async () => {
      useShared?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });

    vi.mocked(useSettings).mockReturnValue(makeSettings(false, 'aisimulate'));
    await act(async () => {
      root.render(<Performance />);
      await Promise.resolve();
    });
    expect(container.querySelector<HTMLSelectElement>('#predict-performance-gpu')?.value).toBe('h100_sxm');

    await act(async () => {
      Array.from(container.querySelectorAll('button'))
        .find(button => button.textContent === 'Calculate')
        ?.click();
    });
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50));
    });

    const input = vi.mocked(fetchEstimateAsInferenceResult).mock.calls[0][0];
    expect(input.system).toBe('h100_sxm');
    expect(input.max_num_seqs).toBe(64);
  });

  it('keeps a serving-policy override through the accepted-system restoration', async () => {
    const blisSettings = makeSettings(false, 'blis');
    vi.mocked(useSettings).mockReturnValue(blisSettings);

    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm' +
      `&simulator=aisimulate&backend=vllm&backendVersion=${BLIS_FIXTURE_VERSION}`,
    );

    // A result gives the manual override a context limit to start from.
    const calculate = async () => {
      await act(async () => {
        Array.from(container.querySelectorAll('button'))
          .find(button => button.textContent === 'Calculate')
          ?.click();
      });
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 50));
      });
    };
    await calculate();

    const servingToggle = Array.from(container.querySelectorAll('button'))
      .find(button => button.textContent?.includes('Serving policy'));
    expect(servingToggle).toBeDefined();
    await act(async () => {
      servingToggle?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      await Promise.resolve();
    });

    // PatternFly links the toggle to the panel it opens, so the override
    // button can be found without guessing which section is expanded.
    // Search inside the accordion item the toggle belongs to: the always-open
    // parallel section offers an override button of its own.
    const servingItem = servingToggle?.closest('[class*="accordion__item"]');
    const override = Array.from(servingItem?.querySelectorAll('button') ?? [])
      .find(button => button.textContent?.includes('Override'));
    expect(override).toBeDefined();
    await act(async () => {
      override?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      await Promise.resolve();
    });

    const useShared = Array.from(container.querySelectorAll('button')).find(button =>
      /Use AISim for this view/.test(button.textContent ?? ''),
    );
    expect(useShared).toBeDefined();
    await act(async () => {
      useShared?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });

    vi.mocked(useSettings).mockReturnValue(makeSettings(false, 'aisimulate'));
    await act(async () => {
      root.render(<Performance />);
      await Promise.resolve();
    });
    expect(container.querySelector<HTMLSelectElement>('#predict-performance-gpu')?.value).toBe('h100_sxm');

    await calculate();

    // The second calculation is the one after acceptance.
    const input = vi.mocked(fetchEstimateAsInferenceResult).mock.calls[1][0];
    expect(input.system).toBe('h100_sxm');
    expect(input.max_seq_len).toBe(4096);
  });
});

describe('performance page remembered model', () => {
  it('keeps the saved default through a shared link, then remembers an explicit pick', async () => {
    let resolveSecond:
      | ((result: Awaited<ReturnType<typeof fetchModelConfig>>) => void)
      | undefined;
    vi.mocked(fetchModelConfig).mockImplementation((candidate: string) => {
      if (candidate === 'org/model-b') {
        return new Promise(resolve => { resolveSecond = resolve; });
      }
      return Promise.resolve({
        success: true,
        config: { hidden_size: 4096 },
        source: 'huggingface',
      });
    });

    const setDefaultModel = vi.mocked(useSettings)().setDefaultModel;

    await mountAt('?model=org%2Fmodel-a&system=h100_sxm');
    // The config lookup is debounced, so let it run before checking.
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 550));
    });
    // A shared link changes this view only; the recipient's remembered default
    // must survive opening it.
    expect(setDefaultModel).not.toHaveBeenCalledWith('org/model-a');

    const input = container.querySelector<HTMLInputElement>('#predict-performance-model')!;
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setter?.call(input, 'org/model-b');
      input.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });
    // The previous model's config must not validate the new one while its own
    // fetch is still pending.
    expect(setDefaultModel).not.toHaveBeenCalledWith('org/model-b');

    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 550));
    });
    expect(resolveSecond).toBeDefined();
    await act(async () => {
      resolveSecond?.({ success: true, config: { hidden_size: 8192 }, source: 'huggingface' });
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    expect(setDefaultModel).toHaveBeenCalledWith('org/model-b');
  });
});

describe('performance page link provenance', () => {
  it('offers a shared simulator instead of switching to it silently', async () => {
    vi.mocked(useSettings).mockReturnValue(makeSettings(false, 'aisimulate'));
    const settings = vi.mocked(useSettings)();

    await mountAt(
      `?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm&simulator=blis&backend=vllm&backendVersion=${BLIS_FIXTURE_VERSION}`,
    );

    // Nothing is applied for the recipient until they choose it.
    expect(settings.applyViewForSession).not.toHaveBeenCalled();
    expect(container.textContent).toContain('This view was shared using');
    expect(container.textContent).toContain('BLIS');

    const useShared = Array.from(container.querySelectorAll('button')).find((button) =>
      /Use BLIS for this view/.test(button.textContent ?? ''),
    );
    expect(useShared).toBeDefined();

    await act(async () => {
      useShared?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });

    expect(settings.applyViewForSession).toHaveBeenCalledWith({
      simulator: 'blis',
      backend: 'vllm',
      backendVersion: BLIS_FIXTURE_VERSION,
    });
  });

  it('applies a second shared link without remounting the page', async () => {
    vi.mocked(useSettings).mockReturnValue(makeSettings(false, 'aisimulate'));

    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');
    const second = await mountAt('?v=1&model=settings%2Fdefault-model&system=h200_sxm');

    expect(second.model?.value).toBe('settings/default-model');
    expect(second.gpu?.value).toBe('h200_sxm');
  });

  it('applies backend and version silently when the simulator matches', async () => {
    vi.mocked(useSettings).mockReturnValue(makeSettings(false, 'aisimulate'));
    const settings = vi.mocked(useSettings)();

    await mountAt('?v=1&system=h100_sxm&simulator=aisimulate&backend=sglang&backendVersion=0.4.0');

    expect(settings.applyViewForSession).toHaveBeenCalledWith({
      backend: 'sglang',
      backendVersion: '0.4.0',
    });
    expect(container.textContent).not.toContain('This view was shared using');
  });
});

describe('performance page parameter defaults', () => {
  it('starts at one concurrent user without pre-selecting a concurrency preset', async () => {
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    const users = container.querySelector<HTMLInputElement>('input[aria-label="Concurrent users"]');
    expect(users?.value).toBe('1');

    const pressed = Array.from(container.querySelectorAll<HTMLButtonElement>('button[aria-pressed="true"]'))
      .map((button) => button.textContent);
    expect(pressed).not.toContain('Small (100)');
  });

  it('applies a concurrency preset to the concurrent users field', async () => {
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    await act(async () => {
      Array.from(container.querySelectorAll('button'))
        .find((button) => button.textContent === 'Small (100)')
        ?.click();
    });

    const users = container.querySelector<HTMLInputElement>('input[aria-label="Concurrent users"]');
    expect(users?.value).toBe('100');
  });

  it('selects a preset when the typed concurrency matches and clears it otherwise', async () => {
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    const typeUsers = async (value: string) => {
      await act(async () => {
        const users = container.querySelector<HTMLInputElement>('input[aria-label="Concurrent users"]')!;
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
        setter?.call(users, value);
        users.dispatchEvent(new Event('input', { bubbles: true }));
        await Promise.resolve();
      });
    };
    const pressed = () => Array.from(container.querySelectorAll<HTMLButtonElement>('button[aria-pressed="true"]'))
      .map((button) => button.textContent);

    await typeUsers('100');
    expect(pressed()).toContain('Small (100)');

    await typeUsers('5');
    expect(pressed()).not.toContain('Small (100)');
  });

  it('applies a workload preset prefix and matches the preset back', async () => {
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    await act(async () => {
      Array.from(container.querySelectorAll('button'))
        .find((button) => button.textContent === 'Interactive chat')
        ?.click();
    });

    const prefix = container.querySelector<HTMLInputElement>('input[aria-label="Shared prefix tokens"]');
    expect(prefix?.value).toBe('3950');
    const pressed = Array.from(container.querySelectorAll<HTMLButtonElement>('button[aria-pressed="true"]'))
      .map((button) => button.textContent);
    expect(pressed).toContain('Interactive chat');
  });

  it('omits the per-user request count by default', async () => {
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    await act(async () => {
      Array.from(container.querySelectorAll<HTMLButtonElement>('button'))
        .find((button) => button.textContent === 'Calculate')
        ?.click();
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });

    expect(vi.mocked(fetchEstimateAsInferenceResult).mock.calls[0][0])
      .not.toHaveProperty('requests_per_load_unit');
  });

  it('sends the prefix a shared link states', async () => {
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm&prefix=200');

    await act(async () => {
      Array.from(container.querySelectorAll<HTMLButtonElement>('button'))
        .find((button) => button.textContent === 'Calculate')
        ?.click();
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });

    expect(vi.mocked(fetchEstimateAsInferenceResult).mock.calls[0][0])
      .toMatchObject({ prefix: 200 });
  });

  it('reads a link that omits the prefix as a cold request', async () => {
    // Links written before the system-prompt default left the field out and
    // meant 0, so the recipient must not inherit the new 200-token default.
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    await act(async () => {
      Array.from(container.querySelectorAll<HTMLButtonElement>('button'))
        .find((button) => button.textContent === 'Calculate')
        ?.click();
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });

    // A cold request is the API default, so the page leaves the field out
    // instead of sending the 200-token system prompt the page starts with.
    expect(vi.mocked(fetchEstimateAsInferenceResult).mock.calls[0][0].prefix)
      .toBeUndefined();
  });

  it('keeps the system-prompt prefix on an unlinked page', async () => {
    await mountAt('');

    await act(async () => {
      Array.from(container.querySelectorAll<HTMLButtonElement>('button'))
        .find((button) => button.textContent === 'Calculate')
        ?.click();
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });

    expect(vi.mocked(fetchEstimateAsInferenceResult).mock.calls[0][0])
      .toMatchObject({ prefix: 200 });
  });

  it('blocks sizing when the shared prefix exceeds the input length', async () => {
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm&isl=128&prefix=200');

    const calculate = Array.from(container.querySelectorAll<HTMLButtonElement>('button'))
      .find((button) => button.textContent === 'Calculate');
    expect(calculate?.disabled).toBe(true);
  });

  it('forwards an explicit per-user request count', async () => {
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    await act(async () => {
      const reuse = container.querySelector<HTMLInputElement>('input[aria-label="Requests per user"]')!;
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setter?.call(reuse, '4');
      reuse.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });

    expect(container.querySelector<HTMLInputElement>('input[aria-label="Requests per user"]')?.value).toBe('4');

    await act(async () => {
      Array.from(container.querySelectorAll<HTMLButtonElement>('button'))
        .find((button) => button.textContent === 'Calculate')
        ?.click();
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });

    expect(vi.mocked(fetchEstimateAsInferenceResult).mock.calls[0][0])
      .toMatchObject({ requests_per_load_unit: 4 });
  });

  it('refuses a per-user request count above the link limit', async () => {
    // A value the link cannot carry must not become a request the page still
    // submits: the form treats it as invalid instead.
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    await act(async () => {
      const reuse = container.querySelector<HTMLInputElement>('input[aria-label="Requests per user"]')!;
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setter?.call(reuse, '1000001');
      reuse.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });

    expect(container.querySelector<HTMLInputElement>('input[aria-label="Requests per user"]')
      ?.getAttribute('aria-invalid')).toBe('true');
    const calculate = Array.from(container.querySelectorAll<HTMLButtonElement>('button'))
      .find((button) => button.textContent === 'Calculate');
    expect(calculate?.disabled).toBe(true);
    expect(fetchEstimateAsInferenceResult).not.toHaveBeenCalled();
  });

  it('keeps current values visible on collapsed assumption rows', async () => {
    await mountAt('?model=Qwen%2FQwen2.5-7B-Instruct&system=h100_sxm');

    const servingToggle = container.querySelector('#acc-serving');
    expect(servingToggle?.textContent).toContain('Serving policy');
    expect(servingToggle?.textContent).toContain('context limit');
  });
});

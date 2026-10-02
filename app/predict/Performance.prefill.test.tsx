// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { InferenceConfigResult } from '@/lib/gpu-math/inference-config';

vi.mock('@/contexts/SettingsContext', () => ({
  useSettings: vi.fn(),
}));

vi.mock('@/lib/app-config', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/app-config')>();
  return {
    ...actual,
    getAppConfig: () => ({
      ...actual.getAppConfig(),
      defaultSystem: 'retired_gpu',
    }),
  };
});

vi.mock('@/lib/hooks/useCatalog', () => ({
  useCatalog: () => ({
    gpuOptions: [
      { systemId: 'h200_sxm', displayName: 'NVIDIA H200', vramGb: 141 },
      { systemId: 'h100_sxm', displayName: 'NVIDIA H100', vramGb: 80 },
    ],
    modelOptions: ['Qwen/Qwen2.5-7B-Instruct', 'settings/default-model'],
    modelSpecs: new Map(),
    isLoading: false,
  }),
}));

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
  EstimateError: class EstimateError extends Error {},
}));

vi.mock('@/components/ui/ModelInput', () => ({
  ModelInput: ({ id, model, onChange }: { id: string; model: string; onChange: (value: string) => void }) => (
    <label>Model<input id={id} aria-label="Model" value={model} onChange={(event) => onChange(event.target.value)} /></label>
  ),
}));

vi.mock('@/components/ModelComboBox/ModelComboBox', () => ({
  ComboBox: ({ id, value, onChange }: { id: string; value: string; onChange: (value: string) => void }) => (
    <label>Model<input id={id} aria-label="Model" value={value} onChange={(event) => onChange(event.target.value)} /></label>
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
import { useSettings } from '@/contexts/SettingsContext';

let container: HTMLDivElement;
let root: ReturnType<typeof createRoot>;

const makeSettings = (costingsEnabled: boolean): ReturnType<typeof useSettings> => ({
  hydrated: true,
  defaultModel: 'settings/default-model',
  hfToken: '',
  inferenceBackend: 'vllm',
  backendVersion: 'latest',
  costingsEnabled,
  preferredCloudProvider: 'aws.us-east-1',
  pricingSource: 'aicostings',
  setDefaultModel: vi.fn(),
  setHfToken: vi.fn(),
  setInferenceBackend: vi.fn(),
  setBackendVersion: vi.fn(),
  setCostingsEnabled: vi.fn(),
  setPreferredCloudProvider: vi.fn(),
  setPricingSource: vi.fn(),
});
const estimateResult: InferenceConfigResult = {
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
  vi.mocked(fetchEstimateAsInferenceResult).mockResolvedValue(estimateResult);
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

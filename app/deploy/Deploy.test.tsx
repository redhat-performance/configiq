// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const startSizing = vi.hoisted(() => vi.fn());
const recommendState = vi.hoisted(() => ({
  result: null as unknown,
  error: null as string | null,
  errorCode: null as string | null,
}));

vi.mock('@/contexts/RecommendContext', () => ({
  useRecommend: () => ({
    params: null,
    runMode: null,
    isLoading: false,
    result: recommendState.result,
    error: recommendState.error,
    errorCode: recommendState.errorCode,
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
    defaultModel: 'Qwen/Qwen3-8B',
    defaultSystem: 'h200_sxm',
    inferenceBackend: 'vllm',
    backendVersion: '0.30.0',
    simulator: 'aisimulate',
    costingsEnabled: false,
    pricingSource: 'merged',
    preferredCloudProvider: null,
    setDefaultSystem: vi.fn(),
    setDefaultModel: vi.fn(),
    applyViewForSession: vi.fn(),
  }),
}));

vi.mock('@/lib/hooks/useTestedModels', () => ({
  useTestedModels: () => ({
    pairs: [{ modelId: 'Qwen/Qwen3-8B', systemId: 'h200_sxm' }],
    modelIds: ['Qwen/Qwen3-8B'],
    validatedModelIds: ['Qwen/Qwen3-8B'],
    measuredModelIds: ['Qwen/Qwen3-8B'],
    approvedModels: ['Qwen/Qwen3-8B'],
    modelStatus: () => 'validated',
    gate: null,
    schemaVersion: null,
    dataset: null,
    isLoading: false,
    isAvailable: true,
  }),
}));

vi.mock('@/lib/hooks/useCatalog', () => ({
  useCatalog: () => ({
    modelOptions: ['Qwen/Qwen3-8B', 'Qwen/Qwen3-72B'],
    gpuOptions: [
      { systemId: 'h200_sxm', label: 'NVIDIA H200', vendor: 'nvidia', vramGb: 141 },
      { systemId: 'h100_sxm', label: 'NVIDIA H100', vendor: 'nvidia', vramGb: 80 },
    ],
    modelSpecs: new Map(),
    networkOptions: [],
    backendOptions: [],
    simulators: [{ id: 'aisimulate', label: 'AISimulate', version: null, default: true }],
    simulatorSupport: {},
    objectives: ['min_gpus', 'throughput_demand', 'concurrency_capacity', 'pareto'],
    timeoutSeconds: 90,
    isLoading: false,
    error: null,
  }),
}));

import { Deploy } from './Deploy';

let host: HTMLDivElement;
let root: ReturnType<typeof createRoot>;

beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  startSizing.mockClear();
  recommendState.result = null;
  recommendState.error = null;
  recommendState.errorCode = null;
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
});

async function renderPage() {
  await act(async () => {
    root.render(<Deploy />);
  });
}

/** React's value tracker ignores direct assignment; set through the native setter. */
async function setLoadValue(value: string) {
  const input = host.querySelector('#deploy-load') as HTMLInputElement;
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    setter?.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    await Promise.resolve();
  });
}

function resultFixture() {
  return {
    requestId: 'req-42',
    status: 'completed' as const,
    simulator: 'aisimulate',
    mode: 'agg' as const,
    recommendation: {
      gpusNeeded: 8, gpusPerReplica: 8, totalGpus: 8, replicasNeeded: 1,
      tensorParallelSize: 8, pipelineParallelSize: 1, dataParallelSize: 1,
      contextParallelSize: 1, moeTensorParallelSize: null, moeExpertParallelSize: null,
      batchSize: null,
    },
    phases: { prefill: null, decode: null, encode: null },
    performance: { ttftLatencyMs: 251, tpotMs: 6.9, requestLatencyMs: 7100, concurrency: 64 },
    throughput: { tokensPerSecond: 4400, tokensPerSecondPerGpu: 550, tokensPerSecondPerUser: 68 },
    memory: { value: 120, unit: 'GB' as const, breakdown: null },
    memoryAvailable: true,
    metadata: {
      modelPath: 'Qwen/Qwen3-8B', system: 'h200_sxm', inputTokens: 1024,
      outputTokens: 1024, targetTtftMs: null, durationMs: 4000,
    },
    warnings: [],
  };
}

describe('Deploy page', () => {
  it('auto-runs once with the validated default model and its tested system', async () => {
    await renderPage();
    expect(startSizing).toHaveBeenCalledTimes(1);
    const params = startSizing.mock.calls[0][0];
    expect(params.model_path).toBe('Qwen/Qwen3-8B');
    expect(params.system).toBe('h200_sxm');
    expect(params.objective).toBe('concurrency_capacity');
    expect(params.sla_mode).toBe('none');
    expect(params.target_concurrency).toBe(64);
  });

  it('does not auto-run again when inputs change', async () => {
    await renderPage();
    await setLoadValue('128');
    expect(startSizing).toHaveBeenCalledTimes(1);
  });

  it('re-runs on Recommend with the edited load and unit', async () => {
    await renderPage();
    await setLoadValue('5');
    const rps = host.querySelector('#deploy-load-rps') as HTMLButtonElement;
    await act(async () => {
      rps.click();
    });
    const button = [...host.querySelectorAll('button')].find(b => b.textContent === 'Recommend');
    await act(async () => {
      button!.click();
    });
    expect(startSizing).toHaveBeenCalledTimes(2);
    const params = startSizing.mock.calls[1][0];
    expect(params.objective).toBe('throughput_demand');
    expect(params.target_request_rate).toBe(5);
  });

  it('shows the no-configuration empty state, distinct from a failure', async () => {
    recommendState.error = 'No configuration found';
    recommendState.errorCode = 'AISIM_NO_CONFIGURATION';
    await renderPage();
    expect(host.textContent).toContain('No configuration for this pair');
    expect(host.querySelector('.pf-v6-c-alert')).toBeNull();
  });

  it('shows an error alert with retry for other failures', async () => {
    recommendState.error = 'gateway unreachable';
    recommendState.errorCode = 'AISIM_UNAVAILABLE';
    await renderPage();
    expect(host.querySelector('.pf-v6-c-alert')).not.toBeNull();
    expect(host.textContent).toContain('gateway unreachable');
    const retry = [...host.querySelectorAll('button')].find(b => b.textContent === 'Retry');
    expect(retry).not.toBeUndefined();
    await act(async () => { retry!.click(); });
    expect(startSizing).toHaveBeenCalledTimes(2);
  });

  it('opens the review panel with the LLMInferenceService YAML', async () => {
    recommendState.result = resultFixture();
    await renderPage();
    const review = [...host.querySelectorAll('button')].find(b => b.textContent === 'Review and deploy');
    expect(review).not.toBeUndefined();
    await act(async () => { review!.click(); });
    // PatternFly Modal portals to document.body, outside the mount host.
    expect(document.body.textContent).toContain('kind: LLMInferenceService');
    expect(document.body.textContent).toContain('inference.redhat.com/objectives');
    expect(document.body.textContent).toContain('concurrentRequests: 64');
    expect(document.body.textContent).toContain('Preview only');
  });

  it('renders the input controls', async () => {
    await renderPage();
    for (const id of ['deploy-model', 'deploy-gpu', 'deploy-load', 'deploy-ttft', 'deploy-tpot', 'deploy-gpus-free', 'deploy-workload']) {
      expect(host.querySelector(`#${id}`), id).not.toBeNull();
    }
  });
});

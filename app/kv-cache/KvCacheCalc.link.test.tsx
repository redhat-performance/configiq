// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// The page reads its shared view through `useSearchParams`. Tests mount it
// directly, so supply the current window search rather than the app router.
const navigation = vi.hoisted(() => ({ search: '' }));

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(navigation.search),
}));

vi.mock('@/contexts/SettingsContext', () => ({
  useSettings: () => ({
    hydrated: true,
    hfToken: '',
    defaultModel: 'Qwen/Qwen2.5-7B-Instruct',
    defaultSystem: 'h200_sxm',
    inferenceBackend: 'vllm',
    backendVersion: '0.9.0',
    setDefaultSystem: vi.fn(),
    setDefaultModel: vi.fn(),
  }),
}));

vi.mock('@/lib/hooks/useCatalog', () => ({
  useCatalog: () => ({
    modelOptions: ['Qwen/Qwen2.5-7B-Instruct'],
    gpuOptions: [
      { systemId: 'h200_sxm', label: 'NVIDIA H200', vendor: 'nvidia', vramGb: 141 },
    ],
    modelSpecs: new Map(),
    backendOptions: [
      {
        id: 'vllm',
        // Deliberately not the page default: without the guard the backend
        // default would replace a fraction kind the link asked for.
        memoryFractionKind: 'of_free',
        versions: ['0.9.0'],
        defaultVersion: '0.9.0',
      },
    ],
    isLoading: false,
  }),
  systemMemoryFraction: () => 0.9,
  hasDefaultMemoryFractionForVersion: () => true,
}));

vi.mock('@/lib/hooks/useTestedModels', () => ({
  useTestedModels: () => ({ validatedModelIds: [], isAvailable: false }),
}));

vi.mock('@/lib/huggingface/fetch-config', () => ({
  fetchModelConfig: vi.fn(async () => ({ success: false, error: 'not used' })),
}));

vi.mock('@/components/help/CopyLinkButton', () => ({ CopyLinkButton: () => null }));
vi.mock('@/components/help/TourButton', () => ({ TourButton: () => null }));

const { default: KvCacheCalc } = await import('./KvCacheCalc');

let root: ReturnType<typeof createRoot>;
let container: HTMLDivElement;
let memoryBodies: Array<Record<string, unknown>>;

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true;
  memoryBodies = [];
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input) === '/api/memory' && init?.body) {
      memoryBodies.push(JSON.parse(String(init.body)) as Record<string, unknown>);
    }
    // A failure response stops the page before it renders results, which is all
    // this test needs: the request body has already been captured.
    return {
      ok: true,
      status: 200,
      json: async () => ({ status: 'failed', error: { message: 'test stop' } }),
    } as unknown as Response;
  }));
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  window.history.replaceState({}, '', '/kv-cache');
  vi.unstubAllGlobals();
});

async function mountAt(search: string) {
  window.history.replaceState({}, '', `/kv-cache${search}`);
  navigation.search = window.location.search;
  await act(async () => {
    root.render(<KvCacheCalc />);
    await Promise.resolve();
  });
}

async function calculate() {
  const button = Array.from(container.querySelectorAll('button'))
    .find(candidate => candidate.textContent === 'Calculate');
  expect(button).toBeDefined();
  expect(button?.disabled).toBe(false);
  await act(async () => {
    button?.click();
    await new Promise(resolve => setTimeout(resolve, 20));
  });
}

describe('KV cache calculator shared views', () => {
  it('sends each disaggregated pool with its own parallelism', async () => {
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&servingMode=disagg' +
      '&prefillTp=2&prefillPp=3&prefillMoeTp=2&prefillMoeEp=2' +
      '&decodeTp=4&decodePp=1&decodeMoeTp=4&decodeMoeEp=4',
    );
    await calculate();

    expect(memoryBodies).toHaveLength(2);
    expect(memoryBodies[0]).toMatchObject({ tp_size: 2, pp_size: 3, moe_tp_size: 2, moe_ep_size: 2 });
    expect(memoryBodies[1]).toMatchObject({ tp_size: 4, pp_size: 1, moe_tp_size: 4, moe_ep_size: 4 });
  });

  it('keeps the fraction kind and value the link states', async () => {
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&memFractionKind=of_total&memFractionValue=0.5',
    );
    await calculate();

    expect(memoryBodies).toHaveLength(1);
    expect(memoryBodies[0]).toMatchObject({
      memory_fraction_kind: 'of_total',
      memory_fraction_value: 0.5,
    });
  });

  it('takes the backend default when the link does not name a fraction', async () => {
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm');
    await calculate();

    expect(memoryBodies).toHaveLength(1);
    expect(memoryBodies[0].memory_fraction_kind).toBe('of_free');
  });

  it('drops a fraction kind the form cannot display', async () => {
    // A malformed kind would otherwise be written to the control, mark it as
    // user-edited, and reach the request outside the documented enum.
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&memFractionKind=bogus',
    );
    await calculate();

    expect(container.textContent).toContain('Ignored "memFractionKind"');
    expect(memoryBodies).toHaveLength(1);
    expect(memoryBodies[0].memory_fraction_kind).toBe('of_free');
  });
});

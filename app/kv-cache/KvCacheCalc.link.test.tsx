// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// The page reads its shared view through `useSearchParams`. Tests mount it
// directly, so supply the current window search rather than the app router.
const navigation = vi.hoisted(() => ({ search: '' }));

// A minimal stand-in for the settings context: the page reads the effective
// backend and version, and a shared view applies them for this session only.
// Saved settings and the session override are kept apart, as the real context
// does, so a test can prove that opening a link does not rewrite the profile.
const settingsStore = vi.hoisted(() => {
  interface State { simulator: string; inferenceBackend: string; backendVersion: string }
  const saved: State = { simulator: 'aisim', inferenceBackend: 'vllm', backendVersion: '0.9.0' };
  let session: Partial<State> = {};
  // `useSyncExternalStore` compares snapshots by identity, so the effective
  // object is rebuilt only when the override changes.
  let effective: State = { ...saved };
  const listeners = new Set<() => void>();
  const recompute = () => { effective = { ...saved, ...session }; };
  const notify = () => {
    for (const listener of listeners) listener();
  };
  return {
    /** The effective values the page reads: saved settings under the override. */
    get: (): State => effective,
    /** The values that would persist; a shared view must never change these. */
    saved: (): State => ({ ...saved }),
    setSession: (next: Partial<State>) => {
      session = { ...session, ...next };
      recompute();
      notify();
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    },
    reset: () => {
      session = {};
      recompute();
      notify();
    },
  };
});

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(navigation.search),
}));

vi.mock('@/contexts/SettingsContext', async () => {
  const React = await import('react');
  return {
    useSettings: () => {
      const state = React.useSyncExternalStore(
        settingsStore.subscribe,
        settingsStore.get,
        settingsStore.get,
      );
      return {
        hydrated: true,
        hfToken: '',
        defaultModel: 'Qwen/Qwen2.5-7B-Instruct',
        defaultSystem: 'h200_sxm',
        simulator: state.simulator,
        inferenceBackend: state.inferenceBackend,
        backendVersion: state.backendVersion,
        setDefaultSystem: vi.fn(),
        setDefaultModel: vi.fn(),
        applyViewForSession: (view: {
          simulator?: string;
          backend?: string;
          backendVersion?: string;
        }) => {
          settingsStore.setSession({
            ...(view.simulator ? { simulator: view.simulator } : {}),
            ...(view.backend ? { inferenceBackend: view.backend } : {}),
            ...(view.backendVersion !== undefined ? { backendVersion: view.backendVersion } : {}),
          });
        },
      };
    },
  };
});

/**
 * Whether the catalog states a memory fraction. A simulator can leave the
 * fraction to its engine, in which case the page shows no number and omits the
 * field from the request.
 */
const catalogState = vi.hoisted(() => ({ memoryFraction: 0.9 as number | null }));

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
    simulatorSupport: {
      aisim: {
        available: true,
        models: ['Qwen/Qwen2.5-7B-Instruct'],
        systems: ['h200_sxm'],
        backends: [],
        networks: [],
        objectives: [],
      },
      blis: {
        available: true,
        models: ['Qwen/Qwen2.5-7B-Instruct'],
        systems: ['h200_sxm'],
        backends: [],
        networks: ['ib-400g'],
        objectives: [],
      },
    },
    simulators: [
      { id: 'aisim', label: 'AISim', version: '0.14.0' },
      { id: 'blis', label: 'BLIS', version: '0.19.0' },
    ],
  }),
  systemMemoryFraction: () => catalogState.memoryFraction,
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

const { default: KvCacheCalc } = await import('./KvCacheCalc');

let root: ReturnType<typeof createRoot>;
let container: HTMLDivElement;
let memoryBodies: Array<Record<string, unknown>>;

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true;
  // Each test starts from the saved profile, with no session override left by
  // the previous link.
  settingsStore.reset();
  catalogState.memoryFraction = 0.9;
  memoryBodies = [];
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input) === '/api/memory' && init?.body) {
      memoryBodies.push(JSON.parse(String(init.body)) as Record<string, unknown>);
    }
    // A completed result, in the shape the same-origin proxy returns, so tests
    // can reach the rendered-result state as well as the request body.
    return {
      ok: true,
      status: 200,
      json: async () => ({
        requestId: 'test',
        status: 'completed',
        kvCache: { totalBytes: 1024, perTokenBytes: 16, totalTokens: 64 },
        memoryBreakdown: {
          weightsBytes: 512,
          activationsBytes: 128,
          runtimeOverheadBytes: 64,
          commOverheadBytes: 0,
        },
        gpuCapacity: { totalBytes: 4096 },
        warnings: [],
        metadata: {
          simulator: 'aisim',
          modelPath: 'Qwen/Qwen2.5-7B-Instruct',
          backend: 'vllm',
          backendVersion: '0.9.0',
          system: 'h200_sxm',
          maxNumTokens: 2048,
          maxBatchSize: 128,
          tpSize: 1,
          ppSize: 1,
          moeTpSize: null,
          moeEpSize: null,
          memoryFractionKind: 'of_free',
          memoryFractionValue: null,
          source: 'test',
          durationMs: 1,
        },
      }),
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

async function openAdjust() {
  const adjust = Array.from(container.querySelectorAll('button'))
    .find(button => (button.textContent ?? '').startsWith('Adjust?'));
  expect(adjust).toBeDefined();
  await act(async () => {
    adjust?.click();
    await Promise.resolve();
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

  it('marks retained results stale when a second link changes the view', async () => {
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&maxNumTokens=1024');
    await calculate();
    expect(container.textContent).not.toContain('Inputs have changed');

    // Same tab, a new link: the visible results now describe the previous
    // request, so the page must say they are stale rather than leaving them
    // looking current.
    const next = '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&maxNumTokens=2048';
    window.history.replaceState({}, '', `/kv-cache${next}`);
    navigation.search = next;
    await act(async () => {
      root.render(<KvCacheCalc />);
      await Promise.resolve();
    });

    expect(container.textContent).toContain('Inputs have changed');
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

  it('sends the backend and version a link names, for this session only', async () => {
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&backend=sglang&backendVersion=0.5.0',
    );
    await calculate();

    expect(memoryBodies).toHaveLength(1);
    expect(memoryBodies[0]).toMatchObject({ backend: 'sglang', backend_version: '0.5.0' });
    // The recipient's saved profile must still be theirs: a shared view is a
    // session override, not a write-through into settings.
    expect(settingsStore.saved()).toMatchObject({ inferenceBackend: 'vllm', backendVersion: '0.9.0' });
  });

  it('reports the settings backend instead of offering a second editor', async () => {
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&backend=sglang&backendVersion=0.5.0',
    );
    await openAdjust();

    // The backend pair is global in Settings. This page must not grow a second
    // editor that would drift from it.
    expect(container.querySelector('#kv-backend')).toBeNull();
    expect(container.querySelector('#kv-backend-ver')).toBeNull();

    expect(container.textContent).toContain('SGLang · 0.5.0');
    const link = Array.from(container.querySelectorAll('a'))
      .find(candidate => (candidate.textContent ?? '').includes('Change backend or version in Settings'));
    expect(link?.getAttribute('href')).toBe('/settings#inference-backend');
  });

  it('sends the simulator selected in Settings', async () => {
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm');
    await calculate();

    expect(memoryBodies).toHaveLength(1);
    expect(memoryBodies[0].simulator).toBe('aisim');
    // The numbers are labelled with the simulator that answered them.
    expect(container.textContent).toContain('Sized by AISim 0.14.0');
  });

  it('offers a shared simulator without switching silently', async () => {
    await mountAt(
      '?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&simulator=blis',
    );

    // The recipient's settings name another simulator, so the page says which
    // one produced the view and offers a view-local switch.
    expect(container.textContent).toContain('This view was shared using');
    expect(settingsStore.get().simulator).toBe('aisim');

    const useShared = Array.from(container.querySelectorAll('button'))
      .find(candidate => /Use BLIS for this view/.test(candidate.textContent ?? ''));
    expect(useShared).toBeDefined();
    await act(async () => {
      useShared?.click();
      await Promise.resolve();
    });

    await calculate();
    expect(memoryBodies).toHaveLength(1);
    expect(memoryBodies[0].simulator).toBe('blis');
    // A shared view applies for this session only.
    expect(settingsStore.saved().simulator).toBe('aisim');
  });

  it('leaves the fraction to the simulator when the catalog states none', async () => {
    // No catalog figure and no link value: the control shows nothing, and the
    // request omits the field rather than sending a number this view never
    // displayed.
    catalogState.memoryFraction = null;
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm');
    await calculate();

    expect(memoryBodies).toHaveLength(1);
    expect(memoryBodies[0]).not.toHaveProperty('memory_fraction_value');
  });

  it('selects the workload whose token total the link states', async () => {    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm&maxNumTokens=16896');

    const pressed = Array.from(container.querySelectorAll('button[aria-pressed="true"]'))
      .map(button => button.textContent);
    expect(pressed).toContain('RAG / Search');
    expect(pressed).not.toContain('Default');
  });

  it('drops the workload selection when the token field is cleared', async () => {
    // The token field keeps its last accepted number while the visible input is
    // empty. Selecting from that number would leave a preset pressed for a
    // workload the form no longer states.
    await mountAt('?v=1&model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm');

    const adjust = Array.from(container.querySelectorAll('button'))
      .find(button => (button.textContent ?? '').startsWith('Adjust?'));
    expect(adjust).toBeDefined();
    await act(async () => {
      adjust?.click();
      await Promise.resolve();
    });

    const tokens = container.querySelector<HTMLInputElement>('#kv-tokens');
    expect(tokens).not.toBeNull();
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setter?.call(tokens, '');
      tokens?.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });

    const row = container.querySelector('[data-tour="workload-presets"]');
    expect(row).not.toBeNull();
    const pressed = Array.from(row?.querySelectorAll('button[aria-pressed="true"]') ?? []);
    expect(pressed).toHaveLength(0);
  });
});

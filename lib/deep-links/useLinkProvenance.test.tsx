// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LinkProvenanceValues } from './useLinkProvenance';

const navigation = vi.hoisted(() => ({ search: '' }));
const settings = vi.hoisted(() => ({
  savedSimulator: 'aisimulate',
  savedBackend: 'vllm',
  savedVersion: '0.9.0',
  override: null as Record<string, string | undefined> | null,
  applyViewForSession: vi.fn(),
}));

// Production replaces the whole session override, so the fake does too: a
// partial link that omits a field is what the regression is about.
settings.applyViewForSession.mockImplementation((view: Record<string, string | undefined>) => {
  settings.override = { ...view };
});

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(navigation.search),
}));

vi.mock('@/contexts/SettingsContext', () => ({
  useSettings: () => ({
    simulator: settings.override?.simulator ?? settings.savedSimulator,
    inferenceBackend: settings.override?.backend ?? settings.savedBackend,
    backendVersion: settings.override?.backendVersion ?? settings.savedVersion,
    applyViewForSession: settings.applyViewForSession,
  }),
}));

const { useLinkProvenance } = await import('./useLinkProvenance');
const { useSettings } = await import('@/contexts/SettingsContext');

let root: ReturnType<typeof createRoot>;
let container: HTMLDivElement;

function Harness({ values, ready = true }: { values: LinkProvenanceValues | null; ready?: boolean }) {
  const { sharedSimulator, acceptSharedSimulator } = useLinkProvenance(() => values, ready);
  const { simulator, inferenceBackend, backendVersion } = useSettings();
  return (
    <>
      <output data-testid="banner">{sharedSimulator ?? ''}</output>
      <output data-testid="effective">{`${simulator}|${inferenceBackend}|${backendVersion}`}</output>
      <button onClick={acceptSharedSimulator}>accept</button>
    </>
  );
}

const banner = () => container.querySelector('[data-testid="banner"]')?.textContent ?? '';
const effective = () => container.querySelector('[data-testid="effective"]')?.textContent ?? '';

async function mount(values: LinkProvenanceValues | null, ready = true) {
  await act(async () => {
    root.render(<Harness values={values} ready={ready} />);
    await Promise.resolve();
  });
}

async function accept() {
  await act(async () => {
    container.querySelector('button')?.click();
    await Promise.resolve();
  });
}

/** Apply a second link in the same tab, as pasting a new URL would. */
async function applyNext(values: LinkProvenanceValues | null, search: string) {
  navigation.search = search;
  window.history.replaceState({}, '', `/predict?${search}`);
  await act(async () => {
    root.render(<Harness values={values} />);
    await Promise.resolve();
  });
  // Production applies the override through context state, which re-renders the
  // consumer. This fake is not reactive, so render once more to read it.
  await act(async () => {
    root.render(<Harness values={values} />);
    await Promise.resolve();
  });
}

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true;
  settings.override = null;
  settings.applyViewForSession.mockClear();
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  window.history.replaceState({}, '', '/predict');
  navigation.search = '';
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

describe('shared-view provenance', () => {
  it('offers a different simulator instead of switching to it', async () => {
    await mount({ simulator: 'blis', backend: 'vllm', backendVersion: '0.9.0' });

    expect(banner()).toBe('blis');
    expect(settings.applyViewForSession).not.toHaveBeenCalled();
  });

  it('switches to the shared simulator only when the reader accepts', async () => {
    await mount({ simulator: 'blis', backend: 'vllm', backendVersion: '0.9.0' });
    await accept();

    expect(settings.applyViewForSession).toHaveBeenCalledWith({
      simulator: 'blis',
      backend: 'vllm',
      backendVersion: '0.9.0',
    });
    expect(banner()).toBe('');
  });

  it('adopts the shared backend and version when the simulator matches', async () => {
    await mount({ simulator: 'aisimulate', backend: 'sglang', backendVersion: '0.5.0' });

    expect(settings.applyViewForSession).toHaveBeenCalledWith({
      backend: 'sglang',
      backendVersion: '0.5.0',
    });
    expect(banner()).toBe('');
  });

  it('keeps the recipient backend version when the link names only a backend', async () => {
    await mount({ backend: 'vllm' });

    // The link says nothing about the version, so the view keeps the recipient's
    // effective one instead of replacing it with an empty value.
    expect(settings.applyViewForSession).toHaveBeenCalledWith({
      backend: 'vllm',
      backendVersion: '0.9.0',
    });
  });

  it('keeps an accepted simulator when a later link names only a backend', async () => {
    await mount({ simulator: 'blis', backend: 'vllm', backendVersion: '0.9.0' });
    await accept();
    expect(effective()).toBe('blis|vllm|0.9.0');

    // A second link in the same tab omits the simulator. Replacing the session
    // override wholesale would silently fall back to the saved simulator.
    await applyNext({ backend: 'sglang' }, 'backend=sglang');

    expect(effective()).toBe('blis|sglang|0.9.0');
    expect(banner()).toBe('');
  });

  it('waits for the page to be ready', async () => {
    await mount({ simulator: 'blis' }, false);

    expect(settings.applyViewForSession).not.toHaveBeenCalled();
    expect(banner()).toBe('');
  });

  it('applies once per link even when the page re-renders', async () => {
    await mount({ backend: 'sglang' });
    await act(async () => {
      // A new values object with the same search string is a re-render, not a
      // new link: re-applying would keep resetting the reader's own changes.
      root.render(<Harness values={{ backend: 'sglang' }} />);
      await Promise.resolve();
    });

    expect(settings.applyViewForSession).toHaveBeenCalledTimes(1);
  });

  it('does nothing for a page with no link', async () => {
    await mount(null);

    expect(settings.applyViewForSession).not.toHaveBeenCalled();
    expect(banner()).toBe('');
  });
});

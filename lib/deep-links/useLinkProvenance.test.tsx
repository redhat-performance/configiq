// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LinkProvenanceValues } from './useLinkProvenance';

const navigation = vi.hoisted(() => ({ search: '' }));
const settings = vi.hoisted(() => ({ simulator: 'aisimulate', applyViewForSession: vi.fn() }));

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(navigation.search),
}));

vi.mock('@/contexts/SettingsContext', () => ({
  useSettings: () => ({
    simulator: settings.simulator,
    applyViewForSession: settings.applyViewForSession,
  }),
}));

const { useLinkProvenance } = await import('./useLinkProvenance');

let root: ReturnType<typeof createRoot>;
let container: HTMLDivElement;

function Harness({ values, ready = true }: { values: LinkProvenanceValues | null; ready?: boolean }) {
  const { sharedSimulator, acceptSharedSimulator } = useLinkProvenance(() => values, ready);
  return (
    <>
      <output>{sharedSimulator ?? ''}</output>
      <button onClick={acceptSharedSimulator}>accept</button>
    </>
  );
}

const banner = () => container.querySelector('output')?.textContent ?? '';

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

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true;
  settings.simulator = 'aisimulate';
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

  it('applies a backend-only link without inventing a version', async () => {
    await mount({ backend: 'vllm' });

    expect(settings.applyViewForSession).toHaveBeenCalledWith({
      backend: 'vllm',
      backendVersion: undefined,
    });
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

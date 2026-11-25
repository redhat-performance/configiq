// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { INFERENCE_BACKENDS, MAX_ID_LENGTH, defineLink, text, type LinkValues } from './common';
import { defineAccessors } from './shared-view';
import { z } from 'zod';

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(''),
}));

const testLink = defineLink('/test', {
  model: text(MAX_ID_LENGTH),
  backend: z.enum(INFERENCE_BACKENDS),
  backendVersion: text(MAX_ID_LENGTH),
  simulator: text(MAX_ID_LENGTH),
});

type TestValues = LinkValues<typeof testLink.fields>;

const { useSharedView } = await import('./useSharedView');

let root: ReturnType<typeof createRoot>;
let container: HTMLDivElement;

function Harness({ overrides }: { overrides?: TestValues }) {
  const fields = React.useRef(defineAccessors(testLink, {
    model: { read: () => 'org/model', write: () => {} },
    backend: { read: () => 'vllm' as const, write: () => {} },
    backendVersion: { read: () => '0.9.0', write: () => {} },
    simulator: { read: () => 'aisimulate', write: () => {} },
  })).current;
  const shared = useSharedView(testLink, { fields, ready: true });
  return <output data-testid="link">{shared.build(overrides)}</output>;
}

const link = () => container.querySelector('[data-testid="link"]')?.textContent ?? '';

async function mount(overrides?: TestValues) {
  await act(async () => {
    root.render(<Harness overrides={overrides} />);
    await Promise.resolve();
  });
}

beforeEach(() => {
  window.history.replaceState({}, '', '/');
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe('useSharedView build overrides', () => {
  it('exports the fields as they stand when no override is given', async () => {
    await mount();

    const url = new URL(link(), 'http://localhost');
    expect(url.searchParams.get('model')).toBe('org/model');
    expect(url.searchParams.get('simulator')).toBe('aisimulate');
    expect(url.searchParams.get('backendVersion')).toBe('0.9.0');
  })

  it('describes a named run instead of the current settings', async () => {
    await mount({ simulator: 'blis', backend: 'sglang', backendVersion: '0.5.0' });

    const url = new URL(link(), 'http://localhost');
    expect(url.searchParams.get('simulator')).toBe('blis');
    expect(url.searchParams.get('backend')).toBe('sglang');
    expect(url.searchParams.get('backendVersion')).toBe('0.5.0');
  })

  it('omits an overridden field the run did not name', async () => {
    await mount({ backendVersion: undefined });

    const url = new URL(link(), 'http://localhost');
    expect(url.searchParams.has('backendVersion')).toBe(false);
  })

  it('uses the concrete location for a dynamic route instead of its template', async () => {
    window.history.replaceState({}, '', '/tested-models/model__h200');
    await mount();

    expect(new URL(link(), 'http://localhost').pathname).toBe('/tested-models/model__h200');
  })

  it('accepts an explicit path for a page that builds another route', async () => {
    function PathHarness() {
      const fields = React.useRef(defineAccessors(testLink, {
        model: { read: () => 'org/model', write: () => {} },
        backend: { read: () => 'vllm' as const, write: () => {} },
        backendVersion: { read: () => '0.9.0', write: () => {} },
        simulator: { read: () => 'aisimulate', write: () => {} },
      })).current;
      const shared = useSharedView(testLink, { fields, ready: true, path: '/test/other' });
      return <output data-testid="link">{shared.build()}</output>;
    }

    await act(async () => {
      root.render(<PathHarness />);
      await Promise.resolve();
    });

    expect(new URL(link(), 'http://localhost').pathname).toBe('/test/other');
  })
})

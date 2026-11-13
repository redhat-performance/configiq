// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { recommendLink } from './pages';
import type { LinkValues } from './common';
import { useDeepLink, type DeepLinkState } from './useDeepLink';

const navigation = vi.hoisted(() => ({ search: '' }));

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(navigation.search),
}));

let root: ReturnType<typeof createRoot> | null = null;
let host: HTMLDivElement;

function mount(state: { current: DeepLinkState<typeof recommendLink.fields> | null }) {
  function Harness({
    ready,
    apply,
  }: {
    ready: boolean;
    apply: (values: LinkValues<typeof recommendLink.fields>) => void;
  }) {
    const deepLink = useDeepLink(recommendLink, ready, apply);
    state.current = deepLink;
    return <span>{deepLink.applied ? 'applied' : 'pending'}</span>;
  }

  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  return Harness;
}

function render(element: React.ReactElement) {
  act(() => {
    root?.render(element);
  });
}

function setSearch(path: string) {
  window.history.replaceState({}, '', path);
  navigation.search = window.location.search;
}

beforeEach(() => {
  setSearch('/recommend');
});

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  host?.remove();
});

describe('useDeepLink', () => {
  it('parses the current search string without applying it before the page is ready', () => {
    setSearch('/recommend?v=1&model=Qwen%2FQwen3-32B&goal=capacity&nope=1');
    const state = { current: null as DeepLinkState<typeof recommendLink.fields> | null };
    const apply = vi.fn();
    const Harness = mount(state);
    render(<Harness ready={false} apply={apply} />);

    expect(state.current?.parsed?.values).toEqual({
      model: 'Qwen/Qwen3-32B',
      goal: 'capacity',
    });
    expect(state.current?.warnings).toEqual(['Ignored unknown parameter "nope".']);
    expect(apply).not.toHaveBeenCalled();
    expect(state.current?.applied).toBe(false);
  });

  it('applies once the ready gate opens, and only once', () => {
    setSearch('/recommend?v=1&concurrency=32&goal=demand');
    const state = { current: null as DeepLinkState<typeof recommendLink.fields> | null };
    const apply = vi.fn();
    const Harness = mount(state);

    render(<Harness ready={false} apply={apply} />);
    render(<Harness ready apply={apply} />);
    render(<Harness ready apply={apply} />);

    expect(apply).toHaveBeenCalledTimes(1);
    expect(apply).toHaveBeenCalledWith({ concurrency: 32, goal: 'demand' });
    expect(state.current?.applied).toBe(true);
  });

  it('rewrites the address bar to the canonical link it understood', () => {
    setSearch('/recommend?v=1&concurrency=32&nope=1&goal=turbo');
    const state = { current: null as DeepLinkState<typeof recommendLink.fields> | null };
    const Harness = mount(state);

    render(<Harness ready apply={vi.fn()} />);

    expect(window.location.pathname).toBe('/recommend');
    expect(window.location.search).toBe('?v=1&concurrency=32');
    // Warnings still describe what the original link asked for.
    expect(state.current?.warnings).toEqual([
      'Ignored unknown parameter "nope".',
      'Ignored "goal": the value was not valid.',
    ]);
  });

  it('applies a new link opened while the page is already mounted', () => {
    setSearch('/recommend?v=1&concurrency=8');
    const state = { current: null as DeepLinkState<typeof recommendLink.fields> | null };
    const apply = vi.fn();
    const Harness = mount(state);

    render(<Harness ready apply={apply} />);
    expect(apply).toHaveBeenCalledWith({ concurrency: 8 });

    setSearch('/recommend?v=1&concurrency=64&goal=latency');
    render(<Harness ready apply={apply} />);

    expect(apply).toHaveBeenCalledTimes(2);
    expect(apply).toHaveBeenLastCalledWith({ concurrency: 64, goal: 'latency' });
  });

  it('refuses a link from an unsupported format version instead of half-applying it', () => {
    setSearch('/recommend?v=2&concurrency=32');
    const state = { current: null as DeepLinkState<typeof recommendLink.fields> | null };
    const apply = vi.fn();
    const Harness = mount(state);

    render(<Harness ready apply={apply} />);

    expect(apply).not.toHaveBeenCalled();
    expect(state.current?.parsed?.values).toEqual({});
    expect(state.current?.warnings[0]).toContain('unsupported format version "2"');
  });

  it('does not apply an empty link but still reports the page as settled', () => {
    const state = { current: null as DeepLinkState<typeof recommendLink.fields> | null };
    const apply = vi.fn();
    const Harness = mount(state);

    render(<Harness ready apply={apply} />);

    expect(apply).not.toHaveBeenCalled();
    expect(state.current?.applied).toBe(true);
    expect(state.current?.warnings).toEqual([]);
  });

  it('never applies a link whose values were all rejected', () => {
    setSearch('/recommend?concurrency=0&goal=turbo');
    const state = { current: null as DeepLinkState<typeof recommendLink.fields> | null };
    const apply = vi.fn();
    const Harness = mount(state);

    render(<Harness ready apply={apply} />);

    expect(apply).not.toHaveBeenCalled();
    expect(state.current?.warnings).toHaveLength(2);
  });
});

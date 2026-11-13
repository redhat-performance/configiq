// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { HelpMenu } from './HelpMenu';

const tourState = vi.hoisted(() => ({
  tour: undefined as { id: string; title: string } | undefined,
  startTour: vi.fn(),
}));

vi.mock('next/navigation', () => ({
  usePathname: () => '/predict',
}));

vi.mock('next/link', () => ({
  default: ({
    href,
    children,
    ...rest
  }: {
    href: string;
    children: React.ReactNode;
  } & React.AnchorHTMLAttributes<HTMLAnchorElement>) =>
    React.createElement('a', { href, ...rest }, children),
}));

vi.mock('./usePageTour', () => ({
  usePageTour: () => tourState,
}));

// The question box reads the active simulator for link provenance; the menu
// itself does not depend on settings.
vi.mock('@/contexts/SettingsContext', () => ({
  useSettings: () => ({ simulator: 'aisimulate', inferenceBackend: 'vllm', backendVersion: '' }),
}));

const INDEX = {
  entries: [
    {
      id: 'docs/settings#sharing-a-view',
      title: 'Sharing a view',
      section: 'Settings and simulators',
      body: 'A shared link records the simulator, backend, and backend version used to build it.',
      url: '/docs/settings#sharing-a-view',
    },
  ],
};

let root: ReturnType<typeof createRoot>;
let host: HTMLDivElement;

const trigger = () =>
  document.querySelector<HTMLButtonElement>('[aria-label="Help, search, and tours"]');

async function render() {
  await act(async () => {
    root.render(<HelpMenu />);
  });
}

async function openMenu() {
  await act(async () => {
    trigger()?.click();
  });
}

async function type(value: string) {
  const input = document.querySelector<HTMLInputElement>('#help-search-input');
  expect(input).not.toBeNull();
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
  await act(async () => {
    setter?.call(input, value);
    input!.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  tourState.tour = undefined;
  tourState.startTour = vi.fn();

  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url === '/api/assistant') {
        return new Response(JSON.stringify({ enabled: false, model: null, reason: 'Assistant is off in tests.' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      return new Response(JSON.stringify(INDEX), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }),
  );

  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  document.querySelectorAll('[data-popper-placement]').forEach((element) => element.remove());
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('HelpMenu', () => {
  it('fetches the docs index lazily on first open', async () => {
    await render();
    expect(fetch).not.toHaveBeenCalled();

    await openMenu();

    expect(fetch).toHaveBeenCalledWith('/help-index.json', expect.anything());
    expect(fetch).toHaveBeenCalledWith('/api/assistant', expect.anything());
    expect(document.querySelector('#help-search-input')).not.toBeNull();
  });

  it('shows a documentation result with its link', async () => {
    await render();
    await openMenu();
    await type('shared link');

    const link = Array.from(document.querySelectorAll<HTMLAnchorElement>('a')).find(
      (anchor) =>
        anchor.getAttribute('href') === '/docs/settings#sharing-a-view' &&
        (anchor.textContent ?? '').includes('Sharing a view'),
    );
    expect(link).not.toBeNull();
  });

  it('offers the mapped documentation beside a control result', async () => {
    await render();
    await openMenu();
    await type('concurrency presets');

    const docLink = Array.from(document.querySelectorAll<HTMLAnchorElement>('a')).find(
      (anchor) => anchor.getAttribute('href') === '/docs/user-scenarios#sizing-multiple-workloads',
    );
    expect(docLink).not.toBeNull();
    expect(docLink?.textContent).toBe('Docs');
    expect(docLink?.getAttribute('aria-label')).toContain('Concurrency presets');
  });

  it('shows a glossary definition without leaving the popover', async () => {
    await render();
    await openMenu();
    await type('KV cache');

    expect(document.body.textContent).toContain('Glossary');
    expect(document.body.textContent).toContain('key/value');
  });

  it('offers the page tour when the route has one', async () => {
    tourState.tour = { id: 'predict-performance', title: 'Predict performance' };
    await render();
    await openMenu();

    const tourRow = Array.from(
      document.querySelectorAll<HTMLButtonElement>('button[data-help-item]'),
    ).find((button) => button.textContent?.includes('Take a tour'));
    expect(tourRow).not.toBeUndefined();

    await act(async () => {
      tourRow!.click();
    });
    expect(tourState.startTour).toHaveBeenCalledTimes(1);
  });

  it('opens from the / shortcut and ignores it while typing', async () => {
    await render();

    const input = document.createElement('input');
    document.body.append(input);
    input.focus();
    await act(async () => {
      input.dispatchEvent(new KeyboardEvent('keydown', { key: '/', bubbles: true }));
    });
    expect(document.querySelector('#help-search-input')).toBeNull();

    input.remove();
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: '/' }));
    });
    expect(document.querySelector('#help-search-input')).not.toBeNull();
  });

  it('explains why Ask is unavailable instead of offering a dead input', async () => {
    await render();
    await openMenu();
    await act(async () => {
      await Promise.resolve();
    });

    expect(document.querySelector('#assistant-question')).toBeNull();
    expect(document.body.textContent).toContain('Ask is unavailable');
  });

  it('offers the question box when the server reports an assistant model', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === 'string' ? input : input.toString();
        if (url === '/api/assistant') {
          return new Response(
            JSON.stringify({ enabled: true, model: 'test/model', reason: null }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          );
        }
        return new Response(JSON.stringify(INDEX), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }),
    );

    await render();
    await openMenu();
    await act(async () => {
      await Promise.resolve();
    });

    expect(document.querySelector('#assistant-question')).not.toBeNull();
    expect(document.body.textContent).toContain('does not run sizing');
  });
});

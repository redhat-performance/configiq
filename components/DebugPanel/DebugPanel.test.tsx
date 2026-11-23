// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DebugPanel } from './DebugPanel';
import type { DebugRun } from '@/lib/debug/types';

let root: ReturnType<typeof createRoot>;
let host: HTMLDivElement;
const writeText = vi.fn(async (_text: string) => undefined);

function exchangeRun(): DebugRun {
  return {
    id: 'predict',
    label: 'Predict performance',
    panes: [
      { id: 'request', label: 'Request', language: 'json', contents: '{"isl":2048}' },
      { id: 'response', label: 'Response', language: 'json', contents: '{"ttft_ms":120}' },
    ],
  };
}

async function render(runs: DebugRun[], open = true) {
  await act(async () => {
    root.render(
      <DebugPanel
        runs={runs}
        status={200}
        duration={42}
        endpoint="POST /api/predict"
        open={open}
        onToggle={() => undefined}
      />,
    );
  });
}

function buttons(): HTMLButtonElement[] {
  return Array.from(host.querySelectorAll<HTMLButtonElement>('button'));
}

function tab(name: string): HTMLButtonElement {
  const found = buttons().find(button => button.getAttribute('role') === 'tab'
    && button.textContent === name);
  if (!found) throw new Error(`No tab named ${name}`);
  return found;
}

async function click(element: HTMLElement) {
  await act(async () => {
    element.click();
  });
}

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText },
  });
  writeText.mockClear();
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
});

describe('DebugPanel', () => {
  it('renders nothing until a run exists', async () => {
    await render([]);
    expect(host.textContent).toBe('');
  });

  it('shows the captured request and response as separate artifacts', async () => {
    await render([exchangeRun()]);

    expect(host.textContent).toContain('Debug panel');
    expect(host.textContent).toContain('POST /api/predict');
    expect(host.textContent).toContain('200');
    expect(host.textContent).toContain('42 ms');

    // Request is the first artifact, so its exact contents are visible.
    expect(host.querySelector('pre')?.textContent).toBe('{"isl":2048}');

    await click(tab('Response'));
    expect(host.querySelector('pre')?.textContent).toBe('{"ttft_ms":120}');
  });

  it('copies only the artifact it was asked for', async () => {
    await render([exchangeRun()]);

    const copy = buttons().find(button => button.getAttribute('aria-label') === 'Copy Request');
    expect(copy).toBeDefined();
    await click(copy!);
    expect(writeText).toHaveBeenCalledWith('{"isl":2048}');

    await click(tab('Response'));
    const copyResponse = buttons().find(button => button.getAttribute('aria-label') === 'Copy Response');
    await click(copyResponse!);
    expect(writeText).toHaveBeenLastCalledWith('{"ttft_ms":120}');
  });

  it('keeps a missing half visible instead of hiding the tab', async () => {
    await render([{
      id: 'run',
      label: 'Run',
      panes: [
        { id: 'request', label: 'Request', language: 'json', contents: '{"isl":2048}' },
        { id: 'response', label: 'Response', language: 'json', contents: '(no response)' },
      ],
    }]);

    await click(tab('Response'));
    expect(host.querySelector('pre')?.textContent).toBe('(no response)');
  });

  it('switches between runs and keeps their artifacts apart', async () => {
    await render([
      exchangeRun(),
      {
        id: 'pareto',
        label: 'Pareto sweep',
        badge: 'BLIS',
        panes: [{ id: 'command', label: 'Command', language: 'shell', contents: 'blis run' }],
      },
    ]);

    expect(host.querySelector('pre')?.textContent).toBe('{"isl":2048}');

    await click(tab('Pareto sweep · BLIS'));
    expect(host.querySelector('pre')?.textContent).toBe('blis run');
  });

  it('groups pool artifacts behind their own sub-tabs', async () => {
    await render([{
      id: 'kv',
      label: 'KV cache',
      panes: [
        { id: 'prefill', label: 'Prefill request', language: 'json', group: 'Prefill', contents: '{"role":"prefill"}' },
        { id: 'decode', label: 'Decode request', language: 'json', group: 'Decode', contents: '{"role":"decode"}' },
      ],
    }]);

    // The first group is active, so only its artifact is on screen.
    expect(host.querySelector('pre')?.textContent).toBe('{"role":"prefill"}');

    await click(tab('Decode'));
    expect(host.querySelector('pre')?.textContent).toBe('{"role":"decode"}');
  });

  it('points every tab at a panel that is rendered', async () => {
    await render([
      exchangeRun(),
      {
        id: 'pareto',
        label: 'Pareto sweep',
        panes: [{ id: 'command', label: 'Command', language: 'shell', contents: 'blis run' }],
      },
    ]);

    const tabs = buttons().filter(button => button.getAttribute('role') === 'tab');
    expect(tabs).toHaveLength(4);
    for (const control of tabs) {
      const target = control.getAttribute('aria-controls');
      expect(target, `${control.textContent} names no panel`).toBeTruthy();
      expect(document.getElementById(target!), `${control.textContent} names a missing panel`).not.toBeNull();
    }

    // The panel is labelled by the artifact tab that switched it.
    const panel = host.querySelector('[role="tabpanel"]');
    expect(document.getElementById(panel?.getAttribute('aria-labelledby') ?? '')).toBe(tab('Request'));

    await click(tab('Pareto sweep'));
    expect(document.getElementById(host.querySelector('[role="tabpanel"]')?.getAttribute('aria-labelledby') ?? '')).toBeNull();
    expect(host.querySelector('pre')?.textContent).toBe('blis run');
  });

  it('does not claim a copy when the clipboard is unavailable', async () => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
    await render([exchangeRun()]);

    const copy = buttons().find(button => button.getAttribute('aria-label') === 'Copy Request');
    expect(copy?.textContent).toBe('Copy');

    await click(copy!);
    expect(copy?.textContent).toBe('Copy');
    expect(writeText).not.toHaveBeenCalled();
  });

  it('moves between artifacts with the arrow keys', async () => {
    await render([exchangeRun()]);

    const request = tab('Request');
    request.focus();
    await act(async () => {
      request.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    });

    expect(tab('Response').getAttribute('aria-selected')).toBe('true');
    expect(host.querySelector('pre')?.textContent).toBe('{"ttft_ms":120}');
  });

  it('stays collapsed while open is false', async () => {
    await render([exchangeRun()], false);
    expect(host.querySelector('pre')).toBeNull();
    expect(host.textContent).toContain('Debug panel');
  });
});

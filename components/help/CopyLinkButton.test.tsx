// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CopyLinkButton } from './CopyLinkButton';

let container: HTMLDivElement;
let root: ReturnType<typeof createRoot>;
const writeText = vi.fn(async () => undefined);

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true;
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText },
  });
  writeText.mockClear();
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.useRealTimers();
});

async function render(getUrl: () => string | null) {
  await act(async () => {
    root.render(<CopyLinkButton getUrl={getUrl} />);
  });
}

function visibleButton(): HTMLButtonElement {
  const found = Array.from(container.querySelectorAll('button'))
    .find(candidate => candidate.closest('[aria-hidden="true"]') == null);
  if (!found) throw new Error('expected a button');
  return found;
}

async function click() {
  await act(async () => {
    visibleButton().dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await Promise.resolve();
  });
}

describe('CopyLinkButton', () => {
  it('copies an absolute link for the current view', async () => {
    await render(() => '/predict?v=1&model=m');
    await click();

    expect(writeText).toHaveBeenCalledWith(`${window.location.origin}/predict?v=1&model=m`);
    expect(visibleButton().textContent).toContain('Copied');
    expect(container.textContent).toContain('Link copied to the clipboard.');
  });

  it('passes an absolute URL through without prefixing the current origin', async () => {
    await render(() => 'https://example.test/scorecard?v=1');
    await click();

    expect(writeText).toHaveBeenCalledWith('https://example.test/scorecard?v=1');
  });

  it('uses the label it was given', async () => {
    await act(async () => {
      root.render(<CopyLinkButton getUrl={() => '/predict'} label="Copy share link" />);
    });

    expect(container.querySelector('button')?.textContent).toContain('Copy share link');
  });

  it('reports a link it could not build instead of copying the current page', async () => {
    await render(() => null);
    await click();

    expect(writeText).not.toHaveBeenCalled();
    expect(visibleButton().textContent).toContain('Copy failed');
    expect(visibleButton().getAttribute('title')).toBe('Could not copy the link');
    expect(container.textContent).toContain('Could not copy the link');
  });

  it('reports a clipboard failure', async () => {
    writeText.mockRejectedValueOnce(new Error('denied'));
    await render(() => '/kv-cache?v=1');
    await click();

    expect(visibleButton().textContent).toContain('Copy failed');
    expect(container.textContent).toContain('Could not copy the link');
  });

  it('reports a browser with no clipboard instead of throwing', async () => {
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: undefined,
    });
    await render(() => '/predict?v=1');
    await click();

    expect(container.textContent).toContain('Could not copy the link');
  });

  it('clears the confirmation after a moment', async () => {
    vi.useFakeTimers();
    await render(() => '/predict');
    await click();
    expect(container.textContent).toContain('Link copied');

    await act(async () => {
      vi.advanceTimersByTime(2600);
    });
    expect(container.textContent).not.toContain('Link copied');
  });

  it('restarts the confirmation timer when the reader copies again', async () => {
    vi.useFakeTimers();
    await render(() => '/predict');
    await click();

    // Two seconds after the first click, copy again.
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    await click();

    // The first timer would have fired by now; the second has not.
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    expect(container.textContent).toContain('Link copied');

    await act(async () => {
      vi.advanceTimersByTime(1600);
    });
    expect(container.textContent).not.toContain('Link copied');
  });

  it('clears its timer on unmount', async () => {
    vi.useFakeTimers();
    const clearTimeout = vi.spyOn(window, 'clearTimeout');
    const local = document.createElement('div');
    document.body.append(local);
    const localRoot = createRoot(local);
    await act(async () => {
      localRoot.render(<CopyLinkButton getUrl={() => '/predict'} />);
    });
    await act(async () => {
      local.querySelector('button')?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      await Promise.resolve();
    });

    await act(async () => localRoot.unmount());
    local.remove();

    expect(clearTimeout).toHaveBeenCalled();
    clearTimeout.mockRestore();
  });
});

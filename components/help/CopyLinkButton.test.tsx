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

async function click() {
  const button = container.querySelector('button');
  expect(button).not.toBeNull();
  await act(async () => {
    button?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await Promise.resolve();
  });
}

describe('CopyLinkButton', () => {
  it('copies an absolute link for the current view', async () => {
    await render(() => '/predict?v=1&model=m');
    await click();

    expect(writeText).toHaveBeenCalledWith(`${window.location.origin}/predict?v=1&model=m`);
    expect(container.textContent).toContain('Link copied');
  });

  it('reports a link it could not build instead of copying the current page', async () => {
    await render(() => null);
    await click();

    expect(writeText).not.toHaveBeenCalled();
    expect(container.textContent).toContain('Could not copy the link');
  });

  it('reports a clipboard failure', async () => {
    writeText.mockRejectedValueOnce(new Error('denied'));
    await render(() => '/kv-cache?v=1');
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
});

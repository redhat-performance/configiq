// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { WarningPanel } from './WarningPanel';

let container: HTMLDivElement;
let root: ReturnType<typeof createRoot>;

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

async function render(warnings: string[]) {
  await act(async () => {
    root.render(<WarningPanel warnings={warnings} />);
  });
}

describe('WarningPanel', () => {
  it('renders nothing when a link was fully understood', async () => {
    await render([]);
    expect(container.textContent).toBe('');
  });

  it('shows one alert per dropped parameter', async () => {
    await render(['Ignored "tp": the value was not valid.', 'Ignored unknown parameter "autorun".']);

    const alerts = container.querySelectorAll('.pf-v6-c-alert');
    expect(alerts).toHaveLength(2);
    expect(container.textContent).toContain('Ignored "tp"');
    expect(container.textContent).toContain('autorun');
  });
});

// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// The page reads its shared view through `useSearchParams`. Tests mount it
// directly, so supply the current window search rather than the app router.
const navigation = vi.hoisted(() => ({ search: '' }));
const clipboard = vi.hoisted(() => ({ text: '' }));

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(navigation.search),
}));

vi.mock('@/lib/hooks/useCatalog', () => ({
  useCatalog: () => ({
    gpuOptions: [
      {
        systemId: 'h200_sxm',
        label: 'NVIDIA H200',
        vendor: 'nvidia',
        vramGb: 141,
        bandwidthTbps: 4.8,
        tflopsBf16: 989,
        architecture: 'hopper',
      },
    ],
    isLoading: false,
  }),
}));

vi.mock('@/contexts/SettingsContext', () => ({
  useSettings: () => ({
    preferredCloudProvider: null,
    pricingSource: 'merged',
  }),
}));

// Hardware costs are per-test state: the page now loads costings regardless of
// the experimental toggle, so coverage is what decides whether cost controls
// appear.
const costingsState = vi.hoisted(() => ({
  hardwareCosts: new Map<string, { new_usd: number }>(),
}));

vi.mock('@/lib/hooks/useCostings', () => ({
  useCostings: () => ({
    models: [],
    gpuCloudRates: new Map(),
    gpuHardwareCosts: costingsState.hardwareCosts,
    isLoading: false,
  }),
}));

// The chart is the only consumer of the selected axes, so it stands in for the
// view: the label pair pins which axes the page settled on.
vi.mock('./GpuBubbleChart', () => ({
  GpuBubbleChart: ({ xLabel, yLabel }: { xLabel: string; yLabel: string }) =>
    React.createElement('div', { 'data-testid': 'chart', 'data-x': xLabel, 'data-y': yLabel }),
}));

vi.mock('@/components/help/TourButton', () => ({ TourButton: () => null }));

const { default: GpuExplorerPage } = await import('./GpuExplorer');

let root: ReturnType<typeof createRoot>;
let container: HTMLDivElement;

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true;
  costingsState.hardwareCosts.clear();
  clipboard.text = '';
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText: vi.fn(async (text: string) => { clipboard.text = text; }) },
  });
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  window.history.replaceState({}, '', '/gpu-explorer');
});

async function mountAt(search: string) {
  window.history.replaceState({}, '', `/gpu-explorer${search}`);
  navigation.search = window.location.search;
  await act(async () => {
    root.render(<GpuExplorerPage />);
    await Promise.resolve();
  });
}

function axes() {
  const chart = container.querySelector<HTMLElement>('[data-testid="chart"]');
  return { x: chart?.dataset.x ?? null, y: chart?.dataset.y ?? null };
}

async function copyLink(): Promise<URL> {
  const button = Array.from(container.querySelectorAll('button'))
    .find(candidate => candidate.textContent?.includes('Copy link'));
  expect(button).toBeDefined();
  await act(async () => {
    button?.click();
    await Promise.resolve();
  });
  expect(clipboard.text).not.toBe('');
  return new URL(clipboard.text);
}

/** Reopen the copied link as a fresh page, the way a recipient would. */
async function remountAt(search: string) {
  await act(async () => root.unmount());
  root = createRoot(container);
  await mountAt(search);
}

describe('GPU explorer shared views', () => {
  it('plots the axes a preset implies', async () => {
    await mountAt('?preset=performance');

    expect(axes()).toEqual({
      x: 'Memory Bandwidth (GB/s)',
      y: 'Throughput Index',
    });
  });

  it('round-trips an axis customised on top of a preset', async () => {
    // The sender picked Performance and then changed X to VRAM. VRAM is the
    // balanced page's starting axis, not this preset's, so a link that omits it
    // would reopen on Memory Bandwidth instead of the view that was shared.
    await mountAt('?preset=performance&x=vram&y=vram');
    expect(axes()).toEqual({ x: 'VRAM (GB)', y: 'VRAM (GB)' });

    const url = await copyLink();
    expect(url.searchParams.get('preset')).toBe('performance');
    expect(url.searchParams.get('x')).toBe('vram');

    await remountAt(url.search);
    expect(axes()).toEqual({ x: 'VRAM (GB)', y: 'VRAM (GB)' });
  });

  it('keeps explicit axes from the link rather than the preset defaults', async () => {
    // A customised view is a preset plus overridden axes. Applying the preset's
    // axes afterwards would throw the override away.
    await mountAt('?preset=performance&x=vram&y=vram');

    expect(axes()).toEqual({ x: 'VRAM (GB)', y: 'VRAM (GB)' });
  });

  it('applies a preset cleanly even when a link set other axes first', async () => {
    await mountAt('?preset=performance&x=vram&y=vram');
    expect(axes()).toEqual({ x: 'VRAM (GB)', y: 'VRAM (GB)' });

    // Choosing a preset is an explicit choice, so it sets both axes.
    await act(async () => {
      Array.from(container.querySelectorAll('button'))
        .find(button => button.textContent === 'Balanced')
        ?.click();
    });

    expect(axes()).toEqual({ x: 'VRAM (GB)', y: 'Throughput Index' });
  });

  it('starts from the balanced view when the link names no axes', async () => {
    await mountAt('');

    expect(axes()).toEqual({ x: 'VRAM (GB)', y: 'Throughput Index' });
  });

  it('replaces a price axis the link names when no hardware costs are loaded', async () => {
    // Costings have already settled when the link applies, so the fallback has
    // to observe the axes themselves. Otherwise the chart keeps a price axis
    // whose control is hidden and plots every GPU at zero cost.
    await mountAt('?x=price&y=price');

    expect(axes()).toEqual({ x: 'VRAM (GB)', y: 'Throughput Index' });
  });

  it('offers the cost-efficiency preset when hardware costs are loaded', async () => {
    // The explorer is not behind the experimental costings toggle: hardware
    // cost coverage alone decides whether the cost controls are available.
    costingsState.hardwareCosts.set('h200_sxm', { new_usd: 32000 });
    await mountAt('?preset=cost-efficiency');

    expect(axes()).toEqual({ x: 'Hardware Cost (USD)', y: 'Throughput Index' });
    const preset = Array.from(container.querySelectorAll('button'))
      .find(button => button.textContent === 'Cost efficiency');
    expect(preset).toBeDefined();
    expect(preset?.hasAttribute('disabled')).toBe(false);

    const url = await copyLink();
    expect(url.searchParams.get('preset')).toBe('cost-efficiency');
  });
});

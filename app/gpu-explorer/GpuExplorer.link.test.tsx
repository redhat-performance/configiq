// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// The page reads its shared view through `useSearchParams`. Tests mount it
// directly, so supply the current window search rather than the app router.
const navigation = vi.hoisted(() => ({ search: '' }));

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
    costingsEnabled: false,
    preferredCloudProvider: null,
    pricingSource: 'merged',
  }),
}));

vi.mock('@/lib/hooks/useCostings', () => ({
  useCostings: () => ({
    models: [],
    gpuCloudRates: new Map(),
    gpuHardwareCosts: new Map(),
    isLoading: false,
  }),
}));

// The chart is the only consumer of the selected axes, so it stands in for the
// view: the label pair pins which axes the page settled on.
vi.mock('./GpuBubbleChart', () => ({
  GpuBubbleChart: ({ xLabel, yLabel }: { xLabel: string; yLabel: string }) =>
    React.createElement('div', { 'data-testid': 'chart', 'data-x': xLabel, 'data-y': yLabel }),
}));

vi.mock('@/components/help/CopyLinkButton', () => ({ CopyLinkButton: () => null }));
vi.mock('@/components/help/TourButton', () => ({ TourButton: () => null }));

const { default: GpuExplorerPage } = await import('./GpuExplorer');

let root: ReturnType<typeof createRoot>;
let container: HTMLDivElement;

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

describe('GPU explorer shared views', () => {
  it('plots the axes a preset implies', async () => {
    await mountAt('?preset=performance');

    expect(axes()).toEqual({
      x: 'Memory Bandwidth (GB/s)',
      y: 'Throughput Index',
    });
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
});

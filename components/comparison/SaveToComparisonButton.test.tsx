// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/comparison/storage', () => ({
  upsertComparisonEntry: vi.fn(),
}));

import { upsertComparisonEntry } from '@/lib/comparison/storage';
import type { ComparisonEntry } from '@/lib/comparison/types';
import { SaveToComparisonButton } from './SaveToComparisonButton';

let root: ReturnType<typeof createRoot>;
let host: HTMLDivElement;

function entry(): ComparisonEntry {
  return {
    id: 'entry-1',
    savedAt: '2026-11-21T00:00:00.000Z',
    name: 'Llama · H200',
    link: '/predict?model=llama&system=h200',
    model: 'meta-llama/Llama-3.1-70B',
    system: 'h200',
    systemLabel: 'H200',
    simulator: 'aisimulate',
    engineVersion: '0.14.0',
    backend: 'vllm',
    backendVersion: '0.9.0',
    mode: 'agg',
    workload: { isl: 2048, osl: 128, concurrency: 32, weightPrecision: 'FP8', kvPrecision: 'FP16' },
    topology: { tp: 8, pp: 1, replicas: 1, prefill: null, decode: null },
    figures: {
      gpus: 8,
      ttftMs: 180,
      tpotMs: 22.5,
      requestLatencyMs: 3060,
      tokensPerSecond: 1600,
      tokensPerSecondPerGpu: 200,
      tokensPerSecondPerUser: 50,
      weightMemoryGb: 140,
      kvCachePerRequestMb: 2500,
      kvCacheUsedGb: 80,
      totalVramGb: 640,
      cloudMonthlyUsd: 14_600,
      cloudFiveYearUsd: 876_000,
      selfHostedMonthlyUsd: 4_000,
      selfHostedFiveYearUsd: 240_000,
    },
  };
}

async function render(buildEntry: () => ComparisonEntry | null, disabledReason: string | null) {
  await act(async () => {
    root.render(<SaveToComparisonButton buildEntry={buildEntry} disabledReason={disabledReason} />);
  });
}

function button(): HTMLButtonElement {
  const found = host.querySelector('button');
  if (!found) throw new Error('expected a button');
  return found;
}

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  vi.mocked(upsertComparisonEntry).mockReset();
  vi.mocked(upsertComparisonEntry).mockReturnValue({ ok: true, document: { schemaVersion: 1, entries: [entry()] } });
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
});

describe('SaveToComparisonButton', () => {
  it('saves the snapshot built on the click and confirms it', async () => {
    await render(entry, null);
    expect(button().textContent).toContain('Save for comparison');

    await act(async () => button().click());

    expect(upsertComparisonEntry).toHaveBeenCalledTimes(1);
    expect(upsertComparisonEntry).toHaveBeenCalledWith(entry());
    expect(button().textContent).toContain('Saved');
    expect(host.textContent).toContain('Saved to comparison.');
    expect(host.querySelector('a')?.getAttribute('href')).toBe('/compare');
  });

  it('explains why saving is unavailable and stays disabled', async () => {
    await render(entry, 'Run a calculation first; there is no result to save.');

    expect(button().disabled).toBe(true);
    expect(button().getAttribute('aria-label')).toContain('Run a calculation first');
    await act(async () => button().click());
    expect(upsertComparisonEntry).not.toHaveBeenCalled();
  });

  it('reports a refused save instead of claiming success', async () => {
    vi.mocked(upsertComparisonEntry).mockReturnValue({
      ok: false,
      problem: 'A comparison holds at most 100 results; this one has 101.',
    });
    await render(entry, null);

    await act(async () => button().click());

    expect(host.textContent).toContain('A comparison holds at most 100 results');
    expect(host.textContent).not.toContain('Saved to comparison.');
  });

  it('refuses to save when the page has no snapshot to build', async () => {
    await render(() => null, null);

    await act(async () => button().click());

    expect(upsertComparisonEntry).not.toHaveBeenCalled();
    expect(host.textContent).toContain('There is no result to save yet.');
  });
});

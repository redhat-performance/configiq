// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { writeComparison } from '@/lib/comparison/storage';
import {
  COMPARISON_SCHEMA_VERSION,
  type ComparisonDocument,
  type ComparisonEntry,
} from '@/lib/comparison/types';
import { CompareResults } from './CompareResults';

let root: ReturnType<typeof createRoot>;
let host: HTMLDivElement;

function memoryStorage(): Storage {
  const values = new Map<string, string>();
  return {
    get length() {
      return values.size;
    },
    clear: () => values.clear(),
    getItem: key => values.get(key) ?? null,
    key: index => Array.from(values.keys())[index] ?? null,
    removeItem: key => {
      values.delete(key);
    },
    setItem: (key, value) => {
      values.set(key, value);
    },
  };
}

function entry(overrides: Partial<ComparisonEntry> = {}): ComparisonEntry {
  return {
    id: 'a',
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
    ...overrides,
  };
}

function doc(entries: ComparisonEntry[]): ComparisonDocument {
  return { schemaVersion: COMPARISON_SCHEMA_VERSION, entries };
}

async function render() {
  await act(async () => {
    root.render(<CompareResults />);
  });
}

function buttonByText(text: string): HTMLButtonElement | undefined {
  return Array.from(host.querySelectorAll('button')).find(button => button.textContent?.includes(text));
}

function rowFor(label: string): HTMLTableRowElement | undefined {
  return Array.from(host.querySelectorAll('tr')).find(
    row => row.querySelector('th')?.textContent === label,
  );
}

function cellsFor(label: string): string[] {
  const row = rowFor(label);
  if (!row) throw new Error(`expected a ${label} row`);
  return Array.from(row.querySelectorAll('td')).map(cell => cell.textContent ?? '');
}

function sectionMarks(label: string): string[] {
  const row = rowFor(label);
  if (!row) throw new Error(`expected a ${label} section row`);
  return Array.from(row.querySelectorAll('td')).map(cell => cell.textContent?.trim() ?? '');
}

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.stubGlobal('localStorage', memoryStorage());
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

describe('CompareResults', () => {
  it('invites the reader to save a result when the set is empty', async () => {
    await render();

    expect(host.textContent).toContain('No results saved yet');
    expect(host.querySelector('a[href="/predict"]')).not.toBeNull();
    expect(host.querySelector('a[href="/recommend"]')).not.toBeNull();
    expect(host.querySelector('table')).toBeNull();
  });

  it('shows one column per saved result with its identity', async () => {
    writeComparison(doc([
      entry(),
      entry({ id: 'b', name: 'Llama · H100', link: '/predict?n=2', systemLabel: 'H100', simulator: 'blis' }),
    ]));
    await render();

    const columns = Array.from(host.querySelectorAll('thead th')).slice(1);
    expect(columns.map(column => column.textContent)).toEqual([
      expect.stringContaining('Llama · H200'),
      expect.stringContaining('Llama · H100'),
    ]);
    expect(columns[0].textContent).toContain('AISim');
    expect(columns[1].textContent).toContain('blis');
  });

  it('shows the simulator, backend, and engine in one deployment row', async () => {
    writeComparison(doc([entry()]));
    await render();

    expect(rowFor('Simulator / backend / engine')?.textContent).toContain('AISim · vLLM 0.14.0');
    expect(rowFor('Simulator')).toBeUndefined();
    expect(rowFor('Engine version')).toBeUndefined();
    expect(rowFor('Backend')).toBeUndefined();
  });

  it('shows absolute values until a baseline is chosen', async () => {
    writeComparison(doc([entry(), entry({ id: 'b', link: '/predict?n=2', figures: { ...entry().figures, gpus: 16 } })]));
    await render();

    expect(cellsFor('GPUs')).toEqual(['8', '16']);
  });

  it('shows a neutral direction against the chosen baseline', async () => {
    writeComparison(doc([entry(), entry({ id: 'b', link: '/predict?n=2', figures: { ...entry().figures, gpus: 16 } })]));
    await render();

    const baseline = host.querySelector<HTMLInputElement>('#baseline-a');
    await act(async () => baseline?.click());

    expect(cellsFor('GPUs')).toEqual(['8', '16▲ 8']);
    expect(sectionMarks('Deployment')).toEqual(['baseline', '']);
    expect(sectionMarks('Cost')).toEqual(['baseline', '']);
    expect(host.textContent).toContain('Clear baseline');
  });

  it('moves the baseline mark above the column when the baseline changes', async () => {
    writeComparison(doc([entry(), entry({ id: 'b', link: '/predict?n=2', figures: { ...entry().figures, gpus: 16 } })]));
    await render();

    await act(async () => host.querySelector<HTMLInputElement>('#baseline-a')?.click());
    expect(sectionMarks('Deployment')).toEqual(['baseline', '']);

    await act(async () => host.querySelector<HTMLInputElement>('#baseline-b')?.click());

    expect(sectionMarks('Deployment')).toEqual(['', 'baseline']);
    expect(cellsFor('GPUs')).toEqual(['8▼ 8', '16']);
  });

  it('removes one result', async () => {
    writeComparison(doc([entry(), entry({ id: 'b', name: 'Second', link: '/predict?n=2' })]));
    await render();

    const remove = host.querySelector<HTMLButtonElement>('[aria-label="Remove Llama · H200 from the comparison"]');
    await act(async () => remove?.click());

    expect(host.textContent).toContain('Removed Llama · H200');
    expect(host.querySelectorAll('[aria-label^="Remove "]')).toHaveLength(1);
  });

  it('clears every result only after confirmation', async () => {
    writeComparison(doc([entry()]));
    await render();

    await act(async () => buttonByText('Clear all')?.click());
    expect(document.body.textContent).toContain('Clear every saved result?');

    const confirm = Array.from(document.querySelectorAll('button'))
      .filter(button => button.textContent?.trim() === 'Clear all')
      .pop();
    await act(async () => confirm?.click());

    expect(host.textContent).toContain('No results saved yet');
  });

  it('imports a saved set while the comparison is empty', async () => {
    await render();
    expect(host.textContent).toContain('No results saved yet');

    const incoming: ComparisonDocument = doc([entry({ id: 'imported', link: '/predict?n=9' })]);
    const input = host.querySelector<HTMLInputElement>('input[type="file"]');
    if (!input) throw new Error('expected a file input');
    const file = new File([JSON.stringify(incoming)], 'set.json', { type: 'application/json' });
    Object.defineProperty(input, 'files', { configurable: true, value: [file] });
    await act(async () => {
      input.dispatchEvent(new Event('change', { bubbles: true }));
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(host.textContent).toContain('Imported 1 result from set.json.');
    expect(host.querySelectorAll('[aria-label^="Remove "]')).toHaveLength(1);
  });

  it('reports an import that fails validation without changing the set', async () => {
    writeComparison(doc([entry()]));
    await render();

    const input = host.querySelector<HTMLInputElement>('input[type="file"]');
    if (!input) throw new Error('expected a file input');
    const file = new File(['{"schemaVersion":2,"entries":[]}'], 'set.json', { type: 'application/json' });
    Object.defineProperty(input, 'files', { configurable: true, value: [file] });
    await act(async () => {
      input.dispatchEvent(new Event('change', { bubbles: true }));
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(host.textContent).toContain('Unsupported comparison version 2');
    expect(host.querySelectorAll('[aria-label^="Remove "]')).toHaveLength(1);
  });

  it('opens the details for one result', async () => {
    writeComparison(doc([entry()]));
    await render();

    await act(async () => buttonByText('Details')?.click());

    expect(document.body.textContent).toContain('Request latency (ms)');
    expect(document.querySelector('a[href="/predict?model=llama&system=h200"]')).not.toBeNull();
  });

  it('exports the set as CSV', async () => {
    writeComparison(doc([entry()]));
    await render();

    const createObjectURL = vi.fn(() => 'blob:comparison');
    const revokeObjectURL = vi.fn();
    vi.stubGlobal('URL', { ...URL, createObjectURL, revokeObjectURL });
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);

    await act(async () => buttonByText('Export CSV')?.click());

    expect(createObjectURL).toHaveBeenCalledTimes(1);
    expect(click).toHaveBeenCalledTimes(1);
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:comparison');
    click.mockRestore();
  });
});

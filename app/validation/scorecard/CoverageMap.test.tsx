// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ScorecardCoverage } from '@/lib/scorecard/types';
import CoverageMap from './CoverageMap';

function coverage(overrides: Partial<ScorecardCoverage> = {}): ScorecardCoverage {
  return {
    simulator: 'aisim',
    source: 'redhat',
    model: 'org/model',
    accelerator: 'H200',
    releaseMatch: 'any',
    supported: 2,
    unsupported: 0,
    inconclusive: 0,
    ineligible: 0,
    total: 2,
    ...overrides,
  };
}

function defaultCoverage(): ScorecardCoverage[] {
  return [
    coverage(),
    coverage({ simulator: 'blis' }),
    coverage({ model: 'org/crescent', accelerator: 'B200' }),
    // Measured but never scored: the coverage gap the map has to show.
    coverage({ model: 'org/gap', accelerator: 'H200', supported: 0, unsupported: 1, total: 1 }),
    coverage({ model: 'org/shared', accelerator: 'H200' }),
    coverage({ simulator: 'blis', model: 'org/shared', accelerator: 'H200' }),
    coverage({ source: 'inferencex', model: 'org/shared', accelerator: 'H200' }),
    coverage({ source: 'inferencex', simulator: 'blis', model: 'org/shared', accelerator: 'H200' }),
    coverage({ source: 'inferencex', model: 'org/other', accelerator: 'B200' }),
  ];
}

const SIMULATORS = [
  { id: 'aisim', label: 'AISim', color: '#76b900' },
  { id: 'blis', label: 'BLIS', color: '#0f62fe' },
];

function normalized(element: Element | null): string {
  return (element?.textContent ?? '').replace(/\s+/g, ' ').trim();
}

let root: Root;
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
});

async function mount(
  options: {
    coverage?: readonly ScorecardCoverage[];
    /** `null` mounts the snapshot that never published a workload count. */
    sharedWorkloads?: number | null;
    source?: 'all' | 'redhat' | 'inferencex';
    releaseMatch?: 'any' | 'matched';
    onSelectPair?: (pair: { model: string; accelerator: string }) => void;
  } = {},
): Promise<ReturnType<typeof vi.fn>> {
  const onSelectPair = vi.fn(options.onSelectPair);
  const sharedWorkloads = options.sharedWorkloads === null
    ? undefined
    : options.sharedWorkloads ?? 0;
  await act(async () => {
    root.render(
      <CoverageMap
        coverage={options.coverage ?? defaultCoverage()}
        sharedWorkloads={sharedWorkloads}
        source={options.source ?? 'all'}
        releaseMatch={options.releaseMatch ?? 'any'}
        simulators={SIMULATORS}
        onSelectPair={onSelectPair}
      />,
    );
    await Promise.resolve();
  });
  return onSelectPair;
}

describe('CoverageMap', () => {
  it('draws both figures and one legend entry per set', async () => {
    await mount();

    const captions = [...container.querySelectorAll('figcaption')].map(caption => normalized(caption));
    expect(captions).toEqual(['Simulator coverage', 'Measured-data coverage']);

    const [simulators, sources] = [...container.querySelectorAll('figure')];
    expect(normalized(simulators)).toContain('AISim 4');
    expect(normalized(simulators)).toContain('BLIS 2');
    expect(normalized(simulators)).toContain('Shared 2');
    expect(normalized(sources)).toContain('Red Hat 4');
    expect(normalized(sources)).toContain('InferenceX 2');
    expect(normalized(sources)).toContain('Shared 1');
    expect(normalized(sources)).toContain('No shared workload shapes');
  });

  it('does not claim a shared pair is a shared workload', async () => {
    await mount({ sharedWorkloads: 0 });

    const sources = container.querySelectorAll('figure')[1];
    // One shared pair, no shared shape: both facts have to appear.
    expect(normalized(sources)).toContain('Shared 1');
    expect(normalized(sources)).toContain('No shared workload shapes');
  });

  it('says when the snapshot does not publish a workload count', async () => {
    await mount({ sharedWorkloads: null });

    expect(normalized(container)).toContain('Workload overlap not published in this snapshot');
  });

  it('draws one mark per measured pair and no simulator circle for an unscored pair', async () => {
    await mount();

    // Four pairs are in a simulator circle; the fifth is measured only.
    const [simulators] = [...container.querySelectorAll('figure')];
    expect(simulators.querySelectorAll('[class*="vennDot"]')).toHaveLength(4);
    expect(simulators.querySelector('button[aria-label="org/gap, H200"]')).toBeNull();

    // The measured-data figure still carries the unscored pair, as a static dot.
    const sources = container.querySelectorAll('figure')[1];
    expect(sources.querySelectorAll('[class*="vennDot"]')).toHaveLength(5);
    expect(sources.querySelector('[class*="vennDotStatic"]')).not.toBeNull();
  });

  it('reports the exact four-way membership in the table', async () => {
    await mount();

    const rows = [...container.querySelectorAll('tbody tr')];
    expect(rows).toHaveLength(5);

    const shared = rows.find(row => normalized(row).startsWith('org/shared'));
    expect(shared).toBeDefined();
    expect(
      [...shared!.querySelectorAll('[role="img"]')].map(mark => mark.getAttribute('aria-label')),
    ).toEqual([
      'AISim scored: yes',
      'BLIS scored: yes',
      'Red Hat measured: yes',
      'InferenceX measured: yes',
    ]);

    const unscored = rows.find(row => normalized(row).startsWith('org/gap'));
    expect(
      [...unscored!.querySelectorAll('[role="img"]')].map(mark => mark.getAttribute('aria-label')),
    ).toEqual([
      'AISim scored: no',
      'BLIS scored: no',
      'Red Hat measured: yes',
      'InferenceX measured: no',
    ]);
  });

  it('selects a pair from a dot', async () => {
    const onSelect = await mount();
    const dot = container.querySelector<HTMLButtonElement>('button[aria-label="org/model, H200"]');

    expect(dot).not.toBeNull();
    await act(async () => {
      dot!.click();
      await Promise.resolve();
    });

    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect.mock.calls[0][0]).toMatchObject({
      model: 'org/model',
      accelerator: 'H200',
    });
  });

  it('selects a pair from the table as well', async () => {
    const onSelect = await mount();
    const button = container.querySelector<HTMLButtonElement>(
      'button[aria-label="Show org/shared on H200"]',
    );

    expect(button).not.toBeNull();
    await act(async () => {
      button!.click();
      await Promise.resolve();
    });

    expect(onSelect.mock.calls[0][0]).toMatchObject({
      model: 'org/shared',
      accelerator: 'H200',
    });
  });

  it('does not make an unscored pair clickable', async () => {
    await mount();

    expect(container.querySelector('button[aria-label="Show org/gap on H200"]')).toBeNull();
    expect(container.textContent).not.toContain('for org/gap');
  });

  it('drops the measured-data figure when one source is selected', async () => {
    await mount({ source: 'redhat' });

    const captions = [...container.querySelectorAll('figcaption')].map(caption => normalized(caption));
    expect(captions).toEqual(['Simulator coverage']);
    expect(container.textContent).not.toContain('InferenceX');
  });

  it('follows the release slice', async () => {
    await mount({
      coverage: [
        coverage({ releaseMatch: 'matched' }),
        coverage({ simulator: 'blis', releaseMatch: 'matched' }),
      ],
      releaseMatch: 'matched',
    });

    expect(container.querySelectorAll('tbody tr')).toHaveLength(1);
    expect(normalized(container)).toContain('org/model');
  });

  it('says plainly when the slice has no pair', async () => {
    await mount({ coverage: [] });

    expect(container.textContent).toContain('No measured model + GPU pair in this view.');
    expect(container.querySelector('figure')).toBeNull();
  });

  it('keeps the page text minimal until a point is hovered or focused', async () => {
    await mount();

    const caption = container.querySelector('[aria-live="polite"]');
    expect(normalized(caption)).toBe('One dot per measured model + GPU pair');

    const dot = container.querySelector<HTMLButtonElement>('button[aria-label="org/model, H200"]');
    await act(async () => {
      dot!.focus();
      await Promise.resolve();
    });

    expect(normalized(container.querySelector('[aria-live="polite"]')))
      .toContain('org/model');
  });
});

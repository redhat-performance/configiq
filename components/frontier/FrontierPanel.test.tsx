// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@patternfly/react-charts/victory', async () => {
  const ReactModule = await import('react');
  const Wrapper = ({ children }: { children?: React.ReactNode }) =>
    ReactModule.createElement('div', null, children);
  const Empty = () => null;
  return {
    Chart: Wrapper,
    ChartAxis: Empty,
    ChartLine: Empty,
    ChartScatter: Empty,
    ChartVoronoiContainer: Empty,
  };
});

import { FrontierPanel } from './FrontierPanel';
import type { DebugRecord } from '@/lib/debug/types';
import type { Candidate, FrontierState } from '@/lib/types/pareto';

let root: ReturnType<typeof createRoot>;
let host: HTMLDivElement;
const writeText = vi.fn(async (_text: string) => undefined);

function candidate(overrides: Partial<Candidate> = {}): Candidate {
  return {
    candidate_id: 'candidate-000001',
    status: 'feasible',
    reason_category: null,
    mode: 'agg',
    topology: { deployment_mode: 'agg', tp: 1, pp: 1, attention_dp: 1, replicas: 1 },
    used_gpus: 1,
    metrics: { mean_ttft_ms: 180, mean_tpot_ms: 22 },
    objectives: { throughput_per_user: 22, throughput_per_gpu: 200 },
    transfer_delay_modeled: false,
    provenance_sources: ['sdk'],
    simulator: 'aisimulate',
    ...overrides,
  };
}

function replay(): DebugRecord {
  return {
    endpoint: '/recommend',
    simulator: 'aisimulate',
    execution: 'in_process',
    created_at: 1_730_000_000,
    versions: {},
    selection: null,
    notes: [],
    invocations: [{
      id: 'candidate-1',
      label: 'Candidate c1',
      command: 'aisimulate predict -c candidate-1.yaml --format json',
      execution: 'in_process',
      group: null,
      candidate_id: 'candidate-000001',
      notes: [],
      files: [{ path: 'candidate-1.yaml', contents: 'engine: {}\n' }],
    }],
    script: null,
  };
}

function ready(candidates: Candidate[], record: DebugRecord | null): FrontierState {
  return {
    status: 'ready',
    candidates,
    views: { pareto_front: ['candidate-000001'], top_n: ['candidate-000001'] },
    sweep: null,
    replay: record,
  };
}

async function render(state: FrontierState) {
  await act(async () => {
    root.render(
      <FrontierPanel state={state} gpuLabel="H200" onRetry={() => undefined} />,
    );
  });
}

function button(label: string): HTMLButtonElement | undefined {
  return Array.from(host.querySelectorAll<HTMLButtonElement>('button'))
    .find(candidate => candidate.getAttribute('aria-label') === label);
}

beforeEach(() => {
  Object.assign(globalThis, {
    IS_REACT_ACT_ENVIRONMENT: true,
    ResizeObserver: class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  });
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

describe('FrontierPanel candidate replay', () => {
  it('copies the command and input files recorded for that candidate', async () => {
    await render(ready([candidate()], replay()));

    expect(Array.from(host.querySelectorAll('th')).map(th => th.textContent)).toContain('Save');

    const copy = button('Copy command for candidate-001');
    expect(copy).toBeDefined();
    await act(async () => copy!.click());

    expect(writeText).toHaveBeenCalledTimes(1);
    const bundle = writeText.mock.calls[0][0];
    expect(bundle).toContain("cat > 'candidate-1.yaml'");
    expect(bundle).toContain('aisimulate predict -c candidate-1.yaml --format json');
    expect(button('Command copied for candidate-001')).toBeDefined();
    expect(host.textContent).toContain('Command for candidate-001 copied with its input files.');
  });

  it('labels the sweep command when a matched row shows sizing figures', async () => {
    await render(ready([
      candidate({ provenance_sources: ['sizing recommendation'], recommended: true }),
    ], replay()));

    expect(button('Copy sweep command for candidate-001')).toBeDefined();
    expect(button('Copy command for candidate-001')).toBeUndefined();
  });

  it('shows a failed copy instead of silently keeping the copy icon', async () => {
    writeText.mockRejectedValueOnce(new Error('denied'));
    await render(ready([candidate()], replay()));

    const copy = button('Copy command for candidate-001');
    await act(async () => copy!.click());

    expect(button('Command copy failed for candidate-001')).toBeDefined();
    expect(host.textContent).toContain('Could not copy the command for candidate-001.');
    const status = host.querySelector('[role="status"]');
    expect(status?.getAttribute('class')).toContain('replayStatusError');
  });

  it('explains a row the sweep recorded no command for instead of copying another', async () => {
    await render(ready([candidate({ candidate_id: 'candidate-000002' })], replay()));

    expect(button('Copy command for candidate-002')).toBeUndefined();
    const unavailable = host.querySelector('[aria-label^="Save unavailable for candidate-002"]');
    expect(unavailable).not.toBeNull();
    expect(unavailable?.getAttribute('aria-label')).toContain('did not report a command to save');
    expect(writeText).not.toHaveBeenCalled();
  });

  it('offers no save column when the response carried no record', async () => {
    await render(ready([candidate()], null));

    expect(Array.from(host.querySelectorAll('th')).map(th => th.textContent)).not.toContain('Save');
    expect(button('Copy command for candidate-001')).toBeUndefined();
  });

  it('does not offer a sweep command for a row projected from the sizing response', async () => {
    await render(ready([candidate({ candidate_id: 'sizing-recommendation', recommended: true })], replay()));

    const unavailable = host.querySelector('[aria-label^="Save unavailable for Sizing recommendation"]');
    expect(unavailable?.getAttribute('aria-label')).toContain('projected from the sizing response');
  });
});

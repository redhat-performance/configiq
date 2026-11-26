// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const charts = vi.hoisted(() => ({
  /** Name given to every rendered comparison curve. */
  lines: [] as Array<{ name?: string }>,
  /** Voronoi components ignored when resolving a hover. */
  voronoiBlacklist: undefined as string[] | undefined,
}));

vi.mock('@patternfly/react-charts/victory', async () => {
  const ReactModule = await import('react');
  const Chart = ({
    children,
    containerComponent,
  }: {
    children?: React.ReactNode;
    containerComponent?: React.ReactElement<{ voronoiBlacklist?: string[] }>;
  }) => {
    charts.voronoiBlacklist = containerComponent?.props.voronoiBlacklist;
    return ReactModule.createElement('div', null, children);
  };
  const Empty = () => null;
  const ChartLine = ({ name }: { name?: string }) => {
    charts.lines.push({ name });
    return null;
  };
  return {
    Chart,
    ChartAxis: Empty,
    ChartLine,
    ChartScatter: Empty,
    ChartVoronoiContainer: Empty,
  };
});

import { CapacityOptionsPanel } from './CapacityOptionsPanel';
import type {
  RecommendCapacityOption,
  RecommendCapacityOutcome,
  RecommendResult,
} from '@/lib/api/recommend';

function result(gpus: number, ttftMs: number, perGpu: number): RecommendResult {
  return {
    requestId: 'size_test',
    status: 'completed',
    simulator: 'aisim',
    mode: 'agg',
    recommendation: {
      gpusNeeded: gpus,
      gpusPerReplica: gpus,
      totalGpus: gpus,
      replicasNeeded: 1,
      tensorParallelSize: gpus,
      pipelineParallelSize: 1,
      dataParallelSize: 1,
      contextParallelSize: 1,
      moeTensorParallelSize: null,
      moeExpertParallelSize: null,
      batchSize: null,
    },
    phases: { prefill: null, decode: null, encode: null },
    performance: { ttftLatencyMs: ttftMs, tpotMs: 20, requestLatencyMs: 3000, concurrency: 8 },
    throughput: {
      tokensPerSecond: perGpu * gpus,
      tokensPerSecondPerGpu: perGpu,
      tokensPerSecondPerUser: 40,
    },
    memory: { value: 80, unit: 'GB', breakdown: null },
    memoryAvailable: true,
    engineVersion: null,
    precision: { weights: null, kvCache: null },
    metadata: {
      modelPath: 'Qwen/Qwen2.5-7B-Instruct',
      system: 'h200_sxm',
      inputTokens: 1024,
      outputTokens: 256,
      targetTtftMs: null,
      durationMs: 1000,
    },
    warnings: [],
  };
}

function capacity(gpus: number): RecommendCapacityOutcome {
  return {
    targetConcurrency: 16,
    contextPerSequenceTokens: 4096,
    sharedPrefixTokens: 0,
    capacityTokens: 65536,
    capacitySequences: 16,
    fits: true,
    shortfallSequences: null,
    shortfallTokens: null,
    minimumGpus: gpus,
    perRankTokens: 4096,
    replicas: 1,
    attentionDp: 1,
    evaluatedShapes: 4,
    feasibleShapes: 2,
    exhaustive: true,
    performanceAvailable: true,
    performanceUnavailableReason: null,
    mode: 'agg',
    prefillReplicas: null,
    decodeReplicas: null,
    prefillGpus: null,
    decodeGpus: null,
    simulator: 'aisim',
  };
}

function option(
  optionId: string,
  rank: number,
  gpus: number,
  ttftMs: number,
  perGpu: number,
): RecommendCapacityOption {
  return {
    optionId,
    rank,
    role: rank === 1 ? 'minimum' : 'larger',
    gpusOverMinimum: gpus - 2,
    capacity: capacity(gpus),
    result: result(gpus, ttftMs, perGpu),
  };
}

let root: ReturnType<typeof createRoot>;
let host: HTMLDivElement;

async function render(options: RecommendCapacityOption[]) {
  await act(async () => {
    root.render(
      <CapacityOptionsPanel
        options={options}
        selectedOptionId={options[0]?.optionId ?? null}
        gpuLabel="H200"
        onSelect={() => undefined}
      />,
    );
  });
}

beforeEach(() => {
  charts.lines.length = 0;
  charts.voronoiBlacklist = undefined;
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
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
});

describe('CapacityOptionsPanel chart hover labels', () => {
  it('keeps the comparison curve out of voronoi resolution so each point labels once', async () => {
    // Neither point dominates the other, so both sit on the comparison curve.
    await render([
      option('option-1', 1, 2, 80, 400),
      option('option-2', 2, 4, 100, 450),
    ]);

    // The curve plots points that the scatter also plots. Victory's voronoi
    // container activates every dataset at the hovered coordinate, so the curve
    // must be excluded or the tooltip repeats the point label.
    const lineNames = new Set(charts.lines.map(line => line.name));
    expect(lineNames.size).toBe(1);
    const [lineName] = [...lineNames];
    expect(lineName).toBeTruthy();
    expect(charts.voronoiBlacklist).toContain(lineName);
  });
});

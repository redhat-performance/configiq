// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { RecommendResult } from '@/lib/api/recommend';
import { AnswerCard } from './AnswerCard';

function resultFixture(overrides: Partial<RecommendResult> = {}): RecommendResult {
  return {
    requestId: 'req-1',
    status: 'completed',
    simulator: 'aisimulate',
    mode: 'agg',
    recommendation: {
      gpusNeeded: 8,
      gpusPerReplica: 8,
      totalGpus: 8,
      replicasNeeded: 1,
      tensorParallelSize: 8,
      pipelineParallelSize: 1,
      dataParallelSize: 1,
      contextParallelSize: 1,
      moeTensorParallelSize: null,
      moeExpertParallelSize: null,
      batchSize: null,
    },
    phases: { prefill: null, decode: null, encode: null },
    performance: { ttftLatencyMs: 251, tpotMs: 6.9, requestLatencyMs: 7100, concurrency: 64 },
    throughput: { tokensPerSecond: 4400, tokensPerSecondPerGpu: 550, tokensPerSecondPerUser: 68 },
    memory: { value: 120, unit: 'GB', breakdown: null },
    memoryAvailable: true,
    metadata: {
      modelPath: 'Qwen/Qwen3-8B',
      system: 'h200_sxm',
      inputTokens: 1024,
      outputTokens: 1024,
      targetTtftMs: null,
      durationMs: 4000,
    },
    warnings: [],
    ...overrides,
  }
}

let host: HTMLDivElement;
let root: ReturnType<typeof createRoot>;

beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
});

async function renderCard(props: Partial<Parameters<typeof AnswerCard>[0]> = {}) {
  const base = {
    result: resultFixture(),
    ttftTarget: 400,
    tpotTarget: 30,
    gpusFree: 8,
    loadUnit: 'users' as const,
    ...props,
  };
  await act(async () => {
    root.render(<AnswerCard {...base} />);
  });
  return base;
}

describe('AnswerCard', () => {
  it('shows the GPU count against what the cluster has free', async () => {
    await renderCard();
    expect(host.textContent).toContain('8');
    expect(host.textContent).toContain('of 8 GPUs free');
  });

  it('titles the fit state and the failure state differently', async () => {
    await renderCard({ gpusFree: 8, result: resultFixture() });
    expect(host.textContent).toContain('fits your cluster');
    await renderCard({ gpusFree: 4 });
    expect(host.textContent).toContain('Needs more GPUs');
    expect(host.textContent).toContain('cluster has 4 free');
  });

  it('shows predicted latency against the target with pass and warn verdicts', async () => {
    await renderCard();
    expect(host.textContent).toContain('251');
    expect(host.textContent).toContain('meets 400');
    expect(host.textContent).toContain('6.9');
    expect(host.textContent).toContain('meets 30');
    await renderCard({ ttftTarget: 200 });
    expect(host.textContent).toContain('over 200');
  });

  it('reports the served load verbatim from the response', async () => {
    await renderCard();
    expect(host.textContent).toContain('64 concurrent');
    await renderCard({ loadUnit: 'rps', result: resultFixture({
      demand: {
        targetRps: 5, achievedRps: 4.7, met: true, observedConcurrency: 60,
        windowSeconds: 60, completedRequests: 280, evaluatedConfigurations: 12,
        bestObservedRps: 4.7, simulator: 'aisimulate',
      },
    }) })
    expect(host.textContent).toContain('4.7 req/s');
  });

  it('names the source of every number in the details disclosure', async () => {
    await renderCard();
    const details = host.querySelector('details');
    expect(details).not.toBeNull();
    expect(details!.textContent).toContain('1 replica × TP 8');
    expect(details!.textContent).toContain('550');
    expect(details!.textContent).toContain('aisimulate');
    expect(details!.textContent).toContain('req-1');
  });

  it('carries the honest best-effort disclaimer', async () => {
    await renderCard();
    expect(host.textContent).toContain('Best-effort');
  });
});

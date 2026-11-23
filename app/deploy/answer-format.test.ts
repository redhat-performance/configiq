import { describe, expect, it } from 'vitest'
import type { RecommendResult } from '@/lib/api/recommend'
import { latencyVerdict, servedLoadLabel, topologySummary } from './answer-format'

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
    performance: {
      ttftLatencyMs: 251,
      tpotMs: 6.9,
      requestLatencyMs: 7100,
      concurrency: 64,
    },
    throughput: {
      tokensPerSecond: 4400,
      tokensPerSecondPerGpu: 550,
      tokensPerSecondPerUser: 68,
    },
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

describe('latencyVerdict', () => {
  it('marks a prediction at or under the target as meeting it', () => {
    expect(latencyVerdict(251, 400)).toBe('meets')
    expect(latencyVerdict(400, 400)).toBe('meets')
  })

  it('marks a prediction over the target as over', () => {
    expect(latencyVerdict(401, 400)).toBe('over')
  })

  it('returns null without a prediction', () => {
    expect(latencyVerdict(null, 400)).toBeNull()
  })
})

describe('servedLoadLabel', () => {
  it('reports the concurrency the response served, verbatim', () => {
    expect(servedLoadLabel(resultFixture(), 'users')).toEqual({ value: 64, unit: 'users' })
  })

  it('reports the achieved request rate for a demand run', () => {
    const result = resultFixture({
      demand: {
        targetRps: 5,
        achievedRps: 4.7,
        met: true,
        observedConcurrency: 60,
        windowSeconds: 60,
        completedRequests: 280,
        evaluatedConfigurations: 12,
        bestObservedRps: 4.7,
        simulator: 'aisimulate',
      },
    })
    expect(servedLoadLabel(result, 'rps')).toEqual({ value: 4.7, unit: 'req/s' })
  })

  it('returns null when the demand run reports no achieved rate', () => {
    const result = resultFixture({ demand: undefined })
    expect(servedLoadLabel(result, 'rps')).toBeNull()
  })
})

describe('topologySummary', () => {
  it('describes an aggregated deployment as replicas with tensor width', () => {
    expect(topologySummary(resultFixture())).toBe('1 replica × TP 8')
  })

  it('describes multiple replicas', () => {
    const result = resultFixture({
      recommendation: {
        ...resultFixture().recommendation,
        replicasNeeded: 2,
        gpusPerReplica: 4,
        gpusNeeded: 8,
        tensorParallelSize: 4,
      },
    })
    expect(topologySummary(result)).toBe('2 replicas × TP 4')
  })

  it('names both pools for a disaggregated deployment', () => {
    const result = resultFixture({ mode: 'disagg' })
    expect(topologySummary(result)).toContain('disaggregated')
  })
})

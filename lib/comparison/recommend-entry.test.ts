import { describe, expect, it } from 'vitest'

import type { RecommendResult } from '../api/recommend'
import { buildRecommendEntry, sizingGoalFrom } from './recommend-entry'

function result(overrides: Partial<RecommendResult> = {}): RecommendResult {
  return {
    requestId: 'rec-1',
    status: 'completed',
    simulator: 'aisimulate',
    mode: 'agg',
    recommendation: {
      gpusNeeded: 4,
      gpusPerReplica: 4,
      totalGpus: 4,
      replicasNeeded: 1,
      tensorParallelSize: 4,
      pipelineParallelSize: 1,
      dataParallelSize: 1,
      contextParallelSize: 1,
      moeTensorParallelSize: null,
      moeExpertParallelSize: null,
      batchSize: 64,
    },
    phases: { prefill: null, decode: null, encode: null },
    performance: {
      ttftLatencyMs: 210,
      tpotMs: 24,
      requestLatencyMs: 12_000,
      concurrency: 64,
    },
    throughput: {
      tokensPerSecond: 900,
      tokensPerSecondPerGpu: 225,
      tokensPerSecondPerUser: 14.06,
    },
    memory: {
      value: 58,
      unit: 'GB',
      breakdown: { weightsGb: 40, kvCacheGb: 16, overheadGb: 2 },
    },
    memoryAvailable: true,
    engineVersion: '0.9.2',
    precision: { weights: 'fp8', kvCache: 'fp8_e5m2' },
    metadata: {
      modelPath: 'meta-llama/Llama-3.1-70B',
      system: 'h200',
      inputTokens: 2048,
      outputTokens: 128,
      targetTtftMs: 300,
      durationMs: 12,
    },
    warnings: [],
    ...overrides,
  }
}

const base = {
  link: '/recommend?model=meta-llama/Llama-3.1-70B&system=h200',
  model: 'meta-llama/Llama-3.1-70B',
  system: 'h200',
  systemLabel: 'H200',
  backend: 'vllm',
  backendVersion: '0.9.0',
  goal: sizingGoalFrom({ sla_mode: 'enforce', ttft: 300, tpot: 30 }),
  cloudRatePerHour: 2.5,
  hardwareCostUsd: 30_000,
  now: new Date('2026-11-21T00:00:00.000Z'),
  id: 'entry-1',
}

describe('buildRecommendEntry', () => {
  it('copies the reported figures, provenance, and goal into a snapshot', () => {
    const entry = buildRecommendEntry({ ...base, result: result() })

    expect(entry.name).toBe('Llama-3.1-70B · H200')
    expect(entry.savedAt).toBe('2026-11-21T00:00:00.000Z')
    expect(entry.simulator).toBe('aisimulate')
    expect(entry.engineVersion).toBe('0.9.2')
    expect(entry.backendVersion).toBe('0.9.0')
    expect(entry.mode).toBe('agg')
    expect(entry.goal).toEqual({
      kind: 'latency',
      targetRequestLatencyMs: null,
      targetTtftMs: 300,
      targetTpotMs: 30,
      targetRequestRate: null,
      targetConcurrency: null,
    })
    expect(entry.workload).toEqual({
      isl: 2048,
      osl: 128,
      concurrency: 64,
      weightPrecision: 'FP8',
      kvPrecision: 'FP8',
    })
    expect(entry.topology).toEqual({ tp: 4, pp: 1, replicas: 1, prefill: null, decode: null })
    expect(entry.figures).toMatchObject({
      gpus: 4,
      ttftMs: 210,
      tpotMs: 24,
      requestLatencyMs: 12_000,
      tokensPerSecond: 900,
      tokensPerSecondPerGpu: 225,
      tokensPerSecondPerUser: 14.06,
      weightMemoryGb: 40,
      kvCacheUsedGb: 16,
      kvCachePerRequestMb: 250,
      totalVramGb: null,
      cloudMonthlyUsd: 4 * 2.5 * 730,
    })
  })

  it('reports an unreported memory breakdown as unavailable instead of zero', () => {
    const entry = buildRecommendEntry({
      ...base,
      cloudRatePerHour: null,
      hardwareCostUsd: null,
      result: result({ memoryAvailable: false, memory: { value: 0, unit: 'GB', breakdown: null } }),
    })

    expect(entry.figures.weightMemoryGb).toBeNull()
    expect(entry.figures.kvCacheUsedGb).toBeNull()
    expect(entry.figures.kvCachePerRequestMb).toBeNull()
    expect(entry.figures.cloudMonthlyUsd).toBeNull()
  })

  it('records both pools of a disaggregated recommendation', () => {
    const phase = (tp: number, workers: number) => ({
      workers,
      gpusPerWorker: tp,
      tensorParallelSize: tp,
      pipelineParallelSize: 1,
      dataParallelSize: 1,
      contextParallelSize: 1,
      moeTensorParallelSize: null,
      moeExpertParallelSize: null,
      batchSize: 8,
      memoryGb: null,
    })
    const entry = buildRecommendEntry({
      ...base,
      result: result({
        mode: 'disagg',
        phases: { prefill: phase(4, 2), decode: phase(8, 4), encode: null },
      }),
    })

    expect(entry.mode).toBe('disagg')
    expect(entry.topology).toEqual({
      tp: null,
      pp: null,
      replicas: null,
      prefill: { tp: 4, pp: 1, workers: 2 },
      decode: { tp: 8, pp: 1, workers: 4 },
    })
  })

  it('says a precision was not reported rather than assuming a default', () => {
    const entry = buildRecommendEntry({
      ...base,
      result: result({ precision: { weights: null, kvCache: null } }),
    })

    expect(entry.workload.weightPrecision).toBe('auto')
    expect(entry.workload.kvPrecision).toBe('auto')
  })

  it('labels the spellings an engine uses for the same precision', () => {
    const entry = buildRecommendEntry({
      ...base,
      result: result({ precision: { weights: 'half', kvCache: 'float8_e4m3' } }),
    })

    expect(entry.workload.weightPrecision).toBe('FP16')
    expect(entry.workload.kvPrecision).toBe('FP8')
  })

  it('keeps an unrecognized quantization mode verbatim', () => {
    const entry = buildRecommendEntry({
      ...base,
      result: result({ precision: { weights: 'custom-kernel', kvCache: null } }),
    })

    expect(entry.workload.weightPrecision).toBe('custom-kernel')
  })
})

describe('sizingGoalFrom', () => {
  it('records a throughput-demand target', () => {
    expect(sizingGoalFrom({
      sla_mode: 'none',
      objective: 'throughput_demand',
      target_request_rate: 12.5,
    })).toEqual({
      kind: 'throughput',
      targetRequestLatencyMs: null,
      targetTtftMs: null,
      targetTpotMs: null,
      targetRequestRate: 12.5,
      targetConcurrency: null,
    })
  })

  it('records a concurrency-capacity target', () => {
    expect(sizingGoalFrom({
      sla_mode: 'none',
      objective: 'concurrency_capacity',
      target_concurrency: 64,
    })).toMatchObject({ kind: 'concurrency', targetConcurrency: 64 })
  })

  it('prefers an end-to-end bound over the decomposed latency pair', () => {
    expect(sizingGoalFrom({
      sla_mode: 'enforce',
      request_latency: 3000,
      ttft: 300,
      tpot: 30,
    })).toMatchObject({
      kind: 'latency',
      targetRequestLatencyMs: 3000,
      targetTtftMs: null,
      targetTpotMs: null,
    })
  })

  it('derives the objective from sla_mode when the request omitted it', () => {
    expect(sizingGoalFrom({ sla_mode: 'none', target_request_rate: 8 }))
      .toMatchObject({ kind: 'throughput', targetRequestRate: 8 })
  })
})

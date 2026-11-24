import { describe, expect, it } from 'vitest'

import {
  COMPARISON_METRICS,
  COMPARISON_SECTIONS,
  backendLabel,
  formatMetric,
  formatNumber,
  formatUsd,
  metricsFor,
  modeLabel,
  precisionLabel,
  simulatorLabel,
  sizingGoalLabel,
  topologyLabel,
  workloadLabel,
} from './metrics'
import type { ComparisonEntry } from './types'

function entry(overrides: Partial<ComparisonEntry> = {}): ComparisonEntry {
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
    workload: {
      isl: 2048,
      osl: 128,
      concurrency: 32,
      weightPrecision: 'FP8',
      kvPrecision: 'FP16',
    },
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
  }
}

describe('comparison labels', () => {
  it('describes an aggregated topology', () => {
    expect(topologyLabel(entry())).toBe('tp8 · 1 replica')
  })

  it('describes both pools of a disaggregated run', () => {
    const disagg = entry({
      mode: 'disagg',
      topology: {
        tp: null,
        pp: null,
        replicas: null,
        prefill: { tp: 4, pp: 2, workers: 2 },
        decode: { tp: 8, pp: 1, workers: 4 },
      },
    })

    expect(topologyLabel(disagg)).toBe('2× tp4·pp2 prefill + 4× tp8 decode')
  })

  it('says when a topology was not reported', () => {
    const unknown = entry({
      topology: { tp: null, pp: null, replicas: null, prefill: null, decode: null },
    })

    expect(topologyLabel(unknown)).toBe('not reported')
  })

  it('names the workload and precision', () => {
    expect(workloadLabel(entry())).toBe('ISL 2048 · OSL 128 · 32 users')
    expect(precisionLabel(entry())).toBe('FP8 weights · FP16 KV')
  })

  it('describes the sizing goal, or shows none for a predicted result', () => {
    const none = {
      targetRequestLatencyMs: null,
      targetTtftMs: null,
      targetTpotMs: null,
      targetRequestRate: null,
      targetConcurrency: null,
    }

    expect(sizingGoalLabel(entry())).toBe('—')
    expect(sizingGoalLabel(entry({
      goal: { ...none, kind: 'latency', targetTtftMs: 300, targetTpotMs: 30 },
    }))).toBe(`TTFT ≤ ${formatNumber(300, 0)} ms · TPOT ≤ ${formatNumber(30, 0)} ms`)
    expect(sizingGoalLabel(entry({
      goal: { ...none, kind: 'latency', targetRequestLatencyMs: 3000 },
    }))).toBe(`Request latency ≤ ${formatNumber(3000, 0)} ms`)
    expect(sizingGoalLabel(entry({
      goal: { ...none, kind: 'throughput', targetRequestRate: 12.5 },
    }))).toBe(`${formatNumber(12.5, 1)} req/s`)
    expect(sizingGoalLabel(entry({
      goal: { ...none, kind: 'concurrency', targetConcurrency: 64 },
    }))).toBe(`${formatNumber(64, 0)} concurrent sequences`)
  })

  it('keeps a fractional latency target visible instead of rounding it to zero', () => {
    const none = {
      targetRequestLatencyMs: null,
      targetTtftMs: null,
      targetTpotMs: null,
      targetRequestRate: null,
      targetConcurrency: null,
    }

    expect(sizingGoalLabel(entry({
      goal: { ...none, kind: 'latency', targetTtftMs: 0.1, targetTpotMs: 0.4 },
    }))).toBe('TTFT ≤ 0.1 ms · TPOT ≤ 0.4 ms')
    expect(sizingGoalLabel(entry({
      goal: { ...none, kind: 'latency', targetRequestLatencyMs: 1.5 },
    }))).toBe('Request latency ≤ 1.5 ms')
  })

  it('names the backend, or says it was not reported', () => {
    expect(backendLabel(entry())).toBe('vllm 0.9.0')
    expect(backendLabel(entry({ backend: 'vllm', backendVersion: null }))).toBe('vllm')
    expect(backendLabel(entry({ backend: null, backendVersion: null }))).toBe('not reported')
  })

  it('shortens the simulator label it was given', () => {
    expect(modeLabel(entry())).toBe('aggregated')
    expect(modeLabel(entry({ mode: 'disagg' }))).toBe('disaggregated')
    expect(simulatorLabel(entry())).toBe('AISim')
    expect(simulatorLabel(entry({ simulator: 'blis' }))).toBe('blis')
  })
})

describe('formatting', () => {
  it('formats numbers with grouped thousands', () => {
    expect(formatNumber(1600, 0)).toBe((1600).toLocaleString(undefined, {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }))
  })

  it('formats currency compactly like the result tiles', () => {
    expect(formatUsd(14_600)).toBe('$14.6K')
    expect(formatUsd(876_000)).toBe('$876.0K')
    expect(formatUsd(1_200_000)).toBe('$1.2M')
    expect(formatUsd(640)).toBe('$640')
    expect(formatUsd(-2_500)).toBe('-$2.5K')
  })

  it('renders a missing value as unavailable', () => {
    const metric = COMPARISON_METRICS.find(candidate => candidate.id === 'ttft')
    if (!metric) throw new Error('expected a TTFT metric')

    expect(formatMetric(metric, null)).toBe('—')
    expect(formatMetric(metric, 180)).toBe(formatNumber(180, 1))
  })
})

describe('metric catalog', () => {
  it('covers every section', () => {
    for (const section of COMPARISON_SECTIONS) {
      expect(metricsFor(section.id).length).toBeGreaterThan(0)
    }
  })

  it('has unique identifiers', () => {
    const ids = COMPARISON_METRICS.map(metric => metric.id)
    expect(new Set(ids).size).toBe(ids.length)
  })
})

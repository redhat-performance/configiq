import { describe, expect, it } from 'vitest'

import type { EngineSection } from '../api/engine'
import { comparisonDeployDisabledReason } from './deployment'
import type { ComparisonEntry } from './types'

function engine(): EngineSection {
  const evaluated = (value: boolean | number | string) => ({ value, source: 'evaluated' as const })
  return {
    model: 'meta-llama/Llama-3.1-70B',
    backend: 'vllm',
    backend_version: '0.31.0',
    mode: 'agg',
    agg: {
      replicas: evaluated(1),
      tensor_parallel_size: evaluated(8),
      pipeline_parallel_size: evaluated(1),
      data_parallel_size: evaluated(1),
      context_parallel_size: evaluated(1),
      decode_context_parallel_size: evaluated(1),
      moe_tensor_parallel_size: evaluated(8),
      moe_expert_parallel_size: evaluated(1),
      max_model_len: evaluated(4096),
      max_num_seqs: evaluated(32),
      max_num_batched_tokens: evaluated(8192),
      block_size: evaluated(64),
      gpu_memory_utilization: evaluated(0.9),
      enable_prefix_caching: evaluated(true),
      enable_chunked_prefill: evaluated(true),
      kv_cache_dtype: evaluated('fp8'),
      quantization: evaluated('fp8'),
    },
    prefill: null,
    decode: null,
  }
}

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
    engineVersion: '0.31.0',
    backend: 'vllm',
    backendVersion: '0.31.0',
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
  }
}

describe('comparisonDeployDisabledReason', () => {
  it('allows an entry that carries evaluated limits', () => {
    const withLimits = entry({ deployment: { engine: engine(), modelConfigData: null } })

    expect(comparisonDeployDisabledReason(withLimits)).toBeNull()
  })

  it('points at the save action for an entry written before artifacts existed', () => {
    expect(comparisonDeployDisabledReason(entry())).toContain('save it again')
  })

  it('does not promise a re-save when the run reported no limits', () => {
    const noLimits = entry({ deployment: null })

    expect(comparisonDeployDisabledReason(noLimits)).toContain('did not report the engine limits')
  })
})

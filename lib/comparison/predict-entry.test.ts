import { describe, expect, it } from 'vitest'

import type { InferenceConfigResult } from '../gpu-math/inference-config/types'
import { buildPredictEntry } from './predict-entry'

function result(overrides: Partial<InferenceConfigResult> = {}): InferenceConfigResult {
  return {
    simulator: 'aisimulate',
    engine_version: '0.14.0',
    memory_available: true,
    memory_analysis: {
      weight_gb: 140,
      weight_gb_per_gpu: 17.5,
      total_vram_gb: 640,
      usable_hbm_per_gpu: 70,
      tp_size: 8,
      replicas: 1,
      kv_cache_budget_gb: 400,
      kv_cache_used_gb: 80,
      max_sequences_from_memory: 32,
    },
    vllm_config: {
      tensor_parallel_size: 8,
      max_model_len: 4096,
      max_num_seqs: 32,
      gpu_memory_utilization: 0.9,
      max_num_batched_tokens: 8192,
      enable_chunked_prefill: true,
      enable_prefix_caching: null,
      quantization: 'fp8',
    },
    parallelism_strategy: { strategy: 'TP_ONLY', pp_size: 1, topology_note: '' },
    bottleneck_analysis: { primary: 'TPOT', risk: '', fix_suggestions: [] },
    performance: {
      ttft_ms: 180,
      tpot_ms: 22.5,
      request_latency_ms: 3060,
      throughput_tokens_per_sec: 1600,
      throughput_tokens_per_sec_per_user: 50,
      concurrency: 32,
    },
    diagnostics: { nvidia_smi_watch: '', dcgm_metrics: [], vllm_metrics: [] },
    warnings: [],
    ...overrides,
  }
}

const base = {
  link: '/predict?model=meta-llama/Llama-3.1-70B&system=h200',
  model: 'meta-llama/Llama-3.1-70B',
  system: 'h200',
  systemLabel: 'H200',
  backend: 'vllm',
  backendVersion: '0.9.0',
  workload: {
    isl: 2048,
    osl: 128,
    concurrency: 32,
    weightPrecision: 'FP8',
    kvPrecision: 'FP16',
  },
  gpus: 8,
  cloudRatePerHour: 2.5,
  hardwareCostUsd: 30_000,
  now: new Date('2026-11-21T00:00:00.000Z'),
  id: 'entry-1',
}

describe('buildPredictEntry', () => {
  it('copies the reported figures and identity into a snapshot', () => {
    const entry = buildPredictEntry({ ...base, result: result() })

    expect(entry.name).toBe('Llama-3.1-70B · H200')
    expect(entry.savedAt).toBe('2026-11-21T00:00:00.000Z')
    expect(entry.simulator).toBe('aisimulate')
    expect(entry.engineVersion).toBe('0.14.0')
    expect(entry.mode).toBe('agg')
    expect(entry.topology).toEqual({ tp: 8, pp: 1, replicas: 1, prefill: null, decode: null })
    expect(entry.figures).toMatchObject({
      gpus: 8,
      ttftMs: 180,
      tpotMs: 22.5,
      tokensPerSecond: 1600,
      tokensPerSecondPerGpu: 200,
      tokensPerSecondPerUser: 50,
      weightMemoryGb: 140,
      kvCacheUsedGb: 80,
      kvCachePerRequestMb: 2500,
      totalVramGb: 640,
      cloudMonthlyUsd: 8 * 2.5 * 730,
    })
  })

  it('reports an unreported metric as unavailable instead of zero', () => {
    const entry = buildPredictEntry({
      ...base,
      cloudRatePerHour: null,
      hardwareCostUsd: null,
      result: result({ performance: undefined, memory_available: false }),
    })

    expect(entry.figures.ttftMs).toBeNull()
    expect(entry.figures.tokensPerSecond).toBeNull()
    expect(entry.figures.tokensPerSecondPerGpu).toBeNull()
    expect(entry.figures.weightMemoryGb).toBeNull()
    expect(entry.figures.kvCachePerRequestMb).toBeNull()
    expect(entry.figures.cloudMonthlyUsd).toBeNull()
  })

  it('records both pools of a disaggregated run', () => {
    const entry = buildPredictEntry({
      ...base,
      result: result({
        mode: 'disagg',
        disagg: {
          prefill: {
            workers: 2,
            gpusPerWorker: 4,
            tp_size: 4,
            pp_size: 1,
            batch_size: 8,
            memory_gb: null,
            moe_tp_size: null,
            moe_ep_size: null,
          },
          decode: {
            workers: 4,
            gpusPerWorker: 8,
            tp_size: 8,
            pp_size: 1,
            batch_size: 16,
            memory_gb: null,
            moe_tp_size: null,
            moe_ep_size: null,
          },
        },
      }),
      gpus: 40,
    })

    expect(entry.mode).toBe('disagg')
    expect(entry.topology).toEqual({
      tp: null,
      pp: null,
      replicas: null,
      prefill: { tp: 4, pp: 1, workers: 2 },
      decode: { tp: 8, pp: 1, workers: 4 },
    })
    expect(entry.figures.tokensPerSecondPerGpu).toBe(1600 / 40)
  })

  it('keeps a missing engine version as null rather than guessing', () => {
    const entry = buildPredictEntry({ ...base, result: result({ engine_version: null }) })

    expect(entry.engineVersion).toBeNull()
  })
})

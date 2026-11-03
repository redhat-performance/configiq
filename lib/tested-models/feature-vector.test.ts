import { describe, expect, it } from 'vitest'
import { buildFeatureVector } from './feature-vector'
import type { CanonicalClassifierInput } from './types'

const input: CanonicalClassifierInput = {
  effective: {
    model_id: 'org/model',
    system_id: 'h200_sxm',
    backend: 'vllm',
    backend_version: '0.24.0',
    isl: 2048,
    osl: 512,
    max_seq_len: null,
    prefill_max_seq_len: null,
    decode_max_seq_len: null,
    concurrency: 32,
    tp_size: 4,
    pp_size: 1,
    dp_size: 1,
    cp_size: 1,
    moe_tp_size: null,
    moe_ep_size: null,
    shared_prefix_tokens: 512,
    prefix_caching_enabled: true,
    weight_precision: 'FP16',
    kv_cache_precision: 'FP16',
    moe_quant_mode: null,
    gpu_memory_utilization: null,
    max_num_seqs: null,
    chunked_prefill_enabled: null,
    serving_mode: 'agg',
    replicas: 1,
    prefill: null,
    decode: null,
    encode: null,
  },
  request: {
    request_kind: 'predict',
    target_request_rate: null,
    target_concurrency: null,
    target_latency_ms: null,
    target_ttft_ms: null,
    target_tpot_ms: null,
    database_mode: null,
    top_n: null,
  },
  model_config: null,
}

describe('buildFeatureVector', () => {
  it('uses the recorded feature order and derived values', () => {
    const vector = buildFeatureVector(input, {
      feature_names: ['model_id=org/model', 'system_id=h200_sxm', 'backend=vllm', 'version=0.24.0', 'accelerator=H200', 'tp', 'memory_pressure', 'prefix_caching', 'unseen_numeric'],
      numeric_medians: { unseen_numeric: 7 },
    })

    expect(vector).toEqual([1, 1, 1, 1, 1, 4, 20480, 1, 7])
  })

  it('matches a kernel mode to the measured quantization category', () => {
    const vector = buildFeatureVector({
      ...input,
      effective: { ...input.effective, gemm_quant_mode: 'int4_wo' },
    }, {
      feature_names: ['weight_quantization=awq', 'weight_quantization=fp8', 'weight_quantization=unknown'],
      numeric_medians: {},
    })

    expect(vector).toEqual([1, 0, 0])
  })

  it('prefers an exact category when a legacy schema lists an alias too', () => {
    const vector = buildFeatureVector({
      ...input,
      effective: { ...input.effective, kvcache_quant_mode: 'fp8' },
    }, {
      feature_names: ['kv_quantization=fp8', 'kv_quantization=fp8_e4m3', 'kv_quantization=unknown'],
      numeric_medians: {},
    })

    expect(vector).toEqual([1, 0, 0])
  })

  it('keeps the exact legacy category when the request names it explicitly', () => {
    const vector = buildFeatureVector({
      ...input,
      effective: { ...input.effective, kvcache_quant_mode: 'fp8_e4m3' },
    }, {
      feature_names: ['kv_quantization=fp8', 'kv_quantization=fp8_e4m3', 'kv_quantization=unknown'],
      numeric_medians: {},
    })

    expect(vector).toEqual([0, 1, 0])
  })

  it('prefers the canonical spelling before another alias', () => {
    const vector = buildFeatureVector({
      ...input,
      effective: { ...input.effective, gemm_quant_mode: 'int4_wo' },
    }, {
      feature_names: ['weight_quantization=awq', 'weight_quantization=int4', 'weight_quantization=fp8'],
      numeric_medians: {},
    })

    expect(vector).toEqual([0, 1, 0])
  })

  it('matches a canonical value to a legacy alias-only category', () => {
    const vector = buildFeatureVector({
      ...input,
      effective: { ...input.effective, kvcache_quant_mode: 'fp8' },
    }, {
      feature_names: ['kv_quantization=fp8_e4m3', 'kv_quantization=unknown'],
      numeric_medians: {},
    })

    expect(vector).toEqual([1, 0])
  })

  it('activates no quantization category for an unseen mode', () => {
    const vector = buildFeatureVector({
      ...input,
      effective: { ...input.effective, gemm_quant_mode: 'nvfp4' },
    }, {
      feature_names: ['weight_quantization=fp8', 'weight_quantization=unknown'],
      numeric_medians: {},
    })

    expect(vector).toEqual([0, 0])
  })
})

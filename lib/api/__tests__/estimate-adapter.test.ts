import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  buildPredictRequestBody,
  fetchEstimateAsInferenceResult,
  isMoeConfig,
  type EstimateAdapterInput,
  EstimateError,
} from '../estimate-adapter'

// ─── Fixtures ────────────────────────────────────────────────────────────────

const VALID_INPUT: EstimateAdapterInput = {
  model_path: 'meta-llama/Llama-3.1-8B-Instruct',
  system: 'h200_sxm',
  isl: 2048,
  osl: 128,
  batch_size: 32,
  tp_size: 1,
  backend: 'vllm',
  simulator: 'aisim',
  gpu_memory_utilization: 0.9,
}

const VALID_RESPONSE = {
  status: 'success',
  simulator: 'aisim',
  memory_breakdown: {
    weights_bytes: 16_000_000_000,
    kv_cache_bytes: 4_000_000_000,
  },
  ttft: 100,
  tpot: 20,
  serving_config: {
    tensor_parallel_size: 1,
    max_model_len: 2176,
    max_num_seqs: 32,
    gpu_memory_utilization: 0.9,
    enable_chunked_prefill: false,
    enable_prefix_caching: false,
    quantization: 'auto',
  },
}

function mockFetchOk(data: unknown) {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: () => Promise.resolve(data),
  })
}

// ─── Request Body Construction Tests ──────────────────────────────────────────

describe('fetchEstimateAsInferenceResult - request body construction', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('includes all required fields in the request', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult(VALID_INPUT)

    const call = mockFetch.mock.calls[0]
    const url = call[0] as string
    const body = JSON.parse(call[1].body as string)

    expect(url).toContain('/api/predict')
    expect(body).toHaveProperty('model_path', 'meta-llama/Llama-3.1-8B-Instruct')
    expect(body).toHaveProperty('system', 'h200_sxm')
    expect(body).toHaveProperty('backend', 'vllm')
    expect(body).toHaveProperty('isl', 2048)
    expect(body).toHaveProperty('osl', 128)
    expect(body).toHaveProperty('batch_size', 32)
    expect(body).toHaveProperty('tp_size', 1)
  })

  it('includes backend_version when provided', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      backend_version: '0.24.0',
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('backend_version', '0.24.0')
  })

  it('forwards the global simulator selection', async () => {
    const mockFetch = mockFetchOk({ ...VALID_RESPONSE, simulator: 'blis' })
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({ ...VALID_INPUT, simulator: 'blis' })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body.simulator).toBe('blis')
  })

  it('forwards the selected fabric and per-pool settings for BLIS P/D', async () => {
    const mockFetch = mockFetchOk({
      ...VALID_RESPONSE,
      simulator: 'blis',
      mode: 'disagg',
      memory_breakdown: null,
      memory_unavailable_reason: 'No memory breakdown.',
      prefill_config: { tp: 4, pp: 1, num_workers: 2, batch_size: 8 },
      decode_config: { tp: 2, pp: 1, num_workers: 1, batch_size: 8 },
    })
    vi.stubGlobal('fetch', mockFetch)

    const result = await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      simulator: 'blis',
      mode: 'disagg',
      fabric: 'ib-400g',
      prefill: { tp: 4, pp: 1, workers: 2, batch: 8 },
      decode: { tp: 2, pp: 1, workers: 1, batch: 8 },
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toMatchObject({
      simulator: 'blis',
      mode: 'disagg',
      fabric: 'ib-400g',
      prefill_tp_size: 4,
      prefill_num_workers: 2,
      decode_tp_size: 2,
      decode_num_workers: 1,
    })
    expect(result.mode).toBe('disagg')
    expect(result.disagg?.prefill.workers).toBe(2)
  })

  it('rejects a BLIS request when the API does not confirm BLIS provenance', async () => {
    vi.stubGlobal('fetch', mockFetchOk(VALID_RESPONSE))

    await expect(fetchEstimateAsInferenceResult({ ...VALID_INPUT, simulator: 'blis' }))
      .rejects.toMatchObject({ code: 'SIMULATOR_MISMATCH' })
  })

  it('includes context window limits when provided', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      max_seq_len: 8192,
      prefill_max_seq_len: 4096,
      decode_max_seq_len: 8192,
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toMatchObject({
      max_seq_len: 8192,
      prefill_max_seq_len: 4096,
      decode_max_seq_len: 8192,
    })
  })

  it('includes GPU memory utilization when provided', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({ ...VALID_INPUT, gpu_memory_utilization: 0.97 })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body.gpu_memory_utilization).toBe(0.97)
  })

  it('includes serving controls when provided', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      max_num_seqs: 64,
      enable_chunked_prefill: true,
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body.max_num_seqs).toBe(64)
    expect(body.enable_chunked_prefill).toBe(true)
  })

  it('keeps BLIS provenance and does not invent memory estimates', async () => {
    const mockFetch = mockFetchOk({
      simulator: 'blis',
      ttft: 80,
      tpot: 20,
      request_latency: 2640,
      tokens_per_second: 500,
      serving_config: {
        backend: 'vllm',
        tensor_parallel_size: 2,
        max_model_len: 2176,
        max_num_seqs: 32,
        max_num_batched_tokens: 8192,
        gpu_memory_utilization: 0.92,
        enable_chunked_prefill: null,
        enable_prefix_caching: true,
        quantization: 'model default',
      },
      memory_breakdown: null,
      memory_unavailable_reason: 'The selected simulator does not provide a KV-cache memory breakdown.',
      provenance: {
        simulator: 'blis',
        requested_model: 'meta-llama/Llama-3.1-8B-Instruct',
        resolved_model: 'llama-3.1-8b-instruct',
        model_source: 'upstream',
        system: 'h200_sxm',
        hardware: 'h200',
        model_config_relation: 'different_quantization',
        engine_version: '0.31.0',
      },
      warnings: ['The selected simulator uses a model configuration marked different_quantization.'],
    })
    vi.stubGlobal('fetch', mockFetch)

    const result = await fetchEstimateAsInferenceResult({ ...VALID_INPUT, simulator: 'blis' })

    expect(result.simulator).toBe('blis')
    expect(result.memory_available).toBe(false)
    expect(result.memory_analysis.weight_gb).toBe(0)
    expect(result.performance?.throughput_tokens_per_sec).toBe(500)
    expect(result.warnings).toContain('The selected simulator does not provide a KV-cache memory breakdown.')
    expect(result.warnings).toContain('The selected simulator uses a model configuration marked different_quantization.')
  })

  it('keeps aggregate and per-user throughput distinct', async () => {
    vi.stubGlobal('fetch', mockFetchOk({
      ...VALID_RESPONSE,
      tokens_per_second: 549.7077676117464,
      tokens_per_second_per_user: 153.80785873894342,
    }))

    const result = await fetchEstimateAsInferenceResult({ ...VALID_INPUT, batch_size: 4 })

    expect(result.performance?.throughput_tokens_per_sec).toBeCloseTo(549.7078)
    expect(result.performance?.throughput_tokens_per_sec_per_user).toBeCloseTo(153.8079)
  })

  it('omits backend_version when not provided', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    const input: EstimateAdapterInput = {
      ...VALID_INPUT,
      backend_version: undefined,
    }
    await fetchEstimateAsInferenceResult(input)

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).not.toHaveProperty('backend_version')
  })

  it('includes prefix when > 0', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      prefix: 512,
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('prefix', 512)
  })

  it('omits prefix when 0 or undefined', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      prefix: 0,
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).not.toHaveProperty('prefix')
  })

  it('includes pp_size when > 1', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      pp_size: 2,
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('pp_size', 2)
  })

  it('omits pp_size when undefined or <= 1', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      pp_size: undefined,
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).not.toHaveProperty('pp_size')
  })

  it('maps FP8 weight precision to gemm_quant_mode: fp8', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      gemm_quant_mode: 'fp8',
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('gemm_quant_mode', 'fp8')
  })

  it('maps INT8 weight precision to gemm_quant_mode: int8_wo', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      gemm_quant_mode: 'int8_wo',
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('gemm_quant_mode', 'int8_wo')
  })

  it('maps INT4 weight precision to gemm_quant_mode: int4_wo', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      gemm_quant_mode: 'int4_wo',
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('gemm_quant_mode', 'int4_wo')
  })

  it('maps MXFP4 weight precision to gemm_quant_mode: mxfp4', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      gemm_quant_mode: 'mxfp4',
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('gemm_quant_mode', 'mxfp4')
  })

  it('maps NVFP4 weight precision to gemm_quant_mode: nvfp4', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      gemm_quant_mode: 'nvfp4',
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('gemm_quant_mode', 'nvfp4')
  })

  it('omits gemm_quant_mode when not provided', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      gemm_quant_mode: undefined,
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).not.toHaveProperty('gemm_quant_mode')
  })

  it('maps FP8 KV cache precision to kvcache_quant_mode: fp8', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      kvcache_quant_mode: 'fp8',
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('kvcache_quant_mode', 'fp8')
  })

  it('maps NVFP4 KV cache precision to kvcache_quant_mode: nvfp4', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      kvcache_quant_mode: 'nvfp4',
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('kvcache_quant_mode', 'nvfp4')
  })

  it('omits kvcache_quant_mode when not provided', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      kvcache_quant_mode: undefined,
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).not.toHaveProperty('kvcache_quant_mode')
  })

  it('includes moe_quant_mode when provided', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      moe_quant_mode: 'w4a16_mxfp4',
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('moe_quant_mode', 'w4a16_mxfp4')
  })

  it('omits moe_quant_mode when not provided', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      moe_quant_mode: undefined,
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).not.toHaveProperty('moe_quant_mode')
  })

  it('omits model_config when hf_model_config is not provided', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      hf_model_config: undefined,
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).not.toHaveProperty('model_config')
  })

  it('includes model_config when hf_model_config is provided', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    const hfConfig = { num_hidden_layers: 32, hidden_size: 4096 }
    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      hf_model_config: hfConfig,
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('model_config', hfConfig)
  })

  it('defaults backend to vllm when not provided', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    const input: EstimateAdapterInput = {
      ...VALID_INPUT,
      backend: undefined,
    }
    await fetchEstimateAsInferenceResult(input)

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('backend', 'vllm')
  })

  it('handles MoE models with moe_ep_size', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      moe_ep_size: 8,
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('moe_ep_size', 8)
  })

  it('handles MoE models with moe_tp_size when moe_ep_size not provided', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      moe_tp_size: 4,
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('moe_tp_size', 4)
  })

  it('does not invent MoE widths when none are given', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({ ...VALID_INPUT, tp_size: 4 })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).not.toHaveProperty('moe_ep_size')
    expect(body).not.toHaveProperty('moe_tp_size')
  })

  it('forwards explicit MoE widths unchanged, regardless of model family', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      tp_size: 4,
      moe_ep_size: 4,
      moe_tp_size: 1,
      hf_model_config: {
        model_type: 'gemma4',
        text_config: { num_experts: null, enable_moe_block: false },
      },
    })

    const body = JSON.parse(mockFetch.mock.calls[0][1].body as string)
    expect(body).toHaveProperty('moe_ep_size', 4)
    expect(body).toHaveProperty('moe_tp_size', 1)
  })

  it('does not send per-pool MoE widths for a disaggregated request', () => {
    const body = buildPredictRequestBody({
      ...VALID_INPUT,
      mode: 'disagg',
      prefill: { tp: 8, pp: 1, workers: 1, batch: 8 },
      decode: { tp: 2, pp: 1, workers: 1, batch: 8 },
    })

    expect(body).not.toHaveProperty('prefill_moe_tp_size')
    expect(body).not.toHaveProperty('prefill_moe_ep_size')
    expect(body).not.toHaveProperty('decode_moe_tp_size')
    expect(body).not.toHaveProperty('decode_moe_ep_size')
  })

  it('does not let a low expert count mask a higher one', () => {
    expect(isMoeConfig({ num_experts: 1, text_config: { num_experts: 128 } })).toBe(true)
    expect(isMoeConfig({ num_experts: 1, num_local_experts: 64 })).toBe(true)
    expect(isMoeConfig({ num_experts: 1, text_config: { num_local_experts: 8 } })).toBe(true)
    expect(isMoeConfig({ num_experts: 1, text_config: { num_experts: 1 } })).toBe(false)
  })

  it('builds the same body that is submitted', () => {
    const input: EstimateAdapterInput = {
      ...VALID_INPUT,
      tp_size: 4,
      hf_model_config: { text_config: { num_experts: 128 } },
    }
    const body = buildPredictRequestBody(input)
    expect(body).not.toHaveProperty('moe_ep_size')
    expect(body).not.toHaveProperty('moe_tp_size')
    expect(body).toHaveProperty('model_config', input.hf_model_config)
    expect(body).not.toHaveProperty('hf_model_config')
  })

  it('omits an unset requests-per-load-unit and forwards an explicit one', () => {
    expect(buildPredictRequestBody(VALID_INPUT)).not.toHaveProperty('requests_per_load_unit')
    expect(
      buildPredictRequestBody({ ...VALID_INPUT, requests_per_load_unit: 4 }),
    ).toHaveProperty('requests_per_load_unit', 4)
  })

  it('asks for the serving config, memory, and engine limits', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult(VALID_INPUT)

    const url = mockFetch.mock.calls[0][0] as string
    expect(url).toContain('include=config,memory,engine')
  })
})

// ─── Response Handling Tests ──────────────────────────────────────────────────

describe('fetchEstimateAsInferenceResult - response handling', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('throws EstimateError with correct code on API error', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: () => Promise.resolve({
        status: 'failed',
        error: { code: 'INVALID_REQUEST', message: 'Bad request' },
      }),
    })
    vi.stubGlobal('fetch', mockFetch)

    await expect(fetchEstimateAsInferenceResult(VALID_INPUT)).rejects.toThrow(
      expect.objectContaining({
        code: 'INVALID_REQUEST',
      })
    )
  })

  it('throws error on non-JSON response', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.reject(new Error('Invalid JSON')),
    })
    vi.stubGlobal('fetch', mockFetch)

    await expect(fetchEstimateAsInferenceResult(VALID_INPUT)).rejects.toThrow(
      'Invalid JSON'
    )
  })

  it('returns a valid InferenceConfigResult on success', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    const result = await fetchEstimateAsInferenceResult(VALID_INPUT)

    expect(result).toHaveProperty('memory_analysis')
    expect(result).toHaveProperty('vllm_config')
    expect(result).toHaveProperty('parallelism_strategy')
    expect(result).toHaveProperty('bottleneck_analysis')
    expect(result).toHaveProperty('diagnostics')
  })

  it('reports the resolved MoE widths from an aggregated estimate', async () => {
    const mockFetch = mockFetchOk({ ...VALID_RESPONSE, moe_tp: 4, moe_ep: 1 })
    vi.stubGlobal('fetch', mockFetch)

    const result = await fetchEstimateAsInferenceResult({ ...VALID_INPUT, tp_size: 4 })

    expect(result.mode).toBe('agg')
    expect(result.moe_tp_size).toBe(4)
    expect(result.moe_ep_size).toBe(1)
  })

  it('reports each pool resolved MoE widths from a disaggregated estimate', async () => {
    const mockFetch = mockFetchOk({
      ...VALID_RESPONSE,
      mode: 'disagg',
      prefill_config: { tp: 8, pp: 1, num_workers: 1, batch_size: 8, moe_tp: 8, moe_ep: 1 },
      decode_config: { tp: 2, pp: 1, num_workers: 1, batch_size: 8, moe_tp: 2, moe_ep: 1 },
    })
    vi.stubGlobal('fetch', mockFetch)

    const result = await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      mode: 'disagg',
      prefill: { tp: 8, pp: 1, workers: 1, batch: 8 },
      decode: { tp: 2, pp: 1, workers: 1, batch: 8 },
    })

    expect(result.mode).toBe('disagg')
    expect(result.moe_tp_size).toBeNull()
    expect(result.disagg?.prefill).toMatchObject({ tp_size: 8, moe_tp_size: 8, moe_ep_size: 1 })
    expect(result.disagg?.decode).toMatchObject({ tp_size: 2, moe_tp_size: 2, moe_ep_size: 1 })
  })

  it('includes weight and KV cache memory in response', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    const result = await fetchEstimateAsInferenceResult(VALID_INPUT)

    expect(result.memory_analysis).toHaveProperty('weight_gb')
    expect(result.memory_analysis).toHaveProperty('kv_cache_used_gb')
  })

  it('computes performance metrics (TTFT, TPOT, request latency, concurrency, throughput)', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    const result = await fetchEstimateAsInferenceResult(VALID_INPUT)

    expect(result.performance).toBeDefined()
    expect(result.performance?.ttft_ms).toBe(100)
    expect(result.performance?.tpot_ms).toBe(20)
    // request_latency_ms = ttft + tpot * osl = 100 + 20 * 128 = 2660
    expect(result.performance?.request_latency_ms).toBe(2660)
    // No aggregate rate was reported, so the fallback is one request's rate.
    // The request omitted `requests_per_load_unit`, which makes `/predict`
    // simulate a single request regardless of the batch width.
    expect(result.performance?.throughput_tokens_per_sec).toBeCloseTo(48.12, 1)
    expect(result.warnings.some(warning => warning.includes('Throughput was not reported'))).toBe(true)
    expect(result.warnings.some(warning => warning.includes('at 1 concurrent request across 1 wave'))).toBe(true)
    // concurrency is the submitted batch the prediction is compared at
    expect(result.performance?.concurrency).toBe(32)
  })

  it('derives throughput from the requests the simulator ran, not the batch width', async () => {
    const mockFetch = mockFetchOk({
      ...VALID_RESPONSE,
      serving_config: { ...VALID_RESPONSE.serving_config, max_num_seqs: 64 },
    })
    vi.stubGlobal('fetch', mockFetch)

    const result = await fetchEstimateAsInferenceResult({ ...VALID_INPUT, batch_size: 16 })

    // Omitting the request count still simulates one request, so the serving
    // cap and the submitted batch do not inflate the derived rate.
    expect(result.performance?.throughput_tokens_per_sec).toBeCloseTo(48.12, 1)
    expect(result.performance?.concurrency).toBe(16)
    expect(result.warnings.some(warning => warning.includes('at 1 concurrent request across 1 wave'))).toBe(true)
  })

  it('scales a derived throughput by the concurrent requests, not the total', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    const result = await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      batch_size: 16,
      requests_per_load_unit: 4,
    })

    // max(1, round(16 × 4)) = 64 requests run as four waves of 16, so the
    // aggregate rate is 16 requests in flight, not all 64 requests.
    expect(result.performance?.throughput_tokens_per_sec).toBeCloseTo(769.9, 1)
    expect(result.warnings.some(warning => warning.includes('at 16 concurrent requests across 4 waves'))).toBe(true)
  })

  it('treats an omitted TTFT or TPOT as missing timing', async () => {
    const withoutTtft: Record<string, unknown> = { ...VALID_RESPONSE }
    delete withoutTtft.ttft
    const mockFetch = mockFetchOk(withoutTtft)
    vi.stubGlobal('fetch', mockFetch)

    const result = await fetchEstimateAsInferenceResult(VALID_INPUT)

    expect(result.performance?.throughput_tokens_per_sec).toBe(0)
    expect(result.warnings.some(warning => warning.includes('performance prediction unavailable'))).toBe(true)
  })

  it('keeps aggregate and per-user throughput distinct', async () => {
    const mockFetch = mockFetchOk({
      ...VALID_RESPONSE,
      tokens_per_second: 1000,
      tokens_per_second_per_user: 31.25,
    })
    vi.stubGlobal('fetch', mockFetch)

    const result = await fetchEstimateAsInferenceResult(VALID_INPUT)

    expect(result.performance?.throughput_tokens_per_sec).toBe(1000)
    expect(result.performance?.throughput_tokens_per_sec_per_user).toBe(31.25)
    expect(result.warnings.some(warning => warning.includes('Throughput was not reported'))).toBe(false)
  })

  it('leaves per-user throughput null when the simulator omits it', async () => {
    const mockFetch = mockFetchOk({ ...VALID_RESPONSE, tokens_per_second: 1000 })
    vi.stubGlobal('fetch', mockFetch)

    const result = await fetchEstimateAsInferenceResult(VALID_INPUT)

    expect(result.performance?.throughput_tokens_per_sec).toBe(1000)
    expect(result.performance?.throughput_tokens_per_sec_per_user).toBeNull()
  })

  it('handles zero output sequence length gracefully', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    const result = await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      osl: 0,
    })

    expect(result.performance).toBeDefined()
    expect(result.performance?.request_latency_ms).toBe(100) // ttft + 0 * tpot
    expect(result.performance?.throughput_tokens_per_sec).toBe(0) // Can't divide by zero
  })

  it('does not split a headline memory figure into weights it cannot see', async () => {
    vi.stubGlobal('fetch', mockFetchOk({
      ...VALID_RESPONSE,
      memory_breakdown: null,
      memory: 72,
    }))

    const result = await fetchEstimateAsInferenceResult(VALID_INPUT)

    // A total with no breakdown says nothing about weights, so it is reported
    // as unavailable rather than half of the total.
    expect(result.memory_available).toBe(true)
    expect(result.memory_analysis.weight_gb).toBe(0)
    expect(result.memory_analysis.kv_cache_used_gb).toBe(0)
    expect(result.memory_analysis.kv_cache_budget_gb).toBeNull()
  })

  it('reports the KV size the simulator stated as the budget, not one derived from device memory', async () => {
    vi.stubGlobal('fetch', mockFetchOk(VALID_RESPONSE))

    const result = await fetchEstimateAsInferenceResult({ ...VALID_INPUT, vram_gb: 141 })

    expect(result.memory_analysis.kv_cache_used_gb).toBeCloseTo(4_000_000_000 / 1_073_741_824)
    expect(result.memory_analysis.kv_cache_budget_gb).toBeCloseTo(4_000_000_000 / 1_073_741_824)
  })

  it('takes the usable figure from the budget the simulator sized the pool against', async () => {
    vi.stubGlobal('fetch', mockFetchOk({
      ...VALID_RESPONSE,
      simulator: 'blis',
      serving_config: {
        tensor_parallel_size: 2,
        max_num_seqs: 32,
        gpu_memory_utilization: null,
      },
      memory_pools: [{
        role: 'colocated',
        budget_bytes: 80 * 1_073_741_824,
      }],
    }))

    const result = await fetchEstimateAsInferenceResult({
      ...VALID_INPUT,
      simulator: 'blis',
      gpu_memory_utilization: undefined,
      tp_size: 2,
      vram_gb: 141,
    })

    // The pool's own budget needs no fraction or device size from the client,
    // and it is what the pool's memory was sized against.
    expect(result.memory_analysis.usable_hbm_per_gpu).toBeCloseTo(80)
    // The simulator reports these per GPU; dividing again would understate them.
    expect(result.memory_analysis.weight_gb_per_gpu).toBeCloseTo(16_000_000_000 / 1_073_741_824)
  })

  it('leaves a setting the run did not state null instead of inventing a default', async () => {
    vi.stubGlobal('fetch', mockFetchOk({
      ...VALID_RESPONSE,
      serving_config: {
        tensor_parallel_size: 1,
        max_model_len: 2176,
        max_num_seqs: 32,
        gpu_memory_utilization: 0.9,
      },
    }))

    const result = await fetchEstimateAsInferenceResult(VALID_INPUT)

    expect(result.vllm_config.max_num_batched_tokens).toBeNull()
    expect(result.vllm_config.quantization).toBeNull()
  })

  it('returns debug info when captureDebug is true', async () => {    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    const result = await fetchEstimateAsInferenceResult(VALID_INPUT, true)

    expect(result).toHaveProperty('result')
    expect(result).toHaveProperty('debugRequest')
    expect(result).toHaveProperty('debugResponse')
    expect(result).toHaveProperty('debugStatus')
    expect(result).toHaveProperty('debugDuration')
    expect((result as any).debugStatus).toBe(200)
    expect((result as any).debugRequest).toHaveProperty('model_path')
    expect((result as any).debugResponse).toEqual(VALID_RESPONSE)
  })
})

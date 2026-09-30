import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { fetchEstimateAsInferenceResult, type EstimateAdapterInput, EstimateError } from '../estimate-adapter'

// ─── Fixtures ────────────────────────────────────────────────────────────────

const VALID_INPUT: EstimateAdapterInput = {
  model_path: 'meta-llama/Llama-3.1-8B-Instruct',
  system: 'h200_sxm',
  isl: 2048,
  osl: 128,
  batch_size: 32,
  tp_size: 1,
  backend: 'vllm',
  gpu_memory_utilization: 0.9,
}

const VALID_RESPONSE = {
  status: 'success',
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

  it('includes include=config,memory in query params', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await fetchEstimateAsInferenceResult(VALID_INPUT)

    const url = mockFetch.mock.calls[0][0] as string
    expect(url).toContain('include=config,memory')
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
    // throughput_tokens_per_sec = (osl * 1000) / request_latency_ms = (128 * 1000) / 2660 ≈ 48.1
    expect(result.performance?.throughput_tokens_per_sec).toBeGreaterThan(0)
    // concurrency = floor(max_num_seqs) = floor(32) from serving_config
    expect(result.performance?.concurrency).toBe(32)
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

  it('returns debug info when captureDebug is true', async () => {
    const mockFetch = mockFetchOk(VALID_RESPONSE)
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

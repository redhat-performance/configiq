import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { KvCacheCalcRequestSchema } from '../schemas'
import { callKvCacheCalc, generateKvRequestId } from '../kv-cache-calc'
import type { KvCacheCalcResult, KvCacheCalcErrorResponse } from '../kv-cache-calc'

// ─── Fixtures ────────────────────────────────────────────────────────────────

const VALID_REQUEST = {
  model_path: 'meta-llama/Llama-3.1-70B-Instruct',
  system: 'h200_sxm',
  backend: 'vllm',
  max_num_tokens: 8192,
  max_batch_size: 128,
  tp_size: 1,
  pp_size: 1,
  memory_fraction_kind: 'of_total' as const,
  memory_fraction_value: 1.0,
}

const EXTERNAL_RESPONSE = {
  total_gpu_capacity_bytes: 85899345920,
  total_kv_size_bytes: 65343062016,
  kv_size_per_token_bytes: 131072,
  total_kv_size_tokens: 498528,
  source: 'native',
  memory_breakdown: {
    weights_bytes: 16059990016,
    activations_bytes: 738197504,
    runtime_overhead_bytes: 3758096384,
    comm_overhead_bytes: 0,
  },
}

function mockFetchOk(data: unknown) {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: () => Promise.resolve(data),
  })
}

// ─── Schema Tests ────────────────────────────────────────────────────────────

describe('KvCacheCalcRequestSchema', () => {
  it('accepts a valid request', () => {
    const result = KvCacheCalcRequestSchema.safeParse(VALID_REQUEST)
    expect(result.success).toBe(true)
  })

  it('rejects missing model_path', () => {
    const { model_path, ...rest } = VALID_REQUEST
    const result = KvCacheCalcRequestSchema.safeParse(rest)
    expect(result.success).toBe(false)
  })

  it('rejects missing system', () => {
    const { system, ...rest } = VALID_REQUEST
    const result = KvCacheCalcRequestSchema.safeParse(rest)
    expect(result.success).toBe(false)
  })

  it('rejects memory_fraction_value above 1', () => {
    const result = KvCacheCalcRequestSchema.safeParse({ ...VALID_REQUEST, memory_fraction_value: 1.5 })
    expect(result.success).toBe(false)
  })

  it('rejects unknown fields (strict mode)', () => {
    const result = KvCacheCalcRequestSchema.safeParse({ ...VALID_REQUEST, extra: 'nope' })
    expect(result.success).toBe(false)
  })

  it('accepts a model_config object', () => {
    const result = KvCacheCalcRequestSchema.safeParse({
      ...VALID_REQUEST,
      model_config: { hidden_size: 8192, architectures: ['LlamaForCausalLM'] },
    })
    expect(result.success).toBe(true)
  })

  it('applies defaults for optional fields', () => {
    const result = KvCacheCalcRequestSchema.safeParse({
      model_path: 'test/model',
      system: 'h200_sxm',
    })
    expect(result.success).toBe(true)
    if (result.success) {
      expect(result.data.backend).toBe('vllm')
      expect(result.data.max_num_tokens).toBe(8192)
      expect(result.data.max_batch_size).toBe(128)
      expect(result.data.tp_size).toBe(1)
      expect(result.data.pp_size).toBe(1)
      expect(result.data.memory_fraction_kind).toBe('of_total')
      expect(result.data.memory_fraction_value).toBe(1.0)
    }
  })
})

// ─── Service Tests ───────────────────────────────────────────────────────────

describe('callKvCacheCalc', () => {
  beforeEach(() => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://aisimulators.dev')
  })

  afterEach(() => {
    vi.unstubAllEnvs()
    vi.restoreAllMocks()
  })

  it('returns a normalized response on success', async () => {
    vi.stubGlobal('fetch', mockFetchOk(EXTERNAL_RESPONSE))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('completed')
    const r = result as KvCacheCalcResult
    expect(r.requestId).toMatch(/^kv_/)
    expect(r.kvCache.totalBytes).toBe(65343062016)
    expect(r.kvCache.perTokenBytes).toBe(131072)
    expect(r.kvCache.totalTokens).toBe(498528)
    expect(r.memoryBreakdown.weightsBytes).toBe(16059990016)
    expect(r.memoryBreakdown.activationsBytes).toBe(738197504)
    expect(r.memoryBreakdown.runtimeOverheadBytes).toBe(3758096384)
    expect(r.memoryBreakdown.commOverheadBytes).toBe(0)
    expect(r.gpuCapacity.totalBytes).toBe(85899345920)
    expect(r.metadata.modelPath).toBe('meta-llama/Llama-3.1-70B-Instruct')
    expect(r.metadata.backend).toBe('vllm')
    expect(r.metadata.source).toBe('native')
    expect(r.metadata.durationMs).toBeGreaterThanOrEqual(0)
  })

  it('calls /memory at the configured API URL', async () => {
    const mockFetch = mockFetchOk(EXTERNAL_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await callKvCacheCalc(VALID_REQUEST)

    expect(mockFetch.mock.calls[0][0]).toBe('https://aisimulators.dev/memory')
  })

  it('sends correct fields without credentials', async () => {
    const mockFetch = mockFetchOk(EXTERNAL_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await callKvCacheCalc(VALID_REQUEST)

    const sentBody = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(sentBody.model_path).toBe('meta-llama/Llama-3.1-70B-Instruct')
    expect(sentBody.backend).toBe('vllm')
    expect(sentBody.tp_size).toBe(1)
    expect(sentBody).not.toHaveProperty('username')
    expect(sentBody).not.toHaveProperty('password')
  })

  it('sends nullable fields only when set', async () => {
    const mockFetch = mockFetchOk(EXTERNAL_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await callKvCacheCalc({ ...VALID_REQUEST, moe_tp_size: 8, backend_version: '0.24.0' })

    const sentBody = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(sentBody.moe_tp_size).toBe(8)
    expect(sentBody.backend_version).toBe('0.24.0')
  })

  it('omits nullable fields when not set', async () => {
    const mockFetch = mockFetchOk(EXTERNAL_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await callKvCacheCalc(VALID_REQUEST)

    const sentBody = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(sentBody).not.toHaveProperty('moe_tp_size')
    expect(sentBody).not.toHaveProperty('moe_ep_size')
    expect(sentBody).not.toHaveProperty('backend_version')
    expect(sentBody).not.toHaveProperty('model_config')
  })

  it('forwards model_config to the upstream request when present', async () => {
    const mockFetch = mockFetchOk(EXTERNAL_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    const model_config = { hidden_size: 8192, architectures: ['LlamaForCausalLM'] }
    await callKvCacheCalc({ ...VALID_REQUEST, model_config })

    const sentBody = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(sentBody.model_config).toEqual(model_config)
  })

  it('returns AISIM_NOT_CONFIGURED when API URL is missing', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', '')

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.code).toBe('AISIM_NOT_CONFIGURED')
  })

  it('returns AISIM_TIMEOUT on fetch timeout', async () => {
    const timeoutError = new Error('signal timed out')
    timeoutError.name = 'TimeoutError'
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(timeoutError))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.code).toBe('AISIM_TIMEOUT')
  })

  it('returns AISIM_UNAVAILABLE on network error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('fetch failed')))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.code).toBe('AISIM_UNAVAILABLE')
  })

  it('returns AISIM_UNSUPPORTED on 422', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: () => Promise.resolve({ detail: 'unsupported model' }),
    }))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.code).toBe('AISIM_UNSUPPORTED')
    expect((result as KvCacheCalcErrorResponse).error.message).toBe('unsupported model')
  })

  it('returns AISIM_UNAVAILABLE on 500', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: () => Promise.resolve({ detail: 'internal error' }),
    }))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.code).toBe('AISIM_UNAVAILABLE')
  })

  it('returns AISIM_INVALID_RESPONSE on non-JSON response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.reject(new Error('invalid json')),
    }))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.code).toBe('AISIM_INVALID_RESPONSE')
  })

  it('handles missing optional response fields gracefully', async () => {
    vi.stubGlobal('fetch', mockFetchOk({ total_kv_size_bytes: 1024 }))

    const result = await callKvCacheCalc(VALID_REQUEST) as KvCacheCalcResult

    expect(result.status).toBe('completed')
    expect(result.kvCache.totalBytes).toBe(1024)
    expect(result.kvCache.perTokenBytes).toBe(0)
    expect(result.kvCache.totalTokens).toBe(0)
    expect(result.memoryBreakdown.weightsBytes).toBe(0)
    expect(result.gpuCapacity.totalBytes).toBe(0)
    expect(result.metadata.source).toBe('unknown')
  })

  it('includes durationMs in metadata', async () => {
    vi.stubGlobal('fetch', mockFetchOk(EXTERNAL_RESPONSE))

    const result = await callKvCacheCalc(VALID_REQUEST) as KvCacheCalcResult

    expect(result.metadata.durationMs).toBeTypeOf('number')
    expect(result.metadata.durationMs).toBeGreaterThanOrEqual(0)
  })
})

// ─── Request ID Tests ────────────────────────────────────────────────────────

describe('generateKvRequestId', () => {
  it('starts with kv_ prefix', () => {
    expect(generateKvRequestId()).toMatch(/^kv_/)
  })

  it('is 15 characters long', () => {
    expect(generateKvRequestId()).toHaveLength(15)
  })

  it('generates unique IDs', () => {
    const ids = new Set(Array.from({ length: 100 }, generateKvRequestId))
    expect(ids.size).toBe(100)
  })
})

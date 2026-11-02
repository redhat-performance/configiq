import { afterEach, describe, expect, it, vi } from 'vitest'
import { readFile } from 'node:fs/promises'
import { POST } from './route'

vi.mock('node:fs/promises', () => ({ readFile: vi.fn() }))

const mockReadFile = vi.mocked(readFile)

const input = {
  effective: {
    model_id: 'org/model', system_id: 'h200_sxm', backend: 'vllm', backend_version: '0.24.0',
    isl: 2048, osl: 512, max_seq_len: null, prefill_max_seq_len: null, decode_max_seq_len: null,
    concurrency: 32, tp_size: 4, pp_size: 1, dp_size: 1, cp_size: 1, moe_tp_size: null, moe_ep_size: null,
    shared_prefix_tokens: 0, prefix_caching_enabled: false, weight_precision: 'FP16', kv_cache_precision: 'FP16',
    moe_quant_mode: null, gpu_memory_utilization: null, max_num_seqs: null, chunked_prefill_enabled: null,
    serving_mode: 'agg', replicas: 1, prefill: null, decode: null, encode: null,
  },
  request: { request_kind: 'predict', target_request_rate: null, target_concurrency: null, target_latency_ms: null, target_ttft_ms: null, target_tpot_ms: null, database_mode: null, top_n: null },
  model_config: null,
}

const gate = { minRecords: 100, minTestAuc: 0.7, minTestF1: 0.3, minConcurrencyLevels: 5 }
const coverage = { records: 600, concurrencyLevels: [1, 2, 4, 8, 16], concurrencyLevelCount: 5, backendVersions: ['vLLM-0.10.2'] }

function registryWith(pairs: unknown[], approvedModels: string[] = ['org/model']) {
  return JSON.stringify({ schemaVersion: 2, gate, approvedModels, pairs })
}

const validatedPair = {
  id: 'model__h200', modelId: 'org/model', systemId: 'h200_sxm',
  status: 'validated', classifier: 'xgboost', tritonModel: 'model__h200',
  featureSchema: { feature_names: ['tp'], numeric_medians: {} },
  thresholds: { within_region_probability: 0.85, uncertain_probability: 0.5 },
  coverage, gate: { passed: true, reasons: [], thresholds: gate },
}

afterEach(() => {
  vi.resetAllMocks()
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
})

describe('POST /api/tested-models/classify', () => {
  it('maps Triton class probabilities using the strict production threshold', async () => {
    mockReadFile.mockResolvedValue(registryWith([validatedPair]) as never)
    const fetchMock = vi.fn(() => Promise.resolve(Response.json({ outputs: [{ data: [0.8] }] })))
    vi.stubGlobal('fetch', fetchMock)

    const response = await POST(new Request('http://localhost/api/tested-models/classify', { method: 'POST', body: JSON.stringify(input) }))
    expect(await response.json()).toMatchObject({
      status: 'near_validated_boundary',
      confidence: 0.8,
      pair_id: 'model__h200',
      dashboard_href: '/tested-models/model__h200',
      boundary_href: '/tested-models/model__h200#boundary',
    })
    expect(fetchMock).toHaveBeenCalledWith('http://host.containers.internal:8000/v2/models/model__h200/infer', expect.objectContaining({ method: 'POST' }))
  })

  it('returns a graph link without calling Triton for a measured pair without an envelope', async () => {
    mockReadFile.mockResolvedValue(registryWith([{
      ...validatedPair,
      status: 'measured_no_envelope',
      gate: { passed: false, reasons: ['test_auc_below_minimum'], thresholds: gate },
      diagnostics: { trainingStatus: 'trained', reasons: ['test_auc_below_minimum'] },
    }]) as never)
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    const response = await POST(new Request('http://localhost/api/tested-models/classify', { method: 'POST', body: JSON.stringify(input) }))
    const body = await response.json()
    expect(body).toMatchObject({
      status: 'measured_no_envelope',
      confidence: null,
      pair_id: 'model__h200',
      dashboard_href: '/tested-models/model__h200',
    })
    expect(body.reasons[0]).toContain('no classifier passes the validation gate')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('distinguishes approved-with-no-data from an unknown model', async () => {
    mockReadFile.mockResolvedValue(registryWith([], ['org/model']) as never)
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    const response = await POST(new Request('http://localhost/api/tested-models/classify', { method: 'POST', body: JSON.stringify(input) }))
    const body = await response.json()
    expect(body).toMatchObject({
      status: 'no_measurements',
      pair_id: null,
      dashboard_href: '/tested-models?model=org%2Fmodel',
    })
    expect(body.reasons[0]).toContain('approved for testing')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('returns unavailable without calling Triton for an unsupported pair', async () => {
    mockReadFile.mockResolvedValue(JSON.stringify({ schemaVersion: 2, gate, approvedModels: [], pairs: [] }) as never)
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    const response = await POST(new Request('http://localhost/api/tested-models/classify', { method: 'POST', body: JSON.stringify(input) }))
    expect(await response.json()).toMatchObject({ status: 'validation_unavailable', pair_id: null })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('keeps registry-unavailable distinct from untested', async () => {
    mockReadFile.mockRejectedValue(new Error('missing'))
    const response = await POST(new Request('http://localhost/api/tested-models/classify', { method: 'POST', body: JSON.stringify(input) }))
    const body = await response.json()
    expect(body.status).toBe('validation_unavailable')
    expect(body.reasons[0]).toContain('unknown, not untested')
  })
})

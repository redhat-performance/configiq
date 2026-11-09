import { describe, it, expect, vi, afterEach } from 'vitest'
import { buildFrontierRequest, fetchFrontier } from '../pareto'

function mockFetchOk(data: unknown) {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: () => Promise.resolve(data),
  })
}

const FRONTIER_BODY = {
  simulator: 'aisimulate',
  candidates: [{
    candidate_id: 'candidate-000001',
    status: 'feasible',
    mode: 'agg',
    topology: { deployment_mode: 'agg', tp: 2 },
    used_gpus: 2,
    metrics: { mean_ttft_ms: 400, mean_tpot_ms: 20 },
    objectives: { throughput_per_user: 30, throughput_per_gpu: 50 },
  }],
  views: { pareto_front: ['candidate-000001'], top_n: ['candidate-000001'] },
  sweep: { max_trials: 20, evaluated: 1, strict_sla: true, window_max_gpus: 32 },
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('buildFrontierRequest', () => {
  it('marks the action as a pareto sweep and drops the sizing load target', () => {
    const body = buildFrontierRequest({
      model_path: 'org/model',
      system: 'h200_sxm',
      isl: 2048,
      osl: 128,
      sla_mode: 'enforce',
      ttft: 1000,
      tpot: 30,
      target_concurrency: 8,
      objective: 'min_gpus',
    })

    expect(body).toMatchObject({
      objective: 'pareto',
      sla_mode: 'enforce',
      ttft: 1000,
      tpot: 30,
    })
    expect(body).not.toHaveProperty('target_concurrency')
    expect(body).not.toHaveProperty('target_request_rate')
    expect(body).not.toHaveProperty('minGpus')
    expect(body).not.toHaveProperty('maxGpus')
  })

  it('never adds a GPU window, so the service resolves the model-aware one', () => {
    const body = buildFrontierRequest({
      model_path: 'org/model',
      system: 'h200_sxm',
      sla_mode: 'none',
    })

    expect(body).not.toHaveProperty('min_candidate_gpus')
    expect(body).not.toHaveProperty('max_candidate_gpus')
  })

  it('preserves the caller simulator rather than substituting AISimulate', () => {
    expect(buildFrontierRequest({ simulator: 'blis' }).simulator).toBe('blis')
    expect(buildFrontierRequest({ simulator: 'aisimulate' }).simulator).toBe('aisimulate')
  })
})

describe('fetchFrontier', () => {
  it('posts the standalone sweep to the same-origin proxy', async () => {
    const fetchMock = mockFetchOk(FRONTIER_BODY)
    vi.stubGlobal('fetch', fetchMock)

    const result = await fetchFrontier({
      model_path: 'org/model',
      system: 'h200_sxm',
      isl: 2048,
      osl: 128,
      sla_mode: 'enforce',
      ttft: 1000,
      tpot: 30,
      target_concurrency: 8,
      objective: 'min_gpus',
      simulator: 'aisimulate',
    })

    expect(result.candidates).toHaveLength(1)
    expect(result.views?.pareto_front).toEqual(['candidate-000001'])

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/recommend')
    const body = JSON.parse(init?.body as string)
    expect(body).toMatchObject({
      objective: 'pareto',
      simulator: 'aisimulate',
      sla_mode: 'enforce',
    })
    // A standalone sweep lets the service choose the GPU window.
    expect(body).not.toHaveProperty('min_candidate_gpus')
    expect(body).not.toHaveProperty('max_candidate_gpus')
    // A sweep drives its own load; the sizing target is not a bound.
    expect(body).not.toHaveProperty('target_concurrency')
    expect(body).not.toHaveProperty('target_request_rate')
    expect(body).not.toHaveProperty('minGpus')
    expect(body).not.toHaveProperty('maxGpus')
  })

  it('drops every latency bound for a demand-mode sweep', async () => {
    const fetchMock = mockFetchOk({ ...FRONTIER_BODY, sweep: { strict_sla: false } })
    vi.stubGlobal('fetch', fetchMock)

    await fetchFrontier({
      model_path: 'org/model',
      system: 'h200_sxm',
      isl: 2048,
      osl: 128,
      sla_mode: 'none',
      target_request_rate: 10,
      ttft: 1000,
      tpot: 30,
      request_latency: 2500,
    })

    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.sla_mode).toBe('none')
    expect(body).not.toHaveProperty('ttft')
    expect(body).not.toHaveProperty('tpot')
    expect(body).not.toHaveProperty('request_latency')
  })

  it('surfaces a service validation detail instead of a bare status', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: () => Promise.resolve({
        detail: [{ loc: ['body', 'objective'], msg: 'unsupported objective' }],
      }),
    }))

    await expect(fetchFrontier({ model_path: 'org/model', system: 'h200_sxm' }))
      .rejects.toThrow('body.objective: unsupported objective')
  })

  it('rejects a response that names a different simulator', async () => {
    vi.stubGlobal('fetch', mockFetchOk({ ...FRONTIER_BODY, simulator: 'blis' }))

    await expect(fetchFrontier({ model_path: 'org/model', system: 'h200_sxm', simulator: 'aisimulate' }))
      .rejects.toThrow('Requested aisimulate, but the API confirmed blis')
  })
})

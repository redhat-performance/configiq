import { afterEach, describe, expect, it, vi } from 'vitest'
import { GET } from './route'

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
})

/**
 * Route every gateway URL to a fixture keyed by pathname + search. A URL with
 * no fixture is treated as an upstream failure so a test cannot accidentally
 * pass on data it did not declare.
 */
function stubGateway(responses: Record<string, unknown>) {
  const fetchMock = vi.fn((url: string) => {
    const parsed = new URL(url)
    const key = parsed.pathname + parsed.search
    if (!(key in responses)) return Promise.resolve(new Response('missing', { status: 404 }))
    return Promise.resolve(Response.json(responses[key]))
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('GET /api/catalog', () => {
  it('discovers simulators and scopes each simulator catalogue to itself', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://aisimulators.dev')
    const responses: Record<string, unknown> = {
      '/simulators': {
        simulators: [
          { id: 'aisimulate', label: 'AISimulate', default: true, version: '1.2.3' },
          { id: 'blis', label: 'BLIS', default: false, version: '0.15.0' },
        ],
        default_simulator: 'aisimulate',
      },
      '/backends?simulator=aisimulate': {
        simulator: 'aisimulate',
        recommend_search: 'windowed',
        objectives: ['min_gpus', 'throughput_demand', 'pareto', 'concurrency_capacity'],
        backends: [{ id: 'vllm' }, { id: 'sglang' }, { id: 'tensorrt-llm' }],
      },
      '/models?simulator=aisimulate&include=specs': {
        simulator: 'aisimulate',
        model_catalog_complete: false,
        models: [{ id: 'Qwen/Qwen3-8B' }],
      },
      '/systems?simulator=aisimulate': { simulator: 'aisimulate', systems: [{ id: 'h200_sxm' }] },
      '/networks?simulator=aisimulate': { simulator: 'aisimulate', networks: [] },
      '/backends?simulator=blis': {
        simulator: 'blis',
        recommend_search: 'single',
        objectives: ['min_gpus', 'throughput_demand'],
        backends: [{ id: 'vllm', versions: ['0.31.0'], disaggregation_requires_fabric: true }],
      },
      '/models?simulator=blis&include=specs': {
        simulator: 'blis',
        model_catalog_complete: true,
        models: [{ id: 'meta-llama/Llama-3.1-70B-Instruct' }],
      },
      '/systems?simulator=blis': { simulator: 'blis', systems: [{ id: 'l40s' }] },
      '/networks?simulator=blis': {
        simulator: 'blis',
        default_network: 'roce-200g',
        networks: ['ib-400g', 'roce-200g'],
      },
    }
    const fetchMock = stubGateway(responses)

    const response = await GET(new Request('https://configiq.dev/api/catalog'))

    expect(response.status).toBe(200)
    const body = await response.json()
    // The default simulator's own catalogue is the top-level catalogue.
    expect(body).toMatchObject({
      systems: [{ id: 'h200_sxm' }],
      models: [{ id: 'Qwen/Qwen3-8B' }],
      backends: [{ id: 'vllm' }, { id: 'sglang' }, { id: 'tensorrt-llm' }],
      networks: [],
      simulators: [
        { id: 'aisimulate', label: 'AISimulate', version: '1.2.3', default: true },
        { id: 'blis', label: 'BLIS', version: '0.15.0', default: false },
      ],
      defaultSimulator: 'aisimulate',
      objectives: ['min_gpus', 'throughput_demand', 'pareto', 'concurrency_capacity'],
    })
    // Capability facts are taken from each simulator's own envelope, never
    // inferred from its ID.
    expect(body.simulatorSupport.aisimulate).toMatchObject({
      id: 'aisimulate',
      label: 'AISimulate',
      version: '1.2.3',
      recommendSearch: 'windowed',
      modelsExhaustive: false,
      systems: [{ id: 'h200_sxm' }],
      models: [{ id: 'Qwen/Qwen3-8B' }],
      networks: [],
      defaultNetwork: null,
    })
    expect(body.simulatorSupport.blis).toMatchObject({
      id: 'blis',
      label: 'BLIS',
      version: '0.15.0',
      recommendSearch: 'single',
      modelsExhaustive: true,
      systems: [{ id: 'l40s' }],
      models: [{ id: 'meta-llama/Llama-3.1-70B-Instruct' }],
      networks: ['ib-400g', 'roce-200g'],
      defaultNetwork: 'roce-200g',
      objectives: ['min_gpus', 'throughput_demand'],
    })
    expect(fetchMock).toHaveBeenCalledTimes(9)
  })

  it('omits a simulator whose scoped responses do not echo it', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://aisimulators.dev')
    stubGateway({
      '/simulators': {
        simulators: [
          { id: 'aisimulate', label: 'AISimulate', default: true, version: null },
          { id: 'blis', label: 'BLIS', default: false, version: null },
        ],
      },
      '/backends?simulator=aisimulate': { simulator: 'aisimulate', backends: [{ id: 'vllm' }] },
      '/models?simulator=aisimulate&include=specs': { simulator: 'aisimulate', models: ['Qwen/Qwen3-8B'] },
      '/systems?simulator=aisimulate': { simulator: 'aisimulate', systems: [{ id: 'h200_sxm' }] },
      '/networks?simulator=aisimulate': { simulator: 'aisimulate', networks: [] },
      // An older build ignores `?simulator=blis` and answers with the default
      // AISimulate data at HTTP 200 without echoing the requested simulator.
      '/backends?simulator=blis': { backends: [{ id: 'vllm' }] },
      '/models?simulator=blis&include=specs': { models: ['Qwen/Qwen3-8B'] },
      '/systems?simulator=blis': { systems: [{ id: 'h200_sxm' }] },
      '/networks?simulator=blis': { networks: [] },
    })

    const response = await GET(new Request('https://configiq.dev/api/catalog'))

    expect(response.status).toBe(200)
    const body = await response.json()
    expect(Object.keys(body.simulatorSupport)).toEqual(['aisimulate'])
    expect(body.simulatorSupport).not.toHaveProperty('blis')
  })

  it('fails when the default simulator catalogue cannot be read', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://aisimulators.dev')
    stubGateway({
      '/simulators': {
        simulators: [{ id: 'aisimulate', label: 'AISimulate', default: true, version: null }],
      },
      // Missing the echo, so this simulator's support is unreadable.
      '/backends?simulator=aisimulate': { backends: [{ id: 'vllm' }] },
      '/models?simulator=aisimulate&include=specs': { models: ['Qwen/Qwen3-8B'] },
      '/systems?simulator=aisimulate': { systems: [{ id: 'h200_sxm' }] },
      '/networks?simulator=aisimulate': { networks: [] },
    })

    const response = await GET(new Request('https://configiq.dev/api/catalog'))

    expect(response.status).toBe(502)
    expect(response.headers.get('cache-control')).toBe('no-store')
    expect(await response.json()).toMatchObject({ error: { code: 'AISIM_INVALID_RESPONSE' } })
  })

  it('falls back to the unscoped catalogue when the service has no /simulators', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://aisimulators.dev')
    stubGateway({
      '/simulators': { detail: 'Not Found' },
      '/systems?include=specs': { systems: [{ id: 'h200_sxm' }] },
      '/models?include=specs': { models: [{ id: 'Qwen/Qwen3-8B' }] },
      '/backends': { backends: [{ id: 'vllm' }], objectives: ['min_gpus'] },
      '/networks': { networks: ['ib-400g'] },
    })

    const response = await GET(new Request('https://configiq.dev/api/catalog'))

    expect(response.status).toBe(200)
    expect(await response.json()).toMatchObject({
      systems: [{ id: 'h200_sxm' }],
      models: [{ id: 'Qwen/Qwen3-8B' }],
      backends: [{ id: 'vllm' }],
      networks: ['ib-400g'],
      simulators: [],
      defaultSimulator: null,
      simulatorSupport: {},
      objectives: ['min_gpus'],
    })
  })

  it('preserves a combined ConfigIQ catalogue gateway envelope', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://configiq.dev/api')
    const fetchMock = vi.fn(() => Promise.resolve(Response.json({
      systems: [{ id: 'h200_sxm' }],
      models: ['Qwen/Qwen3-8B'],
      backends: [{ id: 'vllm' }],
      networks: ['ib-400g'],
      simulators: [{ id: 'aisimulate', label: 'AISimulate', version: '1.2.3', default: true }],
      defaultSimulator: 'aisimulate',
      simulatorSupport: { aisimulate: { id: 'aisimulate' } },
      objectives: ['min_gpus'],
    })))
    vi.stubGlobal('fetch', fetchMock)

    const response = await GET(new Request('https://local.configiq.test/api/catalog'))

    expect(response.status).toBe(200)
    expect(await response.json()).toMatchObject({
      systems: [{ id: 'h200_sxm' }],
      models: ['Qwen/Qwen3-8B'],
      backends: [{ id: 'vllm' }],
      networks: ['ib-400g'],
      simulators: [{ id: 'aisimulate', label: 'AISimulate', version: '1.2.3', default: true }],
      defaultSimulator: 'aisimulate',
      simulatorSupport: { aisimulate: { id: 'aisimulate' } },
      objectives: ['min_gpus'],
    })
    expect(fetchMock).toHaveBeenCalledWith('https://configiq.dev/api/catalog', expect.any(Object))
  })

  it('rejects a combined gateway pointing back to this host, including behind a proxy', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://configiq.dev/api')
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    for (const request of [
      new Request('https://configiq.dev/api/catalog'),
      new Request('http://127.0.0.1:3001/api/catalog', {
        headers: { 'x-forwarded-host': 'configiq.dev' },
      }),
    ]) {
      const response = await GET(request)
      expect(response.status).toBe(503)
      expect(response.headers.get('cache-control')).toBe('no-store')
      expect(await response.json()).toMatchObject({ error: { code: 'AISIM_INVALID_GATEWAY' } })
    }
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it.each([{}, { systems: {}, models: [] }, { systems: [], models: null }])(
    'rejects malformed combined catalogues without caching them: %j',
    async payload => {
      vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://configiq.dev/api')
      vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(Response.json(payload))))

      const response = await GET(new Request('https://local.configiq.test/api/catalog'))
      expect(response.status).toBe(502)
      expect(response.headers.get('cache-control')).toBe('no-store')
      expect(await response.json()).toMatchObject({ error: { code: 'AISIM_INVALID_RESPONSE' } })
    },
  )

  it('rejects a direct service catalogue when a required list is missing', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://aisimulators.dev')
    const fetchMock = vi.fn((url: string) => Promise.resolve(Response.json(
      url.includes('/models') ? { message: 'upstream error' } : { systems: [] },
    )))
    vi.stubGlobal('fetch', fetchMock)

    const response = await GET(new Request('https://configiq.dev/api/catalog'))
    expect(response.status).toBe(502)
    expect(response.headers.get('cache-control')).toBe('no-store')
    expect(await response.json()).toMatchObject({ error: { code: 'AISIM_INVALID_RESPONSE' } })
  })

  it.each([
    { systems: [null], models: ['Qwen/Qwen3-8B'] },
    { systems: [{}], models: ['Qwen/Qwen3-8B'] },
    { systems: [{ id: 'h200_sxm' }], models: [null] },
  ])('rejects malformed entries from a direct service catalogue: %j', async payload => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://aisimulators.dev')
    const fetchMock = vi.fn((url: string) => Promise.resolve(Response.json(
      url.includes('/models') ? { models: payload.models } : { systems: payload.systems },
    )))
    vi.stubGlobal('fetch', fetchMock)

    const response = await GET(new Request('https://configiq.dev/api/catalog'))
    expect(response.status).toBe(502)
    expect(await response.json()).toMatchObject({ error: { code: 'AISIM_INVALID_RESPONSE' } })
  })

  it('rejects malformed entries from a combined catalogue', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://configiq.dev/api')
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(Response.json({
      systems: [null],
      models: ['Qwen/Qwen3-8B'],
    }))))

    const response = await GET(new Request('https://local.configiq.test/api/catalog'))
    expect(response.status).toBe(502)
    expect(await response.json()).toMatchObject({ error: { code: 'AISIM_INVALID_RESPONSE' } })
  })
})

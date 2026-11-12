import { afterEach, describe, expect, it, vi } from 'vitest'
import { GET } from './route'

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
})

// Deliberately non-release strings: the route must forward whatever the
// service's /simulators reports, not know or pin a simulator version. Using
// plausible SemVer here made the fixtures look like real release pins.
const AISIMULATE_FIXTURE_VERSION = 'test-aisimulate-version'
const BLIS_FIXTURE_VERSION = 'test-blis-version'
const AISIMULATE_FIXTURE_REVISION = 'test-aisimulate-revision'
const BLIS_FIXTURE_REVISION = 'test-blis-revision'

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
          { id: 'aisimulate', label: 'AISimulate', description: 'Analytical sizing.', default: true, version: AISIMULATE_FIXTURE_VERSION, revision: AISIMULATE_FIXTURE_REVISION },
          { id: 'blis', label: 'BLIS', description: 'Discrete-event simulation.', default: false, version: BLIS_FIXTURE_VERSION, revision: BLIS_FIXTURE_REVISION },
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
      '/systems?simulator=aisimulate&include=specs': {
        simulator: 'aisimulate',
        systems: [{
          id: 'h200_sxm',
          name: 'NVIDIA H200 SXM',
          vendor: 'nvidia',
          architecture: 'hopper',
          memory_bytes: 151_397_597_184,
          memory_bandwidth_bytes: 4_800_000_000_000,
          bf16_tflops: 989,
          tdp_watts: 700,
          gpus_per_node: 8,
        }],
      },
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
      '/systems?simulator=blis&include=specs': {
        simulator: 'blis',
        systems: [{
          id: 'l40s',
          name: 'NVIDIA L40S',
          vendor: 'nvidia',
          architecture: 'ada',
          memory_bytes: 48_000_000_000,
          memory_bandwidth_bytes: 864_000_000_000,
          bf16_tflops: 362,
          tdp_watts: 350,
          gpus_per_node: 8,
        }],
      },
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
    // The default simulator's own catalogue is exposed through its entry, so
    // the response carries no duplicate top-level lists.
    expect(body).not.toHaveProperty('systems')
    expect(body).not.toHaveProperty('models')
    expect(body).not.toHaveProperty('backends')
    expect(body).not.toHaveProperty('networks')
    expect(body).not.toHaveProperty('objectives')
    expect(body.timeoutSeconds).toEqual(expect.any(Number))
    // Identity, default flag, and capabilities live in one entry per simulator.
    expect(body).not.toHaveProperty('defaultSimulator')
    expect(body).not.toHaveProperty('simulatorSupport')
    expect(body.simulators.map((simulator: { id: string; default: boolean; available: boolean }) => ({
      id: simulator.id,
      default: simulator.default,
      available: simulator.available,
    }))).toEqual([
      { id: 'aisimulate', default: true, available: true },
      { id: 'blis', default: false, available: true },
    ])
    // Capability facts are taken from each simulator's own envelope, never
    // inferred from its ID.
    expect(body.simulators[0]).toMatchObject({
      id: 'aisimulate',
      label: 'AISimulate',
      description: 'Analytical sizing.',
      version: AISIMULATE_FIXTURE_VERSION,
      revision: AISIMULATE_FIXTURE_REVISION,
      recommendSearch: 'windowed',
      modelsExhaustive: false,
      systems: [{ id: 'h200_sxm' }],
      models: [{ id: 'Qwen/Qwen3-8B' }],
      networks: [],
      defaultNetwork: null,
      objectives: ['min_gpus', 'throughput_demand', 'pareto', 'concurrency_capacity'],
    })
    expect(body.simulators[1]).toMatchObject({
      id: 'blis',
      label: 'BLIS',
      description: 'Discrete-event simulation.',
      version: BLIS_FIXTURE_VERSION,
      revision: BLIS_FIXTURE_REVISION,
      recommendSearch: 'single',
      modelsExhaustive: true,
      systems: [{ id: 'l40s' }],
      models: [{ id: 'meta-llama/Llama-3.1-70B-Instruct' }],
      networks: ['ib-400g', 'roce-200g'],
      defaultNetwork: 'roce-200g',
      objectives: ['min_gpus', 'throughput_demand'],
    })
    // The GPU picker groups by vendor/architecture and describes each system
    // with its memory and compute specs, so the proxy must ask for and forward
    // them rather than only the system IDs.
    expect(body.simulators[0].systems[0]).toMatchObject({
      vendor: 'nvidia',
      architecture: 'hopper',
      memory_bytes: 151_397_597_184,
    })
    expect(fetchMock).toHaveBeenCalledWith(
      'https://aisimulators.dev/systems?simulator=aisimulate&include=specs',
      expect.any(Object),
    )
    expect(fetchMock).toHaveBeenCalledTimes(9)
  })

  it('retries a transient capability read instead of dropping the simulator', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://aisimulators.dev')
    const responses: Record<string, unknown> = {
      '/simulators': {
        simulators: [{
          id: 'aisimulate',
          label: 'AISimulate',
          default: true,
          version: AISIMULATE_FIXTURE_VERSION,
          revision: AISIMULATE_FIXTURE_REVISION,
        }],
      },
      '/backends?simulator=aisimulate': { simulator: 'aisimulate', backends: [{ id: 'vllm' }] },
      '/models?simulator=aisimulate&include=specs': {
        simulator: 'aisimulate',
        models: [{ id: 'Qwen/Qwen3-8B' }],
      },
      '/systems?simulator=aisimulate&include=specs': {
        simulator: 'aisimulate',
        systems: [{ id: 'h200_sxm' }],
      },
      '/networks?simulator=aisimulate': { simulator: 'aisimulate', networks: [] },
    }
    // The gateway intermittently answers 503 while this proxy reads several
    // capability endpoints at once. One failed attempt must not make the whole
    // simulator look unavailable to the client.
    let backendsAttempts = 0
    const fetchMock = vi.fn((url: string) => {
      const parsed = new URL(url)
      const key = parsed.pathname + parsed.search
      if (key === '/backends?simulator=aisimulate') {
        backendsAttempts += 1
        if (backendsAttempts === 1) return Promise.resolve(new Response('unavailable', { status: 503 }))
      }
      if (!(key in responses)) return Promise.resolve(new Response('missing', { status: 404 }))
      return Promise.resolve(Response.json(responses[key]))
    })
    vi.stubGlobal('fetch', fetchMock)

    const response = await GET(new Request('https://configiq.dev/api/catalog'))

    expect(response.status).toBe(200)
    const body = await response.json()
    expect(backendsAttempts).toBe(2)
    expect(body.simulators[0]).toMatchObject({
      id: 'aisimulate',
      available: true,
      backends: [{ id: 'vllm' }],
    })
  })

  it('does not retry a capability read the service rejects', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://aisimulators.dev')
    let backendsAttempts = 0
    const failure = new Response('bad request', { status: 400 })
    const cancelFailureBody = vi.spyOn(failure.body!, 'cancel')
    vi.stubGlobal('fetch', vi.fn((url: string) => {
      const parsed = new URL(url)
      if (parsed.pathname === '/simulators') {
        return Promise.resolve(Response.json({
          simulators: [{ id: 'aisimulate', label: 'AISimulate', default: true }],
        }))
      }
      if (parsed.pathname === '/backends') {
        backendsAttempts += 1
        return Promise.resolve(failure)
      }
      return Promise.resolve(Response.json({
        simulator: 'aisimulate',
        models: [{ id: 'Qwen/Qwen3-8B' }],
        systems: [{ id: 'h200_sxm' }],
        networks: [],
      }))
    }))

    const response = await GET(new Request('https://configiq.dev/api/catalog'))

    // A malformed request will fail the same way on every attempt, so the
    // default simulator's catalogue is unreadable and the request fails.
    expect(backendsAttempts).toBe(1)
    expect(cancelFailureBody).toHaveBeenCalled()
    expect(response.status).toBe(502)
    expect(await response.json()).toMatchObject({ error: { code: 'AISIM_INVALID_RESPONSE' } })
  })

  it('reports a blank simulator version as unavailable instead of an empty value', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://aisimulators.dev')
    stubGateway({
      '/simulators': {
        simulators: [{ id: 'aisimulate', label: 'AISimulate', default: true, version: '   ', revision: '   ' }],
      },
      '/backends?simulator=aisimulate': { simulator: 'aisimulate', backends: [{ id: 'vllm' }] },
      '/models?simulator=aisimulate&include=specs': {
        simulator: 'aisimulate',
        models: [{ id: 'Qwen/Qwen3-8B' }],
      },
      '/systems?simulator=aisimulate&include=specs': { simulator: 'aisimulate', systems: [{ id: 'h200_sxm' }] },
      '/networks?simulator=aisimulate': { simulator: 'aisimulate', networks: [] },
    })

    const response = await GET(new Request('https://configiq.dev/api/catalog'))

    expect(response.status).toBe(200)
    const body = await response.json()
    expect(body.simulators).toHaveLength(1)
    expect(body.simulators[0]).toMatchObject({
      id: 'aisimulate',
      label: 'AISimulate',
      version: null,
      revision: null,
      default: true,
      available: true,
    })
  })

  it('marks a simulator whose scoped responses do not echo it as unavailable', async () => {
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
      '/systems?simulator=aisimulate&include=specs': { simulator: 'aisimulate', systems: [{ id: 'h200_sxm' }] },
      '/networks?simulator=aisimulate': { simulator: 'aisimulate', networks: [] },
      // An older build ignores `?simulator=blis` and answers with the default
      // AISimulate data at HTTP 200 without echoing the requested simulator.
      '/backends?simulator=blis': { backends: [{ id: 'vllm' }] },
      '/models?simulator=blis&include=specs': { models: ['Qwen/Qwen3-8B'] },
      '/systems?simulator=blis&include=specs': { systems: [{ id: 'h200_sxm' }] },
      '/networks?simulator=blis': { networks: [] },
    })

    const response = await GET(new Request('https://configiq.dev/api/catalog'))

    expect(response.status).toBe(200)
    const body = await response.json()
    // The identity is kept so it stays listed, but no capability is claimed.
    expect(body.simulators.map((simulator: { id: string; available: boolean }) => ({
      id: simulator.id,
      available: simulator.available,
    }))).toEqual([
      { id: 'aisimulate', available: true },
      { id: 'blis', available: false },
    ])
    expect(body.simulators[1]).toMatchObject({
      id: 'blis',
      default: false,
      available: false,
      models: [],
      systems: [],
      backends: [],
      networks: [],
    })
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
      '/systems?simulator=aisimulate&include=specs': { systems: [{ id: 'h200_sxm' }] },
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
    const body = await response.json()
    expect(body).toMatchObject({
      systems: [{ id: 'h200_sxm' }],
      models: [{ id: 'Qwen/Qwen3-8B' }],
      backends: [{ id: 'vllm' }],
      networks: ['ib-400g'],
      simulators: [],
      objectives: ['min_gpus'],
    })
    expect(body).not.toHaveProperty('defaultSimulator')
    expect(body).not.toHaveProperty('simulatorSupport')
  })

  it('preserves a combined ConfigIQ catalogue gateway envelope', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://configiq.dev/api')
    const fetchMock = vi.fn(() => Promise.resolve(Response.json({
      systems: [{ id: 'h200_sxm' }],
      models: ['Qwen/Qwen3-8B'],
      backends: [{ id: 'vllm' }],
      networks: ['ib-400g'],
      simulators: [{ id: 'aisimulate', label: 'AISimulate', version: AISIMULATE_FIXTURE_VERSION, default: true }],
      defaultSimulator: 'aisimulate',
      simulatorSupport: {
        aisimulate: {
          id: 'aisimulate',
          label: 'AISimulate',
          version: AISIMULATE_FIXTURE_VERSION,
          recommendSearch: 'windowed',
          modelsExhaustive: false,
          backends: [{ id: 'vllm' }],
          models: ['Qwen/Qwen3-8B'],
          systems: [{ id: 'h200_sxm' }],
          networks: ['ib-400g'],
          defaultNetwork: null,
          objectives: ['min_gpus'],
        },
      },
      objectives: ['min_gpus'],
    })))
    vi.stubGlobal('fetch', fetchMock)

    const response = await GET(new Request('https://local.configiq.test/api/catalog'))

    expect(response.status).toBe(200)
    const body = await response.json()
    // The old top-level lists and support map are folded into simulator
    // entries; the reduced response no longer duplicates them.
    expect(body).not.toHaveProperty('systems')
    expect(body).not.toHaveProperty('models')
    expect(body).not.toHaveProperty('backends')
    expect(body).not.toHaveProperty('networks')
    expect(body).not.toHaveProperty('objectives')
    expect(body).not.toHaveProperty('defaultSimulator')
    expect(body).not.toHaveProperty('simulatorSupport')
    expect(body.simulators).toEqual([{
      id: 'aisimulate',
      label: 'AISimulate',
      description: null,
      version: AISIMULATE_FIXTURE_VERSION,
      revision: null,
      default: true,
      available: true,
      recommendSearch: 'windowed',
      modelsExhaustive: false,
      backends: [{ id: 'vllm' }],
      models: ['Qwen/Qwen3-8B'],
      systems: [{ id: 'h200_sxm' }],
      networks: ['ib-400g'],
      defaultNetwork: null,
      objectives: ['min_gpus'],
    }])
    expect(fetchMock).toHaveBeenCalledWith('https://configiq.dev/api/catalog', expect.any(Object))
  })

  it('accepts a combined gateway that already uses the unified simulator shape', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://configiq.dev/api')
    const simulator = {
      id: 'blis',
      label: 'BLIS',
      description: 'Discrete-event simulation.',
      version: BLIS_FIXTURE_VERSION,
      revision: BLIS_FIXTURE_REVISION,
      default: true,
      available: true,
      recommendSearch: 'single',
      modelsExhaustive: true,
      backends: [{ id: 'vllm' }],
      models: ['meta-llama/Llama-3.1-70B-Instruct'],
      systems: [{ id: 'l40s' }],
      networks: ['roce-200g'],
      defaultNetwork: 'roce-200g',
      objectives: ['min_gpus'],
    }
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(Response.json({
      systems: [{ id: 'l40s' }],
      models: simulator.models,
      backends: simulator.backends,
      networks: simulator.networks,
      simulators: [simulator],
      objectives: simulator.objectives,
    }))))

    const response = await GET(new Request('https://local.configiq.test/api/catalog'))

    expect(response.status).toBe(200)
    expect((await response.json()).simulators).toEqual([simulator])
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

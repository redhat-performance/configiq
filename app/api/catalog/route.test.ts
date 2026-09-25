import { afterEach, describe, expect, it, vi } from 'vitest'
import { GET } from './route'

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
})

describe('GET /api/catalog', () => {
  it('preserves the shared backend catalogue alongside systems and models', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://aisimulators.dev')
    const responses: Record<string, unknown> = {
      '/systems?include=specs': { systems: [{ id: 'h200_sxm' }] },
      '/models?include=specs': { models: [{ id: 'Qwen/Qwen3-8B' }] },
      '/backends': { backends: [{ id: 'vllm' }, { id: 'sglang' }, { id: 'tensorrt-llm' }] },
    }
    const fetchMock = vi.fn((url: string) => {
      const path = new URL(url).pathname + new URL(url).search
      return Promise.resolve(Response.json(responses[path]))
    })
    vi.stubGlobal('fetch', fetchMock)

    const response = await GET(new Request('https://configiq.dev/api/catalog'))

    expect(response.status).toBe(200)
    expect(await response.json()).toMatchObject({
      systems: [{ id: 'h200_sxm' }],
      models: [{ id: 'Qwen/Qwen3-8B' }],
      backends: [{ id: 'vllm' }, { id: 'sglang' }, { id: 'tensorrt-llm' }],
    })
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it('preserves backends when using a combined ConfigIQ catalogue gateway', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://configiq.dev/api')
    const fetchMock = vi.fn(() => Promise.resolve(Response.json({
      systems: [{ id: 'h200_sxm' }],
      models: ['Qwen/Qwen3-8B'],
      backends: [{ id: 'vllm' }],
    })))
    vi.stubGlobal('fetch', fetchMock)

    const response = await GET(new Request('https://local.configiq.test/api/catalog'))

    expect(response.status).toBe(200)
    expect(await response.json()).toMatchObject({
      systems: [{ id: 'h200_sxm' }],
      models: ['Qwen/Qwen3-8B'],
      backends: [{ id: 'vllm' }],
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
})

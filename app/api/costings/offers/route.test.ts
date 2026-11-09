import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { GET } from './route'
vi.mock('@/lib/otel-metrics', () => ({ recordErrorCode: vi.fn() }))
beforeEach(() => { vi.stubEnv('AICOSTINGS_GATEWAY_URL', 'https://pricing.example'); vi.stubGlobal('fetch', vi.fn()) })
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals() })
it('uses the server-side gateway and forwards the complete-offer response', async () => {
  const data = { schema_version: 1, rented: [], purchased: [], sources: {} }
  vi.mocked(fetch).mockResolvedValue(new Response(JSON.stringify(data)))
  const response = await GET()
  expect(await response.json()).toEqual(data)
  expect(fetch).toHaveBeenCalledWith('https://pricing.example/offers', expect.objectContaining({ cache: 'no-store' }))
  expect(response.headers.get('cache-control')).toBe('no-store')
})
it.each([404, 500])('keeps %s failures isolated from legacy endpoints', async upstream => {
  vi.mocked(fetch).mockResolvedValue(new Response('private upstream error', { status: upstream }))
  const response = await GET()
  expect(response.status).toBe(upstream === 404 ? 404 : 502)
  expect(response.headers.get('cache-control')).toBe('no-store')
  expect(await response.text()).not.toContain('private upstream error')
})
it('handles timeouts without exposing upstream details', async () => {
  vi.mocked(fetch).mockRejectedValue(Object.assign(new Error('secret URL'), { name: 'TimeoutError' }))
  const response = await GET()
  expect(response.status).toBe(504)
  expect(response.headers.get('cache-control')).toBe('no-store')
  expect(await response.text()).not.toContain('secret URL')
})

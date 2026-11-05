import { NextRequest } from 'next/server'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { GET as models } from './models/route'
import { GET as systems } from './systems/route'

vi.mock('@/lib/otel-metrics', () => ({ recordErrorCode: vi.fn() }))
beforeEach(() => {
  vi.stubEnv('AICOSTINGS_GATEWAY_URL', 'https://pricing.example')
  vi.stubGlobal('fetch', vi.fn())
})
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs() })

it.each([
  { models: [], stale: true },
  { models: [], stale: false },
  { models: [{ id: 'owner/model' }], stale: true },
])('does not HTTP-cache an empty or stale hosted catalogue: %j', async data => {
  vi.mocked(fetch).mockResolvedValue(Response.json(data))
  const response = await models(new NextRequest('http://localhost/api/costings/models?source=openrouter'))
  expect(response.headers.get('Cache-Control')).toBe('no-store')
  expect(await response.json()).toEqual(data)
})

it('retains normal caching and source forwarding for a ready hosted catalogue', async () => {
  vi.mocked(fetch).mockResolvedValue(Response.json({ models: [{ id: 'owner/model' }], stale: false }))
  const response = await models(new NextRequest('http://localhost/api/costings/models?source=openrouter'))
  expect(response.headers.get('Cache-Control')).toContain('s-maxage=3600')
  expect(fetch).toHaveBeenCalledWith('https://pricing.example/models?source=openrouter', expect.any(Object))
})

it.each([
  { systems: [] },
  { systems: [{ id: 'l40s' }] },
  { systems: [{ id: 'l40s', cloud_rates: {}, hardware_cost: { new_usd: 10000 } }] },
  { systems: [{ id: 'l40s', cloud_rates: { 'aws.region': { on_demand: 2 } } }] },
])('does not HTTP-cache requested infrastructure prices before collection: %j', async data => {
  vi.mocked(fetch).mockResolvedValue(Response.json(data))
  const response = await systems(new NextRequest('http://localhost/api/costings/systems?include=cloud,hardware'))
  expect(response.headers.get('Cache-Control')).toBe('no-store')
})

it('preserves the legacy systems payload and normal caching once requested prices exist', async () => {
  const data = { systems: [{ id: 'l40s', cloud_rates: { 'aws.region': { on_demand: 2 } }, hardware_cost: { new_usd: 10000 } }] }
  vi.mocked(fetch).mockResolvedValue(Response.json(data))
  const response = await systems(new NextRequest('http://localhost/api/costings/systems?include=cloud,hardware'))
  expect(response.headers.get('Cache-Control')).toContain('s-maxage=3600')
  expect(await response.json()).toEqual(data)
  expect(fetch).toHaveBeenCalledWith('https://pricing.example/systems?include=cloud%2Chardware', expect.any(Object))
})

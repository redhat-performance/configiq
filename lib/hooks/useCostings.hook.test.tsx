// @vitest-environment happy-dom
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import type { PricingSource } from '@/contexts/SettingsContext'
import type { CostingsData, CostingsScope } from './useCostings'
import { CLIENT_CACHE_TTL_MS } from '@/lib/cache-policy'

const model = { id: 'owner/model', name: 'Model', provider: 'Provider', tier: 'balanced',
  price_per_m_input: 1, price_per_m_output: 2, context_window: null, updated_at: null }
const systems = [{ id: 'l40s', cloud_rates: { 'aws.us-east-1': {
  on_demand: 2, reserved_1yr: null, reserved_3yr: null, spot_median: null,
  rate_basis: 'gpu_hour', gpus_per_instance: 1,
} }, hardware_cost: { new_usd: 10000, indicative: true } }]
let root: ReturnType<typeof createRoot>
let host: HTMLDivElement
let reply: (url: string) => Response | Promise<Response>

function healthyResponse(url: string): Response {
  if (url.includes('/models')) return Response.json({ models: [model], stale: false, updated_at: '2026-10-27T12:00:00Z' })
  if (url.includes('/systems')) return Response.json({ systems })
  return Response.json({ status: 'ok', version: 'test', sources: {} })
}
beforeEach(() => {
  vi.resetModules()
  vi.useFakeTimers()
  vi.setSystemTime(new Date('2026-10-27T12:00:00Z'))
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  reply = healthyResponse
  vi.stubGlobal('fetch', vi.fn((url: string) => Promise.resolve(reply(url))))
  host = document.createElement('div')
  document.body.append(host)
  root = createRoot(host)
})
afterEach(async () => {
  await act(async () => root.unmount())
  host.remove()
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

async function harness(enabled = true, source: PricingSource = 'merged', scope?: CostingsScope) {
  const { useCostings } = await import('./useCostings')
  let state!: CostingsData
  function Harness(props: { enabled: boolean; source: PricingSource; scope?: CostingsScope }) {
    state = useCostings(props.enabled, props.source, props.scope)
    return null
  }
  const render = async (active: boolean, selected: PricingSource = source, selectedScope = scope) => {
    await act(async () => root.render(<Harness enabled={active} source={selected} scope={selectedScope} />))
  }
  await render(enabled)
  return { get state() { return state }, render }
}

it('recovers an empty cold-start catalogue without remounting or reloading', async () => {
  let ready = false
  reply = url => !ready && url.includes('/models')
    ? Response.json({ models: [], stale: true, updated_at: null })
    : !ready && url.includes('/systems') ? Response.json({ systems: [] }) : healthyResponse(url)
  const view = await harness()
  expect(view.state.models).toEqual([])
  ready = true
  await act(async () => vi.advanceTimersByTimeAsync(60_000))
  expect(view.state.models).toHaveLength(1)
  expect(view.state.gpuCloudRates.get('l40s')?.['aws.us-east-1'].on_demand).toBe(2)
  expect(view.state.gpuHardwareCosts.get('l40s')?.new_usd).toBe(10000)
  expect(view.state.modelsStale).toBe(false)
  expect(fetch).toHaveBeenCalledTimes(6)
})

it('retries missing legacy infrastructure even when hosted prices are already ready', async () => {
  let ready = false
  reply = url => !ready && url.includes('/systems') ? Response.json({ systems: [] }) : healthyResponse(url)
  const view = await harness()
  expect(view.state.models).toHaveLength(1)
  expect(view.state.gpuCloudRates.size).toBe(0)
  ready = true
  await act(async () => vi.advanceTimersByTimeAsync(60_000))
  expect(view.state.gpuCloudRates.size).toBe(1)
  expect(view.state.gpuHardwareCosts.size).toBe(1)
})

it('isolates models-only caching and keeps the existing full-data default for other tabs', async () => {
  const view = await harness(true, 'merged', 'models')
  expect(fetch).toHaveBeenCalledTimes(2)
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes('/systems'))).toBe(false)
  await act(async () => vi.advanceTimersByTimeAsync(60_000))
  expect(fetch).toHaveBeenCalledTimes(2)
  await view.render(true, 'merged', 'all')
  expect(fetch).toHaveBeenCalledTimes(5)
  expect(view.state.gpuCloudRates.size).toBe(1)
  expect(view.state.gpuHardwareCosts.size).toBe(1)
  await view.render(true, 'merged', 'models')
  expect(fetch).toHaveBeenCalledTimes(5)
  expect(view.state.models).toHaveLength(1)
  expect(view.state.gpuCloudRates.size).toBe(0)
})

it('deduplicates simultaneous consumers of the same pricing scope', async () => {
  const { useCostings } = await import('./useCostings')
  function Consumer() { useCostings(true); return null }
  await act(async () => root.render(<><Consumer /><Consumer /></>))
  expect(fetch).toHaveBeenCalledTimes(3)
})

it('keeps source changes isolated when an older request finishes later', async () => {
  let complete!: (response: Response) => void
  reply = url => url.includes('source=litellm') ? new Promise(resolve => { complete = resolve }) : healthyResponse(url)
  const view = await harness(true, 'litellm', 'models')
  expect(view.state.isLoading).toBe(true)
  await view.render(true, 'openrouter', 'models')
  expect(view.state.models[0].id).toBe(model.id)
  await act(async () => complete(Response.json({ models: [{ ...model, id: 'litellm/model' }], stale: false })))
  expect(view.state.models[0].id).toBe(model.id)
  await view.render(true, 'litellm', 'models')
  expect(view.state.models[0].id).toBe('litellm/model')
})

it('preserves last-good values during refresh failures and automatically recovers', async () => {
  const view = await harness()
  const previousModels = view.state.models
  reply = () => { throw new Error('Offline') }
  await act(async () => vi.advanceTimersByTimeAsync(CLIENT_CACHE_TTL_MS.costings))
  expect(view.state.models).toBe(previousModels)
  expect(view.state.gpuCloudRates.size).toBe(1)
  expect(view.state.modelsStale).toBe(true)
  expect(view.state.error).toBe('Offline')
  reply = healthyResponse
  await act(async () => vi.advanceTimersByTimeAsync(60_000))
  expect(view.state.error).toBeNull()
  expect(view.state.modelsStale).toBe(false)
})

it('does not reuse another source after a failed switch and retries the selected source', async () => {
  const view = await harness(true, 'openrouter', 'models')
  reply = url => url.includes('source=litellm') ? new Response('', { status: 503 }) : healthyResponse(url)
  await view.render(true, 'litellm', 'models')
  expect(view.state.models).toEqual([])
  expect(view.state.error).toBe('/models 503')
  reply = healthyResponse
  await act(async () => vi.advanceTimersByTimeAsync(60_000))
  expect(view.state.models).toHaveLength(1)
  expect(view.state.error).toBeNull()
})

it('does not fetch while disabled or poll while the page is hidden', async () => {
  const view = await harness(false)
  expect(fetch).not.toHaveBeenCalled()
  await view.render(true)
  expect(fetch).toHaveBeenCalledTimes(3)
  const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
  await act(async () => vi.advanceTimersByTimeAsync(CLIENT_CACHE_TTL_MS.costings))
  expect(fetch).toHaveBeenCalledTimes(3)
  visibility.mockReturnValue('visible')
  await act(async () => document.dispatchEvent(new Event('visibilitychange')))
  expect(fetch).toHaveBeenCalledTimes(6)
  await view.render(false)
  await act(async () => vi.advanceTimersByTimeAsync(CLIENT_CACHE_TTL_MS.costings))
  expect(fetch).toHaveBeenCalledTimes(6)
  expect(view.state.models).toEqual([])
})

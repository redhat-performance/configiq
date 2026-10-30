// @vitest-environment happy-dom
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { fetchInfrastructureOffers } from '@/lib/api/infrastructure-offers'
import { TEST_OFFERS } from '@/lib/hybrid-savings/__fixtures__/offers'
import { CLIENT_CACHE_TTL_MS } from '@/lib/cache-policy'

vi.mock('@/lib/api/infrastructure-offers', () => ({ fetchInfrastructureOffers: vi.fn() }))
let root: ReturnType<typeof createRoot>
let host: HTMLDivElement
beforeEach(() => {
  vi.resetModules(); vi.useFakeTimers(); vi.setSystemTime(new Date('2026-10-27T12:00:00Z'))
  vi.mocked(fetchInfrastructureOffers).mockReset().mockResolvedValue(TEST_OFFERS)
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  host = document.createElement('div'); document.body.append(host); root = createRoot(host)
})
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.useRealTimers() })

async function harness(enabled = true) {
  const { useInfrastructureOffers } = await import('./useInfrastructureOffers')
  let state!: ReturnType<typeof useInfrastructureOffers>
  function Harness({ active }: { active: boolean }) { state = useInfrastructureOffers(active); return null }
  await act(async () => root.render(<Harness active={enabled} />))
  return { get state() { return state }, render: async (active: boolean) => { await act(async () => root.render(<Harness active={active} />)) } }
}
it('does not fetch while costings are disabled', async () => {
  const view = await harness(false)
  expect(fetchInfrastructureOffers).not.toHaveBeenCalled()
  expect(view.state.rented).toEqual([])
  await view.render(true)
  expect(fetchInfrastructureOffers).toHaveBeenCalledTimes(1)
  expect(view.state.rented).toBe(TEST_OFFERS.rented)
})
it('keeps the last-good snapshot on refresh failure without calling it current', async () => {
  const view = await harness()
  vi.mocked(fetchInfrastructureOffers).mockRejectedValue(new Error('Provider offline'))
  await act(async () => vi.advanceTimersByTimeAsync(CLIENT_CACHE_TTL_MS.costings + 60_000))
  expect(view.state.rented).toBe(TEST_OFFERS.rented)
  expect(view.state.stale).toBe(true)
  expect(view.state.error).toBe('Provider offline')
})
it('refreshes prices automatically without clearing valid data', async () => {
  const view = await harness()
  const updated = { ...TEST_OFFERS, rented: TEST_OFFERS.rented.map(offer => ({ ...offer, hourlyCost: offer.hourlyCost * 2 })) }
  let complete!: (data: typeof updated) => void
  vi.mocked(fetchInfrastructureOffers).mockReturnValue(new Promise(resolve => { complete = resolve }))
  await act(async () => vi.advanceTimersByTimeAsync(CLIENT_CACHE_TTL_MS.costings))
  expect(view.state.rented).toBe(TEST_OFFERS.rented)
  expect(view.state.isLoading).toBe(false)
  await act(async () => complete(updated))
  expect(view.state.rented).toBe(updated.rented)
  expect(view.state.error).toBeNull()
})
it('reports initial failure without inventing any rental or purchase prices', async () => {
  vi.mocked(fetchInfrastructureOffers).mockRejectedValue(new Error('Upgrade required'))
  const view = await harness()
  expect(view.state.rented).toEqual([])
  expect(view.state.purchased).toEqual([])
  expect(view.state.error).toBe('Upgrade required')
})
it.each(['unavailable', 'stale', 'unconfigured'] as const)('retries %s source status within a minute', async state => {
  vi.mocked(fetchInfrastructureOffers).mockResolvedValueOnce({ ...TEST_OFFERS,
    sources: { aws: { state, last_success: null, message: 'Collection pending' } } })
  const view = await harness()
  await act(async () => vi.advanceTimersByTimeAsync(60_000))
  expect(fetchInfrastructureOffers).toHaveBeenCalledTimes(2)
  expect(view.state.sources).toEqual(TEST_OFFERS.sources)
})
it('retains an expired cached snapshot on failed re-enable without presenting it as current', async () => {
  const view = await harness()
  await view.render(false)
  expect(view.state.rented).toEqual([])
  await act(async () => vi.advanceTimersByTimeAsync(CLIENT_CACHE_TTL_MS.costings + 60_000))
  vi.mocked(fetchInfrastructureOffers).mockRejectedValue(new Error('Offline'))
  await view.render(true)
  expect(view.state.rented).toBe(TEST_OFFERS.rented)
  expect(view.state.purchased).toBe(TEST_OFFERS.purchased)
  expect(view.state.error).toBe('Offline')
  expect(view.state.stale).toBe(true)
})

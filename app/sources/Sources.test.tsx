// @vitest-environment happy-dom
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Sources from './Sources'
import { useSettings } from '@/contexts/SettingsContext'
import { useCostings } from '@/lib/hooks/useCostings'
import { CostAssumptionsProvider, COST_ASSUMPTIONS_STORAGE_KEY } from '@/contexts/CostAssumptionsContext'

vi.mock('@/contexts/SettingsContext', async importOriginal => ({
  ...await importOriginal<typeof import('@/contexts/SettingsContext')>(), useSettings: vi.fn(),
}))
vi.mock('@/lib/hooks/useCostings', () => ({ useCostings: vi.fn() }))

let root: ReturnType<typeof createRoot>
let host: HTMLDivElement
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  localStorage.clear()
  vi.mocked(useSettings).mockReturnValue({
    hydrated: true, defaultModel: '', testedModels: [], hfToken: '', backendVersion: '', inferenceBackend: 'vllm',
    costingsEnabled: false, preferredCloudProvider: null, pricingSource: 'merged',
    setDefaultModel: vi.fn(), setHfToken: vi.fn(), setInferenceBackend: vi.fn(), setBackendVersion: vi.fn(),
    setCostingsEnabled: vi.fn(), setPreferredCloudProvider: vi.fn(), setPricingSource: vi.fn(),
  })
  vi.mocked(useCostings).mockReturnValue({
    models: [], health: null, modelsUpdatedAt: null, modelsStale: false, isLoading: false, error: null,
    gpuCloudRates: new Map(), gpuHardwareCosts: new Map(),
  })
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}')))
  host = document.createElement('div'); document.body.append(host); root = createRoot(host)
})
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.unstubAllGlobals() })
async function render() { await act(async () => root.render(<CostAssumptionsProvider><Sources /></CostAssumptionsProvider>)) }
async function click(text: string) {
  const button = [...host.querySelectorAll('button')].find(element => element.textContent === text)!
  expect(button).toBeTruthy()
  await act(async () => button.click())
}

describe('Sources shared assumptions', () => {
  it('keeps the editor available without requesting live pricing when costings is disabled', async () => {
    await render()
    expect(host.textContent).toContain('Costings features are disabled')
    expect(host.textContent).not.toContain('Pricing data sources')
    expect(host.querySelectorAll('#cost-assumptions')).toHaveLength(1)
    expect(useCostings).toHaveBeenCalledWith(false, 'merged')
    expect(fetch).not.toHaveBeenCalled()
  })
  it('allows editing and resetting saved overrides with costings disabled', async () => {
    localStorage.setItem(COST_ASSUMPTIONS_STORAGE_KEY, JSON.stringify({ hardwareLifeYears: 6 }))
    await render(); await click('Purchased hardware')
    const field = host.querySelector<HTMLInputElement>('#cost-assumption-hardwareLifeYears')!
    expect(field.value).toBe('6')
    await act(async () => {
      field.focus()
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(field, '7')
      field.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await act(async () => field.blur())
    expect(JSON.parse(localStorage.getItem(COST_ASSUMPTIONS_STORAGE_KEY)!)).toEqual({ hardwareLifeYears: 7 })
    await click('Reset assumptions'); await click('Purchased hardware')
    expect(host.querySelector<HTMLInputElement>('#cost-assumption-hardwareLifeYears')!.value).toBe('4')
    expect(JSON.parse(localStorage.getItem(COST_ASSUMPTIONS_STORAGE_KEY)!)).toEqual({})
    expect(fetch).not.toHaveBeenCalled()
  })
  it('renders only one editor when live costings is enabled', async () => {
    vi.mocked(useSettings).mockReturnValue({ ...vi.mocked(useSettings)(), costingsEnabled: true })
    await render()
    expect(host.textContent).toContain('Pricing data sources')
    expect(host.querySelectorAll('#cost-assumptions')).toHaveLength(1)
    expect(useCostings).toHaveBeenCalledWith(true, 'merged')
  })
})

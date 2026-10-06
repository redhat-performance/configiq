// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ClusterCostPage from './page'
import { CostAssumptionsProvider, COST_ASSUMPTIONS_STORAGE_KEY } from '@/contexts/CostAssumptionsContext'

vi.mock('@/contexts/SettingsContext', () => ({ useSettings: () => ({ costingsEnabled: false, preferredCloudProvider: null, pricingSource: 'merged' }) }))
vi.mock('@/lib/hooks/useCostings', () => ({
  useCostings: () => ({ gpuHardwareCosts: new Map(), gpuCloudRates: new Map() }),
  resolveCloudRate: () => null,
}))
vi.mock('@/lib/pricing/providerPricing', () => ({
  fetchAllProviders: async () => [], getEffectiveRate: () => null,
  loadUserOverrides: () => ({}), setUserOverride: vi.fn(), clearUserOverride: vi.fn(),
  loadSelectedGpus: () => ({}), saveSelectedGpu: vi.fn(),
}))
let root: ReturnType<typeof createRoot>
let host: HTMLDivElement
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  localStorage.clear()
  host = document.createElement('div'); document.body.append(host); root = createRoot(host)
})
afterEach(async () => { await act(async () => root.unmount()); host.remove() })
function inputFor(label: string) {
  const row = [...host.querySelectorAll('[class*="rateRow"]')].find(row => row.querySelector('[class*="rateK"]')?.textContent === label)!
  return row.querySelector<HTMLInputElement>('input')!
}
async function render() { await act(async () => root.render(<CostAssumptionsProvider><ClusterCostPage /></CostAssumptionsProvider>)) }

describe('Cluster Cost shared planning integration', () => {
  it('keeps existing rates editable without shared overrides', async () => {
    await render()
    expect(inputFor('Depreciation').value).toBe('5')
    expect(inputFor('Power rate').value).toBe('0.1')
    expect(inputFor('PUE').value).toBe('1.25')
    expect(inputFor('Staff loaded').value).toBe('200000')
    expect(inputFor('Depreciation').readOnly).toBe(false)
  })
  it('uses saved Sources values with unit conversions and links back to their editor', async () => {
    localStorage.setItem(COST_ASSUMPTIONS_STORAGE_KEY, JSON.stringify({
      hardwareLifeYears: 6, electricityPerKwh: 0.2, pue: 1.5,
      annualMaintenancePct: 5, loadedMonthlyCostPerFte: 10000,
    }))
    await render()
    expect(inputFor('Depreciation').value).toBe('6')
    expect(inputFor('Power rate').value).toBe('0.2')
    expect(inputFor('PUE').value).toBe('1.5')
    expect(inputFor('Staff loaded').value).toBe('120000')
    expect(inputFor('Support/warranty').value).toBe('0.05')
    expect(inputFor('Depreciation').readOnly).toBe(true)
    expect(inputFor('Depreciation').parentElement!.querySelector('a')!.getAttribute('href')).toBe('/sources#cost-assumptions')
    expect(inputFor('Colo/rack').readOnly).toBe(false)
  })
})

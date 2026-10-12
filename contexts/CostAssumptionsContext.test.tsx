// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { CostAssumptionsProvider, COST_ASSUMPTIONS_STORAGE_KEY, useCostAssumptions } from './CostAssumptionsContext'

let root: ReturnType<typeof createRoot>
let host: HTMLDivElement
function Consumer() {
  const { assumptions, overrides, hydrated, storageError, updateAssumption, resetAssumptions } = useCostAssumptions()
  return <>
    <output>{JSON.stringify({ assumptions, overrides, hydrated, storageError })}</output>
    <button onClick={() => updateAssumption('hardwareLifeYears', 6)}>Update</button>
    <button onClick={resetAssumptions}>Reset</button>
  </>
}
const state = () => JSON.parse(host.querySelector('output')!.textContent!)
async function render() { await act(async () => root.render(<CostAssumptionsProvider><Consumer /></CostAssumptionsProvider>)) }
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  localStorage.clear()
  host = document.createElement('div'); document.body.append(host); root = createRoot(host)
})
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.restoreAllMocks() })

describe('Cost assumptions persistence', () => {
  it('hydrates valid persisted overrides without clobbering them with defaults', async () => {
    localStorage.setItem(COST_ASSUMPTIONS_STORAGE_KEY, JSON.stringify({ hardwareLifeYears: 8, pue: -1, analysisMonths: 60 }))
    await render()
    expect(state().hydrated).toBe(true)
    expect(state().assumptions).toMatchObject({ hardwareLifeYears: 8, pue: 1.4, analysisMonths: 60 })
    expect(JSON.parse(localStorage.getItem(COST_ASSUMPTIONS_STORAGE_KEY)!)).toEqual({ hardwareLifeYears: 8, analysisMonths: 60 })
  })
  it('persists updates and resets without touching unrelated settings', async () => {
    localStorage.setItem('settings_pricing_source', 'merged')
    await render(); await act(async () => host.querySelector('button')!.click())
    expect(state().assumptions.hardwareLifeYears).toBe(6)
    expect(JSON.parse(localStorage.getItem(COST_ASSUMPTIONS_STORAGE_KEY)!)).toEqual({ hardwareLifeYears: 6 })
    await act(async () => host.querySelectorAll('button')[1].click())
    expect(state().overrides).toEqual({}); expect(state().assumptions.hardwareLifeYears).toBe(4)
    expect(localStorage.getItem('settings_pricing_source')).toBe('merged')
  })
  it('synchronizes other tabs and cleared storage', async () => {
    await render()
    await act(async () => {
      localStorage.setItem(COST_ASSUMPTIONS_STORAGE_KEY, JSON.stringify({ hardwareLifeYears: 7 }))
      window.dispatchEvent(new StorageEvent('storage', { storageArea: localStorage, key: COST_ASSUMPTIONS_STORAGE_KEY }))
    })
    expect(state().assumptions.hardwareLifeYears).toBe(7)
    await act(async () => {
      localStorage.clear()
      window.dispatchEvent(new StorageEvent('storage', { storageArea: localStorage, key: null }))
    })
    expect(state().assumptions.hardwareLifeYears).toBe(4)
  })
  it('survives malformed data and unavailable storage', async () => {
    localStorage.setItem(COST_ASSUMPTIONS_STORAGE_KEY, '{bad')
    await render()
    expect(state().assumptions.hardwareLifeYears).toBe(4)
    expect(state().storageError).toBe(true)
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('storage disabled') })
    await act(async () => host.querySelector('button')!.click())
    expect(state().assumptions.hardwareLifeYears).toBe(6)
  })
})

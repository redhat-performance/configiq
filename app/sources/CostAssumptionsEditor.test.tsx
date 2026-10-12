// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import CostAssumptionsEditor from './CostAssumptionsEditor'
import { CostAssumptionsProvider, COST_ASSUMPTIONS_STORAGE_KEY } from '@/contexts/CostAssumptionsContext'
import { COST_ASSUMPTION_GROUPS } from '@/lib/costing-assumptions'

let root: ReturnType<typeof createRoot>
let host: HTMLDivElement
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  localStorage.clear()
  host = document.createElement('div'); document.body.append(host); root = createRoot(host)
})
afterEach(async () => { await act(async () => root.unmount()); host.remove() })
async function render() { await act(async () => root.render(<CostAssumptionsProvider><CostAssumptionsEditor /></CostAssumptionsProvider>)) }
async function click(text: string) {
  const button = [...host.querySelectorAll('button')].find(element => element.textContent === text)!
  await act(async () => button.click())
}
async function edit(key: string, value: string) {
  const field = host.querySelector<HTMLInputElement>(`#cost-assumption-${key}`)!
  await act(async () => {
    field.focus(); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(field, value)
    field.dispatchEvent(new Event('input', { bubbles: true }))
  })
  await act(async () => field.blur())
}

describe('Sources cost assumptions editor', () => {
  it('offers every moved field with accessible labels and collapsed groups', async () => {
    await render()
    expect(host.querySelector('#cost-assumptions')!.getAttribute('aria-labelledby')).toBe('cost-assumptions-heading')
    for (const group of COST_ASSUMPTION_GROUPS) {
      const button = [...host.querySelectorAll('button')].find(button => button.textContent === group.title)!
      expect(button.getAttribute('aria-expanded')).toBe('false')
      await click(group.title)
      for (const field of group.fields) {
        const input = host.querySelector<HTMLInputElement>(`#cost-assumption-${field.key}`)!
        expect(input).not.toBeNull()
        expect(host.querySelector(`label[for="${input.id}"]`)!.textContent).toContain(field.label)
      }
    }
    expect(host.querySelectorAll('input')).toHaveLength(COST_ASSUMPTION_GROUPS.flatMap(group => group.fields).length)
  })
  it('saves valid edits but keeps invalid or blank values out of calculations and storage', async () => {
    await render(); await click('Purchased hardware')
    await edit('hardwareLifeYears', '6')
    expect(JSON.parse(localStorage.getItem(COST_ASSUMPTIONS_STORAGE_KEY)!)).toEqual({ hardwareLifeYears: 6 })
    for (const invalid of ['0', '']) {
      await edit('hardwareLifeYears', invalid)
      expect(host.querySelector('#cost-assumption-hardwareLifeYears')!.getAttribute('aria-invalid')).toBe('true')
      expect(host.textContent).toContain('The previous value is still in use')
      expect(JSON.parse(localStorage.getItem(COST_ASSUMPTIONS_STORAGE_KEY)!)).toEqual({ hardwareLifeYears: 6 })
    }
    await edit('hardwareLifeYears', '4')
    expect(host.querySelector('#cost-assumption-hardwareLifeYears')!.getAttribute('aria-invalid')).toBe('false')
    await edit('hardwareResidualPct', '101')
    expect(host.querySelector('#cost-assumption-hardwareResidualPct')!.getAttribute('aria-invalid')).toBe('true')
  })
  it('does not apply default overrides on focus and resets saved values and invalid drafts', async () => {
    await render(); await click('Purchased hardware')
    const field = host.querySelector<HTMLInputElement>('#cost-assumption-hardwareLifeYears')!
    await act(async () => { field.focus(); field.blur() })
    expect(JSON.parse(localStorage.getItem(COST_ASSUMPTIONS_STORAGE_KEY)!)).toEqual({})
    await edit('hardwareLifeYears', '7'); await edit('pue', '0')
    await click('Reset assumptions'); await click('Purchased hardware')
    expect(JSON.parse(localStorage.getItem(COST_ASSUMPTIONS_STORAGE_KEY)!)).toEqual({})
    expect(host.querySelector<HTMLInputElement>('#cost-assumption-hardwareLifeYears')!.value).toBe('4')
    expect(host.querySelector<HTMLInputElement>('#cost-assumption-pue')!.value).toBe('1.4')
    expect(host.querySelector('[aria-invalid="true"]')).toBeNull()
  })
})

// @vitest-environment happy-dom
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { HybridCostCard } from './HybridCostCard'
import type { CostOption, InfrastructureCandidate } from '@/lib/hybrid-savings/calc'

const candidate: InfrastructureCandidate = {
  systemId: 'a100', label: 'A100', gpusPerReplica: 1, replicasNeeded: 1,
  clusterOutputTokensPerSecond: 1000, cloudRatePerGpuHour: 2,
  cloudHourlyCostPerInstance: 2, cloudProvider: 'CoreWeave', cloudRateKind: 'on_demand',
  cloudPriceSource: 'Cloud catalogue', cloudPriceSourceUrl: 'https://example.com/cloud',
  purchasePricePerReplica: 10000, purchaseInstallationPerReplica: 500,
  purchasePriceIndicative: true, purchasePriceSource: 'Hardware catalogue',
  purchasePriceSourceUrl: 'https://example.com/hardware', purchasePriceSourceDate: '2026-10-01',
  tdpWattsPerGpu: 300,
}
function option(key: CostOption['key']): CostOption {
  return {
    key, label: key, monthlyCost: 1467, fullyLoadedMonthlyCost: 1467, marginalMonthlyCost: 356,
    costPerMillionTokens: 0.59, costPerRequest: 0.002,
    candidate: key === 'hosted' ? null : candidate,
    gpuCount: 1, billedGpuCount: 1, replicas: 1, utilizationPct: 5,
    breakdown: [{ label: 'Implementation', monthlyCost: 1111, includedInMarginal: false },
      { label: 'Token usage', monthlyCost: 356, includedInMarginal: true }],
  }
}
let root: ReturnType<typeof createRoot>
let host: HTMLDivElement
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  host = document.createElement('div'); document.body.append(host); root = createRoot(host)
})
afterEach(async () => { await act(async () => root.unmount()); host.remove() })
async function click(label: string) {
  await act(async () => {
    const button = host.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`)!
    button.focus(); button.click()
  })
}

describe('Hybrid cost flip cards', () => {
  it.each(['hosted', 'rented', 'owned'] as const)('flips %s details independently, keeps all values and returns focus', async key => {
    await act(async () => root.render(<HybridCostCard title={key} option={option(key)} cheapest lens="fully-loaded" />))
    const front = host.querySelector<HTMLElement>('[class*="costFront"]')!
    const back = host.querySelector<HTMLElement>('[class*="costBack"]')!
    expect(front.inert).toBe(false)
    expect(back.inert).toBe(true)
    expect(back.getAttribute('aria-hidden')).toBe('true')
    expect(front.textContent).toContain('$1,467')
    await click(`View ${key} details`)
    expect(front.inert).toBe(true)
    expect(front.getAttribute('aria-hidden')).toBe('true')
    expect(back.inert).toBe(false)
    expect(back.getAttribute('aria-hidden')).toBe('false')
    expect(document.activeElement?.getAttribute('aria-label')).toBe(`Back to ${key} summary`)
    expect(back.textContent).toContain('Monthly cost breakdown')
    expect(back.textContent).toContain('Implementation')
    expect(back.textContent).toContain('$1,111')
    if (key !== 'hosted') {
      const link = back.querySelector('a')!
      expect(link.href).toBe(`https://example.com/${key === 'rented' ? 'cloud' : 'hardware'}`)
      link.addEventListener('click', event => event.preventDefault())
      await act(async () => link.click())
      expect(back.getAttribute('aria-hidden')).toBe('false')
      expect(back.textContent).toContain(key === 'rented' ? 'Instance rate' : 'Server price')
    }
    expect(back.querySelector('[role="region"]')).toBeNull()
    expect(back.textContent).not.toContain('Estimated latency')
    expect(front.textContent).toContain('↻ see details')
    expect(back.textContent).toContain('↻ flip back')
    await click(`Back to ${key} summary`)
    expect(front.inert).toBe(false)
    expect(back.inert).toBe(true)
    expect(document.activeElement?.getAttribute('aria-label')).toBe(`View ${key} details`)
  })

  it('keeps selectors independent, preserves their state and permits Escape from the back', async () => {
    const onChange = vi.fn()
    await act(async () => root.render(<HybridCostCard title="Hosted API" option={option('hosted')} cheapest lens="fully-loaded"
      selector={<select aria-label="Provider" defaultValue="first" onChange={onChange}><option value="first">First</option><option value="second">Second</option></select>} />))
    const select = host.querySelector('select')!
    await act(async () => { select.click(); select.value = 'second'; select.dispatchEvent(new Event('change', { bubbles: true })) })
    expect(onChange).toHaveBeenCalledOnce()
    // The surface button is a sibling, never an ancestor, of interactive controls.
    const surfaceButton = host.querySelector('button')!
    expect(surfaceButton.contains(select)).toBe(false)
    expect(surfaceButton.getAttribute('aria-controls')).toBe(host.querySelector('[class*="costBack"]')!.id)
    expect(host.querySelector('[class*="costBack"]')?.getAttribute('aria-hidden')).toBe('true')
    await click('View Hosted API details')
    await act(async () => document.activeElement?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })))
    expect(select.value).toBe('second')
    expect(host.querySelector('[class*="costBack"]')?.getAttribute('aria-hidden')).toBe('true')
    expect(host.querySelector('button select')).toBeNull()
  })

  it('shows only included, nonzero cost lines in the selected view, without altering the engine total', async () => {
    const marginal = { ...option('owned'), monthlyCost: 356, breakdown: [
      ...option('owned').breakdown,
      { label: 'Other owned-platform costs', monthlyCost: 0, includedInMarginal: true },
    ] }
    await act(async () => root.render(<HybridCostCard title="Purchased hardware" option={marginal} cheapest lens="marginal" />))
    await click('View Purchased hardware details')
    const back = host.querySelector('[class*="costBack"]')!
    expect(back.textContent).toContain('Monthly cost breakdown · Marginal')
    expect(back.textContent).not.toContain('Implementation')
    expect(back.textContent).not.toContain('Other platform costs')
    expect(back.querySelectorAll('li')).toHaveLength(1)
    expect(back.querySelector('[class*="costBackTotal"]')!.textContent).toBe('Total / month$356')
    expect(marginal.breakdown).toHaveLength(3)
  })

  it('does not offer a flip for an unavailable option', async () => {
    await act(async () => root.render(<HybridCostCard title="Purchased hardware" cheapest={false} lens="fully-loaded" unavailable="No complete price" />))
    expect(host.textContent).toContain('No complete price')
    expect(host.querySelector('button')).toBeNull()
    expect(host.querySelector('[class*="costBack"]')).toBeNull()
  })

  it('keeps all three cards independent and stable through repeated flips', async () => {
    const keys = ['hosted', 'rented', 'owned'] as const
    await act(async () => root.render(<>{keys.map(key => <HybridCostCard key={key} title={key}
      option={option(key)} cheapest={key === 'hosted'} lens="fully-loaded"
      selector={<select aria-label={`${key} selection`} defaultValue="second">
        <option value="first">First</option><option value="second">Second</option>
      </select>} />)}</>))
    const cards = Array.from(host.querySelectorAll<HTMLElement>('[class*="costFlip"]'))
      .filter(card => card.querySelector('[class*="costFront"]'))
    expect(cards).toHaveLength(3)
    for (const [index, key] of keys.entries()) {
      for (let cycle = 0; cycle < 2; cycle++) {
        await click(`View ${key} details`)
        cards.forEach((card, cardIndex) => {
          expect(card.querySelector('[class*="costBack"]')?.getAttribute('aria-hidden'))
            .toBe(cardIndex === index ? 'false' : 'true')
        })
        await click(`Back to ${key} summary`)
        expect(cards[index].querySelector<HTMLSelectElement>('select')?.value).toBe('second')
        expect(cards[index].querySelector('[class*="costFront"]')?.textContent).toContain('$1,467')
        expect(document.activeElement?.getAttribute('aria-label')).toBe(`View ${key} details`)
      }
    }
  })
})

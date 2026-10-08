// @vitest-environment happy-dom
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { AppliedCostAssumptions } from './AppliedCostAssumptions'
import { COST_ASSUMPTION_GROUPS, DEFAULT_COST_ASSUMPTIONS } from '@/lib/costing-assumptions'
import { bestOwnedAtVolume, bestRentedAtVolume, type CostAssumptions, type InfrastructureCandidate } from '@/lib/hybrid-savings/calc'

const workload = { monthlyInputTokens: 20_000_000, monthlyOutputTokens: 5_000_000, averageInputTokens: 4000, averageOutputTokens: 1000, activeHoursPerMonth: 40, peakToAverage: 1 }
const assumptions: CostAssumptions = { ...DEFAULT_COST_ASSUMPTIONS, costLens: 'fully-loaded', cloudBillingMode: 'scale-to-zero' }
const candidate: InfrastructureCandidate = {
  systemId: 'test', label: 'Test server', gpusPerReplica: 1, replicasNeeded: 1,
  clusterOutputTokensPerSecond: 1000, cloudRatePerGpuHour: 2, cloudHourlyCostPerInstance: 16,
  cloudGpusPerInstance: 8, cloudInstanceName: 'Test 8-GPU instance', cloudProvider: 'test',
  cloudRateKind: 'on_demand', cloudDirectInfrastructureMonthly: 500,
  purchaseGpusPerServer: 4, purchasePricePerReplica: 35000, purchaseInstallationPerReplica: 2500,
  purchasePriceIndicative: true, purchasePriceSource: 'Test catalogue', purchasePriceSourceUrl: null,
  purchasePriceSourceDate: null, tdpWattsPerGpu: 350,
}
let root: ReturnType<typeof createRoot>
let host: HTMLDivElement
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  host = document.createElement('div'); document.body.append(host); root = createRoot(host)
})
afterEach(async () => { await act(async () => root.unmount()); host.remove() })
async function render(inputs = assumptions, hardware: InfrastructureCandidate | null = candidate) {
  await act(async () => root.render(<AppliedCostAssumptions assumptions={inputs} workload={workload}
    rented={hardware ? bestRentedAtVolume(workload, [hardware], inputs, 25_000_000) ?? undefined : undefined}
    owned={hardware ? bestOwnedAtVolume(workload, [hardware], inputs, 25_000_000) ?? undefined : undefined} />))
}
function value(label: string) { return [...host.querySelectorAll('dt')].find(term => term.textContent === label)?.nextElementSibling?.textContent }

describe('Applied cost inputs', () => {
  it('lists every shared editable input once with units, zero values and no editors', async () => {
    await render()
    for (const field of COST_ASSUMPTION_GROUPS.flatMap(group => group.fields)) {
      expect(host.querySelectorAll(`[data-assumption="${field.key}"]`)).toHaveLength(1)
    }
    expect(value('Always-on month')).toBe('730 hours')
    expect(value('Processing window')).toBe('40 hours/month')
    expect(value('Fixed fees per month')).toBe('$0.00 / month')
    expect(value('Planning capacity')).toBe('90%')
    expect([...host.querySelectorAll('h3')].map(heading => heading.textContent)).toEqual(['Comparison basis', 'Cost assumptions'])
    expect([...host.querySelectorAll('h4')].map(heading => heading.textContent)).toEqual(['Hosted API', 'Rented infrastructure', 'Purchased hardware'])
    expect(host.querySelector('input, select, button')).toBeNull()
  })
  it('shows effective catalogue inputs without repeating the selected prices on the flip cards', async () => {
    await render()
    expect(value('Supporting infrastructure')).toContain('$500.00 / monthCatalogue allowance')
    expect(value('Supporting infrastructure')).not.toContain('1,800')
    expect(value('Compute rate')).toBeUndefined()
    expect(value('Complete server price')).toBeUndefined()
    expect(value('Catalogue installation / server')).toContain('$2,500.00')
    await render(assumptions, { ...candidate, cloudDirectInfrastructureMonthly: 0 })
    expect(value('Supporting infrastructure')).toContain('$0.00 / monthCatalogue allowance')
  })
  it('updates effective allowances and Sources inputs when selections change', async () => {
    await render({ ...assumptions, loadedMonthlyCostPerFte: 10000, ownedInstallationPerServer: 750 },
      { ...candidate, cloudDirectInfrastructureMonthly: null, purchaseInstallationPerReplica: 4200 })
    expect(value('Staffing rate / FTE')).toBe('$10,000.00 / month')
    expect(value('Additional installation / server')).toBe('$750.00')
    expect(value('Supporting infrastructure')).toContain('$1,800.00 / monthSources fallback.')
    expect(value('Catalogue installation / server')).toBe('$4,200.00')
  })
  it('distinguishes capacity-block billing and unused runtime buffer in marginal view', async () => {
    await render({ ...assumptions, costLens: 'marginal' }, { ...candidate, cloudRateKind: 'capacity_block' })
    expect(value('Billing mode')).toBe('Always onRequired by capacity block.')
    expect(value('Runtime buffer')).toContain('Not used in this billing mode.')
    expect(host.textContent).toContain('Marginal costs exclude acquisition')
  })
  it('does not claim hardware-dependent inputs are resolved before a valid calculation', async () => {
    await render(assumptions, null)
    expect(value('Catalogue installation / server')).toBe('Available after sizing')
    expect(value('GPU power / GPU')).toBe('Available after sizing')
    expect(value('Supporting infrastructure')).toContain('Fallback; confirmed after sizing')
  })
  it('groups purchased inputs into hardware, power and facilities, then people and setup', async () => {
    await render()
    const purchased = host.querySelector('[aria-labelledby="hybrid-owned-assumptions-heading"]')!
    expect([...purchased.querySelectorAll('h5')].map(heading => heading.textContent)).toEqual(['Hardware', 'Power and facilities', 'People and setup'])
    expect(value('Electricity')).toBe('$0.12 / kWh')
    expect(value('Operations FTE')).toBe('0.05 FTE')
  })
})

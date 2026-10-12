import { describe, expect, it } from 'vitest'
import { COST_ASSUMPTION_GROUPS, DEFAULT_COST_ASSUMPTIONS, parseCostAssumptionOverrides } from './costing-assumptions'
import { applyClusterCostAssumptions } from './cluster-cost/shared-assumptions'

describe('Shared cost assumption validation', () => {
  it('preserves all original Hybrid numeric defaults and exposes every editable field', () => {
    const fields = COST_ASSUMPTION_GROUPS.flatMap(group => group.fields.map(field => field.key))
    expect(new Set(fields).size).toBe(fields.length)
    expect(fields.sort()).toEqual(Object.keys(DEFAULT_COST_ASSUMPTIONS).filter(key => key !== 'hoursPerMonth').sort())
    expect(parseCostAssumptionOverrides(DEFAULT_COST_ASSUMPTIONS)).toEqual(
      Object.fromEntries(Object.entries(DEFAULT_COST_ASSUMPTIONS).filter(([key]) => key !== 'hoursPerMonth')),
    )
    expect(DEFAULT_COST_ASSUMPTIONS).toMatchObject({
      analysisMonths: 36, hardwareLifeYears: 4, hardwareResidualPct: 20,
      loadedMonthlyCostPerFte: 18000, rentedOperationsFte: 0.2, ownedOperationsFte: 0.25,
      rentedImplementation: 40000, ownedImplementation: 50000, hoursPerMonth: 730,
    })
  })
  it('rejects invalid, non-finite, unknown and page-local values but preserves valid zeroes', () => {
    expect(parseCostAssumptionOverrides({
      hardwareLifeYears: 0, hardwareResidualPct: 101, pue: 0.5, planningCapacityUsePct: -1,
      hostedImplementation: Infinity, rentedImplementation: '4000', ownedImplementation: NaN,
      hostedOperationsFte: 0, electricityPerKwh: 0, cloudBillingMode: 'always-on', hoursPerMonth: 0,
      unknown: 10, costLens: 'marginal',
    })).toEqual({ hostedOperationsFte: 0, electricityPerKwh: 0 })
    for (const raw of [null, [], 'bad', 10]) expect(parseCostAssumptionOverrides(raw)).toEqual({})
  })
})

describe('Cluster Cost shared inputs', () => {
  const original = {
    cloud: { opsFix: 1000 },
    onprem: { deprYrs: 5, powerKwh: 0.1, pue: 1.25, supFrac: 0.18, staffAnnual: 200000, storHot: 150 },
  }
  it('retains existing page defaults until explicitly overridden', () => {
    expect(applyClusterCostAssumptions(original, {})).toEqual(original)
    expect(applyClusterCostAssumptions(original, { hostedImplementation: 0, rentedOperationsFte: 1 })).toEqual(original)
  })
  it('converts percentage and monthly staff units without mutating unrelated rates', () => {
    const rates = applyClusterCostAssumptions(original, {
      hardwareLifeYears: 4, electricityPerKwh: 0, pue: 1.4,
      annualMaintenancePct: 5, loadedMonthlyCostPerFte: 18000,
    })
    expect(rates.onprem).toEqual({ ...original.onprem, deprYrs: 4, powerKwh: 0, pue: 1.4, supFrac: 0.05, staffAnnual: 216000 })
    expect(rates.cloud).toEqual(original.cloud)
    expect(original.onprem.staffAnnual).toBe(200000)
  })
})

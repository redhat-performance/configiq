import type { CostAssumptionOverrides } from '@/lib/costing-assumptions'

/** Only same-unit, same-meaning fields are shared; existing tool defaults stay intact. */
export const CLUSTER_SHARED_ASSUMPTIONS = {
  deprYrs: 'hardwareLifeYears',
  powerKwh: 'electricityPerKwh',
  pue: 'pue',
  supFrac: 'annualMaintenancePct',
  staffAnnual: 'loadedMonthlyCostPerFte',
} as const

interface ClusterPlanningRates {
  deprYrs: number
  powerKwh: number
  pue: number
  supFrac: number
  staffAnnual: number
}

export function applyClusterCostAssumptions<T extends { onprem: ClusterPlanningRates }>(
  rates: T, overrides: CostAssumptionOverrides,
): T {
  return {
    ...rates,
    onprem: {
      ...rates.onprem,
      deprYrs: overrides.hardwareLifeYears ?? rates.onprem.deprYrs,
      powerKwh: overrides.electricityPerKwh ?? rates.onprem.powerKwh,
      pue: overrides.pue ?? rates.onprem.pue,
      supFrac: overrides.annualMaintenancePct === undefined ? rates.onprem.supFrac : overrides.annualMaintenancePct / 100,
      staffAnnual: overrides.loadedMonthlyCostPerFte === undefined ? rates.onprem.staffAnnual : overrides.loadedMonthlyCostPerFte * 12,
    },
  }
}

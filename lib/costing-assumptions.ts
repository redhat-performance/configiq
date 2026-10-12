/** Shared planning inputs. Workload billing policy and cost view remain page-specific. */
export const DEFAULT_COST_ASSUMPTIONS = {
  cloudRuntimeBufferPct: 10,
  planningCapacityUsePct: 90,
  hoursPerMonth: 730,
  analysisMonths: 36,
  loadedMonthlyCostPerFte: 18_000,
  hostedOperationsFte: 0.05,
  hostedImplementation: 15_000,
  rentedDirectInfrastructureMonthly: 1_800,
  rentedOperationsFte: 0.2,
  rentedImplementation: 40_000,
  hardwareLifeYears: 4,
  hardwareResidualPct: 20,
  annualCostOfCapitalPct: 8,
  annualMaintenancePct: 5,
  electricityPerKwh: 0.12,
  pue: 1.4,
  ownedBaseSystemPowerWattsPerServer: 450,
  ownedInstallationPerServer: 0,
  ownedFacilityMonthlyPerServer: 200,
  ownedDirectInfrastructureMonthly: 1_500,
  ownedOperationsFte: 0.25,
  ownedImplementation: 50_000,
  hostedFixedMonthly: 0,
  rentedFixedMonthly: 0,
  ownedFixedMonthly: 0,
}

export type PlanningCostAssumptions = typeof DEFAULT_COST_ASSUMPTIONS
export type CostAssumptionKey = keyof PlanningCostAssumptions
export type CostAssumptionOverrides = Partial<PlanningCostAssumptions>

export interface CostAssumptionField {
  key: CostAssumptionKey
  label: string
  min?: number
  max?: number
  step?: number
}

export const COST_ASSUMPTION_GROUPS: { title: string; description: string; fields: CostAssumptionField[] }[] = [
  {
    title: 'Shared planning inputs',
    description: 'Explicit overrides for hardware life, electricity, PUE, maintenance and staff cost also apply to Cluster Cost. Other inputs apply to Hybrid Savings; tools keep their existing defaults until overridden.',
    fields: [
      { key: 'planningCapacityUsePct', label: 'Planning capacity use (%)', min: 1, max: 100 },
      { key: 'analysisMonths', label: 'Analysis period (months)', min: 1 },
      { key: 'loadedMonthlyCostPerFte', label: 'Loaded cost per FTE-month ($)' },
    ],
  },
  {
    title: 'Hosted API',
    description: 'Vendor usage plus internal adoption and management costs.',
    fields: [
      { key: 'hostedFixedMonthly', label: 'Fixed fees per month ($)' },
      { key: 'hostedOperationsFte', label: 'Operations FTE', step: 0.01 },
      { key: 'hostedImplementation', label: 'One-time implementation ($)' },
    ],
  },
  {
    title: 'Rented infrastructure',
    description: 'Cloud compute, supporting infrastructure and model operations.',
    fields: [
      { key: 'cloudRuntimeBufferPct', label: 'Scale-to-zero runtime buffer (%)' },
      { key: 'rentedDirectInfrastructureMonthly', label: 'Fallback infrastructure per month ($)' },
      { key: 'rentedOperationsFte', label: 'Operations FTE', step: 0.01 },
      { key: 'rentedImplementation', label: 'One-time implementation ($)' },
      { key: 'rentedFixedMonthly', label: 'Other monthly costs ($)' },
    ],
  },
  {
    title: 'Purchased hardware',
    description: 'Capital, power, facilities, implementation and platform operations. Catalogue hardware prices are unchanged.',
    fields: [
      { key: 'hardwareLifeYears', label: 'Hardware life (years)', min: 1 },
      { key: 'hardwareResidualPct', label: 'Residual value (%)', max: 100 },
      { key: 'annualCostOfCapitalPct', label: 'Annual cost of capital (%)' },
      { key: 'annualMaintenancePct', label: 'Annual maintenance (%)' },
      { key: 'electricityPerKwh', label: 'Electricity ($ / kWh)', step: 0.01 },
      { key: 'pue', label: 'Power usage effectiveness', min: 1, step: 0.1 },
      { key: 'ownedBaseSystemPowerWattsPerServer', label: 'Base server power per server (W)' },
      { key: 'ownedInstallationPerServer', label: 'Additional installation per server ($)' },
      { key: 'ownedFacilityMonthlyPerServer', label: 'Facilities per server / month ($)' },
      { key: 'ownedDirectInfrastructureMonthly', label: 'Shared infrastructure and platform / month ($)' },
      { key: 'ownedOperationsFte', label: 'Operations FTE', step: 0.01 },
      { key: 'ownedImplementation', label: 'One-time implementation ($)' },
      { key: 'ownedFixedMonthly', label: 'Other monthly costs ($)' },
    ],
  },
]

const fields = COST_ASSUMPTION_GROUPS.flatMap(group => group.fields)

export function isValidCostAssumption(key: CostAssumptionKey, value: unknown): value is number {
  const field = fields.find(field => field.key === key)
  // hoursPerMonth is a fixed engine convention, not an editable shared assumption.
  return !!field && typeof value === 'number' && Number.isFinite(value) &&
    value >= (field.min ?? 0) && (field.max === undefined || value <= field.max)
}

/** Treat browser storage as untrusted: discard unknown, obsolete and invalid fields. */
export function parseCostAssumptionOverrides(raw: unknown): CostAssumptionOverrides {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return {}
  const values = raw as Record<string, unknown>
  const valid: CostAssumptionOverrides = {}
  for (const { key } of fields) {
    if (Object.hasOwn(values, key) && isValidCostAssumption(key, values[key])) valid[key] = values[key]
  }
  return valid
}

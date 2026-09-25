import { firstWinningVolume, TransitionSearchLimitError, type CostCurve } from './transition-search'

export type CloudBillingMode = 'scale-to-zero' | 'active-window' | 'always-on'
export type CostLens = 'fully-loaded' | 'marginal'

export interface HybridWorkload {
  monthlyInputTokens: number
  monthlyOutputTokens: number
  averageInputTokens: number
  averageOutputTokens: number
  activeHoursPerMonth: number
  peakToAverage: number
}

export interface HostedPrice {
  modelId: string
  label: string
  inputPerMillion: number
  outputPerMillion: number
}

export interface InfrastructureCandidate {
  systemId: string
  label: string
  gpusPerReplica: number
  replicasNeeded: number
  clusterOutputTokensPerSecond: number
  /** Whole-instance list price. Preferred over the compatibility GPU rate. */
  cloudHourlyCostPerInstance?: number | null
  /** Offer-specific support allowance; falls back to the global assumption. */
  cloudDirectInfrastructureMonthly?: number | null
  cloudRatePerGpuHour: number | null
  cloudGpusPerInstance?: number | null
  cloudInstanceName?: string | null
  cloudProvider: string | null
  cloudProviderRegion?: string | null
  cloudRateKind: 'on_demand' | 'spot' | 'capacity_block' | null
  cloudMaxInstancesPerReplica?: number | null
  cloudInterconnect?: string | null
  cloudPriceSource?: string | null
  cloudPriceSourceUrl?: string | null
  cloudPriceSourceDate?: string | null
  purchaseGpusPerServer?: number | null
  purchasePricePerReplica: number | null
  purchaseInstallationPerReplica: number | null
  purchasePriceIndicative: boolean
  purchasePriceSource: string | null
  purchasePriceSourceUrl: string | null
  purchasePriceSourceDate: string | null
  tdpWattsPerGpu: number | null
  ttftMs: number
  tpotMs: number
  source: string
}

export interface CostAssumptions {
  costLens: CostLens
  cloudBillingMode: CloudBillingMode
  cloudRuntimeBufferPct: number
  /** Usable share of benchmarked throughput retained for planning headroom. */
  planningCapacityUsePct: number
  hoursPerMonth: number
  analysisMonths: number
  loadedMonthlyCostPerFte: number
  hostedOperationsFte: number
  hostedImplementation: number
  rentedDirectInfrastructureMonthly: number
  rentedOperationsFte: number
  rentedImplementation: number
  hardwareLifeYears: number
  hardwareResidualPct: number
  annualCostOfCapitalPct: number
  annualMaintenancePct: number
  electricityPerKwh: number
  pue: number
  ownedBaseSystemPowerWattsPerServer: number
  /** Additional user-entered installation cost above the catalogue estimate. */
  ownedInstallationPerServer: number
  ownedFacilityMonthlyPerServer: number
  ownedDirectInfrastructureMonthly: number
  ownedOperationsFte: number
  ownedImplementation: number
  hostedFixedMonthly: number
  rentedFixedMonthly: number
  ownedFixedMonthly: number
}

export interface CostBreakdownItem {
  label: string
  monthlyCost: number
  includedInMarginal: boolean
}

export interface CostOption {
  key: 'hosted' | 'rented' | 'owned'
  label: string
  monthlyCost: number
  fullyLoadedMonthlyCost: number
  marginalMonthlyCost: number
  costPerMillionTokens: number
  costPerRequest: number
  candidate: InfrastructureCandidate | null
  gpuCount: number | null
  billedGpuCount: number | null
  replicas: number | null
  utilizationPct: number | null
  breakdown: CostBreakdownItem[]
}

export interface CostPoint {
  tokens: number
  hosted: number | null
  rented: number | null
  owned: number | null
}

export interface HybridComparison {
  monthlyTokens: number
  monthlyRequests: number
  peakRequestsPerSecond: number
  options: CostOption[]
  cheapest: CostOption | null
  rentedBreakEvenTokens: number | null
  ownedBreakEvenTokens: number | null
  rentedLowestCostTokens: number | null
  ownedLowestCostTokens: number | null
  transitionsVerified: boolean
  chartMaximumTokens: number
  chartPoints: CostPoint[]
}

const MILLION = 1_000_000
const SECONDS_PER_HOUR = 3_600
export const HYBRID_PLANNING_HORIZON_TOKENS = 1_000_000_000_000
const DEFAULT_SEARCH_MAXIMUM = HYBRID_PLANNING_HORIZON_TOKENS
const MAX_REPLACEMENT_CYCLES = 100

function finiteNonNegative(value: number): number {
  return Number.isFinite(value) ? Math.max(value, 0) : 0
}

function positive(value: number, fallback = 1): number {
  return Number.isFinite(value) && value > 0 ? value : fallback
}

function optionCosts(
  breakdown: CostBreakdownItem[],
  lens: CostLens,
): Pick<CostOption, 'monthlyCost' | 'fullyLoadedMonthlyCost' | 'marginalMonthlyCost'> {
  const fullyLoadedMonthlyCost = breakdown.reduce((sum, item) => sum + item.monthlyCost, 0)
  const marginalMonthlyCost = breakdown.reduce(
    (sum, item) => sum + (item.includedInMarginal ? item.monthlyCost : 0),
    0,
  )
  return {
    monthlyCost: lens === 'fully-loaded' ? fullyLoadedMonthlyCost : marginalMonthlyCost,
    fullyLoadedMonthlyCost,
    marginalMonthlyCost,
  }
}

function workloadMix(workload: HybridWorkload) {
  const monthlyInputTokens = finiteNonNegative(workload.monthlyInputTokens)
  const monthlyOutputTokens = finiteNonNegative(workload.monthlyOutputTokens)
  const monthlyTokens = monthlyInputTokens + monthlyOutputTokens
  const averageInputTokens = positive(workload.averageInputTokens)
  const averageOutputTokens = positive(workload.averageOutputTokens)
  const averageTokensPerRequest = averageInputTokens + averageOutputTokens
  const inputShare = monthlyTokens > 0
    ? monthlyInputTokens / monthlyTokens
    : averageInputTokens / averageTokensPerRequest
  const outputShare = 1 - inputShare
  /*
   * Monthly input/output totals and a representative request shape are two
   * independent user inputs. When their ratios differ, dividing total tokens
   * by total tokens/request can claim fewer requests than are needed to carry
   * either the input or output volume. Use the larger side's implied request
   * count so the infrastructure comparison never silently drops work.
   */
  const requestsPerBilledToken = Math.max(
    inputShare / averageInputTokens,
    outputShare / averageOutputTokens,
  )

  return {
    monthlyInputTokens,
    monthlyOutputTokens,
    monthlyTokens,
    averageInputTokens,
    averageOutputTokens,
    averageTokensPerRequest,
    inputShare,
    outputShare,
    requestsPerBilledToken,
  }
}

function requestsAtVolume(workload: HybridWorkload, volume: number): number {
  return finiteNonNegative(volume) * workloadMix(workload).requestsPerBilledToken
}

export function workloadFacts(workload: HybridWorkload) {
  const mix = workloadMix(workload)
  const monthlyRequests = mix.monthlyTokens * mix.requestsPerBilledToken
  const peakRequestsPerSecond =
    (monthlyRequests /
      (positive(workload.activeHoursPerMonth) * SECONDS_PER_HOUR)) *
    positive(workload.peakToAverage)

  return {
    monthlyInputTokens: mix.monthlyInputTokens,
    monthlyOutputTokens: mix.monthlyOutputTokens,
    monthlyTokens: mix.monthlyTokens,
    averageTokensPerRequest: mix.averageTokensPerRequest,
    monthlyRequests,
    peakRequestsPerSecond,
  }
}

export function hostedCostAtVolume(
  workload: HybridWorkload,
  hostedPrice: HostedPrice,
  assumptions: CostAssumptions,
  volume: number,
): CostOption {
  const safeVolume = finiteNonNegative(volume)
  const inputShare = workloadMix(workload).inputShare
  const inputTokens = safeVolume * inputShare
  const outputTokens = safeVolume - inputTokens
  const fixed = finiteNonNegative(assumptions.hostedFixedMonthly)
  const operations =
    finiteNonNegative(assumptions.hostedOperationsFte) *
    finiteNonNegative(assumptions.loadedMonthlyCostPerFte)
  const implementation =
    finiteNonNegative(assumptions.hostedImplementation) /
    positive(assumptions.analysisMonths, 36)
  const breakdown: CostBreakdownItem[] = [
    { label: 'Input token usage', monthlyCost: inputTokens / MILLION * finiteNonNegative(hostedPrice.inputPerMillion), includedInMarginal: true },
    { label: 'Output token usage', monthlyCost: outputTokens / MILLION * finiteNonNegative(hostedPrice.outputPerMillion), includedInMarginal: true },
    { label: 'Fixed hosted fees', monthlyCost: fixed, includedInMarginal: true },
    { label: 'Operations and vendor management', monthlyCost: operations, includedInMarginal: false },
    { label: 'Implementation amortization', monthlyCost: implementation, includedInMarginal: false },
  ]
  const costs = optionCosts(breakdown, assumptions.costLens)
  const requests = requestsAtVolume(workload, safeVolume)

  return {
    key: 'hosted',
    label: hostedPrice.label,
    ...costs,
    costPerMillionTokens: safeVolume > 0 ? costs.monthlyCost / safeVolume * MILLION : 0,
    costPerRequest: requests > 0 ? costs.monthlyCost / requests : 0,
    candidate: null,
    gpuCount: null,
    billedGpuCount: null,
    replicas: null,
    utilizationPct: null,
    breakdown,
  }
}

export function candidateCapacityTokens(
  workload: HybridWorkload,
  candidate: InfrastructureCandidate,
  planningCapacityUsePct = 100,
): number {
  if (
    !Number.isFinite(candidate.clusterOutputTokensPerSecond) ||
    candidate.clusterOutputTokensPerSecond <= 0 ||
    !Number.isFinite(candidate.replicasNeeded) ||
    candidate.replicasNeeded <= 0 ||
    !Number.isFinite(candidate.gpusPerReplica) ||
    candidate.gpusPerReplica <= 0
  ) {
    return 0
  }

  const replicas = Math.max(Math.ceil(candidate.replicasNeeded), 1)
  const outputTokensPerSecondPerReplica =
    candidate.clusterOutputTokensPerSecond / replicas
  const mix = workloadMix(workload)
  const requestsPerSecondPerReplica =
    outputTokensPerSecondPerReplica / mix.averageOutputTokens
  const monthlyRequestCapacity =
    requestsPerSecondPerReplica *
    Math.min(positive(planningCapacityUsePct, 100), 100) / 100 *
    positive(workload.activeHoursPerMonth) *
    SECONDS_PER_HOUR /
    positive(workload.peakToAverage)

  return monthlyRequestCapacity / positive(mix.requestsPerBilledToken)
}

function deploymentForVolume(
  workload: HybridWorkload,
  candidate: InfrastructureCandidate,
  assumptions: CostAssumptions,
  volume: number,
) {
  const safeVolume = finiteNonNegative(volume)
  const capacityPerReplica = candidateCapacityTokens(
    workload,
    candidate,
    assumptions.planningCapacityUsePct,
  )
  const replicas = safeVolume > 0
    ? Math.max(Math.ceil(safeVolume / positive(capacityPerReplica)), 1)
    : 0
  const gpusPerReplica = Math.max(Math.ceil(positive(candidate.gpusPerReplica)), 1)
  return {
    capacityPerReplica,
    replicas,
    gpuCount: replicas * gpusPerReplica,
    gpusPerReplica,
  }
}

function rentedInstanceLayout(
  candidate: InfrastructureCandidate,
  replicas: number,
): {
  billedInstances: number
  billedGpuCount: number
  replicasPerInstance: number
  instancesPerReplica: number
} | null {
  const instanceGpuCount = Math.max(
    Math.ceil(positive(candidate.cloudGpusPerInstance ?? 1)),
    1,
  )
  const gpusPerReplica = Math.max(Math.ceil(positive(candidate.gpusPerReplica)), 1)

  if (gpusPerReplica <= instanceGpuCount) {
    const replicasPerInstance = Math.max(Math.floor(instanceGpuCount / gpusPerReplica), 1)
    const billedInstances = replicas > 0 ? Math.ceil(replicas / replicasPerInstance) : 0
    return {
      billedInstances,
      billedGpuCount: billedInstances * instanceGpuCount,
      replicasPerInstance,
      instancesPerReplica: 1,
    }
  }

  const instancesPerReplica = Math.ceil(gpusPerReplica / instanceGpuCount)
  const supportedMaximum = candidate.cloudMaxInstancesPerReplica ?? 1
  if (instancesPerReplica > supportedMaximum) return null
  const billedInstances = replicas * instancesPerReplica
  return {
    billedInstances,
    billedGpuCount: billedInstances * instanceGpuCount,
    replicasPerInstance: 1,
    instancesPerReplica,
  }
}

function rentedCostForCandidate(
  workload: HybridWorkload,
  candidate: InfrastructureCandidate,
  assumptions: CostAssumptions,
  volume: number,
): CostOption | null {
  if (candidateCapacityTokens(workload, candidate, assumptions.planningCapacityUsePct) <= 0) {
    return null
  }
  const cloudGpusPerInstance = Math.max(
    Math.ceil(positive(candidate.cloudGpusPerInstance ?? 1)),
    1,
  )
  const hourlyInstanceCost = candidate.cloudHourlyCostPerInstance ?? (
    candidate.cloudRatePerGpuHour == null
      ? null
      : candidate.cloudRatePerGpuHour * cloudGpusPerInstance
  )
  if (hourlyInstanceCost == null || hourlyInstanceCost <= 0) {
    return null
  }

  const deployment = deploymentForVolume(workload, candidate, assumptions, volume)
  const safeVolume = finiteNonNegative(volume)
  const layout = rentedInstanceLayout(candidate, deployment.replicas)
  if (!layout) return null
  let billableInstanceHours = 0
  const effectiveBillingMode: CloudBillingMode = candidate.cloudRateKind === 'capacity_block'
    ? 'always-on'
    : assumptions.cloudBillingMode

  if (safeVolume > 0 && effectiveBillingMode === 'scale-to-zero') {
    const equivalentOutputTokens =
      requestsAtVolume(workload, safeVolume) * positive(workload.averageOutputTokens)
    const outputTokensPerSecondPerReplica =
      positive(candidate.clusterOutputTokensPerSecond) /
      Math.max(Math.ceil(positive(candidate.replicasNeeded)), 1) *
      Math.min(positive(assumptions.planningCapacityUsePct, 100), 100) / 100
    const replicaHours = equivalentOutputTokens /
      outputTokensPerSecondPerReplica /
      SECONDS_PER_HOUR
    /*
     * Use the continuous replica load rather than the rounded replica count
     * when crediting parallel work inside a whole cloud instance. Dividing by
     * ceil(load) made billable runtime fall every time another replica became
     * necessary, producing a saw-tooth curve where more demand could cost
     * less. The continuous load keeps sparse traffic on one replica, credits
     * genuinely usable parallelism as demand grows, and makes each offer's
     * scale-to-zero cost monotonic.
     */
    const continuousReplicaLoad = safeVolume / positive(deployment.capacityPerReplica)
    // capacityPerReplica is peak-adjusted, so continuousReplicaLoad is the
    // number of replicas needed at the workload peak. Instance runtime is
    // driven by average work, not by assuming that peak parallelism persists
    // for the entire month. Divide the peak load by the peak-to-average ratio
    // before crediting simultaneous replicas inside one whole instance.
    const averageReplicaLoad = continuousReplicaLoad / positive(workload.peakToAverage)
    const parallelReplicasPerInstance = Math.min(
      layout.replicasPerInstance,
      Math.max(averageReplicaLoad, 1),
    )
    billableInstanceHours =
      replicaHours /
      parallelReplicasPerInstance *
      layout.instancesPerReplica *
      (1 + finiteNonNegative(assumptions.cloudRuntimeBufferPct) / 100)
  } else if (safeVolume > 0 && effectiveBillingMode === 'active-window') {
    billableInstanceHours =
      layout.billedInstances * positive(workload.activeHoursPerMonth)
  } else if (safeVolume > 0) {
    billableInstanceHours =
      layout.billedInstances * positive(assumptions.hoursPerMonth, 730)
  }

  const compute = billableInstanceHours * hourlyInstanceCost
  const fixed = finiteNonNegative(assumptions.rentedFixedMonthly)
  const directInfrastructure = finiteNonNegative(
    candidate.cloudDirectInfrastructureMonthly ?? assumptions.rentedDirectInfrastructureMonthly,
  )
  const operations =
    finiteNonNegative(assumptions.rentedOperationsFte) *
    finiteNonNegative(assumptions.loadedMonthlyCostPerFte)
  const implementation =
    finiteNonNegative(assumptions.rentedImplementation) /
    positive(assumptions.analysisMonths, 36)
  const breakdown: CostBreakdownItem[] = [
    { label: 'Cloud GPU compute', monthlyCost: compute, includedInMarginal: true },
    { label: 'Infrastructure and observability', monthlyCost: directInfrastructure, includedInMarginal: true },
    { label: 'Other rented-platform costs', monthlyCost: fixed, includedInMarginal: true },
    { label: 'Platform and model operations', monthlyCost: operations, includedInMarginal: false },
    { label: 'Implementation amortization', monthlyCost: implementation, includedInMarginal: false },
  ]
  const costs = optionCosts(breakdown, assumptions.costLens)
  const requests = requestsAtVolume(workload, safeVolume)

  return {
    key: 'rented',
    label: candidate.label,
    ...costs,
    costPerMillionTokens: safeVolume > 0 ? costs.monthlyCost / safeVolume * MILLION : 0,
    costPerRequest: requests > 0 ? costs.monthlyCost / requests : 0,
    candidate,
    gpuCount: deployment.gpuCount,
    billedGpuCount: layout.billedGpuCount,
    replicas: deployment.replicas,
    utilizationPct: deployment.capacityPerReplica > 0 && deployment.replicas > 0
      ? Math.min(safeVolume / (deployment.capacityPerReplica * deployment.replicas) * 100, 100)
      : null,
    breakdown,
  }
}

function ownedCostForCandidate(
  workload: HybridWorkload,
  candidate: InfrastructureCandidate,
  assumptions: CostAssumptions,
  volume: number,
): CostOption | null {
  if (candidate.purchasePricePerReplica == null || candidate.purchasePricePerReplica <= 0) {
    return null
  }
  if (candidateCapacityTokens(workload, candidate, assumptions.planningCapacityUsePct) <= 0) {
    return null
  }

  const deployment = deploymentForVolume(workload, candidate, assumptions, volume)
  const safeVolume = finiteNonNegative(volume)
  const purchaseGpusPerServer = Math.max(
    Math.ceil(positive(candidate.purchaseGpusPerServer ?? deployment.gpusPerReplica)),
    1,
  )
  if (deployment.gpusPerReplica > purchaseGpusPerServer) return null
  const replicasPerServer = Math.max(
    Math.floor(purchaseGpusPerServer / deployment.gpusPerReplica),
    1,
  )
  const serverCount = deployment.replicas > 0
    ? Math.ceil(deployment.replicas / replicasPerServer)
    : 0
  const billedGpuCount = serverCount * purchaseGpusPerServer
  const acquisition = serverCount * candidate.purchasePricePerReplica
  const lifeMonths = positive(assumptions.hardwareLifeYears, 4) * 12
  const analysisMonths = positive(assumptions.analysisMonths, 36)
  const installationPerServer =
    finiteNonNegative(candidate.purchaseInstallationPerReplica ?? 0) +
    finiteNonNegative(assumptions.ownedInstallationPerServer)
  const installationAcquisition = serverCount * installationPerServer
  const residualValue = acquisition * Math.min(finiteNonNegative(assumptions.hardwareResidualPct), 100) / 100
  const annualCostOfCapital = finiteNonNegative(assumptions.annualCostOfCapitalPct) / 100

  // Match the standalone estimator's lifecycle treatment. This matters when
  // the analysis period is shorter than the useful life and when replacement
  // purchases are needed during a longer analysis.
  const completedCycles = Math.min(
    Math.max(Math.ceil(analysisMonths / lifeMonths) - 1, 0),
    MAX_REPLACEMENT_CYCLES,
  )
  let elapsedMonths = completedCycles * lifeMonths
  let depreciationConsumed = completedCycles * (acquisition - residualValue)
  let installationConsumed = completedCycles * installationAcquisition
  let capitalChargeTotal = completedCycles *
    ((acquisition + installationAcquisition + residualValue) / 2) *
    annualCostOfCapital *
    (lifeMonths / 12)

  if (elapsedMonths < analysisMonths) {
    const heldMonths = Math.min(lifeMonths, analysisMonths - elapsedMonths)
    const heldFraction = heldMonths / lifeMonths
    const endingHardwareBookValue = acquisition - (acquisition - residualValue) * heldFraction
    const endingInstallationBookValue = installationAcquisition * (1 - heldFraction)
    const openingBookValue = acquisition + installationAcquisition
    const endingBookValue = endingHardwareBookValue + endingInstallationBookValue

    depreciationConsumed += acquisition - endingHardwareBookValue
    installationConsumed += installationAcquisition - endingInstallationBookValue
    capitalChargeTotal +=
      ((openingBookValue + endingBookValue) / 2) *
      annualCostOfCapital *
      (heldMonths / 12)
    elapsedMonths += heldMonths
  }

  const amortization = depreciationConsumed / analysisMonths
  const installation = installationConsumed / analysisMonths
  const capitalCharge = capitalChargeTotal / analysisMonths
  const maintenance =
    acquisition * finiteNonNegative(assumptions.annualMaintenancePct) / 100 / 12
  const energy =
    (serverCount * finiteNonNegative(assumptions.ownedBaseSystemPowerWattsPerServer) +
      // The acquisition price and capacity represent the complete installed
      // server. Charge every installed GPU consistently instead of assigning
      // zero power to spare GPUs that are still part of that server. TDP is a
      // planning proxy; a production case should replace it with metered draw.
      billedGpuCount * finiteNonNegative(candidate.tdpWattsPerGpu ?? 0)) /
    1_000 *
    positive(assumptions.hoursPerMonth, 730) *
    positive(assumptions.pue, 1) *
    finiteNonNegative(assumptions.electricityPerKwh)
  const facilities =
    serverCount * finiteNonNegative(assumptions.ownedFacilityMonthlyPerServer) +
    finiteNonNegative(assumptions.ownedDirectInfrastructureMonthly)
  const operations =
    finiteNonNegative(assumptions.ownedOperationsFte) *
    finiteNonNegative(assumptions.loadedMonthlyCostPerFte)
  const implementation =
    finiteNonNegative(assumptions.ownedImplementation) /
    positive(assumptions.analysisMonths, 36)
  const fixed = finiteNonNegative(assumptions.ownedFixedMonthly)
  const breakdown: CostBreakdownItem[] = [
    { label: 'Hardware depreciation net of residual', monthlyCost: amortization, includedInMarginal: false },
    { label: 'Cost of capital', monthlyCost: capitalCharge, includedInMarginal: false },
    { label: 'Installation and commissioning', monthlyCost: installation, includedInMarginal: false },
    { label: 'Maintenance and spares', monthlyCost: maintenance, includedInMarginal: true },
    { label: 'Power including PUE', monthlyCost: energy, includedInMarginal: true },
    { label: 'Rack, facilities and infrastructure', monthlyCost: facilities, includedInMarginal: true },
    { label: 'Other owned-platform costs', monthlyCost: fixed, includedInMarginal: true },
    { label: 'Platform and model operations', monthlyCost: operations, includedInMarginal: false },
    { label: 'Implementation amortization', monthlyCost: implementation, includedInMarginal: false },
  ]
  const costs = optionCosts(breakdown, assumptions.costLens)
  const requests = requestsAtVolume(workload, safeVolume)

  return {
    key: 'owned',
    label: candidate.label,
    ...costs,
    costPerMillionTokens: safeVolume > 0 ? costs.monthlyCost / safeVolume * MILLION : 0,
    costPerRequest: requests > 0 ? costs.monthlyCost / requests : 0,
    candidate,
    gpuCount: deployment.gpuCount,
    billedGpuCount,
    replicas: deployment.replicas,
    utilizationPct: deployment.capacityPerReplica > 0 && serverCount > 0
      ? Math.min(safeVolume / (deployment.capacityPerReplica * serverCount * replicasPerServer) * 100, 100)
      : null,
    breakdown,
  }
}

function lowestCostOption(options: Array<CostOption | null>): CostOption | null {
  return options
    .filter((option): option is CostOption => option !== null)
    .sort((left, right) => left.monthlyCost - right.monthlyCost)[0] ?? null
}

export function bestRentedAtVolume(
  workload: HybridWorkload,
  candidates: InfrastructureCandidate[],
  assumptions: CostAssumptions,
  volume: number,
): CostOption | null {
  return lowestCostOption(
    candidates.map(candidate => rentedCostForCandidate(workload, candidate, assumptions, volume)),
  )
}

export function bestOwnedAtVolume(
  workload: HybridWorkload,
  candidates: InfrastructureCandidate[],
  assumptions: CostAssumptions,
  volume: number,
): CostOption | null {
  return lowestCostOption(
    candidates.map(candidate => ownedCostForCandidate(workload, candidate, assumptions, volume)),
  )
}

function candidateCurves(
  workload: HybridWorkload,
  candidates: InfrastructureCandidate[],
  assumptions: CostAssumptions,
): { rented: CostCurve[]; owned: CostCurve[] } {
  const rented: CostCurve[] = []
  const owned: CostCurve[] = []
  for (const candidate of candidates) {
    const replicaCapacity = candidateCapacityTokens(workload, candidate, assumptions.planningCapacityUsePct)
    if (!Number.isFinite(replicaCapacity) || replicaCapacity <= 0) continue

    const rentedAtZero = rentedCostForCandidate(workload, candidate, assumptions, 0)
    if (rentedAtZero) {
      const layout = rentedInstanceLayout(candidate, 1)
      if (layout) {
        const billingMode = candidate.cloudRateKind === 'capacity_block'
          ? 'always-on' : assumptions.cloudBillingMode
        if (billingMode === 'scale-to-zero') {
          const firstKnot = replicaCapacity * positive(workload.peakToAverage)
          const sample = Math.min(1, firstKnot / 2)
          const sampled = rentedCostForCandidate(workload, candidate, assumptions, sample)
          if (sample > 0 && sampled) rented.push({
            kind: 'scale-to-zero',
            fixed: rentedAtZero.monthlyCost,
            rate: (sampled.monthlyCost - rentedAtZero.monthlyCost) / sample,
            firstKnot,
            parallelism: layout.replicasPerInstance,
          })
        } else {
          const capacity = replicaCapacity * layout.replicasPerInstance
          const sample = Math.min(1, capacity / 2)
          const sampled = rentedCostForCandidate(workload, candidate, assumptions, sample)
          if (sample > 0 && sampled) rented.push({
            kind: 'step',
            fixed: rentedAtZero.monthlyCost,
            increment: sampled.monthlyCost - rentedAtZero.monthlyCost,
            capacity,
          })
        }
      }
    }

    const ownedAtZero = ownedCostForCandidate(workload, candidate, assumptions, 0)
    if (ownedAtZero) {
      const gpusPerReplica = Math.max(Math.ceil(positive(candidate.gpusPerReplica)), 1)
      const gpusPerServer = Math.max(Math.ceil(positive(candidate.purchaseGpusPerServer ?? gpusPerReplica)), 1)
      const capacity = replicaCapacity * Math.max(Math.floor(gpusPerServer / gpusPerReplica), 1)
      const sample = Math.min(1, capacity / 2)
      const sampled = ownedCostForCandidate(workload, candidate, assumptions, sample)
      if (sample > 0 && sampled) owned.push({
        kind: 'step',
        fixed: ownedAtZero.monthlyCost,
        increment: sampled.monthlyCost - ownedAtZero.monthlyCost,
        capacity,
      })
    }
  }
  return { rented, owned }
}

function chartMaximum(
  currentVolume: number,
  crossovers: Array<number | null>,
): number {
  const found = crossovers.filter((value): value is number => value !== null)
  const raw = found.length > 0
    ? Math.max(currentVolume * 1.25, ...found.map(value => value * 1.18), 10_000_000)
    : Math.max(currentVolume * 5, 100_000_000_000)
  const capped = Math.min(raw, DEFAULT_SEARCH_MAXIMUM)
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(capped, 1)))
  const normalized = capped / magnitude
  const rounded = Math.ceil(normalized * 10) / 10
  return rounded * magnitude
}

function chartVolumes(
  curves: CostCurve[],
  maximum: number,
  anchors: number[],
): number[] {
  // Include the first non-zero workload explicitly. Whole-instance rented
  // modes and purchased hardware jump from zero deployed capacity to one
  // deployment at this point; without it, a line chart draws a false diagonal
  // from zero to the first coarse sample instead of the near-vertical step.
  const volumes = new Set<number>([0, Math.min(1, maximum), maximum, ...anchors])
  const uniformSamples = 160
  for (let index = 0; index <= uniformSamples; index += 1) {
    volumes.add(Math.round(maximum * index / uniformSamples))
  }
  const stepped = curves.filter((curve): curve is Extract<CostCurve, { kind: 'step' }> => curve.kind === 'step')
  const stepsPerCurve = Math.max(Math.floor(120 / Math.max(stepped.length, 1)), 1)
  for (const curve of curves) {
    if (curve.kind === 'scale-to-zero') {
      for (const knot of [curve.firstKnot, curve.firstKnot * curve.parallelism]) {
        volumes.add(Math.floor(knot))
        volumes.add(Math.ceil(knot))
      }
      continue
    }
    if (curve.kind !== 'step') continue
    const addBoundary = (multiplier: number) => {
      if (multiplier < 1) return
      const lastTokenBeforeStep = Math.floor(curve.capacity * multiplier)
      volumes.add(lastTokenBeforeStep)
      volumes.add(lastTokenBeforeStep + 1)
    }
    const boundaryCount = Math.floor(maximum / curve.capacity)
    const shown = Math.min(boundaryCount, stepsPerCurve)
    for (let index = 1; index <= shown; index += 1) {
      addBoundary(Math.ceil(index * boundaryCount / shown))
    }
    for (const anchor of anchors) {
      addBoundary(Math.floor(anchor / curve.capacity))
      addBoundary(Math.ceil(anchor / curve.capacity))
    }
  }
  return [...volumes]
    .filter(volume => volume >= 0 && volume <= maximum)
    .sort((left, right) => left - right)
}

export function calculateHybridComparison(
  workload: HybridWorkload,
  hostedPrice: HostedPrice,
  candidates: InfrastructureCandidate[],
  assumptions: CostAssumptions,
): HybridComparison {
  const facts = workloadFacts(workload)
  const hosted = hostedCostAtVolume(
    workload,
    hostedPrice,
    assumptions,
    facts.monthlyTokens,
  )
  const rented = bestRentedAtVolume(
    workload,
    candidates,
    assumptions,
    facts.monthlyTokens,
  )
  const owned = bestOwnedAtVolume(
    workload,
    candidates,
    assumptions,
    facts.monthlyTokens,
  )
  const options = [hosted, rented, owned].filter(
    (option): option is CostOption => option !== null,
  )
  const cheapest = [...options].sort((left, right) => left.monthlyCost - right.monthlyCost)[0] ?? null
  const curves = candidateCurves(workload, candidates, assumptions)
  const inputShare = workloadMix(workload).inputShare
  const hostedCurve: CostCurve = {
    kind: 'linear',
    fixed: hostedCostAtVolume(workload, hostedPrice, assumptions, 0).monthlyCost,
    rate: (inputShare * finiteNonNegative(hostedPrice.inputPerMillion)
      + (1 - inputShare) * finiteNonNegative(hostedPrice.outputPerMillion)) / MILLION,
  }
  let rentedBreakEvenTokens: number | null = null
  let ownedBreakEvenTokens: number | null = null
  let rentedLowestCostTokens: number | null = null
  let ownedLowestCostTokens: number | null = null
  let transitionsVerified = true
  try {
    // Keep an adversarial near-tie from monopolizing the browser's main thread.
    // In that rare case the current-workload costs remain valid, but no
    // unverified transition may be labelled "not reached".
    const budget = { remaining: 10_000 }
    rentedBreakEvenTokens = firstWinningVolume(curves.rented, [hostedCurve], DEFAULT_SEARCH_MAXIMUM, budget)
    ownedBreakEvenTokens = firstWinningVolume(curves.owned, [hostedCurve], DEFAULT_SEARCH_MAXIMUM, budget)
    rentedLowestCostTokens = firstWinningVolume(
      curves.rented, [hostedCurve, ...curves.owned], DEFAULT_SEARCH_MAXIMUM, budget,
    )
    ownedLowestCostTokens = firstWinningVolume(
      curves.owned, [hostedCurve, ...curves.rented], DEFAULT_SEARCH_MAXIMUM, budget,
    )
  } catch (error) {
    if (!(error instanceof TransitionSearchLimitError)) throw error
    transitionsVerified = false
    rentedBreakEvenTokens = null
    ownedBreakEvenTokens = null
    rentedLowestCostTokens = null
    ownedLowestCostTokens = null
  }
  const chartMaximumTokens = chartMaximum(
    facts.monthlyTokens,
    [
      rentedBreakEvenTokens,
      ownedBreakEvenTokens,
      rentedLowestCostTokens,
      ownedLowestCostTokens,
    ],
  )
  const anchors = [
    facts.monthlyTokens,
    rentedBreakEvenTokens,
    ownedBreakEvenTokens,
    rentedLowestCostTokens,
    ownedLowestCostTokens,
  ].filter((value): value is number => value !== null)
  const chartPoints = chartVolumes(
    [...curves.rented, ...curves.owned],
    chartMaximumTokens,
    anchors,
  )
    .map(tokens => ({
      tokens,
      hosted: hostedCostAtVolume(workload, hostedPrice, assumptions, tokens).monthlyCost,
      rented: bestRentedAtVolume(workload, candidates, assumptions, tokens)?.monthlyCost ?? null,
      owned: bestOwnedAtVolume(workload, candidates, assumptions, tokens)?.monthlyCost ?? null,
    }))

  return {
    monthlyTokens: facts.monthlyTokens,
    monthlyRequests: facts.monthlyRequests,
    peakRequestsPerSecond: facts.peakRequestsPerSecond,
    options,
    cheapest,
    rentedBreakEvenTokens,
    ownedBreakEvenTokens,
    rentedLowestCostTokens,
    ownedLowestCostTokens,
    transitionsVerified,
    chartMaximumTokens,
    chartPoints,
  }
}

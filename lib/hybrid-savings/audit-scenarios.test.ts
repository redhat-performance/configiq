import { describe, expect, it } from 'vitest'
import {
  bestOwnedAtVolume,
  bestRentedAtVolume,
  calculateHybridComparison,
  candidateCapacityTokens,
  hostedCostAtVolume,
  type CostAssumptions,
  type HostedPrice,
  type HybridWorkload,
  type InfrastructureCandidate,
} from './calc'

const workload: HybridWorkload = {
  monthlyInputTokens: 800_000,
  monthlyOutputTokens: 200_000,
  averageInputTokens: 4_000,
  averageOutputTokens: 1_000,
  activeHoursPerMonth: 100,
  peakToAverage: 2,
}

const hostedPrice: HostedPrice = {
  modelId: 'audit/model',
  label: 'Audit hosted API',
  inputPerMillion: 1,
  outputPerMillion: 3,
}

const candidate: InfrastructureCandidate = {
  systemId: 'audit_gpu',
  label: 'Audit GPU',
  gpusPerReplica: 1,
  replicasNeeded: 1,
  clusterOutputTokensPerSecond: 100,
  cloudHourlyCostPerInstance: 2,
  cloudDirectInfrastructureMonthly: 0,
  cloudRatePerGpuHour: 2,
  cloudGpusPerInstance: 1,
  cloudProvider: 'audit.region',
  cloudRateKind: 'on_demand',
  purchaseGpusPerServer: 1,
  purchasePricePerReplica: 24_000,
  purchaseInstallationPerReplica: 0,
  purchasePriceIndicative: false,
  purchasePriceSource: 'Audit fixture',
  purchasePriceSourceUrl: null,
  purchasePriceSourceDate: '2026-09-23',
  tdpWattsPerGpu: 0,
  ttftMs: 100,
  tpotMs: 20,
  source: 'Audit fixture',
}

const assumptions: CostAssumptions = {
  costLens: 'fully-loaded',
  cloudBillingMode: 'active-window',
  cloudRuntimeBufferPct: 0,
  planningCapacityUsePct: 90,
  hoursPerMonth: 730,
  analysisMonths: 36,
  loadedMonthlyCostPerFte: 0,
  hostedOperationsFte: 0,
  hostedImplementation: 0,
  rentedDirectInfrastructureMonthly: 0,
  rentedOperationsFte: 0,
  rentedImplementation: 0,
  hardwareLifeYears: 5,
  hardwareResidualPct: 0,
  annualCostOfCapitalPct: 0,
  annualMaintenancePct: 0,
  electricityPerKwh: 0,
  pue: 1,
  ownedBaseSystemPowerWattsPerServer: 0,
  ownedInstallationPerServer: 0,
  ownedFacilityMonthlyPerServer: 0,
  ownedDirectInfrastructureMonthly: 0,
  ownedOperationsFte: 0,
  ownedImplementation: 0,
  hostedFixedMonthly: 0,
  rentedFixedMonthly: 0,
  ownedFixedMonthly: 0,
}

describe('hybrid savings independently calculated audit scenarios', () => {
  // Capacity = 100 output tok/s * 5 total/output ratio * 90% planning
  //          * 100 hours * 3,600 seconds / 2x peak = 81M billed tokens.
  const expectedCapacity = 81_000_000

  it('matches a very small workload calculation', () => {
    const volume = 1_000_000
    const hosted = hostedCostAtVolume(workload, hostedPrice, assumptions, volume)
    const rented = bestRentedAtVolume(workload, [candidate], assumptions, volume)
    const owned = bestOwnedAtVolume(workload, [candidate], assumptions, volume)

    // 80% input * $1/M + 20% output * $3/M = $1.40/M.
    expect(hosted.monthlyCost).toBeCloseTo(1.4)
    // One instance * 100 active hours * $2/hour.
    expect(rented?.monthlyCost).toBeCloseTo(200)
    // $24,000 / (5 years * 12 months).
    expect(owned?.monthlyCost).toBeCloseTo(400)
  })

  it('matches a moderate workload calculation', () => {
    const volume = 40_500_000
    const hosted = hostedCostAtVolume(workload, hostedPrice, assumptions, volume)
    const rented = bestRentedAtVolume(workload, [candidate], assumptions, volume)
    const owned = bestOwnedAtVolume(workload, [candidate], assumptions, volume)

    expect(candidateCapacityTokens(workload, candidate, 90)).toBe(expectedCapacity)
    expect(hosted.monthlyCost).toBeCloseTo(56.7)
    expect(rented?.monthlyCost).toBeCloseTo(200)
    expect(rented?.utilizationPct).toBeCloseTo(50)
    expect(owned?.monthlyCost).toBeCloseTo(400)
  })

  it('keeps one deployment at high utilization below the boundary', () => {
    const volume = expectedCapacity * 0.99
    const rented = bestRentedAtVolume(workload, [candidate], assumptions, volume)
    const owned = bestOwnedAtVolume(workload, [candidate], assumptions, volume)

    expect(rented?.replicas).toBe(1)
    expect(rented?.monthlyCost).toBeCloseTo(200)
    expect(rented?.utilizationPct).toBeCloseTo(99)
    expect(owned?.replicas).toBe(1)
    expect(owned?.monthlyCost).toBeCloseTo(400)
  })

  it('adds a whole deployment immediately after a capacity boundary', () => {
    const below = expectedCapacity
    const above = expectedCapacity + 1

    expect(bestRentedAtVolume(workload, [candidate], assumptions, below)?.monthlyCost)
      .toBeCloseTo(200)
    expect(bestRentedAtVolume(workload, [candidate], assumptions, above)?.monthlyCost)
      .toBeCloseTo(400)
    expect(bestOwnedAtVolume(workload, [candidate], assumptions, below)?.monthlyCost)
      .toBeCloseTo(400)
    expect(bestOwnedAtVolume(workload, [candidate], assumptions, above)?.monthlyCost)
      .toBeCloseTo(800)
  })

  it('finds the exact hosted-to-rented crossover from the same piecewise costs', () => {
    const crossoverCandidate = {
      ...candidate,
      cloudHourlyCostPerInstance: 1,
      cloudRatePerGpuHour: 1,
      cloudDirectInfrastructureMonthly: null,
    }
    const crossoverAssumptions = {
      ...assumptions,
      rentedDirectInfrastructureMonthly: 30,
    }
    const result = calculateHybridComparison(
      workload,
      hostedPrice,
      [crossoverCandidate],
      crossoverAssumptions,
    )

    // Hosted costs $1.40/M. Rented costs $100 per whole capacity step plus
    // $30 direct infrastructure. Steps one and two end before their rented
    // cost is recovered. Step three costs $330, so its exact first crossover
    // is ceil($330 / $1.40 * 1M) = 235,714,286 tokens.
    expect(result.rentedBreakEvenTokens).toBe(235_714_286)
    const crossoverPoint = result.chartPoints.find(
      point => point.tokens === result.rentedBreakEvenTokens,
    )
    expect(crossoverPoint).toBeDefined()
    expect(crossoverPoint?.rented).toBeLessThanOrEqual(crossoverPoint?.hosted ?? 0)
    const previousHosted = hostedCostAtVolume(
      workload,
      hostedPrice,
      assumptions,
      result.rentedBreakEvenTokens! - 1,
    )
    const previousRented = bestRentedAtVolume(
      workload,
      [crossoverCandidate],
      crossoverAssumptions,
      result.rentedBreakEvenTokens! - 1,
    )
    expect(previousRented?.monthlyCost).toBeGreaterThan(previousHosted.monthlyCost)
  })
})

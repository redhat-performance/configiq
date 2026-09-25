import { describe, expect, it } from 'vitest'
import {
  bestOwnedAtVolume,
  bestRentedAtVolume,
  calculateHybridComparison,
  candidateCapacityTokens,
  hostedCostAtVolume,
  workloadFacts,
  type CostAssumptions,
  type HostedPrice,
  type HybridWorkload,
  type InfrastructureCandidate,
} from './calc'

const workload: HybridWorkload = {
  monthlyInputTokens: 20_000_000,
  monthlyOutputTokens: 5_000_000,
  averageInputTokens: 4_000,
  averageOutputTokens: 1_000,
  activeHoursPerMonth: 730,
  peakToAverage: 1,
}

const hostedPrice: HostedPrice = {
  modelId: 'test/model',
  label: 'Hosted API',
  inputPerMillion: 1,
  outputPerMillion: 3,
}

const assumptions: CostAssumptions = {
  costLens: 'fully-loaded',
  cloudBillingMode: 'always-on',
  cloudRuntimeBufferPct: 15,
  planningCapacityUsePct: 100,
  hoursPerMonth: 730,
  analysisMonths: 36,
  loadedMonthlyCostPerFte: 18_000,
  hostedOperationsFte: 0,
  hostedImplementation: 0,
  rentedDirectInfrastructureMonthly: 0,
  rentedOperationsFte: 0,
  rentedImplementation: 0,
  hardwareLifeYears: 5,
  hardwareResidualPct: 0,
  annualCostOfCapitalPct: 0,
  annualMaintenancePct: 8,
  electricityPerKwh: 0.12,
  pue: 1.4,
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

const candidate: InfrastructureCandidate = {
  systemId: 'test_gpu',
  label: 'Test GPU',
  gpusPerReplica: 1,
  replicasNeeded: 1,
  clusterOutputTokensPerSecond: 100,
  cloudRatePerGpuHour: 2,
  cloudGpusPerInstance: 1,
  cloudProvider: 'test.region',
  cloudRateKind: 'on_demand',
  purchasePricePerReplica: 12_000,
  purchaseInstallationPerReplica: 0,
  purchasePriceIndicative: true,
  purchasePriceSource: 'Test catalogue',
  purchasePriceSourceUrl: null,
  purchasePriceSourceDate: '2026-09-16',
  tdpWattsPerGpu: 500,
  ttftMs: 300,
  tpotMs: 40,
  source: 'AISimulators',
}

describe('hybrid savings calculations', () => {
  it('derives request demand from billed tokens and the request shape', () => {
    const facts = workloadFacts(workload)
    expect(facts.monthlyTokens).toBe(25_000_000)
    expect(facts.monthlyRequests).toBe(5_000)
    expect(facts.peakRequestsPerSecond).toBeCloseTo(5_000 / (730 * 3_600))
  })

  it('does not drop work when the monthly token mix differs from the average request', () => {
    const mismatched: HybridWorkload = {
      ...workload,
      monthlyInputTokens: 800_000,
      monthlyOutputTokens: 200_000,
      averageInputTokens: 900,
      averageOutputTokens: 100,
      activeHoursPerMonth: 1,
    }
    const facts = workloadFacts(mismatched)

    // Input implies 888.89 requests, while output implies 2,000. The larger
    // count is required to avoid silently omitting output work.
    expect(facts.monthlyRequests).toBeCloseTo(2_000)
    // 100 output tok/s / 100 output tok/request = 1 request/s. At the
    // workload mix, 2,000 requests carry 1M billed tokens, so one hour of
    // capacity is 3,600 / 2,000 * 1M = 1.8M billed tokens.
    expect(candidateCapacityTokens(mismatched, candidate)).toBeCloseTo(1_800_000)
  })

  it('prices hosted input and output tokens independently', () => {
    const result = hostedCostAtVolume(workload, hostedPrice, assumptions, 25_000_000)
    expect(result.monthlyCost).toBe(35)
    expect(result.costPerMillionTokens).toBe(1.4)
  })

  it('treats invalid hosted feed prices as zero in totals and breakdown rows', () => {
    const result = hostedCostAtVolume(
      workload,
      { ...hostedPrice, inputPerMillion: Number.NaN, outputPerMillion: Number.POSITIVE_INFINITY },
      assumptions,
      25_000_000,
    )

    expect(result.monthlyCost).toBe(0)
    expect(result.breakdown.every(item => Number.isFinite(item.monthlyCost))).toBe(true)
  })

  it('uses an offer-specific rented infrastructure allowance before the global fallback', () => {
    const priced = bestRentedAtVolume(
      workload,
      [{ ...candidate, cloudDirectInfrastructureMonthly: 500 }],
      { ...assumptions, rentedDirectInfrastructureMonthly: 1_800 },
      25_000_000,
    )

    expect(
      priced?.breakdown.find(item => item.label === 'Infrastructure and observability')?.monthlyCost,
    ).toBe(500)
  })

  it('converts AISimulators output throughput into billed-token capacity', () => {
    const capacity = candidateCapacityTokens(workload, candidate)
    expect(capacity).toBe(100 * 5 * 730 * 3_600)
  })

  it('excludes incomplete AISimulators capacity instead of inventing a fallback rate', () => {
    const incomplete = { ...candidate, clusterOutputTokensPerSecond: 0 }

    expect(candidateCapacityTokens(workload, incomplete)).toBe(0)
    expect(bestRentedAtVolume(workload, [incomplete], assumptions, 25_000_000)).toBeNull()
    expect(bestOwnedAtVolume(workload, [incomplete], assumptions, 25_000_000)).toBeNull()
  })

  it('selects the least-cost eligible rented and owned candidates independently', () => {
    const expensive = {
      ...candidate,
      systemId: 'expensive',
      label: 'Expensive GPU',
      cloudRatePerGpuHour: 4,
      purchasePricePerReplica: 24_000,
    }
    expect(bestRentedAtVolume(workload, [expensive, candidate], assumptions, 25_000_000)?.candidate?.systemId)
      .toBe('test_gpu')
    expect(bestOwnedAtVolume(workload, [expensive, candidate], assumptions, 25_000_000)?.candidate?.systemId)
      .toBe('test_gpu')
  })

  it('uses workload-linked GPU hours for scale-to-zero billing', () => {
    const result = bestRentedAtVolume(
      workload,
      [candidate],
      { ...assumptions, cloudBillingMode: 'scale-to-zero', cloudRuntimeBufferPct: 0 },
      25_000_000,
    )
    const expectedGpuHours = 5_000_000 / 100 / 3_600
    expect(result?.monthlyCost).toBeCloseTo(expectedGpuHours * 2)
  })

  it('bills a whole multi-GPU instance for sparse scale-to-zero demand', () => {
    const result = bestRentedAtVolume(
      workload,
      [{
        ...candidate,
        cloudGpusPerInstance: 8,
        cloudHourlyCostPerInstance: 16,
      }],
      { ...assumptions, cloudBillingMode: 'scale-to-zero', cloudRuntimeBufferPct: 0 },
      25_000_000,
    )
    const expectedInstanceHours = 5_000_000 / 100 / 3_600
    expect(result?.monthlyCost).toBeCloseTo(expectedInstanceHours * 16)
    expect(result?.billedGpuCount).toBe(8)
  })

  it('does not mis-rank an eight-GPU B200 instance as a one-GPU bargain', () => {
    const l40s = {
      ...candidate,
      systemId: 'l40s',
      label: 'NVIDIA L40S',
      clusterOutputTokensPerSecond: 698.279,
      cloudHourlyCostPerInstance: 1.861,
      cloudRatePerGpuHour: 1.861,
      cloudGpusPerInstance: 1,
    }
    const b200 = {
      ...candidate,
      systemId: 'b200_sxm',
      label: 'NVIDIA B200',
      clusterOutputTokensPerSecond: 9_164.991,
      cloudHourlyCostPerInstance: 68.8,
      cloudRatePerGpuHour: 8.6,
      cloudGpusPerInstance: 8,
    }
    const result = bestRentedAtVolume(
      workload,
      [b200, l40s],
      {
        ...assumptions,
        cloudBillingMode: 'scale-to-zero',
        cloudRuntimeBufferPct: 10,
        planningCapacityUsePct: 90,
      },
      2_500_000_000,
    )
    expect(result?.candidate?.systemId).toBe('l40s')
    expect(result?.candidate?.cloudGpusPerInstance).toBe(1)
  })

  it('does not price Qwen3 8B rented capacity from single-user throughput', () => {
    const planningWorkload: HybridWorkload = {
      monthlyInputTokens: 2_000_000_000,
      monthlyOutputTokens: 500_000_000,
      averageInputTokens: 2_048,
      averageOutputTokens: 512,
      activeHoursPerMonth: 730,
      peakToAverage: 1,
    }
    const qwenPrice: HostedPrice = {
      modelId: 'Qwen/Qwen3-8B',
      label: 'Qwen hosted API',
      inputPerMillion: 0.117,
      outputPerMillion: 0.455,
    }
    const a100CapacityCandidate: InfrastructureCandidate = {
      ...candidate,
      systemId: 'a100_sxm',
      label: 'NVIDIA A100-SXM4-80GB',
      // Live AISimulators result at the standardized concurrency-32 capacity
      // load. Concurrency 1 returns only 74.63 tok/s and is a latency result,
      // not the serving capacity used for infrastructure planning.
      clusterOutputTokensPerSecond: 1_072.1655488481815,
      cloudHourlyCostPerInstance: 2.7,
      cloudRatePerGpuHour: 2.7,
      cloudGpusPerInstance: 1,
      cloudDirectInfrastructureMonthly: 500,
      purchaseGpusPerServer: 1,
      purchasePricePerReplica: 34_000,
      purchaseInstallationPerReplica: 2_400,
      tdpWattsPerGpu: 400,
    }
    const planningAssumptions: CostAssumptions = {
      ...assumptions,
      costLens: 'fully-loaded',
      cloudBillingMode: 'scale-to-zero',
      cloudRuntimeBufferPct: 10,
      planningCapacityUsePct: 90,
      analysisMonths: 36,
      hardwareLifeYears: 4,
      hardwareResidualPct: 20,
      annualCostOfCapitalPct: 8,
      annualMaintenancePct: 5,
      loadedMonthlyCostPerFte: 18_000,
      hostedOperationsFte: 0.05,
      hostedImplementation: 15_000,
      rentedOperationsFte: 0.2,
      rentedImplementation: 40_000,
      ownedOperationsFte: 0.25,
      ownedImplementation: 50_000,
      ownedDirectInfrastructureMonthly: 1_500,
      ownedFacilityMonthlyPerServer: 200,
      ownedBaseSystemPowerWattsPerServer: 450,
    }

    const result = calculateHybridComparison(
      planningWorkload,
      qwenPrice,
      [a100CapacityCandidate],
      planningAssumptions,
    )
    const hosted = result.options.find(option => option.key === 'hosted')
    const rented = result.options.find(option => option.key === 'rented')
    const owned = result.options.find(option => option.key === 'owned')

    expect(hosted?.monthlyCost).toBeLessThan(rented?.monthlyCost ?? Infinity)
    expect(rented?.monthlyCost).toBeLessThan(owned?.monthlyCost ?? Infinity)
    expect(rented?.replicas).toBe(1)
    expect(owned?.replicas).toBe(1)
  })

  it('shares a whole instance only across replicas the workload actually needs', () => {
    const capacity = candidateCapacityTokens(workload, candidate)
    const result = bestRentedAtVolume(
      workload,
      [{
        ...candidate,
        cloudGpusPerInstance: 8,
        cloudHourlyCostPerInstance: 16,
      }],
      { ...assumptions, cloudBillingMode: 'scale-to-zero', cloudRuntimeBufferPct: 0 },
      capacity * 8,
    )
    const outputShare = workload.monthlyOutputTokens /
      (workload.monthlyInputTokens + workload.monthlyOutputTokens)
    const replicaHours = capacity * 8 * outputShare / 100 / 3_600
    expect(result?.replicas).toBe(8)
    expect(result?.monthlyCost).toBeCloseTo(replicaHours / 8 * 16)
  })

  it('does not treat burst peak replicas as continuously parallel scale-to-zero work', () => {
    const burstWorkload = { ...workload, peakToAverage: 2 }
    const packedCandidate = {
      ...candidate,
      cloudGpusPerInstance: 8,
      cloudHourlyCostPerInstance: 16,
    }
    const capacity = candidateCapacityTokens(burstWorkload, packedCandidate)
    const volume = capacity * 2
    const result = bestRentedAtVolume(
      burstWorkload,
      [packedCandidate],
      { ...assumptions, cloudBillingMode: 'scale-to-zero', cloudRuntimeBufferPct: 0 },
      volume,
    )
    const outputShare = burstWorkload.monthlyOutputTokens /
      (burstWorkload.monthlyInputTokens + burstWorkload.monthlyOutputTokens)
    const replicaHours = volume * outputShare / 100 / 3_600

    // Two replicas are needed at the 2x peak, but average demand uses only one
    // replica. The whole instance therefore runs for the full 730-hour active
    // window rather than receiving an impossible 2x full-month packing credit.
    expect(result?.replicas).toBe(2)
    expect(replicaHours).toBeCloseTo(730)
    expect(result?.monthlyCost).toBeCloseTo(replicaHours * 16)
  })

  it('never lowers scale-to-zero rented cost when demand crosses replica boundaries', () => {
    const packedCandidate = {
      ...candidate,
      cloudGpusPerInstance: 8,
      cloudHourlyCostPerInstance: 16,
    }
    const scaleToZero = {
      ...assumptions,
      cloudBillingMode: 'scale-to-zero' as const,
      cloudRuntimeBufferPct: 0,
    }
    const capacity = candidateCapacityTokens(workload, packedCandidate)
    const volumes = [
      capacity * 0.5,
      capacity,
      capacity + 1,
      capacity * 2,
      capacity * 4,
      capacity * 8,
      capacity * 8 + 1,
      capacity * 12,
      capacity * 16,
    ]
    const costs = volumes.map(volume => (
      bestRentedAtVolume(workload, [packedCandidate], scaleToZero, volume)?.monthlyCost ?? -1
    ))

    for (let index = 1; index < costs.length; index += 1) {
      expect(costs[index]).toBeGreaterThanOrEqual(costs[index - 1] - 0.000001)
    }
  })

  it('adds whole replicas when demand exceeds one replica capacity', () => {
    const capacity = candidateCapacityTokens(workload, candidate)
    const result = bestOwnedAtVolume(workload, [candidate], assumptions, capacity + 1)
    expect(result?.replicas).toBe(2)
    expect(result?.gpuCount).toBe(2)
  })

  it('packs compatible purchased replicas into the same complete server', () => {
    const packedCandidate = {
      ...candidate,
      purchaseGpusPerServer: 8,
      purchasePricePerReplica: 80_000,
    }
    const capacity = candidateCapacityTokens(workload, packedCandidate)
    const oneReplica = bestOwnedAtVolume(workload, [packedCandidate], assumptions, capacity)
    const eightReplicas = bestOwnedAtVolume(workload, [packedCandidate], assumptions, capacity * 8)

    expect(oneReplica?.billedGpuCount).toBe(8)
    expect(eightReplicas?.replicas).toBe(8)
    expect(eightReplicas?.billedGpuCount).toBe(8)
    expect(eightReplicas?.breakdown.find(item => item.label.includes('depreciation'))?.monthlyCost)
      .toBeCloseTo(oneReplica?.breakdown.find(item => item.label.includes('depreciation'))?.monthlyCost ?? 0)
    const installedServerEnergy = 8 * 500 / 1_000 * 730 * 1.4 * 0.12
    expect(oneReplica?.breakdown.find(item => item.label === 'Power including PUE')?.monthlyCost)
      .toBeCloseTo(installedServerEnergy)
    expect(eightReplicas?.breakdown.find(item => item.label === 'Power including PUE')?.monthlyCost)
      .toBeCloseTo(installedServerEnergy)
  })

  it('rounds active-window rented capacity to a whole cloud instance', () => {
    const result = bestRentedAtVolume(
      workload,
      [{ ...candidate, cloudGpusPerInstance: 8 }],
      { ...assumptions, cloudBillingMode: 'active-window' },
      25_000_000,
    )

    expect(result?.gpuCount).toBe(1)
    expect(result?.billedGpuCount).toBe(8)
    expect(result?.monthlyCost).toBe(8 * 730 * 2)
  })

  it('uses exact replica packing rather than aggregate GPU rounding', () => {
    const fiveGpuReplica = {
      ...candidate,
      gpusPerReplica: 5,
      cloudGpusPerInstance: 8,
      cloudHourlyCostPerInstance: 16,
      clusterOutputTokensPerSecond: 300,
    }
    const capacity = candidateCapacityTokens(workload, fiveGpuReplica)
    const result = bestRentedAtVolume(
      workload,
      [fiveGpuReplica],
      { ...assumptions, cloudBillingMode: 'always-on' },
      capacity * 3,
    )
    expect(result?.replicas).toBe(3)
    expect(result?.billedGpuCount).toBe(24)
    expect(result?.monthlyCost).toBe(3 * 730 * 16)
  })

  it('excludes a cloud shape that cannot span enough instances for one replica', () => {
    const result = bestRentedAtVolume(
      workload,
      [{
        ...candidate,
        gpusPerReplica: 16,
        cloudGpusPerInstance: 8,
        cloudHourlyCostPerInstance: 16,
        cloudMaxInstancesPerReplica: 1,
      }],
      assumptions,
      25_000_000,
    )
    expect(result).toBeNull()
  })

  it('forces capacity-block offers to remain on for the full month', () => {
    const result = bestRentedAtVolume(
      workload,
      [{
        ...candidate,
        cloudHourlyCostPerInstance: 10,
        cloudRateKind: 'capacity_block',
      }],
      { ...assumptions, cloudBillingMode: 'scale-to-zero' },
      25_000_000,
    )
    expect(result?.monthlyCost).toBe(730 * 10)
  })

  it('applies explicit planning headroom once to modeled capacity', () => {
    expect(candidateCapacityTokens(workload, candidate, 90)).toBeCloseTo(
      candidateCapacityTokens(workload, candidate) * 0.9,
    )
  })

  it('builds a zero-based chart and finds infrastructure crossovers', () => {
    const highApiPrice = { ...hostedPrice, inputPerMillion: 100, outputPerMillion: 100 }
    const result = calculateHybridComparison(workload, highApiPrice, [candidate], assumptions)
    expect(result.transitionsVerified).toBe(true)
    expect(result.chartPoints[0].tokens).toBe(0)
    expect(result.chartPoints[1].tokens).toBe(1)
    expect(result.rentedBreakEvenTokens).not.toBeNull()
    expect(result.ownedBreakEvenTokens).not.toBeNull()
    const lowestCostTransitions = [result.rentedLowestCostTokens, result.ownedLowestCostTokens]
      .filter((value): value is number => value !== null)
    expect(lowestCostTransitions.length).toBeGreaterThan(0)
    expect(result.chartPoints.some(point => point.tokens === result.rentedBreakEvenTokens)).toBe(true)
    expect(lowestCostTransitions.every(transition =>
      result.chartPoints.some(point => point.tokens === transition),
    )).toBe(true)
    expect(result.chartMaximumTokens).toBeGreaterThanOrEqual(result.monthlyTokens)
    const furthestRelevantPoint = Math.max(
      result.monthlyTokens,
      result.rentedBreakEvenTokens ?? 0,
      result.ownedBreakEvenTokens ?? 0,
      result.rentedLowestCostTokens ?? 0,
      result.ownedLowestCostTokens ?? 0,
    )
    expect(result.chartMaximumTokens).toBeGreaterThan(furthestRelevantPoint)
    expect(result.chartMaximumTokens).toBeLessThanOrEqual(furthestRelevantPoint * 2)
  })

  it('finds a narrow crossover after the eightieth capacity boundary', () => {
    const tenBillionTokenStep = 10_000_000_000
    const lateCandidate: InfrastructureCandidate = {
      ...candidate,
      clusterOutputTokensPerSecond:
        tenBillionTokenStep /
        ((workload.averageInputTokens + workload.averageOutputTokens) /
          workload.averageOutputTokens * workload.activeHoursPerMonth * 3_600),
      cloudRatePerGpuHour: 99 / 730,
      cloudGpusPerInstance: 1,
      cloudHourlyCostPerInstance: 99 / 730,
      purchasePricePerReplica: null,
    }
    const lateAssumptions: CostAssumptions = {
      ...assumptions,
      cloudBillingMode: 'always-on',
      rentedDirectInfrastructureMonthly: 80.5,
    }
    const result = calculateHybridComparison(
      workload,
      { ...hostedPrice, inputPerMillion: 0.01, outputPerMillion: 0.01 },
      [lateCandidate],
      lateAssumptions,
    )

    // At the end of step 80 the hosted cost is $8,000 and rented is
    // $8,000.50. In step 81, rented is $8,099.50, so the exact first whole
    // token where hosted reaches that amount is 809.95B tokens.
    expect(result.rentedBreakEvenTokens).toBe(809_950_000_000)
    expect(result.chartPoints.some(point => point.tokens === tenBillionTokenStep * 81)).toBe(true)
  })

  it('matches exhaustive whole-token transitions for every rented billing mode', () => {
    const smallWorkload: HybridWorkload = {
      ...workload, monthlyInputTokens: 400, monthlyOutputTokens: 100,
      averageInputTokens: 4, averageOutputTokens: 1, activeHoursPerMonth: 1,
    }
    const offers = [
      {
        ...candidate, clusterOutputTokensPerSecond: 100 / (5 * 3_600),
        cloudGpusPerInstance: 2, cloudHourlyCostPerInstance: 150 / 730,
        purchaseGpusPerServer: 2, purchasePricePerReplica: 9_600,
      },
      {
        ...candidate, systemId: 'alternative', clusterOutputTokensPerSecond: 140 / (5 * 3_600),
        cloudGpusPerInstance: 1, cloudHourlyCostPerInstance: 100 / 730,
        purchaseGpusPerServer: 1, purchasePricePerReplica: 7_200,
      },
    ]
    const price = { ...hostedPrice, inputPerMillion: 1_000_000, outputPerMillion: 1_000_000 }

    for (const cloudBillingMode of ['scale-to-zero', 'active-window', 'always-on'] as const) {
      const planning = {
        ...assumptions, cloudBillingMode, cloudRuntimeBufferPct: 0,
        annualMaintenancePct: 0, electricityPerKwh: 0,
      }
      const result = calculateHybridComparison(smallWorkload, price, offers, planning)
      const expected: Array<number | null> = [null, null, null, null]
      for (let volume = 1; volume <= 1_000; volume += 1) {
        const hosted = hostedCostAtVolume(smallWorkload, price, planning, volume).monthlyCost
        const rented = bestRentedAtVolume(smallWorkload, offers, planning, volume)?.monthlyCost ?? Infinity
        const owned = bestOwnedAtVolume(smallWorkload, offers, planning, volume)?.monthlyCost ?? Infinity
        if (expected[0] === null && rented <= hosted) expected[0] = volume
        if (expected[1] === null && owned <= hosted) expected[1] = volume
        if (expected[2] === null && rented <= hosted && rented <= owned) expected[2] = volume
        if (expected[3] === null && owned <= hosted && owned <= rented) expected[3] = volume
      }
      const actual = [
        result.rentedBreakEvenTokens, result.ownedBreakEvenTokens,
        result.rentedLowestCostTokens, result.ownedLowestCostTokens,
      ]
      expected.forEach((value, index) => {
        if (value !== null) expect(actual[index]).toBe(value)
        else expect(actual[index] === null || actual[index]! > 1_000).toBe(true)
      })
    }
  })

  it('keeps a one-hour, 1T-token comparison and its chart bounded', () => {
    const largeWorkload: HybridWorkload = {
      ...workload, monthlyInputTokens: 800_000_000_000,
      monthlyOutputTokens: 200_000_000_000, activeHoursPerMonth: 1,
    }
    const offers = Array.from({ length: 10 }, (_, index) => ({
      ...candidate, systemId: `gpu_${index}`, clusterOutputTokensPerSecond: 100 + index * 10,
    }))
    const started = performance.now()
    const result = calculateHybridComparison(largeWorkload, hostedPrice, offers, assumptions)

    expect(result.monthlyTokens).toBe(1_000_000_000_000)
    expect(result.chartPoints[0].tokens).toBe(0)
    expect(result.chartPoints.some(point => point.tokens === result.monthlyTokens)).toBe(true)
    expect(result.chartPoints.length).toBeLessThan(1_000)
    expect(performance.now() - started).toBeLessThan(2_000)
  })

  it('uses the same lowest-cost formulas for every plotted chart point', () => {
    const alternative = {
      ...candidate,
      systemId: 'alternative_gpu',
      label: 'Alternative GPU',
      clusterOutputTokensPerSecond: 250,
      cloudRatePerGpuHour: 3,
      purchasePricePerReplica: 20_000,
    }
    const candidates = [candidate, alternative]
    const result = calculateHybridComparison(workload, hostedPrice, candidates, assumptions)
    const currentPoint = result.chartPoints.find(point => point.tokens === result.monthlyTokens)

    expect(currentPoint).toBeDefined()
    for (const point of result.chartPoints) {
      expect(point.hosted).toBeCloseTo(
        hostedCostAtVolume(workload, hostedPrice, assumptions, point.tokens).monthlyCost,
      )
      expect(point.rented).toBeCloseTo(
        bestRentedAtVolume(workload, candidates, assumptions, point.tokens)?.monthlyCost ?? 0,
      )
      expect(point.owned).toBeCloseTo(
        bestOwnedAtVolume(workload, candidates, assumptions, point.tokens)?.monthlyCost ?? 0,
      )
    }
  })

  it('keeps every plotted cost path monotonic as workload increases', () => {
    const packed = {
      ...candidate,
      cloudGpusPerInstance: 8,
      cloudHourlyCostPerInstance: 16,
      purchaseGpusPerServer: 8,
      purchasePricePerReplica: 80_000,
    }
    const alternative = {
      ...candidate,
      systemId: 'alternative_gpu',
      label: 'Alternative GPU',
      clusterOutputTokensPerSecond: 160,
      cloudGpusPerInstance: 4,
      cloudHourlyCostPerInstance: 13,
      purchaseGpusPerServer: 4,
      purchasePricePerReplica: 58_000,
    }
    const result = calculateHybridComparison(
      workload,
      hostedPrice,
      [packed, alternative],
      {
        ...assumptions,
        cloudBillingMode: 'scale-to-zero',
        cloudRuntimeBufferPct: 10,
      },
    )

    for (let index = 1; index < result.chartPoints.length; index += 1) {
      const previous = result.chartPoints[index - 1]
      const current = result.chartPoints[index]
      expect(current.hosted ?? Infinity).toBeGreaterThanOrEqual((previous.hosted ?? 0) - 0.000001)
      expect(current.rented ?? Infinity).toBeGreaterThanOrEqual((previous.rented ?? 0) - 0.000001)
      expect(current.owned ?? Infinity).toBeGreaterThanOrEqual((previous.owned ?? 0) - 0.000001)
    }
  })

  it('keeps full TCO and marginal costs distinct without changing GPU capacity', () => {
    const fullTco = calculateHybridComparison(
      workload,
      hostedPrice,
      [candidate],
      {
        ...assumptions,
        hostedOperationsFte: 0.05,
        hostedImplementation: 15_000,
        rentedDirectInfrastructureMonthly: 1_800,
        rentedOperationsFte: 0.2,
        rentedImplementation: 40_000,
        ownedOperationsFte: 0.25,
        ownedImplementation: 50_000,
        ownedDirectInfrastructureMonthly: 1_500,
      },
    )
    const marginal = calculateHybridComparison(
      workload,
      hostedPrice,
      [candidate],
      {
        ...assumptions,
        costLens: 'marginal',
        hostedOperationsFte: 0.05,
        hostedImplementation: 15_000,
        rentedDirectInfrastructureMonthly: 1_800,
        rentedOperationsFte: 0.2,
        rentedImplementation: 40_000,
        ownedOperationsFte: 0.25,
        ownedImplementation: 50_000,
        ownedDirectInfrastructureMonthly: 1_500,
      },
    )

    expect(fullTco.options.find(option => option.key === 'hosted')?.monthlyCost).toBeGreaterThan(
      marginal.options.find(option => option.key === 'hosted')?.monthlyCost ?? 0,
    )
    expect(fullTco.options.find(option => option.key === 'owned')?.monthlyCost).toBeGreaterThan(
      marginal.options.find(option => option.key === 'owned')?.monthlyCost ?? 0,
    )
    expect(fullTco.options.find(option => option.key === 'owned')?.billedGpuCount).toBe(
      marginal.options.find(option => option.key === 'owned')?.billedGpuCount,
    )
  })

  it('uses a complete-server Qwen3 8B estimate for purchased TCO', () => {
    const qwenPrice: HostedPrice = {
      modelId: 'Qwen/Qwen3-8B',
      label: 'Qwen hosted API',
      inputPerMillion: 0.117,
      outputPerMillion: 0.455,
    }
    const l40s: InfrastructureCandidate = {
      ...candidate,
      systemId: 'l40s',
      label: 'NVIDIA L40S',
      clusterOutputTokensPerSecond: 470.039,
      cloudRatePerGpuHour: null,
      purchasePricePerReplica: 35_000,
      purchaseInstallationPerReplica: 2_500,
      tdpWattsPerGpu: 350,
      ttftMs: 590.247,
      tpotMs: 58.979,
    }
    const result = calculateHybridComparison(
      workload,
      qwenPrice,
      [l40s],
      {
        ...assumptions,
        hardwareLifeYears: 4,
        hardwareResidualPct: 20,
        annualCostOfCapitalPct: 8,
        annualMaintenancePct: 5,
        ownedInstallationPerServer: 0,
        ownedFacilityMonthlyPerServer: 200,
        ownedDirectInfrastructureMonthly: 1_500,
        ownedOperationsFte: 0.25,
        ownedImplementation: 50_000,
      },
    )
    const hosted = result.options.find(option => option.key === 'hosted')
    const owned = result.options.find(option => option.key === 'owned')

    expect(hosted?.monthlyCost).toBeCloseTo(4.615, 3)
    expect(owned?.candidate?.systemId).toBe('l40s')
    expect(owned?.monthlyCost).toBeGreaterThan(7_500)
    expect(owned?.monthlyCost).toBe(owned?.fullyLoadedMonthlyCost)
    expect(owned?.marginalMonthlyCost).toBeLessThan(owned?.fullyLoadedMonthlyCost ?? 0)
    expect(owned?.breakdown.find(item => item.label === 'Hardware depreciation net of residual')?.monthlyCost)
      .toBeCloseTo((35_000 - 7_000) / 48)
  })

  it('prices each complete server once rather than multiplying a bare GPU price', () => {
    const fourGpuServer: InfrastructureCandidate = {
      ...candidate,
      gpusPerReplica: 4,
      purchasePricePerReplica: 94_000,
      purchaseInstallationPerReplica: 6_600,
      tdpWattsPerGpu: 0,
    }
    const result = bestOwnedAtVolume(
      workload,
      [fourGpuServer],
      {
        ...assumptions,
        hardwareLifeYears: 4,
        annualMaintenancePct: 0,
      },
      25_000_000,
    )

    expect(result?.breakdown.find(item => item.label === 'Hardware depreciation net of residual')?.monthlyCost)
      .toBeCloseTo(94_000 / 48)
    expect(result?.breakdown.find(item => item.label === 'Installation and commissioning')?.monthlyCost)
      .toBeCloseTo(6_600 / 48)
  })

  it('uses the four-year UI default when hardware life is invalid', () => {
    const result = bestOwnedAtVolume(
      workload,
      [{ ...candidate, purchasePricePerReplica: 48_000, tdpWattsPerGpu: 0 }],
      {
        ...assumptions,
        hardwareLifeYears: 0,
        hardwareResidualPct: 0,
        annualMaintenancePct: 0,
      },
      25_000_000,
    )

    expect(result?.breakdown.find(item => item.label === 'Hardware depreciation net of residual')?.monthlyCost)
      .toBeCloseTo(1_000)
  })

  it('accounts for replacement purchases when the analysis exceeds useful life', () => {
    const result = bestOwnedAtVolume(
      workload,
      [{ ...candidate, purchasePricePerReplica: 12_000, purchaseInstallationPerReplica: 1_200, tdpWattsPerGpu: 0 }],
      {
        ...assumptions,
        analysisMonths: 24,
        hardwareLifeYears: 1,
        annualMaintenancePct: 0,
      },
      25_000_000,
    )

    expect(result?.breakdown.find(item => item.label === 'Hardware depreciation net of residual')?.monthlyCost)
      .toBeCloseTo(1_000)
    expect(result?.breakdown.find(item => item.label === 'Installation and commissioning')?.monthlyCost)
      .toBeCloseTo(100)
  })
})

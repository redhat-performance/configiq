import { describe, expect, it } from 'vitest'
import type { CostAssumptions, CostOption, HybridWorkload, InfrastructureCandidate } from './calc'
import { captureFixedDeployment, fixedDeploymentSamples } from './fixed-deployment'

const workload: HybridWorkload = {
  monthlyInputTokens: 8_000_000,
  monthlyOutputTokens: 2_000_000,
  averageInputTokens: 4_000,
  averageOutputTokens: 1_000,
  activeHoursPerMonth: 730,
  peakToAverage: 1,
}

const assumptions: CostAssumptions = {
  costLens: 'fully-loaded',
  cloudBillingMode: 'always-on',
  cloudRuntimeBufferPct: 0,
  planningCapacityUsePct: 100,
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

const candidate: InfrastructureCandidate = {
  systemId: 'fixed_gpu',
  label: 'Fixed GPU',
  gpusPerReplica: 1,
  replicasNeeded: 1,
  clusterOutputTokensPerSecond: 1000,
  cloudRatePerGpuHour: 10,
  cloudGpusPerInstance: 1,
  cloudHourlyCostPerInstance: 10,
  cloudProvider: 'test',
  cloudRateKind: 'on_demand',
  purchasePricePerReplica: 60_000,
  purchaseInstallationPerReplica: 0,
  purchasePriceIndicative: false,
  purchasePriceSource: 'Test catalogue',
  purchasePriceSourceUrl: null,
  purchasePriceSourceDate: null,
  tdpWattsPerGpu: 0,
  purchaseGpusPerServer: 1,
}

const hostedPrice = {
  modelId: 'test/model',
  label: 'Hosted API',
  inputPerMillion: 1,
  outputPerMillion: 2,
}

function captureFor(option: CostOption, path: 'rented' | 'owned' = 'rented') {
  return captureFixedDeployment(path, option, 'basis-signature')
}

describe('captureFixedDeployment', () => {
  it('captures the actual chosen deployment, never an automatic label', () => {
    const option = {
      key: 'rented' as const,
      label: 'Fixed GPU',
      monthlyCost: 7_300,
      fullyLoadedMonthlyCost: 7_300,
      marginalMonthlyCost: 7_300,
      costPerMillionTokens: 0.725,
      costPerMillionOutputTokens: 3.65,
      costPerRequest: 1,
      candidate,
      gpuCount: 1,
      billedGpuCount: 1,
      replicas: 1,
      utilizationPct: 100,
      breakdown: [],
      deployment: {
        requiredReplicas: 1,
        requiredGpus: 1,
        billedGpus: 1,
        wholeUnits: 1,
        availableReplicaSlots: 1,
        planningTokenCapacity: 10_000_000,
        planningOutputCapacity: 2_000_000,
        planningCapacityUsedPct: 100,
        effectiveBillingMode: 'always-on' as const,
        billableInstanceHours: 730,
      },
    }
    const capture = captureFor(option)
    expect(capture).not.toBeNull()
    expect(capture?.path).toBe('rented')
    expect(capture?.candidateKey).toBe(JSON.stringify(['rented', 'fixed_gpu',
      ['test', undefined, undefined, 1, 'on_demand', null, null]]))
    expect(capture?.wholeUnits).toBe(1)
    expect(capture?.availableReplicaSlots).toBe(1)
    expect(capture?.planningTokenCapacity).toBe(10_000_000)
    expect(capture?.basisSignature).toBe('basis-signature')
  })

  it('refuses options without a freezable deployment', () => {
    const hosted: CostOption = {
      key: 'hosted', label: 'Hosted', monthlyCost: 0, fullyLoadedMonthlyCost: 0,
      marginalMonthlyCost: 0, costPerMillionTokens: 0, costPerMillionOutputTokens: null,
      costPerRequest: 0, candidate: null, gpuCount: null, billedGpuCount: null,
      replicas: null, utilizationPct: null, breakdown: [], deployment: null,
    }
    expect(captureFor(hosted)).toBeNull()
  })
})

describe('fixedDeploymentSamples', () => {
  it('samples whole percentages plus the exact current workload percentage', () => {
    const option = {
      key: 'rented' as const,
      label: 'Fixed GPU', monthlyCost: 1, fullyLoadedMonthlyCost: 1, marginalMonthlyCost: 1,
      costPerMillionTokens: 0, costPerMillionOutputTokens: 0, costPerRequest: 0,
      candidate, gpuCount: 1, billedGpuCount: 1, replicas: 1, utilizationPct: 100,
      breakdown: [], deployment: {
        requiredReplicas: 1, requiredGpus: 1, billedGpus: 1, wholeUnits: 1,
        availableReplicaSlots: 1, planningTokenCapacity: 10_000_000,
        planningOutputCapacity: 2_000_000, planningCapacityUsedPct: 100,
        effectiveBillingMode: 'always-on' as const, billableInstanceHours: 730,
      },
    }
    const capture = captureFor(option)!
    // 6.44M input + 1.61M output = 8.05M combined, exactly 80.5% of capacity.
    const partial: HybridWorkload = {
      ...workload, monthlyInputTokens: 6_440_000, monthlyOutputTokens: 1_610_000,
    }
    const samples = fixedDeploymentSamples(partial, hostedPrice, assumptions, candidate, capture)

    // 101 whole percentages plus the exact non-integer current percentage.
    expect(samples).toHaveLength(102)
    expect(samples[0]?.capacityUsedPct).toBe(0)
    expect(samples.at(-1)?.capacityUsedPct).toBe(100)
    const exact = samples.find(sample => sample.capacityUsedPct === 80.5)
    expect(exact).toBeDefined()
    expect(exact?.combinedTokens).toBe(8_050_000)
    expect(samples.map(sample => sample.capacityUsedPct))
      .toEqual([...samples.map(sample => sample.capacityUsedPct)].sort((a, b) => a - b))
  })

  it('keeps the frozen deployment fixed and reprices hosted against identical demand', () => {
    const option = {
      key: 'rented' as const,
      label: 'Fixed GPU', monthlyCost: 1, fullyLoadedMonthlyCost: 1, marginalMonthlyCost: 1,
      costPerMillionTokens: 0, costPerMillionOutputTokens: 0, costPerRequest: 0,
      candidate, gpuCount: 1, billedGpuCount: 1, replicas: 1, utilizationPct: 100,
      breakdown: [], deployment: {
        requiredReplicas: 1, requiredGpus: 1, billedGpus: 1, wholeUnits: 1,
        availableReplicaSlots: 1, planningTokenCapacity: 10_000_000,
        planningOutputCapacity: 2_000_000, planningCapacityUsedPct: 100,
        effectiveBillingMode: 'always-on' as const, billableInstanceHours: 730,
      },
    }
    const capture = captureFor(option)!
    const samples = fixedDeploymentSamples(workload, hostedPrice, assumptions, candidate, capture)

    for (const sample of samples) {
      expect(sample.hardware?.deployment?.wholeUnits).toBe(1)
      expect(sample.hardware?.deployment?.billedGpus).toBe(1)
      expect(sample.hardware?.deployment?.availableReplicaSlots).toBe(1)
      // Always-on fixed deployment: identical monthly cost at every fill level.
      expect(sample.hardware?.monthlyCost).toBe(730 * 10)
      // Hosted is billed for exactly the sampled demand, mix-adjusted.
      const inputTokens = sample.combinedTokens * 0.8
      expect(sample.hosted?.monthlyCost).toBeCloseTo(
        inputTokens / 1e6 * 1 + sample.outputTokens / 1e6 * 2, 6)
    }

    // Zero output carries no defined unit cost.
    expect(samples[0]?.outputTokens).toBe(0)
    expect(samples[0]?.hardware?.costPerMillionOutputTokens).toBeNull()
  })

  it('matches the fixed-deployment worked example: $1,000 fixed over a 10M-token capacity', () => {
    // One always-on instance at $1/hour over a 1,000-hour month is a fixed
    // $1,000; the frozen deployment carries a 10M combined-token capacity and
    // the workload mix is entirely output tokens.
    const outputOnly: HybridWorkload = {
      ...workload,
      monthlyInputTokens: 0,
      monthlyOutputTokens: 10_000_000,
      averageInputTokens: 1,
      averageOutputTokens: 1_000,
    }
    const capture: Parameters<typeof fixedDeploymentSamples>[4] = {
      path: 'rented',
      candidateKey: 'example',
      systemId: 'fixed_gpu',
      gpusPerReplica: 1,
      wholeUnits: 1,
      billedGpus: 1,
      availableReplicaSlots: 1,
      planningTokenCapacity: 10_000_000,
      basisSignature: 'example',
    }
    const samples = fixedDeploymentSamples(
      outputOnly,
      hostedPrice,
      { ...assumptions, hoursPerMonth: 1_000 },
      { ...candidate, cloudHourlyCostPerInstance: 1, cloudRatePerGpuHour: 1 },
      capture,
    )

    for (const pct of [25, 50, 100] as const) {
      const sample = samples.find(entry => entry.capacityUsedPct === pct)!
      expect(sample.hardware?.monthlyCost).toBe(1_000)
      expect(sample.hardware?.costPerMillionOutputTokens)
        .toBeCloseTo(1_000 * 1_000_000 / (10_000_000 * pct / 100), 6)
    }
    // 50% used → 5M output → $200 / 1M output; 100% used → $100 / 1M.
    expect(samples.find(entry => entry.capacityUsedPct === 50)?.hardware?.costPerMillionOutputTokens)
      .toBeCloseTo(200, 6)
    expect(samples.find(entry => entry.capacityUsedPct === 100)?.hardware?.costPerMillionOutputTokens)
      .toBeCloseTo(100, 6)
  })

  it('bills frozen scale-to-zero runtime proportionally to the sampled demand', () => {
    // Single-replica fixture: the frozen deployment is one 1-GPU instance at
    // $10/hour; every sampled combined token is an output token flowing
    // through one replica at 1,000 tokens/s.
    const outputOnly: HybridWorkload = {
      ...workload,
      monthlyInputTokens: 0,
      monthlyOutputTokens: 10_000_000,
      averageInputTokens: 1,
      averageOutputTokens: 1_000,
    }
    const capture: Parameters<typeof fixedDeploymentSamples>[4] = {
      path: 'rented',
      candidateKey: 'example',
      systemId: 'fixed_gpu',
      gpusPerReplica: 1,
      wholeUnits: 1,
      billedGpus: 1,
      availableReplicaSlots: 1,
      planningTokenCapacity: 10_000_000,
      basisSignature: 'example',
    }
    const samples = fixedDeploymentSamples(
      outputOnly,
      hostedPrice,
      { ...assumptions, cloudBillingMode: 'scale-to-zero' },
      candidate,
      capture,
    )

    // 10M output tokens at 1,000 tokens/s on one replica = 10M/3.6M hours.
    const fullHours = 10_000_000 / 1_000 / 3_600
    expect(samples.find(entry => entry.capacityUsedPct === 100)?.hardware?.deployment?.billableInstanceHours)
      .toBeCloseTo(fullHours, 6)
    expect(samples.find(entry => entry.capacityUsedPct === 100)?.hardware?.monthlyCost)
      .toBeCloseTo(fullHours * 10, 6)
    // Runtime tracks the sampled demand, not the frozen instance count: half
    // the demand bills half the hours on the same single instance.
    expect(samples.find(entry => entry.capacityUsedPct === 50)?.hardware?.deployment?.billableInstanceHours)
      .toBeCloseTo(fullHours / 2, 6)
    expect(samples.find(entry => entry.capacityUsedPct === 25)?.hardware?.deployment?.billableInstanceHours)
      .toBeCloseTo(fullHours / 4, 6)
    // Zero demand bills nothing under scale-to-zero, frozen deployment or not.
    expect(samples[0]?.hardware?.monthlyCost).toBe(0)
    // The frozen layout never resizes while runtime scales.
    for (const sample of samples) {
      expect(sample.hardware?.deployment?.wholeUnits).toBe(1)
      expect(sample.hardware?.deployment?.availableReplicaSlots).toBe(1)
    }
  })

  it('applies the runtime buffer to frozen scale-to-zero runtime at every fill level', () => {
    const outputOnly: HybridWorkload = {
      ...workload,
      monthlyInputTokens: 0,
      monthlyOutputTokens: 10_000_000,
      averageInputTokens: 1,
      averageOutputTokens: 1_000,
    }
    const capture: Parameters<typeof fixedDeploymentSamples>[4] = {
      path: 'rented',
      candidateKey: 'example',
      systemId: 'fixed_gpu',
      gpusPerReplica: 1,
      wholeUnits: 1,
      billedGpus: 1,
      availableReplicaSlots: 1,
      planningTokenCapacity: 10_000_000,
      basisSignature: 'example',
    }
    const samples = fixedDeploymentSamples(
      outputOnly,
      hostedPrice,
      { ...assumptions, cloudBillingMode: 'scale-to-zero', cloudRuntimeBufferPct: 25 },
      candidate,
      capture,
    )

    // The existing buffer multiplies the proportional runtime by 1.25.
    const fullHours = 10_000_000 / 1_000 / 3_600
    for (const pct of [25, 50, 100] as const) {
      expect(samples.find(entry => entry.capacityUsedPct === pct)?.hardware?.deployment?.billableInstanceHours)
        .toBeCloseTo(fullHours * pct / 100 * 1.25, 6)
    }
    expect(samples.find(entry => entry.capacityUsedPct === 50)?.hardware?.monthlyCost)
      .toBeCloseTo(fullHours / 2 * 1.25 * 10, 6)
  })

  it('keeps active-window compute fixed across the frozen range', () => {
    const option = {
      key: 'rented' as const,
      label: 'Fixed GPU', monthlyCost: 1, fullyLoadedMonthlyCost: 1, marginalMonthlyCost: 1,
      costPerMillionTokens: 0, costPerMillionOutputTokens: 0, costPerRequest: 0,
      candidate, gpuCount: 1, billedGpuCount: 1, replicas: 1, utilizationPct: 100,
      breakdown: [], deployment: {
        requiredReplicas: 1, requiredGpus: 1, billedGpus: 1, wholeUnits: 1,
        availableReplicaSlots: 1, planningTokenCapacity: 10_000_000,
        planningOutputCapacity: 2_000_000, planningCapacityUsedPct: 100,
        effectiveBillingMode: 'active-window' as const, billableInstanceHours: 730,
      },
    }
    const capture = captureFor(option)!
    const samples = fixedDeploymentSamples(
      workload,
      hostedPrice,
      { ...assumptions, cloudBillingMode: 'active-window' },
      candidate,
      capture,
    )

    // One whole instance billed for the workload's processing window at every
    // fill level, including zero demand: the frozen deployment keeps billing.
    expect(samples.length).toBeGreaterThan(1)
    for (const sample of samples) {
      expect(sample.hardware?.deployment?.billableInstanceHours).toBe(730)
      expect(sample.hardware?.monthlyCost).toBe(730 * 10)
      expect(sample.hardware?.deployment?.wholeUnits).toBe(1)
    }
  })
})

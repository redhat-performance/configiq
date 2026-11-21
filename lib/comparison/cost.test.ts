import { describe, expect, it } from 'vitest'

import { comparisonCost } from './cost'

describe('comparisonCost', () => {
  it('uses the same month length and amortization window as the result page', () => {
    const cost = comparisonCost({ gpus: 8, cloudRatePerHour: 2.5, hardwareCostUsd: 20_000 })

    expect(cost.cloudMonthlyUsd).toBe(8 * 2.5 * 730)
    expect(cost.cloudFiveYearUsd).toBe(8 * 2.5 * 730 * 60)
    expect(cost.selfHostedMonthlyUsd).toBe((8 * 20_000) / 60)
    expect(cost.selfHostedFiveYearUsd).toBe(8 * 20_000)
  })

  it('leaves a figure null when its rate or price is unknown', () => {
    const cost = comparisonCost({ gpus: 4, cloudRatePerHour: null, hardwareCostUsd: null })

    expect(cost).toEqual({
      cloudMonthlyUsd: null,
      cloudFiveYearUsd: null,
      selfHostedMonthlyUsd: null,
      selfHostedFiveYearUsd: null,
    })
  })

  it('treats a negative rate or price as unknown rather than as a credit', () => {
    const cost = comparisonCost({ gpus: 4, cloudRatePerHour: -1, hardwareCostUsd: -1 })

    expect(cost.cloudMonthlyUsd).toBeNull()
    expect(cost.selfHostedMonthlyUsd).toBeNull()
  })

  it('never divides by zero GPUs', () => {
    const cost = comparisonCost({ gpus: 0, cloudRatePerHour: 2, hardwareCostUsd: 100 })

    expect(cost.cloudMonthlyUsd).toBe(0)
    expect(cost.selfHostedMonthlyUsd).toBe(0)
  })
})

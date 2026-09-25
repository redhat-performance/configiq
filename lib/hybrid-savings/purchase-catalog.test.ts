import { describe, expect, it } from 'vitest'
import {
  hasServerPurchaseConfiguration,
  resolveCompatibleServerPurchaseConfigurations,
} from './purchase-catalog'

describe('purchased server catalogue', () => {
  it('keeps unsupported or differently packaged GPUs out of purchased comparisons', () => {
    expect(hasServerPurchaseConfiguration('gb300')).toBe(false)
    expect(hasServerPurchaseConfiguration('a100_pcie')).toBe(false)
    expect(hasServerPurchaseConfiguration('h100_pcie')).toBe(false)
    expect(resolveCompatibleServerPurchaseConfigurations('a100_pcie', 1)).toEqual([])
    expect(resolveCompatibleServerPurchaseConfigurations('h100_pcie', 1)).toEqual([])
  })

  it('retains larger complete servers as explicit packing candidates', () => {
    const configurations = resolveCompatibleServerPurchaseConfigurations('l40s', 1)
    expect(configurations.map(configuration => configuration.gpuCount)).toEqual([1, 2, 4, 8])
    expect(configurations[0]).toMatchObject({
      purchasePrice: 35_000,
      installationCost: 2_500,
    })
    expect(configurations.find(configuration => configuration.gpuCount === 8)?.purchasePrice)
      .toBe(150_000)
  })

  it('does not infer purchased coverage from per-GPU pricing data', () => {
    expect(hasServerPurchaseConfiguration('l40s')).toBe(true)
    expect(hasServerPurchaseConfiguration('future_gpu')).toBe(false)
  })
})

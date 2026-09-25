import { describe, expect, it } from 'vitest'
import { normalizeCloudRates, resolveCloudRate } from './useCostings'

describe('cloud-rate normalization', () => {
  it('converts legacy AWS eight-GPU instance rates into GPU-hour rates', () => {
    const normalized = normalizeCloudRates('h100_sxm', 'aws.us-east-1', {
      on_demand: 55.04,
      reserved_1yr: null,
      reserved_3yr: null,
      spot_median: null,
    })

    expect(normalized?.on_demand).toBeCloseTo(6.88)
    expect(normalized?.gpus_per_instance).toBe(8)
  })

  it('does not present the legacy Azure low-priority value as on-demand', () => {
    const normalized = normalizeCloudRates('h100_sxm', 'azure.eastus', {
      on_demand: 2.792,
      reserved_1yr: null,
      reserved_3yr: null,
      spot_median: 2.579808,
    })

    expect(normalized?.on_demand).toBeNull()
    expect(normalized?.spot_median).toBeCloseTo(1.289904)
    expect(normalized?.gpus_per_instance).toBe(2)
  })

  it('does not divide records already declared per GPU-hour', () => {
    const normalized = normalizeCloudRates('h100_sxm', 'aws.us-east-1', {
      on_demand: 6.88,
      reserved_1yr: null,
      reserved_3yr: null,
      spot_median: null,
      rate_basis: 'gpu_hour',
      gpus_per_instance: 8,
    })

    expect(normalized?.on_demand).toBeCloseTo(6.88)
  })

  it('rejects an instance-hour rate without an instance GPU count', () => {
    const normalized = normalizeCloudRates('unknown_gpu', 'provider.region', {
      on_demand: 24,
      reserved_1yr: null,
      reserved_3yr: null,
      spot_median: null,
      rate_basis: 'instance_hour',
    })

    expect(normalized).toBeNull()
  })

  it('selects an on-demand rate without claiming instance topology', () => {
    const selected = resolveCloudRate({
      'aws.us-east-1': {
        on_demand: 6.88,
        reserved_1yr: null,
        reserved_3yr: null,
        spot_median: null,
        rate_basis: 'gpu_hour',
        gpus_per_instance: 8,
      },
    })

    expect(selected).toEqual({ rate: 6.88, provider: 'aws.us-east-1', kind: 'on_demand' })
  })
})

import { describe, expect, it } from 'vitest'
import {
  hasRentedCloudOffer,
  resolveRentedCloudOffers,
} from './cloud-offer-catalog'

const liveSystems = [
  'a100_sxm',
  'b200_sxm',
  'b300_sxm',
  'b60',
  'gb200',
  'gb300',
  'h100_sxm',
  'h200_sxm',
  'l40s',
  'rtx_pro_6000_server',
]

describe('complete rented cloud offers', () => {
  it('has defensible stable whole-instance offers for seven of the ten live systems', () => {
    const ready = liveSystems.filter(systemId => hasRentedCloudOffer(systemId))
    expect(ready).toEqual([
      'a100_sxm',
      'b200_sxm',
      'gb200',
      'h100_sxm',
      'h200_sxm',
      'l40s',
      'rtx_pro_6000_server',
    ])
  })

  it('retains different whole-instance sizes until after sizing', () => {
    const offers = resolveRentedCloudOffers('l40s')
    expect(new Set(offers.map(offer => offer.gpuCount))).toEqual(new Set([1, 4, 8]))
    expect(offers.every(offer => offer.hourlyCost > 0)).toBe(true)
  })

  it('keeps the exact instance price and provenance rather than an aggregate family rate', () => {
    const offers = resolveRentedCloudOffers('a100_sxm')
    const awsOffer = offers.find(offer => offer.id === 'aws-p4de-24xlarge')

    expect(awsOffer).toMatchObject({
      providerRegion: 'aws.us-east-1',
      instanceName: 'EC2 p4de.24xlarge',
      hourlyCost: 40.96,
      sourceLabel: 'AWS P4de public On-Demand price',
    })
  })

  it('honors the selected provider without collapsing its instance shapes', () => {
    const offers = resolveRentedCloudOffers('a100_sxm', 'gcp.us-central1')
    expect(offers.every(offer => offer.provider === 'gcp')).toBe(true)
    expect(new Set(offers.map(offer => offer.gpuCount))).toEqual(new Set([1, 2, 4, 8]))
  })

  it('does not let a spot rate silently beat stable on-demand offers', () => {
    const offers = resolveRentedCloudOffers('h200_sxm')
    expect(offers.some(offer => offer.rateKind === 'spot')).toBe(false)
  })

  it('does not treat a spot-only system as cost-ready by default', () => {
    expect(resolveRentedCloudOffers('b300_sxm')).toEqual([])
    expect(hasRentedCloudOffer('b300_sxm')).toBe(false)
  })

  it('does not rank a system without a complete instance identity', () => {
    expect(resolveRentedCloudOffers('gb300')).toHaveLength(0)
  })
})

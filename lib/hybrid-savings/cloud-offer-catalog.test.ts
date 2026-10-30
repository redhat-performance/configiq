import { describe, expect, it } from 'vitest'
import { hasRentedCloudOffer, resolveRentedCloudOffers } from './cloud-offer-catalog'
import { TEST_OFFERS, rentalOffer } from './__fixtures__/offers'

describe('API-supplied rental eligibility', () => {
  it('discovers supported systems from supplied offers, including future validated systems', () => {
    expect(hasRentedCloudOffer([], 'l40s')).toBe(false)
    expect(hasRentedCloudOffer(TEST_OFFERS.rented, 'l40s')).toBe(true)
    expect(hasRentedCloudOffer([rentalOffer({ systemId: 'future' })], 'future')).toBe(true)
  })
  it('retains whole-instance sizes and exact prices', () => {
    const offers = resolveRentedCloudOffers(TEST_OFFERS.rented, 'l40s')
    expect(new Set(offers.map(offer => offer.gpuCount))).toEqual(new Set([1, 4, 8]))
    expect(offers.find(offer => offer.id === 'aws-g6e-xlarge')?.hourlyCost).toBe(1.861)
  })
  it('honors exact provider regions, then provider, without inventing missing offers', () => {
    for (const [preference, provider] of [['aws.us-east-1', 'aws'], ['coreweave.other', 'coreweave']]) {
      const offers = resolveRentedCloudOffers(TEST_OFFERS.rented, 'l40s', preference)
      expect(offers.length).toBeGreaterThan(0)
      expect(offers.every(offer => offer.provider === provider)).toBe(true)
    }
    expect(resolveRentedCloudOffers([], 'l40s', 'aws.us-east-1')).toEqual([])
  })
  it('does not silently rank spot, committed blocks or inference-only prices as general rentals', () => {
    expect(resolveRentedCloudOffers(TEST_OFFERS.rented, 'l40s').some(offer => offer.id === 'restricted')).toBe(false)
    expect(hasRentedCloudOffer(TEST_OFFERS.rented, 'b300_sxm')).toBe(false)
    expect(resolveRentedCloudOffers(TEST_OFFERS.rented, 'gb300')).toEqual([])
    const restrictedProvider = [rentalOffer({ provider: 'restricted', providerRegion: 'restricted.region', eligibility: 'inference_platform' }), rentalOffer()]
    expect(resolveRentedCloudOffers(restrictedProvider, 'l40s', 'restricted.region')).toEqual([restrictedProvider[1]])
    const conditional = [rentalOffer({ provider: 'restricted', providerRegion: 'restricted.region', rateKind: 'capacity_block' }), rentalOffer()]
    expect(resolveRentedCloudOffers(conditional, 'l40s', 'restricted.region')).toEqual([conditional[1]])
    expect(hasRentedCloudOffer(conditional.slice(0, 1), 'l40s')).toBe(false)
  })
})

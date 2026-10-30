import { describe, expect, it } from 'vitest'
import { hasServerPurchaseConfiguration, resolveCompatibleServerPurchaseConfigurations } from './purchase-catalog'
import { TEST_OFFERS, serverOffer } from './__fixtures__/offers'

describe('API-supplied server configurations', () => {
  it('requires an explicitly supplied complete server, not a GPU-card price', () => {
    expect(hasServerPurchaseConfiguration([], 'l40s')).toBe(false)
    expect(hasServerPurchaseConfiguration(TEST_OFFERS.purchased, 'l40s')).toBe(true)
    expect(resolveCompatibleServerPurchaseConfigurations(TEST_OFFERS.purchased, 'a100_pcie', 1)).toEqual([])
  })
  it('retains larger servers for packing and preserves complete acquisition prices', () => {
    const servers = resolveCompatibleServerPurchaseConfigurations(TEST_OFFERS.purchased, 'l40s', 1)
    expect(servers.map(server => server.gpuCount)).toEqual([1, 2, 4, 8])
    expect(servers[0]).toMatchObject({ purchasePrice: 35000, installationCost: 2500 })
    expect(servers.at(-1)?.purchasePrice).toBe(150000)
    expect(resolveCompatibleServerPurchaseConfigurations(TEST_OFFERS.purchased, 'l40s', 9)).toEqual([])
  })
  it('keeps distinct server identities even when their GPU counts match', () => {
    const servers = [serverOffer(), serverOffer({ id: 'another-vendor', purchasePrice: 40000 })]
    expect(resolveCompatibleServerPurchaseConfigurations(servers, 'l40s', 1)).toHaveLength(2)
    for (const invalid of [0, NaN, 1.5]) expect(resolveCompatibleServerPurchaseConfigurations(servers, 'l40s', invalid)).toEqual([])
  })
})

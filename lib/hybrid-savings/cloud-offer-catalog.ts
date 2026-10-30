/** Validated complete-instance records supplied by aicostings, not a local catalogue. */
export type RentedRateKind = 'on_demand' | 'spot' | 'capacity_block'
export interface RentedCloudOffer {
  id: string
  systemId: string
  provider: string
  providerRegion: string
  instanceName: string
  gpuCount: number
  hourlyCost: number
  rateKind: RentedRateKind
  eligibility: 'standard' | 'inference_platform'
  sourceLabel: string
  sourceUrl: string | null
  sourceDate: string | null
  collectionKind: 'automated' | 'curated'
  retrievedAt: string | null
  stale: boolean
  maxInstancesPerReplica: number | null
  interconnect: string | null
}
function providerId(providerRegion: string): string {
  return providerRegion.split('.')[0].toLowerCase()
}
export function resolveRentedCloudOffers(
  offers: readonly RentedCloudOffer[], systemId: string, preferredProviderRegion?: string | null,
): RentedCloudOffer[] {
  // Apply eligibility before provider preference: a provider's conditional
  // rate must not hide another provider's usable offer.
  let resolved = offers.filter(offer => offer.systemId === systemId && offer.rateKind === 'on_demand' && offer.eligibility === 'standard')
  if (preferredProviderRegion) {
    const exact = resolved.filter(offer => offer.providerRegion === preferredProviderRegion)
    const provider = resolved.filter(offer => offer.provider === providerId(preferredProviderRegion))
    if (exact.length) resolved = exact
    else if (provider.length) resolved = provider
  }
  return resolved
}
export function hasRentedCloudOffer(offers: readonly RentedCloudOffer[], systemId: string): boolean {
  return resolveRentedCloudOffers(offers, systemId).length > 0
}

/** Test-only snapshots: never imported by production components. */
import type { InfrastructureOffers } from '@/lib/api/infrastructure-offers'
import type { RentedCloudOffer } from '../cloud-offer-catalog'
import type { ServerPurchaseConfiguration } from '../purchase-catalog'

export function rentalOffer(overrides: Partial<RentedCloudOffer> = {}): RentedCloudOffer {
  return { id: 'aws-g6e-xlarge', systemId: 'l40s', provider: 'aws', providerRegion: 'aws.us-east-1',
    instanceName: 'EC2 g6e.xlarge', gpuCount: 1, hourlyCost: 1.861, rateKind: 'on_demand', eligibility: 'standard',
    sourceLabel: 'AWS EC2 Linux On-Demand catalogue', sourceUrl: 'https://aws.amazon.com/ec2/pricing/on-demand/',
    sourceDate: '2026-08-23', collectionKind: 'curated', retrievedAt: null, stale: false,
    maxInstancesPerReplica: null, interconnect: null, ...overrides }
}
export function serverOffer(overrides: Partial<ServerPurchaseConfiguration> = {}): ServerPurchaseConfiguration {
  return { id: 'l40s:server:1', systemId: 'l40s', gpuCount: 1, purchasePrice: 35_000, installationCost: 2500,
    sourceLabel: 'Test complete server', sourceUrl: 'https://example.com/server', sourceDate: '2026-08-01',
    indicative: true, collectionKind: 'curated', retrievedAt: null, stale: false, ...overrides }
}
export const TEST_OFFERS: InfrastructureOffers = {
  rented: [rentalOffer(), rentalOffer({ id: 'aws-g6e-12xlarge', instanceName: 'g6e.12xlarge', gpuCount: 4, hourlyCost: 10.49264 }),
    rentalOffer({ id: 'aws-g6e-48xlarge', instanceName: 'g6e.48xlarge', gpuCount: 8, hourlyCost: 30.13118 }),
    rentalOffer({ id: 'coreweave-l40s-8', provider: 'coreweave', providerRegion: 'coreweave.north-america',
      instanceName: 'CoreWeave 8× L40S', gpuCount: 8, hourlyCost: 18 }),
    rentalOffer({ id: 'restricted', eligibility: 'inference_platform', hourlyCost: 0.01 }),
    rentalOffer({ id: 'spot', systemId: 'b300_sxm', rateKind: 'spot', hourlyCost: 0.01 })],
  purchased: [serverOffer(), serverOffer({ id: 'l40s:server:2', gpuCount: 2, purchasePrice: 56000, installationCost: 3900 }),
    serverOffer({ id: 'l40s:server:4', gpuCount: 4, purchasePrice: 94000, installationCost: 6600 }),
    serverOffer({ id: 'l40s:server:8', gpuCount: 8, purchasePrice: 150000, installationCost: 10500 })],
  sources: { curated: { state: 'curated', last_success: null, message: 'Manually maintained estimates.' } }, rejectedRecords: 0,
}

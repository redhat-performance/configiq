export type RentedRateKind = 'on_demand' | 'spot' | 'capacity_block'

/**
 * One complete, billable cloud shape. Prices are for the whole instance,
 * never for a bare GPU. This is deliberately more specific than the
 * aicostings aggregate cloud-rate response because instance topology is
 * required to price sparse and multi-GPU workloads correctly.
 */
export interface RentedCloudOffer {
  id: string
  systemId: string
  provider: string
  providerRegion: string
  instanceName: string
  gpuCount: number
  hourlyCost: number
  /** Supporting infrastructure and observability allowance for this offer. */
  directInfrastructureMonthly: number
  rateKind: RentedRateKind
  sourceLabel: string
  sourceUrl: string | null
  sourceDate: string | null
  maxInstancesPerReplica: number | null
  interconnect: string | null
}

const AWS_SOURCE = 'https://aws.amazon.com/ec2/pricing/on-demand/'
const AWS_CAPACITY_SOURCE = 'https://aws.amazon.com/ec2/capacityblocks/pricing/'
const GCP_SOURCE = 'https://cloud.google.com/products/compute/pricing/accelerator-optimized'
const COREWEAVE_SOURCE = 'https://www.coreweave.com/pricing'
const SNAPSHOT_DATE = '2026-08-23'
const NEW_OFFER_DATE = '2026-09-07'
const CURRENT_OFFER_DATE = '2026-09-20'

function offer(
  id: string,
  systemId: string,
  provider: string,
  providerRegion: string,
  instanceName: string,
  gpuCount: number,
  hourlyCost: number,
  sourceLabel: string,
  sourceUrl: string | null,
  sourceDate: string | null = SNAPSHOT_DATE,
  rateKind: RentedRateKind = 'on_demand',
  maxInstancesPerReplica: number | null = null,
  interconnect: string | null = null,
): RentedCloudOffer {
  return {
    id,
    systemId,
    provider,
    providerRegion,
    instanceName,
    gpuCount,
    hourlyCost,
    // Keep this aligned with the provider-backed profiles in the standalone
    // LLM Cost Calculator. The global assumption remains a fallback for
    // candidates that do not have a complete cloud-offer record.
    directInfrastructureMonthly: 500,
    rateKind,
    sourceLabel,
    sourceUrl,
    sourceDate,
    maxInstancesPerReplica,
    interconnect,
  }
}

/**
 * Dated complete-instance snapshots shared with the standalone LLM Cost
 * Calculator. They fill the topology/SKU gap in the current aggregate
 * aicostings response and make the two calculators compare the same shapes.
 */
export const RENTED_CLOUD_OFFERS: RentedCloudOffer[] = [
  offer('aws-g6e-xlarge', 'l40s', 'aws', 'aws.us-east-1', 'EC2 g6e.xlarge', 1, 1.861, 'AWS EC2 Linux On-Demand catalogue', AWS_SOURCE),
  offer('aws-g6e-12xlarge', 'l40s', 'aws', 'aws.us-east-1', 'EC2 g6e.12xlarge', 4, 10.49264, 'AWS EC2 Linux On-Demand catalogue', AWS_SOURCE),
  offer('aws-g6e-48xlarge', 'l40s', 'aws', 'aws.us-east-1', 'EC2 g6e.48xlarge', 8, 30.13118, 'AWS EC2 Linux On-Demand catalogue', AWS_SOURCE),
  offer('coreweave-l40s-1', 'l40s', 'coreweave', 'coreweave.north-america', 'CoreWeave Inference 1× L40S', 1, 2.25, 'CoreWeave Inference On-Demand price', COREWEAVE_SOURCE),
  offer('coreweave-l40s-8', 'l40s', 'coreweave', 'coreweave.north-america', 'CoreWeave 8× L40S', 8, 18, 'CoreWeave On-Demand GPU instance price', COREWEAVE_SOURCE),

  offer('aws-p4de-24xlarge', 'a100_sxm', 'aws', 'aws.us-east-1', 'EC2 p4de.24xlarge', 8, 40.96, 'AWS P4de public On-Demand price', 'https://aws.amazon.com/ec2/instance-types/p4/'),
  offer('gcp-a2-ultragpu-1g', 'a100_sxm', 'gcp', 'gcp.public-reference', 'GCP a2-ultragpu-1g', 1, 5.06879789, 'Google Cloud accelerator-optimized On-Demand price', GCP_SOURCE),
  offer('gcp-a2-ultragpu-2g', 'a100_sxm', 'gcp', 'gcp.public-reference', 'GCP a2-ultragpu-2g', 2, 10.137595781, 'Google Cloud accelerator-optimized On-Demand price', GCP_SOURCE),
  offer('gcp-a2-ultragpu-4g', 'a100_sxm', 'gcp', 'gcp.public-reference', 'GCP a2-ultragpu-4g', 4, 20.275191562, 'Google Cloud accelerator-optimized On-Demand price', GCP_SOURCE),
  offer('gcp-a2-ultragpu-8g', 'a100_sxm', 'gcp', 'gcp.public-reference', 'GCP a2-ultragpu-8g', 8, 40.550383123, 'Google Cloud accelerator-optimized On-Demand price', GCP_SOURCE),
  offer('coreweave-a100-1', 'a100_sxm', 'coreweave', 'coreweave.north-america', 'CoreWeave Inference 1× A100 80 GB', 1, 2.7, 'CoreWeave Inference On-Demand price', COREWEAVE_SOURCE),
  offer('coreweave-a100-8', 'a100_sxm', 'coreweave', 'coreweave.north-america', 'CoreWeave 8× A100 80 GB', 8, 21.6, 'CoreWeave On-Demand GPU instance price', COREWEAVE_SOURCE),

  offer('aws-p5-4xlarge', 'h100_sxm', 'aws', 'aws.us-east-1', 'EC2 p5.4xlarge', 1, 6.88, 'AWS EC2 Linux On-Demand catalogue', 'https://aws.amazon.com/ec2/instance-types/p5/'),
  offer('aws-p5-48xlarge', 'h100_sxm', 'aws', 'aws.us-east-1', 'EC2 p5.48xlarge', 8, 55.04, 'AWS EC2 Linux On-Demand catalogue', 'https://aws.amazon.com/ec2/instance-types/p5/', SNAPSHOT_DATE, 'on_demand', 8, 'EFA and GPUDirect RDMA capable cluster'),
  offer('gcp-a3-highgpu-8g', 'h100_sxm', 'gcp', 'gcp.public-reference', 'GCP a3-highgpu-8g', 8, 88.490000119, 'Google Cloud accelerator-optimized On-Demand price', GCP_SOURCE),
  offer('coreweave-h100-1', 'h100_sxm', 'coreweave', 'coreweave.north-america', 'CoreWeave Inference 1× H100', 1, 6.16, 'CoreWeave Inference On-Demand price', COREWEAVE_SOURCE),
  offer('coreweave-h100-8', 'h100_sxm', 'coreweave', 'coreweave.north-america', 'CoreWeave 8× H100', 8, 49.24, 'CoreWeave On-Demand GPU instance price', COREWEAVE_SOURCE),

  offer('aws-p5en-48xlarge', 'h200_sxm', 'aws', 'aws.us-east-1', 'EC2 p5en.48xlarge', 8, 63.296, 'AWS EC2 Linux On-Demand catalogue', 'https://aws.amazon.com/ec2/instance-types/p5/', SNAPSHOT_DATE, 'on_demand', 8, 'EFA and GPUDirect RDMA capable cluster'),
  offer('gcp-a3-ultragpu-8g', 'h200_sxm', 'gcp', 'gcp.public-reference', 'GCP a3-ultragpu-8g', 8, 84.806908493, 'Google Cloud accelerator-optimized On-Demand price', GCP_SOURCE),
  offer('coreweave-h200-1', 'h200_sxm', 'coreweave', 'coreweave.north-america', 'CoreWeave Inference 1× H200', 1, 6.31, 'CoreWeave Inference On-Demand price', COREWEAVE_SOURCE),
  offer('coreweave-h200-8', 'h200_sxm', 'coreweave', 'coreweave.north-america', 'CoreWeave 8× H200', 8, 50.44, 'CoreWeave On-Demand GPU instance price', COREWEAVE_SOURCE),

  offer('aws-p6-b200-48xlarge', 'b200_sxm', 'aws', 'aws.us-east-1', 'EC2 p6-b200.48xlarge', 8, 98.84, 'AWS EC2 Capacity Blocks effective instance-hour price', AWS_CAPACITY_SOURCE, NEW_OFFER_DATE, 'capacity_block', 8, 'EFA and GPUDirect RDMA capable cluster'),
  offer('coreweave-b200-8', 'b200_sxm', 'coreweave', 'coreweave.north-america', 'CoreWeave 8× HGX B200', 8, 68.8, 'CoreWeave On-Demand GPU instance price', COREWEAVE_SOURCE, NEW_OFFER_DATE, 'on_demand', 8, 'InfiniBand-capable GPU cluster'),

  // CoreWeave currently publishes a B300 spot price but asks customers to
  // contact sales for its on-demand price. Keep the rate explicitly marked as
  // spot so the UI does not present it as a stable list-price commitment.
  offer('coreweave-b300-8-spot', 'b300_sxm', 'coreweave', 'coreweave.north-america', 'CoreWeave 8× HGX B300', 8, 35.84, 'CoreWeave public spot GPU instance price', COREWEAVE_SOURCE, CURRENT_OFFER_DATE, 'spot'),

  offer('aws-p6e-gb200x36', 'gb200', 'aws', 'aws.dallas-local-zone', 'EC2 u-p6e-gb200x36 UltraServer', 36, 380.952, 'AWS EC2 Capacity Blocks effective UltraServer-hour price', AWS_CAPACITY_SOURCE, NEW_OFFER_DATE, 'capacity_block', 1, 'NVLink and EFAv4 UltraServer fabric'),
  offer('coreweave-gb200-4', 'gb200', 'coreweave', 'coreweave.north-america', 'CoreWeave 4× GB200 NVL72', 4, 42, 'CoreWeave On-Demand GPU instance price', COREWEAVE_SOURCE, NEW_OFFER_DATE, 'on_demand', 18, 'NVLink and 400 Gb/s NDR InfiniBand'),

  offer('gcp-g4-standard-48', 'rtx_pro_6000_server', 'gcp', 'gcp.public-reference', 'GCP g4-standard-48', 1, 4.49993, 'Google Cloud accelerator-optimized On-Demand price', GCP_SOURCE, NEW_OFFER_DATE),
  offer('coreweave-rtx-pro-6000-8', 'rtx_pro_6000_server', 'coreweave', 'coreweave.north-america', 'CoreWeave 8× RTX PRO 6000 Blackwell', 8, 20, 'CoreWeave On-Demand GPU instance price', COREWEAVE_SOURCE, NEW_OFFER_DATE, 'on_demand', 8, 'InfiniBand-capable GPU cluster'),
]

function providerId(providerRegion: string): string {
  return providerRegion.split('.')[0].toLowerCase()
}

/** Return every complete offer for a system; topology filtering happens after sizing. */
export function resolveRentedCloudOffers(
  systemId: string,
  preferredProviderRegion?: string | null,
): RentedCloudOffer[] {
  // The shared aicostings rate is aggregated by GPU family and provider region.
  // Without an instance SKU it cannot safely replace a complete offer's price.
  let resolved = RENTED_CLOUD_OFFERS.filter(candidate => candidate.systemId === systemId)

  if (preferredProviderRegion) {
    const exact = resolved.filter(candidate => candidate.providerRegion === preferredProviderRegion)
    const preferredProvider = providerId(preferredProviderRegion)
    const providerMatches = resolved.filter(candidate => candidate.provider === preferredProvider)
    if (exact.length > 0) resolved = exact
    else if (providerMatches.length > 0) resolved = providerMatches
  }

  // Spot/marketplace prices are interruption-prone and must not silently beat
  // stable list-price offers. A spot-only system remains available to
  // AISimulators for sizing, but it is not cost-ready for the default
  // on-demand comparison. Spot can be added later as an explicit user-selected
  // scenario with its interruption and availability assumptions made visible.
  const stable = resolved.filter(candidate => candidate.rateKind !== 'spot')
  return stable
    .filter(candidate => candidate.hourlyCost > 0 && candidate.gpuCount > 0)
}

export function hasRentedCloudOffer(
  systemId: string,
): boolean {
  return resolveRentedCloudOffers(systemId).length > 0
}

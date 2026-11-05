import { z } from 'zod'
import type { RentedCloudOffer } from '@/lib/hybrid-savings/cloud-offer-catalog'
import type { ServerPurchaseConfiguration } from '@/lib/hybrid-savings/purchase-catalog'

const sourceSchema = z.object({
  kind: z.enum(['automated', 'curated']), label: z.string().min(1),
  url: z.url().refine(value => /^https?:\/\//.test(value)).nullable().default(null),
  price_date: z.iso.date().nullable().default(null),
  retrieved_at: z.iso.datetime({ offset: true }).nullable().default(null),
  effective_at: z.iso.datetime({ offset: true }).nullable().default(null),
  stale: z.boolean().default(false),
})
const rentalSchema = z.object({
  id: z.string().min(1), system_id: z.string().min(1), provider: z.string().min(1),
  provider_region: z.string().min(1), instance_name: z.string().min(1),
  gpu_count: z.number().int().positive(), hourly_cost: z.number().positive(),
  currency: z.literal('USD'), rate_basis: z.literal('instance_hour'),
  rate_kind: z.enum(['on_demand', 'spot', 'capacity_block']),
  eligibility: z.enum(['standard', 'inference_platform']),
  max_instances_per_replica: z.number().int().positive().nullable().default(null),
  interconnect: z.string().nullable().default(null), source: sourceSchema,
})
const serverSchema = z.object({
  id: z.string().min(1), system_id: z.string().min(1), gpu_count: z.number().int().positive(),
  purchase_price: z.number().positive(), installation_cost: z.number().nonnegative().nullable().default(null),
  currency: z.literal('USD'), price_basis: z.literal('complete_server'), includes: z.string().min(1),
  indicative: z.boolean(), source: sourceSchema,
})
const responseSchema = z.object({
  schema_version: z.literal(1), rented: z.array(z.unknown()), purchased: z.array(z.unknown()),
  sources: z.record(z.string(), z.object({
    state: z.enum(['fresh', 'stale', 'unconfigured', 'unavailable', 'curated']),
    last_success: z.iso.datetime({ offset: true }).nullable().default(null),
    message: z.string().nullable().default(null),
  })),
})

export interface InfrastructureOffers {
  rented: RentedCloudOffer[]
  purchased: ServerPurchaseConfiguration[]
  sources: z.infer<typeof responseSchema>['sources']
  rejectedRecords: number
}

export function parseInfrastructureOffers(raw: unknown): InfrastructureOffers {
  const envelope = responseSchema.parse(raw)
  let rejectedRecords = 0
  const ids = new Set<string>()
  const rented = envelope.rented.flatMap(record => {
    const parsed = rentalSchema.safeParse(record)
    if (!parsed.success || ids.has(parsed.data.id)) { rejectedRecords++; return [] }
    const offer = parsed.data
    ids.add(offer.id)
    return [{
      id: offer.id, systemId: offer.system_id, provider: offer.provider, providerRegion: offer.provider_region,
      instanceName: offer.instance_name,
      gpuCount: offer.gpu_count, hourlyCost: offer.hourly_cost,
      rateKind: offer.rate_kind, eligibility: offer.eligibility, maxInstancesPerReplica: offer.max_instances_per_replica,
      interconnect: offer.interconnect, sourceLabel: offer.source.label, sourceUrl: offer.source.url,
      sourceDate: offer.source.price_date ?? offer.source.effective_at?.slice(0, 10) ?? null,
      collectionKind: offer.source.kind, retrievedAt: offer.source.retrieved_at, stale: offer.source.stale,
    }]
  })
  const purchased = envelope.purchased.flatMap(record => {
    const parsed = serverSchema.safeParse(record)
    if (!parsed.success || ids.has(parsed.data.id)) { rejectedRecords++; return [] }
    const offer = parsed.data
    ids.add(offer.id)
    return [{
      id: offer.id, systemId: offer.system_id, gpuCount: offer.gpu_count, purchasePrice: offer.purchase_price,
      installationCost: offer.installation_cost, indicative: offer.indicative,
      sourceLabel: offer.source.label, sourceUrl: offer.source.url,
      sourceDate: offer.source.price_date ?? offer.source.effective_at?.slice(0, 10) ?? null,
      collectionKind: offer.source.kind, retrievedAt: offer.source.retrieved_at, stale: offer.source.stale,
    }]
  })
  return { rented, purchased, sources: envelope.sources, rejectedRecords }
}

export async function fetchInfrastructureOffers(): Promise<InfrastructureOffers> {
  const response = await fetch('/api/costings/offers', {
    headers: { Accept: 'application/json' }, signal: AbortSignal.timeout(30_000),
  })
  if (!response.ok) throw new Error(response.status === 404
    ? 'The pricing service needs the complete-offers API upgrade. No local pricing fallback is used.'
    : `Infrastructure pricing unavailable (${response.status}).`)
  try {
    return parseInfrastructureOffers(await response.json())
  } catch {
    throw new Error('Infrastructure pricing returned an incompatible response. No local pricing fallback is used.')
  }
}

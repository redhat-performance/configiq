import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchInfrastructureOffers, parseInfrastructureOffers } from './infrastructure-offers'

const source = { kind: 'automated', label: 'Public provider', url: 'https://example.com/prices',
  price_date: null, retrieved_at: '2026-10-27T12:00:00+00:00', effective_at: null, stale: false }
const rental = { id: 'aws:region:sku', system_id: 'l40s', provider: 'aws', provider_region: 'aws.us-east-1',
  instance_name: 'g6e.48xlarge', gpu_count: 8, hourly_cost: 16, currency: 'USD', rate_basis: 'instance_hour',
  rate_kind: 'on_demand', eligibility: 'standard', source }
const server = { id: 'vendor:server', system_id: 'l40s', gpu_count: 8, purchase_price: 150000,
  installation_cost: 10500, currency: 'USD', price_basis: 'complete_server', includes: 'Chassis, CPU, RAM and GPUs',
  indicative: true, source: { ...source, kind: 'curated', retrieved_at: null, price_date: '2026-08-01' } }
const envelope = { schema_version: 1, rented: [rental], purchased: [server], sources: {} }
afterEach(() => vi.unstubAllGlobals())

describe('complete-offer API validation', () => {
  it('preserves complete prices, identities and distinct date meanings', () => {
    const data = parseInfrastructureOffers(envelope)
    expect(data.rejectedRecords).toBe(0)
    expect(data.rented[0]).toMatchObject({ id: rental.id, gpuCount: 8, hourlyCost: 16,
      instanceName: 'g6e.48xlarge', sourceDate: null, collectionKind: 'automated', retrievedAt: source.retrieved_at })
    expect(data.purchased[0]).toMatchObject({ purchasePrice: 150000, installationCost: 10500,
      sourceDate: '2026-08-01', retrievedAt: null, collectionKind: 'curated' })
  })
  it.each([
    { hourly_cost: 0 }, { gpu_count: 1.5 }, { hourly_cost: Infinity }, { currency: 'EUR' },
    { rate_basis: 'gpu_hour' }, { source: { ...source, url: 'javascript:alert(1)' } },
  ])('excludes incompatible records without dropping valid paths: %j', invalid => {
    const data = parseInfrastructureOffers({ ...envelope, rented: [{ ...rental, ...invalid }] })
    expect(data.rejectedRecords).toBe(1)
    expect(data.rented).toEqual([])
    expect(data.purchased).toHaveLength(1)
  })
  it('does not treat GPU-card prices as complete servers or rank duplicate IDs', () => {
    const data = parseInfrastructureOffers({ ...envelope, rented: [rental, rental],
      purchased: [{ ...server, price_basis: 'gpu' }] })
    expect(data.rejectedRecords).toBe(2)
    expect(data.rented).toHaveLength(1)
    expect(data.purchased).toEqual([])
  })
  it('rejects unsupported response versions instead of manufacturing prices', () => {
    expect(() => parseInfrastructureOffers({ ...envelope, schema_version: 2 })).toThrow()
  })
  it('reports an undeployed API without reading any local price fallback', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 404 })))
    await expect(fetchInfrastructureOffers()).rejects.toThrow('complete-offers API upgrade')
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe('/api/costings/offers')
  })
})

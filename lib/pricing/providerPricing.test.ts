import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchAllProviders } from './providerPricing'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('fetchAllProviders', () => {
  it('loads and normalizes provider-region rates from aicostings', async () => {
    const fetchMock = vi.fn(() => Promise.resolve(Response.json({
      systems: [{
        id: 'h100_sxm',
        name: 'H100 80GB',
        cloud_rates: {
          'aws.us-east-1': {
            on_demand: 8,
            rate_basis: 'instance_hour',
            gpus_per_instance: 8,
          },
          'gcp.us-central1': { on_demand: 3.2, rate_basis: 'gpu_hour' },
        },
      }],
    })))
    vi.stubGlobal('fetch', fetchMock)

    const providers = await fetchAllProviders()

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/costings/systems?include=cloud',
      expect.objectContaining({ headers: { Accept: 'application/json' } }),
    )
    expect(providers).toEqual([
      { id: 'aws.us-east-1', label: 'AWS (us-east-1)', gpus: [{ model: 'H100 80GB', price: 1 }] },
      { id: 'gcp.us-central1', label: 'GCP (us-central1)', gpus: [{ model: 'H100 80GB', price: 3.2 }] },
    ])
  })
})

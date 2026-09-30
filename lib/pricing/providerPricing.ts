// Provider pricing adapter for the aicostings systems API.

import { normalizeCloudRates } from '@/lib/hooks/useCostings'
import { CLIENT_CACHE_TTL_MS } from '@/lib/cache-policy'

export interface ProviderGpu {
  model: string
  price: number | null // $/hr, null if unavailable
}

export interface Provider {
  id: string
  label: string
  gpus: ProviderGpu[]
}

interface CacheEntry {
  data: Provider[]
  timestamp: number
}

let cache: CacheEntry | null = null

interface AicostingsSystem {
  id: string
  name: string
  cloud_rates: Record<string, {
    on_demand?: unknown
    rate_basis?: unknown
    gpus_per_instance?: unknown
    spot_median?: unknown
  }>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object'
}

function parseSystems(data: unknown): AicostingsSystem[] {
  if (!isRecord(data) || !Array.isArray(data.systems)) return []

  return data.systems.flatMap((value): AicostingsSystem[] => {
    if (!isRecord(value) || typeof value.id !== 'string' || !value.id.trim() ||
      typeof value.name !== 'string' || !isRecord(value.cloud_rates)) return []

    const cloudRates = Object.fromEntries(
      Object.entries(value.cloud_rates).flatMap(([providerRegion, rawRates]) => {
        if (!isRecord(rawRates)) return []
        return [[providerRegion, {
          on_demand: rawRates.on_demand,
          rate_basis: rawRates.rate_basis,
          gpus_per_instance: rawRates.gpus_per_instance,
          spot_median: rawRates.spot_median,
        }]]
      }),
    )
    return [{ id: value.id, name: value.name, cloud_rates: cloudRates }]
  })
}

function providerLabel(providerRegion: string): string {
  const [provider, ...region] = providerRegion.split('.')
  const label = provider.toUpperCase()
  return region.length > 0 ? `${label} (${region.join('.')})` : label
}

function parseAicostingsResponse(data: unknown): Provider[] {
  const providers = new Map<string, Provider>()
  for (const system of parseSystems(data)) {
    for (const [providerRegion, rates] of Object.entries(system.cloud_rates)) {
      const normalized = normalizeCloudRates(system.id, providerRegion, {
        on_demand: typeof rates.on_demand === 'number' ? rates.on_demand : null,
        reserved_1yr: null,
        reserved_3yr: null,
        spot_median: typeof rates.spot_median === 'number' ? rates.spot_median : null,
        rate_basis: rates.rate_basis === 'gpu_hour' || rates.rate_basis === 'instance_hour'
          ? rates.rate_basis
          : undefined,
        gpus_per_instance: typeof rates.gpus_per_instance === 'number'
          ? rates.gpus_per_instance
          : null,
      })
      if (!normalized) continue
      const price = normalized.on_demand ?? normalized.spot_median
      const provider = providers.get(providerRegion) ?? {
        id: providerRegion,
        label: providerLabel(providerRegion),
        gpus: [],
      }
      provider.gpus.push({ model: system.name, price })
      providers.set(providerRegion, provider)
    }
  }
  return [...providers.values()].filter(provider => provider.gpus.length > 0)
}

/** Fetch provider-region GPU rates from the same-origin aicostings proxy. */
export async function fetchAllProviders(): Promise<Provider[]> {
  if (cache && Date.now() - cache.timestamp < CLIENT_CACHE_TTL_MS.costings) {
    return cache.data
  }

  const response = await fetch('/api/costings/systems?include=cloud', {
    headers: { 'Accept': 'application/json' },
    signal: AbortSignal.timeout(10000),
  })
  if (!response.ok) {
    throw new Error(`aicostings /systems returned ${response.status}`)
  }

  const providers = parseAicostingsResponse(await response.json())
  cache = { data: providers, timestamp: Date.now() }
  return providers
}

/**
 * Get effective rate for a provider/GPU combination.
 * Checks overrides first, then aicostings data.
 */
export function getEffectiveRate(
  providerId: string,
  gpuModel: string,
  providers: Provider[],
  overrides: Record<string, number | undefined>
): number | null {
  const key = `${providerId}_${gpuModel}`

  // Check override first
  if (overrides[key] !== undefined) {
    return overrides[key]!
  }

  // Check aicostings data
  const provider = providers.find(p => p.id === providerId)
  const gpu = provider?.gpus.find(g => g.model === gpuModel)
  return gpu?.price ?? null
}

/**
 * Load user overrides from localStorage
 */
export function loadUserOverrides(): Record<string, number | undefined> {
  if (typeof window === 'undefined') return {}

  const overrides: Record<string, number | undefined> = {}

  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i)
    if (key?.startsWith('price_')) {
      const priceKey = key.substring(6) // Remove 'price_' prefix
      const value = localStorage.getItem(key)
      if (value) {
        const parsed = parseFloat(value)
        if (!isNaN(parsed)) {
          overrides[priceKey] = parsed
        }
      }
    }
  }

  return overrides
}

/**
 * Save user override to localStorage
 */
export function setUserOverride(providerId: string, gpuModel: string, price: number | undefined): void {
  if (typeof window === 'undefined') return

  const key = `price_${providerId}_${gpuModel}`

  if (price === undefined) {
    localStorage.removeItem(key)
  } else {
    localStorage.setItem(key, String(price))
  }
}

/**
 * Clear user override from localStorage
 */
export function clearUserOverride(providerId: string, gpuModel: string): void {
  setUserOverride(providerId, gpuModel, undefined)
}

/**
 * Load selected GPU per provider from localStorage
 */
export function loadSelectedGpus(providers: Provider[]): Record<string, string> {
  if (typeof window === 'undefined') return {}

  const selected: Record<string, string> = {}

  providers.forEach(p => {
    const stored = localStorage.getItem(`selectedGpu_${p.id}`)
    if (stored && p.gpus.some(g => g.model === stored)) {
      selected[p.id] = stored
    } else {
      // Default to first GPU
      selected[p.id] = p.gpus[0]?.model || ''
    }
  })

  return selected
}

/**
 * Save selected GPU for a provider to localStorage
 */
export function saveSelectedGpu(providerId: string, gpuModel: string): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(`selectedGpu_${providerId}`, gpuModel)
}

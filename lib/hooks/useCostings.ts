'use client'

import { useState, useEffect } from 'react'
import type { PricingSource } from '@/contexts/SettingsContext'
import { CLIENT_CACHE_TTL_MS } from '@/lib/cache-policy'

// ── Types ──────────────────────────────────────────────────────────────────

export interface FrontierModel {
  id: string
  name: string
  provider: string
  tier: 'fast' | 'balanced' | 'frontier'
  price_per_m_input: number
  price_per_m_output: number
  context_window: number | null
  updated_at: string | null
  /** Which feed this model's pricing came from (present in ?source=merged). */
  source?: 'openrouter' | 'litellm' | 'override'
}

export interface CloudRates {
  on_demand: number | null
  reserved_1yr: number | null
  reserved_3yr: number | null
  spot_median: number | null
  /** Unit of the published hourly values. Older aicostings responses omit it. */
  rate_basis?: 'gpu_hour' | 'instance_hour'
  /** Minimum whole-instance GPU count when the rate comes from a VM SKU. */
  gpus_per_instance?: number | null
}

export interface HardwareCost {
  new_usd: number | null
  new_usd_low: number | null
  new_usd_high: number | null
  indicative: boolean
  source_label: string | null
  source_url: string | null
  source_date: string | null
}

export interface SourceStatus {
  last_success: string | null
  last_error: { at: string; message: string } | null
  stale: boolean
}

export interface CostingsHealth {
  status: string
  version: string
  sources: Record<string, SourceStatus>
}

export interface CostingsData {
  models: FrontierModel[]
  // systemId → provider.region → rates
  gpuCloudRates: Map<string, Record<string, CloudRates>>
  gpuHardwareCosts: Map<string, HardwareCost>
  health: CostingsHealth | null
  modelsUpdatedAt: string | null
  modelsStale: boolean
  isLoading: boolean
  error: string | null
}

// A single cloud rate chosen for a GPU, tagged with where it came from and
// whether it is an on-demand or spot price, so callers can label it honestly.
export interface ResolvedCloudRate {
  rate: number
  provider: string // provider.region key, e.g. "aws.us-east-1"
  kind: 'on_demand' | 'spot'
}

// Compatibility for the current prototype API, whose AWS and Azure values are
// instance-hour prices but do not yet identify their unit or topology. The
// local aicostings implementation now emits these fields explicitly; this map
// prevents older shared responses from being multiplied as if they were
// already per-GPU rates.
const LEGACY_INSTANCE_GPU_COUNTS: Record<string, Partial<Record<'aws' | 'azure', number>>> = {
  h100_sxm: { aws: 8, azure: 2 },
  a100_sxm: { aws: 8, azure: 8 },
}

function positiveRate(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null
}

export function normalizeCloudRates(
  systemId: string,
  providerRegion: string,
  rates: CloudRates,
): CloudRates | null {
  const provider = providerRegion.split('.')[0] as 'aws' | 'azure' | string
  const explicitGpuCount = typeof rates.gpus_per_instance === 'number' && rates.gpus_per_instance > 0
    ? Math.ceil(rates.gpus_per_instance)
    : null
  const legacyGpuCount = provider === 'aws' || provider === 'azure'
    ? LEGACY_INSTANCE_GPU_COUNTS[systemId]?.[provider]
    : undefined
  // An instance-hour rate cannot be converted safely without its topology.
  // Reject incomplete explicit records rather than silently treating a whole
  // multi-GPU VM as a one-GPU offer. Legacy records retain the narrow mappings
  // above until the shared prototype consistently publishes both fields.
  if (rates.rate_basis === 'instance_hour' && explicitGpuCount === null) return null
  const gpusPerInstance = explicitGpuCount ?? legacyGpuCount ?? 1
  const divisor = rates.rate_basis === 'gpu_hour' ? 1 : gpusPerInstance

  let onDemand = positiveRate(rates.on_demand)
  let reservedOneYear = positiveRate(rates.reserved_1yr)
  let reservedThreeYear = positiveRate(rates.reserved_3yr)
  let spotMedian = positiveRate(rates.spot_median)

  // The prototype Azure scraper historically classified "Low Priority" as
  // on-demand. Legacy responses cannot distinguish it from the true Linux
  // consumption meter, so exclude that field until a response declares its
  // rate basis. The spot/low-priority value remains available and is labelled.
  if (provider === 'azure' && rates.rate_basis === undefined) onDemand = null

  onDemand = onDemand === null ? null : onDemand / divisor
  reservedOneYear = reservedOneYear === null ? null : reservedOneYear / divisor
  reservedThreeYear = reservedThreeYear === null ? null : reservedThreeYear / divisor
  spotMedian = spotMedian === null ? null : spotMedian / divisor
  if (onDemand === null && spotMedian === null) return null

  return {
    on_demand: onDemand,
    reserved_1yr: reservedOneYear,
    reserved_3yr: reservedThreeYear,
    spot_median: spotMedian,
    rate_basis: 'gpu_hour',
    gpus_per_instance: gpusPerInstance,
  }
}

// Pick a single representative cloud $/hr for a GPU from its per-provider rates.
// Preference order: the preferred provider's on-demand (then its spot), then the
// cheapest on-demand across all providers, then the cheapest spot. Returns null
// when no provider publishes any usable rate. Spot is only used as a fallback so
// callers can surface that it is not an on-demand price.
export function resolveCloudRate(
  cloudRates: Record<string, CloudRates> | undefined,
  preferredProvider?: string | null,
): ResolvedCloudRate | null {
  if (!cloudRates) return null

  if (preferredProvider) {
    const pr = cloudRates[preferredProvider]
    if (pr?.on_demand != null) return { rate: pr.on_demand, provider: preferredProvider, kind: 'on_demand' }
    if (pr?.spot_median != null) return { rate: pr.spot_median, provider: preferredProvider, kind: 'spot' }
  }

  let cheapestOnDemand: ResolvedCloudRate | null = null
  let cheapestSpot: ResolvedCloudRate | null = null
  for (const [provider, r] of Object.entries(cloudRates)) {
    if (r.on_demand != null && (cheapestOnDemand == null || r.on_demand < cheapestOnDemand.rate)) {
      cheapestOnDemand = { rate: r.on_demand, provider, kind: 'on_demand' }
    }
    if (r.spot_median != null && (cheapestSpot == null || r.spot_median < cheapestSpot.rate)) {
      cheapestSpot = { rate: r.spot_median, provider, kind: 'spot' }
    }
  }
  return cheapestOnDemand ?? cheapestSpot
}

// ── Module-level cache ─────────────────────────────────────────────────────
// Survives re-renders and navigation. Healthy data lasts one hour; incomplete
// startup snapshots retry after one minute without requiring a page reload.

export type CostingsScope = 'all' | 'models'
type CacheKey = `${PricingSource}:${CostingsScope}`
const RETRY_INTERVAL_MS = 60_000

// The fully-parsed result of one fetch, so a source's data is published as a
// single unit rather than via several shared mutable globals that could tear.
interface CostingsResult {
  models: FrontierModel[]
  cloudRates: Map<string, Record<string, CloudRates>>
  hardwareCosts: Map<string, HardwareCost>
  health: CostingsHealth | null
  modelsUpdatedAt: string | null
  modelsStale: boolean
}

// A models-only consumer must not seed empty infrastructure maps for other
// pages. Source/scope keys also isolate source changes during in-flight work.
const cached = new Map<CacheKey, { result: CostingsResult; updatedAt: number }>()
const inFlight = new Map<CacheKey, Promise<CostingsResult>>()

function cacheTtl(result: CostingsResult, scope: CostingsScope): number {
  const incomplete = result.modelsStale || result.models.length === 0 ||
    (scope === 'all' && (result.cloudRates.size === 0 || result.hardwareCosts.size === 0))
  return incomplete ? RETRY_INTERVAL_MS : CLIENT_CACHE_TTL_MS.costings
}

async function fetchCostings(source: PricingSource, scope: CostingsScope): Promise<CostingsResult> {
  const [modelsRes, systemsRes, healthRes] = await Promise.all([
    fetch(`/api/costings/models?source=${source}`),
    scope === 'all' ? fetch('/api/costings/systems?include=cloud,hardware') : null,
    fetch('/api/costings/health'),
  ])

  if (!modelsRes.ok) throw new Error(`/models ${modelsRes.status}`)
  if (systemsRes && !systemsRes.ok) throw new Error(`/systems ${systemsRes.status}`)

  const modelsData = await modelsRes.json()
  const systemsData = systemsRes ? await systemsRes.json() : { systems: [] }
  const healthData = healthRes.ok ? await healthRes.json() : null

  const systems = (systemsData.systems ?? []) as Record<string, unknown>[]

  return {
    models: (modelsData.models ?? []) as FrontierModel[],
    cloudRates: parseCloudRates(systems),
    hardwareCosts: parseHardwareCosts(systems),
    health: healthData as CostingsHealth | null,
    modelsUpdatedAt: modelsData.updated_at ?? null,
    modelsStale: modelsData.stale ?? false,
  }
}

function getCostingsFetch(source: PricingSource, scope: CostingsScope): Promise<CostingsResult> {
  const key: CacheKey = `${source}:${scope}`
  let promise = inFlight.get(key)
  if (!promise) {
    promise = fetchCostings(source, scope).then(result => {
      cached.set(key, { result, updatedAt: Date.now() })
      return result
    }).finally(() => { inFlight.delete(key) })
    inFlight.set(key, promise)
  }
  return promise
}

function resultToData(result: CostingsResult): CostingsData {
  return {
    models: result.models,
    gpuCloudRates: result.cloudRates,
    gpuHardwareCosts: result.hardwareCosts,
    health: result.health,
    modelsUpdatedAt: result.modelsUpdatedAt,
    modelsStale: result.modelsStale,
    isLoading: false,
    error: null,
  }
}

function parseCloudRates(
  systems: Record<string, unknown>[],
): Map<string, Record<string, CloudRates>> {
  const result = new Map<string, Record<string, CloudRates>>()
  for (const sys of systems) {
    const id = sys.id as string
    const rates = sys.cloud_rates as Record<string, CloudRates> | undefined
    if (!id || !rates) continue
    const normalized: Record<string, CloudRates> = {}
    for (const [providerRegion, providerRates] of Object.entries(rates)) {
      const value = normalizeCloudRates(id, providerRegion, providerRates)
      if (value) normalized[providerRegion] = value
    }
    if (Object.keys(normalized).length > 0) result.set(id, normalized)
  }
  return result
}

function parseHardwareCosts(
  systems: Record<string, unknown>[],
): Map<string, HardwareCost> {
  const result = new Map<string, HardwareCost>()
  for (const sys of systems) {
    const id = sys.id as string
    const hw = sys.hardware_cost as HardwareCost | undefined
    if (id && hw) result.set(id, hw)
  }
  return result
}

// ── Hook ───────────────────────────────────────────────────────────────────

const EMPTY: CostingsData = {
  models: [],
  gpuCloudRates: new Map(),
  gpuHardwareCosts: new Map(),
  health: null,
  modelsUpdatedAt: null,
  modelsStale: false,
  isLoading: false,
  error: null,
}

export function useCostings(
  enabled: boolean,
  pricingSource: PricingSource = 'merged',
  scope: CostingsScope = 'all',
): CostingsData {
  const key: CacheKey = `${pricingSource}:${scope}`
  const [data, setData] = useState<CostingsData>(() => {
    const entry = enabled ? cached.get(key) : undefined
    return entry ? resultToData(entry.result) : { ...EMPTY, isLoading: enabled }
  })

  useEffect(() => {
    if (!enabled) {
      setData(EMPTY)
      return
    }

    let active = true
    let lastAttempt = 0
    const refresh = () => {
      const entry = cached.get(key)
      if (entry && Date.now() - entry.updatedAt < cacheTtl(entry.result, scope)) {
        setData(resultToData(entry.result))
        return
      }
      if (lastAttempt && Date.now() - lastAttempt < RETRY_INTERVAL_MS) return
      lastAttempt = Date.now()
      // Keep the last-good result during refresh, but never carry prices from
      // a different source or scope into the newly selected comparison.
      setData(entry ? resultToData(entry.result) : { ...EMPTY, isLoading: true })
      void getCostingsFetch(pricingSource, scope).then(result => {
        if (active) setData(resultToData(result))
      }).catch((error: unknown) => {
        if (active) setData({
          ...(entry ? resultToData(entry.result) : EMPTY),
          modelsStale: true,
          error: error instanceof Error ? error.message : 'Failed to fetch costings data',
        })
      })
    }
    const revalidate = () => { if (document.visibilityState !== 'hidden') refresh() }
    refresh()
    const timer = setInterval(revalidate, RETRY_INTERVAL_MS)
    window.addEventListener('focus', revalidate)
    document.addEventListener('visibilitychange', revalidate)
    return () => {
      active = false
      clearInterval(timer)
      window.removeEventListener('focus', revalidate)
      document.removeEventListener('visibilitychange', revalidate)
    }
  }, [enabled, pricingSource, scope, key])

  return data
}

'use client'

import { useState, useEffect } from 'react'
import { DEFAULT_GATEWAY_TIMEOUT_SECONDS } from '@/lib/api/timeout'
import { CLIENT_CACHE_TTL_MS } from '@/lib/cache-policy'

export interface GpuOption {
  systemId: string
  label: string
  vendor: string | null
  architecture: string | null
  vramGb: number | null
  bandwidthTbps: number | null
  tflopsBf16: number | null
  tdpWatts: number | null
  gpusPerNode: number | null
}

/**
 * Per-model metadata returned by GET /models?include=specs.
 *
 * MoE detection:  num_experts != null && num_experts > 1
 * Vision model:   architecture?.endsWith('ForConditionalGeneration')
 */
export interface ModelSpec {
  id: string
  /** Total number of experts (null for dense models). */
  num_experts: number | null
  /** Experts activated per token — top-k (null for dense models). */
  num_experts_per_tok: number | null
  /** Maximum context window in tokens. */
  context_length: number | null
  /** Number of query attention heads. */
  num_attn_heads: number | null
  /** Number of KV heads (< num_attn_heads when GQA is used). */
  num_kv_heads: number | null
  /** Raw HuggingFace architecture class name. */
  architecture: string | null
}

/** The version a single system completed, with the memory fraction it implies. */
export interface SystemDefault {
  version: string | null
  memoryFraction: number | null
}

export interface BackendOption {
  id: string
  aisimulateId: string
  versions: string[]
  /** Version namespace the API resolves per system (`current`), or a literal for BLIS. */
  defaultVersion: string | null
  memoryFraction: number | null
  memoryFractionKind: 'of_total' | 'of_free'
  runtimeField: string | null
  systems: Record<string, string[]>
  /** Per-system defaults — the only safe source for a system-specific fraction. */
  systemDefaults: Record<string, SystemDefault>
  /**
   * What each engine version can express, when the simulator reports it. BLIS
   * models disaggregation and `max_model_len: auto` per vLLM release, so a
   * client must not offer a topology the selected release cannot run.
   */
  versionSupport: Record<string, EngineVersionSupport>
  /**
   * Whether a P/D topology needs an inter-node fabric selected first. The
   * simulator reports this so the client does not infer it from its ID.
   */
  disaggregationRequiresFabric: boolean
}

/** Per-version capabilities reported by a simulator that models them itself. */
export interface EngineVersionSupport {
  disaggregated: boolean
  maxModelLenAuto: boolean
}

/** Simulator-specific overrides of the top-level catalog support. */
export interface SimulatorSupport {
  /** Build/release identifier for the simulator, when it has one. */
  version: string | null
  /** Model IDs this simulator can price. */
  models: string[]
  /** GPU system IDs this simulator can price. */
  systems: string[]
  backends: BackendOption[]
}

/** Capabilities for one engine version, if the simulator scopes support that way. */
export function engineVersionSupport(
  backend: BackendOption | null | undefined,
  version: string | null | undefined,
): EngineVersionSupport | null {
  if (!backend || !version) return null
  return backend.versionSupport[version] ?? null
}

export interface Catalog {
  gpuOptions: GpuOption[]
  modelOptions: string[]
  modelSpecs: Map<string, ModelSpec>
  backendOptions: BackendOption[]
  networkOptions: string[]
  /**
   * Per-simulator overrides keyed by the simulator ID the API uses. A missing
   * entry means the top-level catalog is that simulator's support; a present
   * entry narrows the models, systems, and versions it can price.
   */
  simulatorSupport: Record<string, SimulatorSupport>
  /** Effective AISimulators request timeout (seconds) reported by the server; falls back
   *  to DEFAULT_GATEWAY_TIMEOUT_SECONDS until the catalog resolves. */
  timeoutSeconds: number
  isLoading: boolean
  error: string | null
}

function mapSystemDefault(raw: unknown): SystemDefault | null {
  if (!raw || typeof raw !== 'object') return null
  const entry = raw as Record<string, unknown>
  return {
    version: typeof entry.version === 'string' ? entry.version : null,
    memoryFraction: typeof entry.memory_fraction === 'number' ? entry.memory_fraction : null,
  }
}

function mapVersionSupport(raw: unknown): Record<string, EngineVersionSupport> {
  if (!raw || typeof raw !== 'object') return {}
  const support: Record<string, EngineVersionSupport> = {}
  for (const [version, value] of Object.entries(raw as Record<string, unknown>)) {
    if (!value || typeof value !== 'object') continue
    const entry = value as Record<string, unknown>
    support[version] = {
      disaggregated: entry.disaggregated === true,
      maxModelLenAuto: entry.max_model_len_auto === true,
    }
  }
  return support
}

export function mapBackendOption(backend: Record<string, unknown>): BackendOption | null {
  if (typeof backend.id !== 'string' || !Array.isArray(backend.versions)) return null
  const rawSystemDefaults = backend.system_defaults && typeof backend.system_defaults === 'object'
    ? backend.system_defaults as Record<string, unknown>
    : {}
  const systemDefaults: Record<string, SystemDefault> = {}
  for (const [system, raw] of Object.entries(rawSystemDefaults)) {
    const mapped = mapSystemDefault(raw)
    if (mapped) systemDefaults[system] = mapped
  }
  return {
    id: backend.id,
    aisimulateId: typeof backend.aisimulate_id === 'string' ? backend.aisimulate_id : backend.id,
    versions: backend.versions.filter((v): v is string => typeof v === 'string'),
    defaultVersion: typeof backend.default_version === 'string' ? backend.default_version : null,
    memoryFraction: typeof backend.memory_fraction === 'number' ? backend.memory_fraction : null,
    memoryFractionKind: backend.memory_fraction_kind === 'of_free' ? 'of_free' : 'of_total',
    runtimeField: typeof backend.runtime_field === 'string' ? backend.runtime_field : null,
    systems: backend.systems && typeof backend.systems === 'object'
      ? backend.systems as Record<string, string[]>
      : {},
    systemDefaults,
    versionSupport: mapVersionSupport(backend.version_support),
    disaggregationRequiresFabric: backend.disaggregation_requires_fabric === true,
  }
}

/**
 * The memory fraction to seed a system's controls with.
 *
 * Fractions are version-keyed, so the system's own default wins; the
 * backend-level figure only describes the fleet's majority version.
 */
export function systemMemoryFraction(backend: BackendOption | undefined, system: string): number | null {
  if (!backend) return null
  return backend.systemDefaults[system]?.memoryFraction ?? backend.memoryFraction
}

/** Whether the catalog's system default fraction is known to match a version pin. */
export function hasDefaultMemoryFractionForVersion(
  backend: BackendOption | undefined,
  system: string,
  backendVersion: string,
): boolean {
  const version = backendVersion.trim()
  if (!version) return true
  if (!backend) return false

  const systemDefault = backend.systemDefaults[system]
  if (!systemDefault?.version || systemDefault.memoryFraction == null) return false
  if (version === 'current') return backend.defaultVersion === 'current'
  return version === systemDefault.version
}

function mapSimulatorSupport(raw: unknown): Record<string, SimulatorSupport> {
  if (!raw || typeof raw !== 'object') return {}
  const support: Record<string, SimulatorSupport> = {}
  for (const [simulator, value] of Object.entries(raw as Record<string, unknown>)) {
    if (!value || typeof value !== 'object') continue
    const entry = value as Record<string, unknown>
    const backends = (Array.isArray(entry.backends) ? entry.backends : [])
      .filter((candidate): candidate is Record<string, unknown> => !!candidate && typeof candidate === 'object')
      .map(mapBackendOption)
      .filter((candidate): candidate is BackendOption => candidate !== null)
    const models = Array.isArray(entry.models)
      ? entry.models
        .map(model => typeof model === 'string' ? model : (model && typeof model === 'object'
          ? (model as Record<string, unknown>).id
          : null))
        .filter((id): id is string => typeof id === 'string' && id.length > 0)
      : []
    const systems = Array.isArray(entry.systems)
      ? entry.systems
        .map(system => system && typeof system === 'object'
          ? (system as Record<string, unknown>).id
          : system)
        .filter((id): id is string => typeof id === 'string' && id.length > 0)
      : []
    if (backends.length === 0 && models.length === 0 && systems.length === 0) continue
    support[simulator] = {
      version: typeof entry.version === 'string' ? entry.version : null,
      models,
      systems,
      backends,
    }
  }
  return support
}

const GB = 1_073_741_824

function mapSystem(s: Record<string, unknown>): GpuOption {
  const memBytes = typeof s.memory_bytes === 'number' ? s.memory_bytes : 0
  const bwBytes = typeof s.memory_bandwidth_bytes === 'number' ? s.memory_bandwidth_bytes : 0
  return {
    systemId: typeof s.id === 'string' ? s.id : '',
    label: typeof s.name === 'string' ? s.name : (typeof s.id === 'string' ? s.id : ''),
    vendor: typeof s.vendor === 'string' ? s.vendor : null,
    architecture: typeof s.architecture === 'string' ? s.architecture : null,
    vramGb: memBytes > 0 ? Math.round(memBytes / GB) : null,
    bandwidthTbps: bwBytes > 0 ? bwBytes / 1e12 : null,
    tflopsBf16: typeof s.bf16_tflops === 'number' && s.bf16_tflops > 0 ? s.bf16_tflops : null,
    tdpWatts: typeof s.tdp_watts === 'number' ? s.tdp_watts : null,
    gpusPerNode: typeof s.gpus_per_node === 'number' ? s.gpus_per_node : null,
  }
}

// Module-level cache — shared across all components, survives re-renders
// Ten minutes so new models appear without a hard refresh.
let cachedGpus: GpuOption[] | null = null
let cachedModels: string[] | null = null
let cachedModelSpecs: Map<string, ModelSpec> | null = null
let cachedBackends: BackendOption[] | null = null
let cachedNetworks: string[] | null = null
let cachedSimulatorSupport: Record<string, SimulatorSupport> = {}
let cachedTimeoutSeconds: number = DEFAULT_GATEWAY_TIMEOUT_SECONDS
let cacheTimestamp: number | null = null
let fetchPromise: Promise<void> | null = null

function isCacheValid(): boolean {
  return cachedGpus !== null && cachedModels !== null && cachedModelSpecs !== null && cachedBackends !== null && cachedNetworks !== null &&
    cacheTimestamp !== null && Date.now() - cacheTimestamp < CLIENT_CACHE_TTL_MS.catalog
}

export function useCatalog(): Catalog {
  const [gpuOptions, setGpuOptions] = useState<GpuOption[]>(isCacheValid() ? cachedGpus! : [])
  const [modelOptions, setModelOptions] = useState<string[]>(isCacheValid() ? cachedModels! : [])
  const [modelSpecs, setModelSpecs] = useState<Map<string, ModelSpec>>(isCacheValid() ? cachedModelSpecs! : new Map())
  const [backendOptions, setBackendOptions] = useState<BackendOption[]>(isCacheValid() ? cachedBackends! : [])
  const [networkOptions, setNetworkOptions] = useState<string[]>(isCacheValid() ? cachedNetworks! : [])
  const [simulatorSupport, setSimulatorSupport] = useState<Record<string, SimulatorSupport>>(
    isCacheValid() ? cachedSimulatorSupport : {},
  )
  const [timeoutSeconds, setTimeoutSeconds] = useState<number>(cachedTimeoutSeconds)
  const [isLoading, setIsLoading] = useState(!isCacheValid())
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (isCacheValid()) {
      setGpuOptions(cachedGpus!)
      setModelOptions(cachedModels!)
      setModelSpecs(cachedModelSpecs!)
      setBackendOptions(cachedBackends!)
      setNetworkOptions(cachedNetworks!)
      setSimulatorSupport(cachedSimulatorSupport)
      setTimeoutSeconds(cachedTimeoutSeconds)
      setIsLoading(false)
      return
    }

    fetchPromise = null  // reset so stale cache triggers a fresh fetch

    let cancelled = false

    async function fetchCatalog() {
      if (!fetchPromise) {
        fetchPromise = (async () => {
          // Same-origin proxy (app/api/catalog) resolves the per-host gateway
          // server-side, so no build-time NEXT_PUBLIC_AISIMULATORS_API_URL.
          const res = await fetch('/api/catalog')
          if (!res.ok) throw new Error(`Catalog fetch failed (${res.status})`)

          const data = await res.json()

          const systems = (data.systems ?? []) as Record<string, unknown>[]
          const gpus = systems.map(mapSystem)

          const rawModels = (data.models ?? []) as unknown[]
          const backendList = (data.backends ?? []) as Record<string, unknown>[]
          const networks = Array.isArray(data.networks)
            ? data.networks.filter((network: unknown): network is string => typeof network === 'string' && network.trim().length > 0)
            : []
          const backends: BackendOption[] = backendList.flatMap(backend => {
            const option = mapBackendOption(backend)
            return option ? [option] : []
          })
          const modelList: string[] = []
          const specsMap = new Map<string, ModelSpec>()
          for (const m of rawModels) {
            if (typeof m === 'string') {
              modelList.push(m)
            } else if (m && typeof m === 'object' && typeof (m as Record<string, unknown>).id === 'string') {
              const spec = m as ModelSpec
              modelList.push(spec.id)
              specsMap.set(spec.id, spec)
            }
          }

          if (gpus.length === 0 || modelList.length === 0) {
            throw new Error('AISimulators returned empty catalog')
          }

          cachedGpus = gpus
          cachedModels = modelList
          cachedModelSpecs = specsMap
          cachedBackends = backends
          cachedNetworks = networks
          cachedSimulatorSupport = mapSimulatorSupport(data.simulatorSupport)
          if (typeof data.timeoutSeconds === 'number' && data.timeoutSeconds > 0) {
            cachedTimeoutSeconds = data.timeoutSeconds
          }
          cacheTimestamp = Date.now()
        })()
      }

      try {
        await fetchPromise
        if (!cancelled) {
          setGpuOptions(cachedGpus!)
          setModelOptions(cachedModels!)
          setModelSpecs(cachedModelSpecs!)
          setBackendOptions(cachedBackends!)
          setNetworkOptions(cachedNetworks!)
          setSimulatorSupport(cachedSimulatorSupport)
          setTimeoutSeconds(cachedTimeoutSeconds)
        }
      } catch (err) {
        fetchPromise = null
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to fetch catalog')
        }
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    fetchCatalog()
    return () => { cancelled = true }
  }, [])

  return { gpuOptions, modelOptions, modelSpecs, backendOptions, networkOptions, simulatorSupport, timeoutSeconds, isLoading, error }
}

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
  /**
   * Which comparison options a capacity run can return. The client gates the
   * disaggregated option presentation on this, never on a simulator ID.
   */
  capacityOptions: { disaggregated: boolean }
}

/** Per-version capabilities reported by a simulator that models them itself. */
export interface EngineVersionSupport {
  disaggregated: boolean
  maxModelLenAuto: boolean
}

/** Identity and release of one simulator, as listed by GET /simulators. */
export interface SimulatorDescriptor {
  id: string
  label: string
  /** Short, service-authored explanation of what this simulator models. */
  description: string | null
  version: string | null
  default: boolean
}

/** Everything one simulator reports about the catalogue it can price. */
export interface SimulatorSupport {
  id: string
  label: string
  /** Release of the simulator itself (not the engine/backend version). */
  version: string | null
  /**
   * How the client requests a recommendation: `windowed` streams progressive
   * GPU windows, `single` makes one bounded search call. The simulator reports
   * this so the client never selects a search shape by simulator ID.
   */
  recommendSearch: 'windowed' | 'single'
  /** Model IDs this simulator can price. */
  models: string[]
  /**
   * Whether `models` is the complete set this simulator can price. When false
   * (or not reported), a model outside the list may still resolve, so the UI
   * must not call it unsupported.
   */
  modelsExhaustive: boolean
  /** Per-model metadata for the models above. */
  modelSpecs: Map<string, ModelSpec>
  /** GPU system IDs this simulator can price. */
  systems: string[]
  backends: BackendOption[]
  /** Network fabrics this simulator models; empty when it models none. */
  networks: string[]
  /** Fabric used when a P/D request omits one; null when none is modeled. */
  defaultNetwork: string | null
  /**
   * Sizing objectives this simulator advertises. The client gates its goal
   * selector on this list rather than on the simulator's ID.
   */
  objectives: string[]
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
   * Capabilities for each simulator whose catalogue could be read, keyed by
   * ID. An unavailable simulator is absent rather than represented by empty
   * lists, so pages fall back to the top-level catalogue.
   */
  simulatorSupport: Record<string, SimulatorSupport>
  /** Simulator identity list from the combined catalog, in service order. */
  simulators: SimulatorDescriptor[]
  /**
   * Sizing objectives the default simulator (the top-level catalog) advertises.
   * Empty when an older API does not report them.
   */
  objectives: string[]
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
    capacityOptions: {
      disaggregated:
        backend.capacity_options != null &&
        typeof backend.capacity_options === 'object' &&
        (backend.capacity_options as Record<string, unknown>).disaggregated === true,
    },
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

function mapSimulatorSupportEntry(entry: Record<string, unknown>): SimulatorSupport {
  const backends = (Array.isArray(entry.backends) ? entry.backends : [])
    .filter((candidate): candidate is Record<string, unknown> => !!candidate && typeof candidate === 'object')
    .map(mapBackendOption)
    .filter((candidate): candidate is BackendOption => candidate !== null)
  const models: string[] = []
  const modelSpecs = new Map<string, ModelSpec>()
  for (const model of Array.isArray(entry.models) ? entry.models : []) {
    if (typeof model === 'string' && model.length > 0) {
      models.push(model)
    } else if (model && typeof model === 'object') {
      const spec = model as Record<string, unknown>
      if (typeof spec.id === 'string' && spec.id.length > 0) {
        models.push(spec.id)
        modelSpecs.set(spec.id, spec as unknown as ModelSpec)
      }
    }
  }
  const systems = Array.isArray(entry.systems)
    ? entry.systems
      .map(system => system && typeof system === 'object'
        ? (system as Record<string, unknown>).id
        : system)
      .filter((id): id is string => typeof id === 'string' && id.length > 0)
    : []
  const networks = Array.isArray(entry.networks)
    ? entry.networks.filter((network): network is string => typeof network === 'string' && network.trim().length > 0)
    : []
  return {
    id: entry.id as string,
    label: typeof entry.label === 'string' && entry.label.length > 0 ? entry.label : entry.id as string,
    version: typeof entry.version === 'string' && entry.version.trim().length > 0
      ? entry.version.trim()
      : null,
    recommendSearch: entry.recommend_search === 'single' || entry.recommendSearch === 'single'
      ? 'single'
      : 'windowed',
    models,
    modelsExhaustive: entry.modelsExhaustive === true || entry.model_catalog_complete === true,
    modelSpecs,
    systems,
    backends,
    networks,
    defaultNetwork: typeof entry.default_network === 'string'
      ? entry.default_network
      : typeof entry.defaultNetwork === 'string' ? entry.defaultNetwork : null,
    objectives: Array.isArray(entry.objectives)
      ? entry.objectives.filter((objective): objective is string => typeof objective === 'string' && objective.trim().length > 0)
      : [],
  }
}

interface MappedSimulators {
  descriptors: SimulatorDescriptor[]
  support: Record<string, SimulatorSupport>
}

/**
 * Split the combined catalogue's self-describing simulator entries into the
 * identity list used for selection and the capability map used by pages.
 *
 * An entry marked unavailable keeps its identity but contributes no support,
 * so callers fall back to the top-level catalogue instead of reading empty
 * capability lists as "this simulator prices nothing".
 */
function mapSimulators(raw: unknown): MappedSimulators {
  const descriptors: SimulatorDescriptor[] = []
  const support: Record<string, SimulatorSupport> = {}
  if (!Array.isArray(raw)) return { descriptors, support }
  for (const candidate of raw) {
    if (!candidate || typeof candidate !== 'object') continue
    const entry = candidate as Record<string, unknown>
    if (typeof entry.id !== 'string' || entry.id.length === 0) continue
    descriptors.push({
      id: entry.id,
      label: typeof entry.label === 'string' && entry.label.length > 0 ? entry.label : entry.id,
      description: typeof entry.description === 'string' && entry.description.trim().length > 0
        ? entry.description.trim()
        : null,
      version: typeof entry.version === 'string' && entry.version.trim().length > 0
        ? entry.version.trim()
        : null,
      default: entry.default === true,
    })
    if (entry.available === false) continue
    if (!Array.isArray(entry.backends) || !Array.isArray(entry.models) || !Array.isArray(entry.systems)) continue
    support[entry.id] = mapSimulatorSupportEntry(entry)
  }
  return { descriptors, support }
}

/** The page-ready view of a `/api/catalog` payload. */
export interface MappedCatalog {
  gpuOptions: GpuOption[]
  modelOptions: string[]
  modelSpecs: Map<string, ModelSpec>
  backendOptions: BackendOption[]
  networkOptions: string[]
  simulatorSupport: Record<string, SimulatorSupport>
  simulators: SimulatorDescriptor[]
  objectives: string[]
  timeoutSeconds: number | null
}

/**
 * Turn a `/api/catalog` payload into the view pages consume.
 *
 * The default simulator's entry is authoritative: its systems, models,
 * backends, fabrics, and objectives become the top-level catalogue. A response
 * without simulator entries (an older gateway) keeps reporting those facts at
 * the top level, which is the only fallback.
 *
 * Throws when the resolved catalogue cannot render a page, matching the hook's
 * existing "empty catalog" failure.
 */
export function mapCatalogResponse(data: unknown): MappedCatalog {
  const record = (data && typeof data === 'object' ? data : {}) as Record<string, unknown>
  const rawSimulators = Array.isArray(record.simulators)
    ? record.simulators as Record<string, unknown>[]
    : []
  const mapped = mapSimulators(rawSimulators)
  const defaultRaw = rawSimulators.find(entry => entry.default === true) ?? rawSimulators[0]
  const defaultEntry = defaultRaw !== undefined
    && defaultRaw.available !== false
    && Array.isArray(defaultRaw.systems)
    && Array.isArray(defaultRaw.models)
    && Array.isArray(defaultRaw.backends)
    ? defaultRaw
    : null

  const systems = (defaultEntry?.systems ?? record.systems ?? []) as Record<string, unknown>[]
  const gpus = systems.map(mapSystem)

  const rawModels = (defaultEntry?.models ?? record.models ?? []) as unknown[]
  const backendList = (defaultEntry?.backends ?? record.backends ?? []) as Record<string, unknown>[]
  const rawNetworks = defaultEntry?.networks ?? record.networks
  const networks = Array.isArray(rawNetworks)
    ? rawNetworks.filter((network: unknown): network is string => typeof network === 'string' && network.trim().length > 0)
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

  const defaultSupport = defaultEntry
    ? mapped.support[defaultEntry.id as string]
    : undefined
  const objectives = defaultSupport?.objectives
    ?? (Array.isArray(record.objectives)
      ? record.objectives.filter((objective): objective is string => typeof objective === 'string' && objective.trim().length > 0)
      : [])
  const timeoutSeconds = typeof record.timeoutSeconds === 'number' && record.timeoutSeconds > 0
    ? record.timeoutSeconds
    : null

  return {
    gpuOptions: gpus,
    modelOptions: modelList,
    modelSpecs: specsMap,
    backendOptions: backends,
    networkOptions: networks,
    simulatorSupport: mapped.support,
    simulators: mapped.descriptors,
    objectives,
    timeoutSeconds,
  }
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
let cachedSimulators: SimulatorDescriptor[] = []
let cachedObjectives: string[] = []
let cachedTimeoutSeconds: number = DEFAULT_GATEWAY_TIMEOUT_SECONDS
let cacheTimestamp: number | null = null
let fetchPromise: Promise<void> | null = null

function isCacheValid(): boolean {
  return cachedGpus !== null && cachedModels !== null && cachedModelSpecs !== null && cachedBackends !== null && cachedBackends.length > 0 && cachedNetworks !== null &&
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
  const [simulators, setSimulators] = useState<SimulatorDescriptor[]>(
    isCacheValid() ? cachedSimulators : [],
  )
  const [objectives, setObjectives] = useState<string[]>(isCacheValid() ? cachedObjectives : [])
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
      setSimulators(cachedSimulators)
      setObjectives(cachedObjectives)
      setTimeoutSeconds(cachedTimeoutSeconds)
      setIsLoading(false)
      return
    }

    fetchPromise = null  // reset so stale cache triggers a fresh fetch

    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | null = null

    async function fetchCatalog() {
      if (!fetchPromise) {
        fetchPromise = (async () => {
          // Same-origin proxy (app/api/catalog) resolves the per-host gateway
          // server-side, so no build-time NEXT_PUBLIC_AISIMULATORS_API_URL.
          const res = await fetch('/api/catalog')
          if (!res.ok) throw new Error(`Catalog fetch failed (${res.status})`)

          const data = await res.json()
          const catalog = mapCatalogResponse(data)

          cachedGpus = catalog.gpuOptions
          cachedModels = catalog.modelOptions
          cachedModelSpecs = catalog.modelSpecs
          cachedBackends = catalog.backendOptions
          cachedNetworks = catalog.networkOptions
          cachedSimulatorSupport = catalog.simulatorSupport
          cachedSimulators = catalog.simulators
          cachedObjectives = catalog.objectives
          if (catalog.timeoutSeconds !== null) {
            cachedTimeoutSeconds = catalog.timeoutSeconds
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
          setSimulators(cachedSimulators)
          setObjectives(cachedObjectives)
          setTimeoutSeconds(cachedTimeoutSeconds)
          if (cachedBackends!.length === 0) {
            setError('AISimulators returned no serving backends. Retrying the catalog.')
            retryTimer = setTimeout(() => {
              fetchPromise = null
              void fetchCatalog()
            }, 15_000)
          } else {
            setError(null)
          }
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
    return () => {
      cancelled = true
      if (retryTimer) clearTimeout(retryTimer)
    }
  }, [])

  return {
    gpuOptions, modelOptions, modelSpecs, backendOptions, networkOptions,
    simulatorSupport, simulators, objectives, timeoutSeconds,
    isLoading, error,
  }
}

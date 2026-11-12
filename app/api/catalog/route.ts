// GET /api/catalog
// Same-origin proxy for the AISimulators catalog. Discovers the simulators the
// service exposes through /simulators, then fetches each simulator's own
// /models, /systems, /backends, and /networks envelopes server-side via
// AISIMULATORS_GATEWAY_URL and folds them into one list of self-describing
// simulator entries for useCatalog.
//
// The browser must call this route rather than AISimulators directly, so the
// per-host gateway (host.containers.internal on each deployment) is resolved
// server-side. That keeps the .dev/.xyz two-host split correct client-side and
// removes the need for a build-time NEXT_PUBLIC_AISIMULATORS_API_URL.
//
// Nothing here names a simulator. The default simulator, its label, version,
// models, systems, backends, fabrics, and sizing objectives all come from the
// service; adding a simulator must not require a change in this file.

import { NextResponse } from 'next/server'
import { gatewayTimeoutSeconds } from '@/lib/api/timeout'
import { recordErrorCode } from '@/lib/otel-metrics'

// This is the catalog fetch's own timeout (30s). The value surfaced to the
// client below is gatewayTimeoutSeconds() — the longer recommend/predict timeout.
const DEFAULT_TIMEOUT_SECONDS = 30

export interface SimulatorDescriptor {
  id: string
  label: string
  /** Short, service-authored explanation of what this simulator models. */
  description: string | null
  version: string | null
  /** Git commit of the simulator artifact, when the service reports one. */
  revision: string | null
  default: boolean
}

/** Everything one simulator reports about the catalogue it can price. */
export interface SimulatorCapabilities {
  recommendSearch: 'windowed' | 'single'
  modelsExhaustive: boolean
  models: unknown[]
  systems: unknown[]
  backends: unknown[]
  networks: string[]
  defaultNetwork: string | null
  objectives: string[]
}

/**
 * One simulator as the combined catalogue exposes it: identity, the default
 * flag, and capabilities in a single entry. Clients do not have to correlate a
 * separate identity list and support map by ID.
 *
 * An entry whose capabilities could not be read keeps its identity so it stays
 * selectable, with `available: false` and empty capability lists. Empty lists
 * alone never mean "supports nothing"; `available` carries that distinction.
 */
export interface SimulatorCatalogEntry extends SimulatorDescriptor, SimulatorCapabilities {
  available: boolean
}

function emptyCapabilities(): SimulatorCapabilities {
  return {
    recommendSearch: 'windowed',
    modelsExhaustive: false,
    models: [],
    systems: [],
    backends: [],
    networks: [],
    defaultNetwork: null,
    objectives: [],
  }
}

/** The unscoped catalogue shape from a service older than /simulators. */
interface LegacyCatalog {
  systems: unknown[]
  models: unknown[]
  backends: unknown[]
  networks: string[]
  objectives: string[]
}

function catalogResponse(simulators: SimulatorCatalogEntry[], legacy?: LegacyCatalog) {
  return NextResponse.json(
    {
      // Only a service that cannot report simulators falls back to this view.
      ...(legacy ?? {}),
      simulators,
      timeoutSeconds: gatewayTimeoutSeconds(),
    },
    {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        // Mirrors the useCatalog client-side cache TTL (10 min).
        'Cache-Control': 'public, s-maxage=600, stale-while-revalidate=3600',
      },
    },
  )
}

/** Attempts for one upstream read: the gateway intermittently answers 503. */
const UPSTREAM_ATTEMPTS = 3
/** Backoff between attempts, multiplied by the attempt number. */
const UPSTREAM_RETRY_DELAY_MS = 250

async function fetchJson(
  url: string,
  timeoutSeconds: number,
): Promise<unknown | null> {
  for (let attempt = 1; attempt <= UPSTREAM_ATTEMPTS; attempt++) {
    try {
      const response = await fetch(url, {
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        signal: AbortSignal.timeout(timeoutSeconds * 1000),
      })
      if (response.ok) return await response.json()
      // Retry only failures a later attempt can plausibly fix. A 4xx response
      // means the request itself is wrong.
      if (response.status !== 429 && response.status < 500) return null
    } catch (error) {
      // A timeout means the service is not answering; retrying only multiplies
      // the wait. Network resets and transient 5xx responses are worth a retry.
      if (error instanceof Error && (error.name === 'TimeoutError' || error.name === 'AbortError')) return null
    }
    if (attempt < UPSTREAM_ATTEMPTS) {
      await new Promise(resolve => setTimeout(resolve, UPSTREAM_RETRY_DELAY_MS * attempt))
    }
  }
  return null
}

/** A release string, or null when absent or blank. */
function normalizedVersion(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null
}

/** The simulator identity list from GET /simulators, in service order. */
function simulatorDescriptors(data: unknown): SimulatorDescriptor[] {
  if (!data || typeof data !== 'object') return []
  const raw = (data as Record<string, unknown>).simulators
  if (!Array.isArray(raw)) return []
  return raw.flatMap((entry): SimulatorDescriptor[] => {
    if (!entry || typeof entry !== 'object') return []
    const value = entry as Record<string, unknown>
    if (typeof value.id !== 'string' || value.id.trim().length === 0) return []
    return [{
      id: value.id,
      label: typeof value.label === 'string' && value.label.trim().length > 0 ? value.label : value.id,
      description: catalogString(value, 'description'),
      version: normalizedVersion(value.version),
      revision: normalizedVersion(value.revision),
      default: value.default === true,
    }]
  })
}

function catalogList(data: unknown, key: 'systems' | 'models' | 'backends' | 'networks'): unknown[] | null {
  if (!data || typeof data !== 'object') return null
  const value = (data as Record<string, unknown>)[key]
  return Array.isArray(value) ? value : null
}

/** Sizing goals a simulator advertises through its /backends envelope. */
function catalogObjectives(data: unknown): string[] {
  if (!data || typeof data !== 'object') return []
  const value = (data as Record<string, unknown>).objectives
  return Array.isArray(value)
    ? value.filter((objective): objective is string => typeof objective === 'string' && objective.trim().length > 0)
    : []
}

function catalogString(data: unknown, key: string): string | null {
  if (!data || typeof data !== 'object') return null
  const value = (data as Record<string, unknown>)[key]
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

/**
 * Read one simulator's capability block from either the direct service
 * envelopes or a newer combined catalogue, where it is flattened onto the
 * simulator entry. Returns null for an unavailable entry or one that lacks the
 * required lists entirely.
 */
function capabilityEnvelope(raw: unknown): SimulatorCapabilities | null {
  if (!raw || typeof raw !== 'object') return null
  const entry = raw as Record<string, unknown>
  if (entry.available === false) return null
  const backends = catalogList(entry, 'backends')
  const models = catalogList(entry, 'models')
  const systems = catalogList(entry, 'systems')
  if (!backends || !models || !systems) return null
  const networks = catalogList(entry, 'networks') ?? []
  return {
    recommendSearch: catalogString(entry, 'recommendSearch') === 'single'
      || catalogString(entry, 'recommend_search') === 'single'
      ? 'single'
      : 'windowed',
    modelsExhaustive: entry.modelsExhaustive === true || entry.model_catalog_complete === true,
    models,
    systems,
    backends,
    networks: networks.filter(
      (network): network is string => typeof network === 'string' && network.trim().length > 0,
    ),
    defaultNetwork: catalogString(entry, 'defaultNetwork') ?? catalogString(entry, 'default_network'),
    objectives: catalogObjectives(entry),
  }
}

/**
 * Fold a combined upstream catalogue (this proxy's old two-field shape or the
 * newer one-entry shape) into the single simulator list clients consume.
 * Capabilities come from the entry itself when present, otherwise from the
 * legacy `simulatorSupport` map keyed by ID.
 */
function combinedSimulators(data: Record<string, unknown>): SimulatorCatalogEntry[] {
  const rawList = Array.isArray(data.simulators) ? data.simulators : []
  const support = data.simulatorSupport && typeof data.simulatorSupport === 'object'
    ? data.simulatorSupport as Record<string, unknown>
    : {}
  const entries: SimulatorCatalogEntry[] = []

  for (const raw of rawList) {
    if (!raw || typeof raw !== 'object') continue
    const value = raw as Record<string, unknown>
    const id = catalogString(value, 'id')
    if (!id) continue
    const capabilities = capabilityEnvelope(value) ?? capabilityEnvelope(support[id])
    entries.push({
      id,
      label: catalogString(value, 'label') ?? id,
      description: catalogString(value, 'description') ?? catalogString(support[id], 'description'),
      version: normalizedVersion(value.version),
      revision: normalizedVersion(value.revision),
      default: value.default === true,
      available: capabilities !== null,
      ...(capabilities ?? emptyCapabilities()),
    })
  }

  // A pre-/simulators upstream may expose only the support map.
  if (entries.length === 0) {
    for (const [id, raw] of Object.entries(support)) {
      if (!raw || typeof raw !== 'object') continue
      const value = raw as Record<string, unknown>
      const capabilities = capabilityEnvelope(value)
      if (!capabilities) continue
      entries.push({
        id,
        label: catalogString(value, 'label') ?? id,
        description: catalogString(value, 'description'),
        version: normalizedVersion(value.version),
        revision: normalizedVersion(value.revision),
        default: false,
        available: true,
        ...capabilities,
      })
    }
  }

  // Resolve exactly one default so clients never have to guess: an explicit
  // upstream field wins, then the entry's own flag, then the first entry.
  const defaultId = catalogString(data, 'defaultSimulator') ?? catalogString(data, 'default_simulator')
  const resolved = (defaultId ? entries.find(entry => entry.id === defaultId) : undefined)
    ?? entries.find(entry => entry.default)
    ?? entries[0]
  for (const entry of entries) entry.default = entry === resolved
  return entries
}

/**
 * Everything one simulator reports about itself: the models and systems it can
 * price, its backend/version table, its fabrics and default fabric, and the
 * sizing objectives it implements.
 *
 * The envelope must echo the simulator we asked for. An older AISimulators
 * build ignores an unknown `simulator` parameter and answers with its default
 * data at HTTP 200; without this check a rolling deploy could advertise one
 * simulator's support as another's.
 */
async function simulatorCapabilities(
  baseUrl: string,
  descriptor: SimulatorDescriptor,
  timeoutSeconds: number,
): Promise<SimulatorCapabilities | null> {
  const simulator = encodeURIComponent(descriptor.id)
  const [backendsData, modelsData, systemsData, networksData] = await Promise.all([
    fetchJson(`${baseUrl}/backends?simulator=${simulator}`, timeoutSeconds),
    fetchJson(`${baseUrl}/models?simulator=${simulator}&include=specs`, timeoutSeconds),
    // The client's GPU picker groups systems by vendor/architecture and shows
    // memory, bandwidth, and compute below the selection. Those fields are only
    // present when the service is asked for specs.
    fetchJson(`${baseUrl}/systems?simulator=${simulator}&include=specs`, timeoutSeconds),
    fetchJson(`${baseUrl}/networks?simulator=${simulator}`, timeoutSeconds),
  ])
  if (![backendsData, modelsData, systemsData].every(data => catalogString(data, 'simulator') === descriptor.id)) {
    return null
  }
  const backends = catalogList(backendsData, 'backends')
  const models = catalogList(modelsData, 'models')
  const systems = catalogList(systemsData, 'systems')
  if (!backends || !models || !systems) return null
  // A simulator that models no fabric still returns an envelope; a missing
  // networks response means "no fabrics" rather than a reason to drop support.
  const networks = catalogString(networksData, 'simulator') === descriptor.id
    ? (catalogList(networksData, 'networks') ?? [])
        .filter((network): network is string => typeof network === 'string' && network.trim().length > 0)
    : []
  return {
    recommendSearch: catalogString(backendsData, 'recommend_search') === 'single' ? 'single' : 'windowed',
    // Whether the model list is the complete set the simulator can price.
    // Absent means "unknown", which falls back to per-model resolution.
    modelsExhaustive: modelsData != null && typeof modelsData === 'object'
      && (modelsData as Record<string, unknown>).model_catalog_complete === true,
    backends,
    models,
    systems,
    networks,
    defaultNetwork: catalogString(networksData, 'default_network'),
    objectives: catalogObjectives(backendsData),
  }
}

function hasUsableId(value: unknown): boolean {
  if (value === null || typeof value !== 'object') return false
  const id = (value as Record<string, unknown>).id
  return typeof id === 'string' && id.trim().length > 0
}

function validCatalogEntries(
  systems: unknown[],
  models: unknown[],
  backends: unknown[],
): boolean {
  return systems.every(hasUsableId) &&
    models.every(model =>
      (typeof model === 'string' && model.trim().length > 0) || hasUsableId(model)) &&
    backends.every(hasUsableId)
}

function invalidCatalogResponse() {
  return NextResponse.json(
    { status: 'failed', error: { code: 'AISIM_INVALID_RESPONSE', message: 'AISimulators catalog is missing systems or models' } },
    { status: 502, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
  )
}

/**
 * Read the unscoped catalogue an older service or shared gateway exposes.
 * Used only when no simulator list is available; the current API reports the
 * same facts inside each simulator entry.
 */
function legacyCatalog(data: unknown): LegacyCatalog | null {
  const systems = catalogList(data, 'systems')
  const models = catalogList(data, 'models')
  const backends = catalogList(data, 'backends') ?? []
  if (!systems || !models || !validCatalogEntries(systems, models, backends)) return null
  const networks = (catalogList(data, 'networks') ?? [])
    .filter((network): network is string => typeof network === 'string' && network.trim().length > 0)
  return { systems, models, backends, networks, objectives: catalogObjectives(data) }
}

function isSelfReferentialGateway(gatewayUrl: string, request: Request): boolean {
  const gatewayHost = new URL(gatewayUrl).host.toLowerCase()
  const forwardedHost = request.headers.get('x-forwarded-host')?.split(',')[0].trim()
  return [new URL(request.url).host, request.headers.get('host'), forwardedHost]
    .some(host => host?.toLowerCase() === gatewayHost)
}

export async function GET(request: Request) {
  // Require the gateway to be configured; fail loud (like /recommend) rather
  // than silently falling back to the public domain, which masks a misconfig
  // and bypasses the intended per-host internal gateway.
  const baseUrl = process.env.AISIMULATORS_GATEWAY_URL
  if (!baseUrl) {
    recordErrorCode('/api/catalog', 'AISIM_NOT_CONFIGURED', 503)
    return NextResponse.json(
      { status: 'failed', error: { code: 'AISIM_NOT_CONFIGURED', message: 'AISimulators API URL is not configured' } },
      { status: 503, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
    )
  }
  // Shares the shared resolver's positive-integer validation, with the catalog
  // fetch's own 30s baseline (a negative env value would break AbortSignal).
  const timeoutSeconds = gatewayTimeoutSeconds(DEFAULT_TIMEOUT_SECONDS)

  try {
    // A shared ConfigIQ deployment can expose the same AISimulators service
    // through its combined same-origin /api/catalog proxy. That response is
    // already normalized, so pass it through instead of re-parsing it.
    //
    // The reduced { simulators, timeoutSeconds } shape is this deployment's
    // same-origin contract, shipped with its own frontend. When ConfigIQ
    // instances are chained, upgrade the downstream reader before the upstream
    // it reads: this route still accepts an older upstream's legacy top-level
    // lists, but an older downstream expects lists this route no longer sends.
    const combinedCatalogUrl = /\/api\/?$/.test(baseUrl)
      ? `${baseUrl.replace(/\/$/, '')}/catalog`
      : null
    if (combinedCatalogUrl) {
      if (isSelfReferentialGateway(combinedCatalogUrl, request)) {
        return NextResponse.json(
          { status: 'failed', error: { code: 'AISIM_INVALID_GATEWAY', message: 'AISimulators gateway points back to this ConfigIQ catalogue' } },
          { status: 503, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
        )
      }
      const catalogRes = await fetch(combinedCatalogUrl, {
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        signal: AbortSignal.timeout(timeoutSeconds * 1000),
      })
      if (!catalogRes.ok) {
        return NextResponse.json(
          {
            status: 'failed',
            error: {
              code: 'AISIM_ERROR',
              message: `AISimulators catalog fetch failed (${catalogRes.status})`,
            },
          },
          { status: 502, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
        )
      }
      let catalogData: unknown
      try {
        catalogData = await catalogRes.json()
      } catch {
        return NextResponse.json(
          { status: 'failed', error: { code: 'AISIM_INVALID_RESPONSE', message: 'AISimulators returned non-JSON response' } },
          { status: 502, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
        )
      }
      const simulatorEntries = combinedSimulators(catalogData as Record<string, unknown>)
      if (simulatorEntries.length === 0) {
        // No simulator list: an older gateway exposes only the unscoped
        // catalogue. Pass it through for clients that predate simulator
        // selection.
        const legacy = legacyCatalog(catalogData)
        if (!legacy) return invalidCatalogResponse()
        return catalogResponse([], legacy)
      }

      // The default simulator's own catalogue is what the client renders; if
      // it cannot be read, every page that lists models or systems is unusable.
      const defaultEntry = simulatorEntries.find(entry => entry.default)
      if (!defaultEntry || !defaultEntry.available) return invalidCatalogResponse()
      if (!validCatalogEntries(defaultEntry.systems, defaultEntry.models, defaultEntry.backends)) {
        return invalidCatalogResponse()
      }
      return catalogResponse(simulatorEntries)
    }

    const simulators = simulatorDescriptors(await fetchJson(`${baseUrl}/simulators`, timeoutSeconds))
    const defaultDescriptor = simulators.find(simulator => simulator.default) ?? simulators[0] ?? null

    if (!defaultDescriptor) {
      // Older service without /simulators: fall back to its unscoped catalog.
      // There is no simulator list to advertise, so support stays empty.
      const [systemsData, modelsData, backendsData, networksData] = await Promise.all([
        fetchJson(`${baseUrl}/systems?include=specs`, timeoutSeconds),
        fetchJson(`${baseUrl}/models?include=specs`, timeoutSeconds),
        fetchJson(`${baseUrl}/backends`, timeoutSeconds),
        fetchJson(`${baseUrl}/networks`, timeoutSeconds),
      ])
      const systems = catalogList(systemsData, 'systems')
      const models = catalogList(modelsData, 'models')
      const backends = catalogList(backendsData, 'backends') ?? []
      if (!systems || !models || !validCatalogEntries(systems, models, backends)) return invalidCatalogResponse()
      const networks = (catalogList(networksData, 'networks') ?? [])
        .filter((network): network is string => typeof network === 'string' && network.trim().length > 0)
      const legacy = legacyCatalog({
        systems,
        models,
        backends,
        networks,
        objectives: catalogObjectives(backendsData),
      })
      if (!legacy) return invalidCatalogResponse()
      return catalogResponse([], legacy)
    }

    const capabilities = await Promise.all(
      simulators.map(simulator => simulatorCapabilities(baseUrl, simulator, timeoutSeconds)),
    )
    const simulatorEntries: SimulatorCatalogEntry[] = simulators.map((simulator, index) => {
      const support = capabilities[index]
      return {
        ...simulator,
        default: simulator.id === defaultDescriptor.id,
        available: support !== null,
        ...(support ?? emptyCapabilities()),
      }
    })

    const defaultEntry = simulatorEntries.find(entry => entry.id === defaultDescriptor.id)
    if (!defaultEntry || !defaultEntry.available) {
      // The default simulator's own catalogue is the top-level catalogue; if it
      // cannot be read, every page that lists models or systems is unusable.
      return invalidCatalogResponse()
    }
    if (!validCatalogEntries(defaultEntry.systems, defaultEntry.models, defaultEntry.backends)) {
      return invalidCatalogResponse()
    }
    return catalogResponse(simulatorEntries)
  } catch (err: unknown) {
    if (err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      recordErrorCode('/api/catalog', 'AISIM_TIMEOUT', 504)
      return NextResponse.json(
        { status: 'failed', error: { code: 'AISIM_TIMEOUT', message: 'AISimulators API timed out' } },
        { status: 504 },
      )
    }
    recordErrorCode('/api/catalog', 'AISIM_UNAVAILABLE', 502)
    return NextResponse.json(
      { status: 'failed', error: { code: 'AISIM_UNAVAILABLE', message: 'AISimulators API is unreachable' } },
      { status: 502 },
    )
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}

// GET /api/catalog
// Same-origin proxy for the AISimulators catalog. Discovers the simulators the
// service exposes through /simulators, then fetches each simulator's own
// /models, /systems, /backends, and /networks envelopes server-side via
// AISIMULATORS_GATEWAY_URL and combines their raw shapes for useCatalog.
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
  version: string | null
  default: boolean
}

function catalogResponse(
  systems: unknown[],
  models: unknown[],
  backends: unknown[],
  networks: string[],
  simulators: SimulatorDescriptor[],
  defaultSimulator: string | null,
  simulatorSupport: Record<string, unknown>,
  objectives: string[],
) {
  return NextResponse.json(
    {
      systems,
      models,
      backends,
      networks,
      simulators,
      defaultSimulator,
      simulatorSupport,
      objectives,
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

async function fetchJson(
  url: string,
  timeoutSeconds: number,
): Promise<unknown | null> {
  try {
    const response = await fetch(url, {
      headers: { Accept: 'application/json' },
      cache: 'no-store',
      signal: AbortSignal.timeout(timeoutSeconds * 1000),
    })
    if (!response.ok) return null
    return await response.json()
  } catch {
    return null
  }
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
      version: typeof value.version === 'string' ? value.version : null,
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
): Promise<Record<string, unknown> | null> {
  const simulator = encodeURIComponent(descriptor.id)
  const [backendsData, modelsData, systemsData, networksData] = await Promise.all([
    fetchJson(`${baseUrl}/backends?simulator=${simulator}`, timeoutSeconds),
    fetchJson(`${baseUrl}/models?simulator=${simulator}&include=specs`, timeoutSeconds),
    fetchJson(`${baseUrl}/systems?simulator=${simulator}`, timeoutSeconds),
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
    id: descriptor.id,
    label: descriptor.label,
    version: descriptor.version,
    recommendSearch: catalogString(backendsData, 'recommend_search') ?? 'windowed',
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
      const systems = catalogList(catalogData, 'systems')
      const models = catalogList(catalogData, 'models')
      const backends = catalogList(catalogData, 'backends') ?? []
      const networks = (catalogList(catalogData, 'networks') ?? [])
        .filter((network): network is string => typeof network === 'string')
      if (!systems || !models || !validCatalogEntries(systems, models, backends)) return invalidCatalogResponse()
      const rawSupport = (catalogData as Record<string, unknown>).simulatorSupport
      const simulatorSupport = rawSupport && typeof rawSupport === 'object' ? rawSupport : {}
      const simulators = simulatorDescriptors(catalogData)
      const defaultSimulator = catalogString(catalogData, 'defaultSimulator')
        ?? simulators.find(simulator => simulator.default)?.id
        ?? null
      return catalogResponse(
        systems, models, backends, networks, simulators, defaultSimulator,
        simulatorSupport as Record<string, unknown>, catalogObjectives(catalogData),
      )
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
      return catalogResponse(
        systems, models, backends, networks, [], null, {}, catalogObjectives(backendsData),
      )
    }

    const capabilities = await Promise.all(
      simulators.map(simulator => simulatorCapabilities(baseUrl, simulator, timeoutSeconds)),
    )
    const simulatorSupport: Record<string, Record<string, unknown>> = {}
    simulators.forEach((simulator, index) => {
      const support = capabilities[index]
      if (support) simulatorSupport[simulator.id] = support
    })

    const defaultSupport = simulatorSupport[defaultDescriptor.id]
    if (!defaultSupport) {
      // The default simulator's own catalogue is the top-level catalogue; if it
      // cannot be read, every page that lists models or systems is unusable.
      return invalidCatalogResponse()
    }
    const systems = defaultSupport.systems as unknown[]
    const models = defaultSupport.models as unknown[]
    const backends = defaultSupport.backends as unknown[]
    if (!validCatalogEntries(systems, models, backends)) return invalidCatalogResponse()
    return catalogResponse(
      systems,
      models,
      backends,
      defaultSupport.networks as string[],
      simulators,
      defaultDescriptor.id,
      simulatorSupport,
      defaultSupport.objectives as string[],
    )
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

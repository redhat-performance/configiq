// GET /api/catalog
// Same-origin proxy for the AISimulators catalog. Fetches /systems, /models,
// /backends, and /networks server-side via AISIMULATORS_GATEWAY_URL and combines
// their raw shapes for useCatalog.
//
// The browser must call this route rather than AISimulators directly, so the
// per-host gateway (host.containers.internal on each deployment) is resolved
// server-side. That keeps the .dev/.xyz two-host split correct client-side and
// removes the need for a build-time NEXT_PUBLIC_AISIMULATORS_API_URL.

import { NextResponse } from 'next/server'
import { gatewayTimeoutSeconds } from '@/lib/api/timeout'
import { recordErrorCode } from '@/lib/otel-metrics'

// This is the catalog fetch's own timeout (30s). The value surfaced to the
// client below is gatewayTimeoutSeconds() — the longer recommend/predict timeout.
const DEFAULT_TIMEOUT_SECONDS = 30

function catalogResponse(
  systems: unknown[],
  models: unknown[],
  backends: unknown[],
  networks: string[],
  simulatorSupport: unknown,
) {
  return NextResponse.json(
    { systems, models, backends, networks, simulatorSupport, timeoutSeconds: gatewayTimeoutSeconds() },
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

/**
 * Support reported by a simulator other than the default one: the models and
 * systems it can price, plus its backend/version table. AISimulate is the
 * top-level catalog, so this only adds per-simulator overrides.
 *
 * Best effort: an older AISimulators build has no `simulator` parameter, and a
 * simulator failure must not take down the catalog. Any failed part omits that
 * simulator, which the client treats as "no override".
 */
async function simulatorCapabilities(
  baseUrl: string,
  simulator: string,
  timeoutSeconds: number,
): Promise<unknown> {
  const options = (): RequestInit => ({
    headers: { Accept: 'application/json' },
    cache: 'no-store',
    signal: AbortSignal.timeout(timeoutSeconds * 1000),
  })
  try {
    const responses = await Promise.all([
      fetch(`${baseUrl}/backends?simulator=${simulator}`, options()),
      fetch(`${baseUrl}/models?simulator=${simulator}`, options()),
      fetch(`${baseUrl}/systems?simulator=${simulator}`, options()),
    ])
    if (!responses.every(response => response.ok)) return null
    const [backendsData, modelsData, systemsData] = await Promise.all(
      responses.map(response => response.json()),
    )
    // An older AISimulators build ignores the unknown `simulator` parameter and
    // answers with its AISimulate data at HTTP 200. Require each envelope to echo
    // the simulator we asked for, so a rolling deploy cannot advertise AISimulate
    // support as BLIS.
    const echoesSimulator = (data: unknown) =>
      !!data && typeof data === 'object' &&
      (data as Record<string, unknown>).simulator === simulator
    if (![backendsData, modelsData, systemsData].every(echoesSimulator)) return null
    const backends = catalogList(backendsData, 'backends')
    const models = catalogList(modelsData, 'models')
    const systems = catalogList(systemsData, 'systems')
    if (!backends || !models || !systems) return null
    const release = (backendsData as Record<string, unknown>).blis_version
    return {
      version: typeof release === 'string' ? release : null,
      backends,
      models,
      systems,
    }
  } catch {
    return null
  }
}

function catalogList(data: unknown, key: 'systems' | 'models' | 'backends' | 'networks'): unknown[] | null {
  if (!data || typeof data !== 'object') return null
  const value = (data as Record<string, unknown>)[key]
  return Array.isArray(value) ? value : null
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
    // through its combined same-origin /api/catalog proxy. This is useful for
    // local consumers when the standalone service hostname is being migrated.
    // Direct service deployments continue to use /systems, /models, /backends,
    // and /networks below.
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
      return catalogResponse(systems, models, backends, networks, simulatorSupport)
    }

    const [systemsRes, modelsRes, backendsResult, networksResult, blisSupport] = await Promise.all([
      fetch(`${baseUrl}/systems?include=specs`, {
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        signal: AbortSignal.timeout(timeoutSeconds * 1000),
      }),
      fetch(`${baseUrl}/models?include=specs`, {
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        signal: AbortSignal.timeout(timeoutSeconds * 1000),
      }),
      fetch(`${baseUrl}/backends`, {
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        signal: AbortSignal.timeout(timeoutSeconds * 1000),
      }).then(response => ({ response })).catch(() => ({ response: null })),
      fetch(`${baseUrl}/networks`, {
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        signal: AbortSignal.timeout(timeoutSeconds * 1000),
      }).then(response => ({ response })).catch(() => ({ response: null })),
      // BLIS reports its own support surface through the simulator-scoped
      // endpoints. It is the only non-default simulator today.
      simulatorCapabilities(baseUrl, 'blis', timeoutSeconds),
    ])

    if (!systemsRes.ok || !modelsRes.ok) {
      recordErrorCode('/api/catalog', 'AISIM_ERROR', 502)
      return NextResponse.json(
        {
          status: 'failed',
          error: {
            code: 'AISIM_ERROR',
            message: `AISimulators catalog fetch failed (systems ${systemsRes.status}, models ${modelsRes.status})`,
          },
        },
        { status: 502, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
      )
    }

    let systemsData: unknown
    let modelsData: unknown
    let backendsData: unknown = {}
    try {
      systemsData = await systemsRes.json()
      modelsData = await modelsRes.json()
    } catch {
      recordErrorCode('/api/catalog', 'AISIM_INVALID_RESPONSE', 502)
      return NextResponse.json(
        { status: 'failed', error: { code: 'AISIM_INVALID_RESPONSE', message: 'AISimulators returned non-JSON response' } },
        { status: 502, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
      )
    }
    if (backendsResult.response?.ok) {
      try {
        backendsData = await backendsResult.response.json()
      } catch {
        backendsData = {}
      }
    }

    let networks: string[] = []
    if (networksResult.response?.ok) {
      try {
        const networksData: unknown = await networksResult.response.json()
        networks = (catalogList(networksData, 'networks') ?? [])
          .filter((network): network is string => typeof network === 'string' && network.trim().length > 0)
      } catch {
        networks = []
      }
    }

    const systems = catalogList(systemsData, 'systems')
    const models = catalogList(modelsData, 'models')
    const backends = catalogList(backendsData, 'backends') ?? []
    if (!systems || !models || !validCatalogEntries(systems, models, backends)) return invalidCatalogResponse()
    const simulatorSupport = blisSupport ? { blis: blisSupport } : {}
    return catalogResponse(systems, models, backends, networks, simulatorSupport)
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

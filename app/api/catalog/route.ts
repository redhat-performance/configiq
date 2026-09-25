// GET /api/catalog
// Same-origin proxy for the AISimulators catalog. Fetches /systems, /models,
// and /backends server-side via AISIMULATORS_GATEWAY_URL and combines their
// raw shapes for useCatalog.
//
// The browser must call this route rather than AISimulators directly, so the
// per-host gateway (host.containers.internal on each deployment) is resolved
// server-side. That keeps the .dev/.xyz two-host split correct client-side and
// removes the need for a build-time NEXT_PUBLIC_AISIMULATORS_API_URL.

import { NextResponse } from 'next/server'
import { gatewayTimeoutSeconds } from '@/lib/api/timeout'

// This is the catalog fetch's own timeout (30s). The value surfaced to the
// client below is gatewayTimeoutSeconds() — the longer recommend/predict timeout.
const DEFAULT_TIMEOUT_SECONDS = 30

function catalogResponse(systems: unknown[], models: unknown[], backends: unknown[]) {
  return NextResponse.json(
    { systems, models, backends, timeoutSeconds: gatewayTimeoutSeconds() },
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

function catalogList(data: unknown, key: 'systems' | 'models' | 'backends'): unknown[] | null {
  if (!data || typeof data !== 'object') return null
  const value = (data as Record<string, unknown>)[key]
  return Array.isArray(value) ? value : null
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
    // Direct service deployments continue to use /systems, /models, and
    // /backends below.
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
      if (!systems || !models) return invalidCatalogResponse()
      return catalogResponse(systems, models, catalogList(catalogData, 'backends') ?? [])
    }

    const [systemsRes, modelsRes, backendsResult] = await Promise.all([
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
    ])

    if (!systemsRes.ok || !modelsRes.ok) {
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

    const systems = catalogList(systemsData, 'systems')
    const models = catalogList(modelsData, 'models')
    if (!systems || !models) return invalidCatalogResponse()
    return catalogResponse(systems, models, catalogList(backendsData, 'backends') ?? [])
  } catch (err: unknown) {
    if (err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      return NextResponse.json(
        { status: 'failed', error: { code: 'AISIM_TIMEOUT', message: 'AISimulators API timed out' } },
        { status: 504 },
      )
    }
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

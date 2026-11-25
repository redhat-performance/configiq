// GET /api/catalog
// Same-origin proxy for the AISimulators catalog. Discovers the simulators the
// service exposes through /simulators, then fetches each simulator's own
// /models, /systems, /backends, and /networks envelopes server-side via
// AISIMULATORS_GATEWAY_URL and folds them into one list of self-describing
// simulator entries for useCatalog.
//
// The browser must call this route rather than AISimulators directly, so the
// per-host gateway (host.containers.internal on each deployment) is resolved
// server-side, so the browser never needs a public API URL or a rewritten
// environment-specific hostname.
//
// The read itself lives in `lib/api/catalog-server` because the scheduled
// scorecard job needs exactly the same capabilities the pages see. Nothing in
// either file names a simulator: the default simulator, its label, version,
// models, systems, backends, fabrics, and sizing objectives all come from the
// service.

import { NextResponse } from 'next/server'
import { gatewayTimeoutSeconds } from '@/lib/api/timeout'
import { readCatalog, type LegacyCatalog, type SimulatorCatalogEntry } from '@/lib/api/catalog-server'
import { recordErrorCode } from '@/lib/otel-metrics'

export type {
  SimulatorCapabilities,
  SimulatorCatalogEntry,
  SimulatorDescriptor,
} from '@/lib/api/catalog-server'

// This is the catalog fetch's own timeout (30s). The value surfaced to the
// client below is gatewayTimeoutSeconds() — the longer recommend/predict timeout.
const DEFAULT_TIMEOUT_SECONDS = 30

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

  const result = await readCatalog(baseUrl, timeoutSeconds, {
    isSelfReferential: url => isSelfReferentialGateway(url, request),
  })

  if (!result.ok) {
    recordErrorCode('/api/catalog', result.error.code, result.error.status)
    return NextResponse.json(
      { status: 'failed', error: { code: result.error.code, message: result.error.message } },
      { status: result.error.status, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
    )
  }

  if (result.simulators.length === 0) {
    if (!result.legacy) {
      return NextResponse.json(
        {
          status: 'failed',
          error: { code: 'AISIM_INVALID_RESPONSE', message: 'AISimulators catalog is missing systems or models' },
        },
        { status: 502, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
      )
    }
    return catalogResponse([], result.legacy)
  }

  return catalogResponse(result.simulators)
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

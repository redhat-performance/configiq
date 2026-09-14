// GET /api/catalog
// Same-origin proxy for the AIConfigurator catalog. Fetches /systems and
// /models (both with specs) server-side via AICONFIGURATOR_GATEWAY_URL and
// returns their raw shapes combined as { systems, models }.
//
// The browser must call this route rather than AIConfigurator directly, so the
// per-host gateway (host.containers.internal on each deployment) is resolved
// server-side. That keeps the .dev/.xyz two-host split correct client-side and
// removes the need for a build-time NEXT_PUBLIC_AICONFIGURATOR_API_URL.

import { NextResponse } from 'next/server'
import { aicTimeoutSeconds } from '@/lib/api/timeout'

// This is the catalog fetch's own timeout (30s). The value surfaced to the
// client below is aicTimeoutSeconds() — the longer recommend/estimate timeout.
const DEFAULT_TIMEOUT_SECONDS = 30

export async function GET() {
  // Dev default mirrors app/api/gpus/route.ts so local dev works without the
  // gateway env set; production sets AICONFIGURATOR_GATEWAY_URL per host.
  const baseUrl = process.env.AICONFIGURATOR_GATEWAY_URL || 'https://aiconfigurator.dev'
  const timeoutSeconds =
    parseInt(process.env.AICONFIGURATOR_TIMEOUT_SECONDS || '', 10) ||
    DEFAULT_TIMEOUT_SECONDS

  try {
    const [systemsRes, modelsRes] = await Promise.all([
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
    ])

    if (!systemsRes.ok || !modelsRes.ok) {
      return NextResponse.json(
        {
          status: 'failed',
          error: {
            code: 'AIC_ERROR',
            message: `AIConfigurator catalog fetch failed (systems ${systemsRes.status}, models ${modelsRes.status})`,
          },
        },
        { status: 502, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
      )
    }

    let systemsData: { systems?: unknown[] }
    let modelsData: { models?: unknown[] }
    try {
      systemsData = await systemsRes.json()
      modelsData = await modelsRes.json()
    } catch {
      return NextResponse.json(
        { status: 'failed', error: { code: 'AIC_INVALID_RESPONSE', message: 'AIConfigurator returned non-JSON response' } },
        { status: 502, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
      )
    }

    return NextResponse.json(
      {
        systems: systemsData.systems ?? [],
        models: modelsData.models ?? [],
        // Effective AIC request timeout (recommend/estimate), for the loader hint.
        timeoutSeconds: aicTimeoutSeconds(),
      },
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          // Mirrors the useAicCatalog client-side cache TTL (10 min).
          'Cache-Control': 'public, s-maxage=600, stale-while-revalidate=3600',
        },
      },
    )
  } catch (err: unknown) {
    if (err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      return NextResponse.json(
        { status: 'failed', error: { code: 'AIC_TIMEOUT', message: 'AIConfigurator API timed out' } },
        { status: 504 },
      )
    }
    return NextResponse.json(
      { status: 'failed', error: { code: 'AIC_UNAVAILABLE', message: 'AIConfigurator API is unreachable' } },
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

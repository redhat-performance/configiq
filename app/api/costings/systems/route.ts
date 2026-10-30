// GET /api/costings/systems
// Same-origin proxy for the aicostings /systems endpoint. Forwards the optional
// ?include= extras (cloud, hardware) to the upstream and returns its raw shape
// { systems }.
//
// The browser must call this route rather than aicostings directly, so the
// per-host gateway is resolved server-side. Keeps the .dev/.xyz two-host split
// correct client-side (no build-time NEXT_PUBLIC_AICOSTINGS_API_URL).

import { NextRequest, NextResponse } from 'next/server'
import { recordErrorCode } from '@/lib/otel-metrics'

const DEFAULT_TIMEOUT_SECONDS = 30

export async function GET(req: NextRequest) {
  // Dev default mirrors app/api/gpus/route.ts so local dev works without the
  // gateway env set; production sets AICOSTINGS_GATEWAY_URL per host.
  const baseUrl = process.env.AICOSTINGS_GATEWAY_URL || 'https://aicostings.dev'
  const timeoutSeconds =
    parseInt(process.env.AICOSTINGS_TIMEOUT_SECONDS || '', 10) || DEFAULT_TIMEOUT_SECONDS

  const include = new URL(req.url).searchParams.get('include')
  const query = include ? `?include=${encodeURIComponent(include)}` : ''

  try {
    const res = await fetch(`${baseUrl}/systems${query}`, {
      headers: { Accept: 'application/json' },
      cache: 'no-store',
      signal: AbortSignal.timeout(timeoutSeconds * 1000),
    })

    if (!res.ok) {
      recordErrorCode('/api/costings/systems', 'COSTINGS_ERROR', 502)
      return NextResponse.json(
        {
          status: 'failed',
          error: { code: 'COSTINGS_ERROR', message: `aicostings /systems returned ${res.status}` },
        },
        { status: 502, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
      )
    }

    const data = await res.json()
    const systems: Record<string, unknown>[] = Array.isArray(data.systems) ? data.systems : []
    const extras = include?.split(',') ?? []
    const cacheable = systems.length > 0 &&
      (!extras.includes('cloud') || systems.some(system => system.cloud_rates && Object.keys(system.cloud_rates).length > 0)) &&
      (!extras.includes('hardware') || systems.some(system => system.hardware_cost))
    return NextResponse.json(data, {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        // Base system IDs can be ready before background pricing collection.
        'Cache-Control': cacheable ? 'public, s-maxage=3600, stale-while-revalidate=7200' : 'no-store',
      },
    })
  } catch (err: unknown) {
    if (err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      recordErrorCode('/api/costings/systems', 'COSTINGS_TIMEOUT', 504)
      return NextResponse.json(
        { status: 'failed', error: { code: 'COSTINGS_TIMEOUT', message: 'aicostings API timed out' } },
        { status: 504, headers: { 'Access-Control-Allow-Origin': '*' } },
      )
    }
    recordErrorCode('/api/costings/systems', 'COSTINGS_UNAVAILABLE', 502)
    return NextResponse.json(
      { status: 'failed', error: { code: 'COSTINGS_UNAVAILABLE', message: 'aicostings API is unreachable' } },
      { status: 502, headers: { 'Access-Control-Allow-Origin': '*' } },
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

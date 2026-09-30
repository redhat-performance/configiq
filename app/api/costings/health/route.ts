// GET /api/costings/health
// Same-origin proxy for the aicostings /health endpoint. Returns the raw shape
// { status, version, sources } describing per-source scrape freshness.
//
// The browser must call this route rather than aicostings directly, so the
// per-host gateway is resolved server-side. Keeps the .dev/.xyz two-host split
// correct client-side (no build-time NEXT_PUBLIC_AICOSTINGS_API_URL).

import { NextResponse } from 'next/server'
import { recordErrorCode } from '@/lib/otel-metrics'

const DEFAULT_TIMEOUT_SECONDS = 30

export async function GET() {
  // Dev default mirrors app/api/gpus/route.ts so local dev works without the
  // gateway env set; production sets AICOSTINGS_GATEWAY_URL per host.
  const baseUrl = process.env.AICOSTINGS_GATEWAY_URL || 'https://aicostings.dev'
  const timeoutSeconds =
    parseInt(process.env.AICOSTINGS_TIMEOUT_SECONDS || '', 10) || DEFAULT_TIMEOUT_SECONDS

  try {
    const res = await fetch(`${baseUrl}/health`, {
      headers: { Accept: 'application/json' },
      cache: 'no-store',
      signal: AbortSignal.timeout(timeoutSeconds * 1000),
    })

    if (!res.ok) {
      recordErrorCode('/api/costings/health', 'COSTINGS_ERROR', 502)
      return NextResponse.json(
        {
          status: 'failed',
          error: { code: 'COSTINGS_ERROR', message: `aicostings /health returned ${res.status}` },
        },
        { status: 502, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
      )
    }

    const data = await res.json()
    return NextResponse.json(data, {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        // Health is cheap and changes on each scrape tick; keep it short-lived.
        'Cache-Control': 'public, s-maxage=60, stale-while-revalidate=300',
      },
    })
  } catch (err: unknown) {
    if (err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      recordErrorCode('/api/costings/health', 'COSTINGS_TIMEOUT', 504)
      return NextResponse.json(
        { status: 'failed', error: { code: 'COSTINGS_TIMEOUT', message: 'aicostings API timed out' } },
        { status: 504, headers: { 'Access-Control-Allow-Origin': '*' } },
      )
    }
    recordErrorCode('/api/costings/health', 'COSTINGS_UNAVAILABLE', 502)
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

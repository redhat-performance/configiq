// GET /api/costings/models
// Same-origin proxy for the aicostings /models endpoint. Forwards the optional
// ?source= selector to the upstream and returns its raw shape
// { models, source, updated_at, stale }.
//
// The set of valid feeds is owned by the aicostings API (it 400s on an unknown
// source), not hardcoded here — that keeps new/retired feeds working without a
// proxy change. The source is URL-encoded before interpolation.
//
// The browser must call this route rather than aicostings directly, so the
// per-host gateway is resolved server-side. That keeps the .dev/.xyz two-host
// split correct client-side and removes the need for a build-time
// NEXT_PUBLIC_AICOSTINGS_API_URL.

import { NextRequest, NextResponse } from 'next/server'
import { recordErrorCode } from '@/lib/otel-metrics'

const DEFAULT_TIMEOUT_SECONDS = 30

export async function GET(req: NextRequest) {
  // Dev default mirrors app/api/gpus/route.ts so local dev works without the
  // gateway env set; production sets AICOSTINGS_GATEWAY_URL per host.
  const baseUrl = process.env.AICOSTINGS_GATEWAY_URL || 'https://aicostings.dev'
  const timeoutSeconds =
    parseInt(process.env.AICOSTINGS_TIMEOUT_SECONDS || '', 10) || DEFAULT_TIMEOUT_SECONDS

  const source = (new URL(req.url).searchParams.get('source') || 'merged').toLowerCase()

  try {
    const res = await fetch(`${baseUrl}/models?source=${encodeURIComponent(source)}`, {
      headers: { Accept: 'application/json' },
      cache: 'no-store',
      signal: AbortSignal.timeout(timeoutSeconds * 1000),
    })

    if (!res.ok) {
      recordErrorCode('/api/costings/models', 'COSTINGS_ERROR', res.status >= 400 && res.status < 500 ? res.status : 502)
      // Relay client errors (e.g. 400 unknown source) verbatim so callers see
      // the real reason; wrap upstream server errors as a 502.
      if (res.status >= 400 && res.status < 500) {
        const body = await res.text()
        return new NextResponse(body, {
          status: res.status,
          headers: {
            'Content-Type': res.headers.get('content-type') ?? 'application/json',
            'Cache-Control': 'no-store',
            'Access-Control-Allow-Origin': '*',
          },
        })
      }
      return NextResponse.json(
        {
          status: 'failed',
          error: { code: 'COSTINGS_ERROR', message: `aicostings /models returned ${res.status}` },
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
        // Mirrors the useCostings client-side cache TTL (1 hour).
        'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=7200',
      },
    })
  } catch (err: unknown) {
    if (err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      recordErrorCode('/api/costings/models', 'COSTINGS_TIMEOUT', 504)
      return NextResponse.json(
        { status: 'failed', error: { code: 'COSTINGS_TIMEOUT', message: 'aicostings API timed out' } },
        { status: 504, headers: { 'Access-Control-Allow-Origin': '*' } },
      )
    }
    recordErrorCode('/api/costings/models', 'COSTINGS_UNAVAILABLE', 502)
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

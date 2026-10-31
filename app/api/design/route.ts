import { type NextRequest, NextResponse } from 'next/server'

/**
 * GET /api/design/stream — SSE proxy to the FastAPI /recommend/stream endpoint.
 *
 * Forwards all query params directly and streams the response body without
 * buffering. nginx must have proxy_buffering off for this path.
 */
export async function GET(req: NextRequest): Promise<NextResponse | Response> {
  const baseUrl = process.env.AISIMULATORS_GATEWAY_URL
  if (!baseUrl) {
    return NextResponse.json(
      { error: 'AISimulators API URL is not configured' },
      { status: 503 }
    )
  }

  const upstreamUrl = `${baseUrl}/recommend/stream?${req.nextUrl.searchParams.toString()}`

  let upstream: Response
  try {
    upstream = await fetch(upstreamUrl, {
      method: 'GET',
      headers: { Accept: 'text/event-stream' },
      signal: AbortSignal.any([req.signal, AbortSignal.timeout(90_000)]),
    })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'upstream fetch failed'
    return NextResponse.json({ error: msg }, { status: 502 })
  }

  if (!upstream.ok || !upstream.body) {
    const body = await upstream.text().catch(() => '')
    return NextResponse.json({ error: body || 'upstream error' }, { status: upstream.status })
  }

  // Stream the response body directly — do not buffer.
  return new Response(upstream.body, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'X-Accel-Buffering': 'no',
    },
  })
}

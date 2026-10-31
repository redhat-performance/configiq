import { type NextRequest, NextResponse } from 'next/server'

/**
 * POST /api/design — proxy to the FastAPI /recommend endpoint with
 * objective=pareto. Returns the full candidate ledger and SDK views.
 *
 * The aisimulate sweep is synchronous and bounded (20 trials / 8s per
 * candidate) to fit the gateway timeout, so a single request returns the
 * whole frontier. No streaming.
 */
export async function POST(req: NextRequest): Promise<NextResponse> {
  const baseUrl = process.env.AISIMULATORS_GATEWAY_URL
  if (!baseUrl) {
    return NextResponse.json(
      { error: 'AISimulators API URL is not configured' },
      { status: 503 }
    )
  }

  let body: Record<string, unknown>
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: 'invalid JSON body' }, { status: 400 })
  }

  try {
    const res = await fetch(`${baseUrl}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ ...body, objective: 'pareto', simulator: 'aisimulate' }),
      signal: AbortSignal.any([req.signal, AbortSignal.timeout(90_000)]),
    })
    const text = await res.text()
    return new NextResponse(text, {
      status: res.status,
      headers: { 'Content-Type': 'application/json' },
    })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'upstream fetch failed'
    return NextResponse.json({ error: msg }, { status: 502 })
  }
}

import { NextResponse } from 'next/server'
import { recordErrorCode } from '@/lib/otel-metrics'

/** Additive proxy: no browser credentials, no legacy systems-response changes. */
export async function GET() {
  const base = process.env.AICOSTINGS_GATEWAY_URL || 'https://aicostings.dev'
  const timeout = Number.parseInt(process.env.AICOSTINGS_TIMEOUT_SECONDS || '', 10) || 30
  try {
    const response = await fetch(`${base}/offers`, {
      headers: { Accept: 'application/json' }, cache: 'no-store', signal: AbortSignal.timeout(timeout * 1000),
    })
    if (!response.ok) {
      const status = response.status === 404 ? 404 : 502
      recordErrorCode('/api/costings/offers', 'OFFERS_ERROR', status)
      return NextResponse.json({ error: { message: 'Complete-offers pricing unavailable.' } }, { status, headers: { 'Cache-Control': 'no-store' } })
    }
    return NextResponse.json(await response.json(), { headers: { 'Cache-Control': 'no-store' } })
  } catch (error: unknown) {
    const status = error instanceof Error && ['AbortError', 'TimeoutError'].includes(error.name) ? 504 : 502
    recordErrorCode('/api/costings/offers', 'OFFERS_UNAVAILABLE', status)
    return NextResponse.json({ error: { message: 'Infrastructure pricing service unavailable.' } }, { status, headers: { 'Cache-Control': 'no-store' } })
  }
}

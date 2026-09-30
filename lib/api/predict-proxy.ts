import { NextRequest, NextResponse } from 'next/server'
import { gatewayTimeoutSeconds } from '@/lib/api/timeout'
import { recordErrorCode, recordModelRequest } from '@/lib/otel-metrics'

const DEPRECATION_HEADERS = {
  Deprecation: '@1790208000',
  Sunset: 'Wed, 30 Sep 2026 00:00:00 GMT',
  Link: '</api/predict>; rel="successor-version"',
}

export async function handlePredict(
  req: NextRequest,
  deprecated = false,
  metricRoute: 'predict' | 'estimate' = 'predict',
): Promise<NextResponse> {
  const route = `/api/${metricRoute}`
  const baseUrl = process.env.AISIMULATORS_GATEWAY_URL
  const timeoutSeconds = gatewayTimeoutSeconds()
  const extraHeaders = deprecated ? DEPRECATION_HEADERS : {}

  if (!baseUrl) {
    recordErrorCode(route, 'AISIM_NOT_CONFIGURED', 503)
    return NextResponse.json(
      { status: 'failed', error: { code: 'AISIM_NOT_CONFIGURED', message: 'AISimulators API URL is not configured' } },
      { status: 503, headers: extraHeaders },
    )
  }

  let body: unknown
  try {
    body = await req.json()
  } catch {
    recordErrorCode(route, 'INVALID_REQUEST', 400)
    return NextResponse.json(
      { status: 'failed', error: { code: 'INVALID_REQUEST', message: 'Invalid JSON body' } },
      { status: 400, headers: extraHeaders },
    )
  }
  const model = body && typeof body === 'object' && 'model_path' in body
    ? (body as { model_path?: unknown }).model_path
    : undefined
  if (metricRoute === 'predict') recordModelRequest('predict', model)

  const { searchParams } = new URL(req.url)
  const include = searchParams.get('include')
  const gatewayUrl = include
    ? `${baseUrl}/predict?include=${encodeURIComponent(include)}`
    : `${baseUrl}/predict`

  try {
    const res = await fetch(gatewayUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutSeconds * 1000),
    })

    const text = await res.text()
    let data: unknown
    try {
      data = JSON.parse(text)
    } catch {
    recordErrorCode(route, 'AISIM_INVALID_RESPONSE', 502, model)
      return NextResponse.json(
        { status: 'failed', error: { code: 'AISIM_INVALID_RESPONSE', message: 'AISimulators returned non-JSON response' } },
        { status: 502, headers: { ...extraHeaders, 'Cache-Control': 'no-store' } },
      )
    }

    if (!res.ok) {
      const d = data as Record<string, unknown>
      const raw = ((d?.error as Record<string, unknown>)?.message ?? d?.detail ?? '').toString().toLowerCase()
      let code = 'AISIM_NO_CONFIGURATION'
      if (raw.includes('oom') || raw.includes('does not fit in gpu memory')) code = 'OOM'
      else if (raw.includes('moe_ep_size') || raw.includes('moe_tp_size') || raw.includes('moe models')) code = 'MOE_PARAMS_REQUIRED'
      else if (res.status === 401 || raw.includes('authentication') || raw.includes('gated')) code = 'AUTH_REQUIRED'
      else if (res.status === 404 || raw.includes('not found')) code = 'MODEL_NOT_FOUND'
      const message = ((d?.error as Record<string, unknown>)?.message ?? d?.detail ?? 'Unknown error').toString()
      recordErrorCode(route, code, res.status, model)
      return NextResponse.json(
        { status: 'failed', error: { code, message } },
        { status: res.status, headers: { ...extraHeaders, 'Cache-Control': 'no-store' } },
      )
    }

    return NextResponse.json(data, {
      status: 200,
      headers: { ...extraHeaders, 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' },
    })
  } catch (err: unknown) {
    if (err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      recordErrorCode(route, 'AISIM_TIMEOUT', 504, model)
      return NextResponse.json(
        { status: 'failed', error: { code: 'AISIM_TIMEOUT', message: 'AISimulators API timed out' } },
        { status: 504, headers: extraHeaders },
      )
    }
    recordErrorCode(route, 'AISIM_UNAVAILABLE', 502, model)
    return NextResponse.json(
      { status: 'failed', error: { code: 'AISIM_UNAVAILABLE', message: 'AISimulators API is unreachable' } },
      { status: 502, headers: extraHeaders },
    )
  }
}

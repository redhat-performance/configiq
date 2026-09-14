import { NextRequest, NextResponse } from 'next/server'
import { aicTimeoutSeconds } from '@/lib/api/timeout'

export async function POST(req: NextRequest) {
  const baseUrl = process.env.AICONFIGURATOR_GATEWAY_URL
  const timeoutSeconds = aicTimeoutSeconds()

  if (!baseUrl) {
    return NextResponse.json(
      { status: 'failed', error: { code: 'AIC_NOT_CONFIGURED', message: 'AIConfigurator API URL is not configured' } },
      { status: 503 },
    )
  }

  const { searchParams } = new URL(req.url)
  const include = searchParams.get('include')
  const aicUrl = include
    ? `${baseUrl}/estimate?include=${encodeURIComponent(include)}`
    : `${baseUrl}/estimate`

  let body: unknown
  try {
    body = await req.json()
  } catch {
    return NextResponse.json(
      { status: 'failed', error: { code: 'INVALID_REQUEST', message: 'Invalid JSON body' } },
      { status: 400 },
    )
  }

  try {
    const res = await fetch(aicUrl, {
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
      return NextResponse.json(
        { status: 'failed', error: { code: 'AIC_INVALID_RESPONSE', message: 'AIConfigurator returned non-JSON response' } },
        { status: 502 },
      )
    }

    if (!res.ok) {
      const d = data as Record<string, unknown>
      const raw = (
        (d?.error as Record<string, unknown>)?.message ??
        d?.detail ??
        ''
      ).toString().toLowerCase()
      let code = 'AIC_NO_CONFIGURATION'
      if (raw.includes('oom') || raw.includes('does not fit in gpu memory')) code = 'OOM'
      else if (raw.includes('moe_ep_size') || raw.includes('moe_tp_size') || raw.includes('moe models')) code = 'MOE_PARAMS_REQUIRED'
      else if (res.status === 401 || raw.includes('authentication') || raw.includes('gated')) code = 'AUTH_REQUIRED'
      else if (res.status === 404 || raw.includes('not found')) code = 'MODEL_NOT_FOUND'
      const message = ((d?.error as Record<string, unknown>)?.message ?? d?.detail ?? 'Unknown error').toString()
      return NextResponse.json(
        { status: 'failed', error: { code, message } },
        { status: res.status, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
      )
    }

    return NextResponse.json(data, {
      status: 200,
      headers: {
        'Cache-Control': 'no-store',
        'Access-Control-Allow-Origin': '*',
      },
    })
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
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}

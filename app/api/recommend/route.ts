import { NextRequest, NextResponse } from 'next/server'
import { RecommendRequestSchema } from '@/lib/api/schemas'
import { callRecommend, generateRequestId } from '@/lib/api/recommend'
import { aicTimeoutSeconds } from '@/lib/api/timeout'

const ERROR_STATUS_MAP: Record<string, number> = {
  INVALID_REQUEST: 400,
  AIC_NOT_CONFIGURED: 503,
  AIC_UNAVAILABLE: 502,
  AIC_TIMEOUT: 504,
  AIC_INVALID_RESPONSE: 502,
  AIC_NO_CONFIGURATION: 422,
  INTERNAL_ERROR: 500,
}

async function proxyToAic(body: Record<string, unknown>, include: string): Promise<NextResponse> {
  const baseUrl = process.env.AICONFIGURATOR_GATEWAY_URL
  const timeoutSeconds = aicTimeoutSeconds()

  if (!baseUrl) {
    return NextResponse.json(
      { status: 'failed', error: { code: 'AIC_NOT_CONFIGURED', message: 'AIConfigurator API URL is not configured' } },
      { status: 503 },
    )
  }

  try {
    const res = await fetch(`${baseUrl}/recommend?include=${encodeURIComponent(include)}`, {
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
    return NextResponse.json(data, {
      status: res.ok ? 200 : res.status,
      headers: { 'Cache-Control': 'no-store' },
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

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const include = req.nextUrl.searchParams.get('include')

    if (include) {
      return proxyToAic(body, include)
    }

    const validated = RecommendRequestSchema.parse(body)
    const result = await callRecommend(validated)

    if (result.status === 'failed') {
      const httpStatus = ERROR_STATUS_MAP[result.error.code] ?? 500
      return NextResponse.json(result, { status: httpStatus })
    }

    return NextResponse.json(result, {
      status: 200,
      headers: { 'Cache-Control': 'no-store' },
    })
  } catch (err: unknown) {
    const requestId = generateRequestId()

    if (err instanceof Error && err.constructor.name === 'ZodError') {
      const zodErr = err as Error & { issues: unknown[] }
      return NextResponse.json(
        {
          requestId,
          status: 'failed',
          error: {
            code: 'INVALID_REQUEST',
            message: 'Request validation failed',
            details: zodErr.issues,
          },
        },
        { status: 400 }
      )
    }

    return NextResponse.json(
      {
        requestId,
        status: 'failed',
        error: {
          code: 'INTERNAL_ERROR',
          message: err instanceof Error ? err.message : 'An unexpected error occurred',
        },
      },
      { status: 500 }
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

import { NextRequest, NextResponse } from 'next/server'
import { RecommendRequestSchema } from '@/lib/api/schemas'
import { callRecommend, generateRequestId, incrementalRecommend, type RecommendProgressEvent } from '@/lib/api/recommend'
import { gatewayTimeoutSeconds } from '@/lib/api/timeout'

const ERROR_STATUS_MAP: Record<string, number> = {
  INVALID_REQUEST: 400,
  AISIM_NOT_CONFIGURED: 503,
  AISIM_UNAVAILABLE: 502,
  AISIM_TIMEOUT: 504,
  AISIM_INVALID_RESPONSE: 502,
  AISIM_NO_CONFIGURATION: 422,
  INTERNAL_ERROR: 500,
}

async function proxyToGateway(body: Record<string, unknown>, req: NextRequest, include?: string): Promise<NextResponse> {
  const baseUrl = process.env.AISIMULATORS_GATEWAY_URL
  const timeoutSeconds = gatewayTimeoutSeconds()
  const wantsStream = req.headers.get('accept')?.includes('text/event-stream') ?? false

  if (!baseUrl) {
    return NextResponse.json(
      { status: 'failed', error: { code: 'AISIM_NOT_CONFIGURED', message: 'AISimulators API URL is not configured' } },
      { status: 503 },
    )
  }

  try {
    const upstreamUrl = include
      ? `${baseUrl}/recommend?include=${encodeURIComponent(include)}`
      : `${baseUrl}/recommend`
    const res = await fetch(upstreamUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': wantsStream ? 'text/event-stream' : 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.any([req.signal, AbortSignal.timeout(timeoutSeconds * 1000)]),
    })

    if (wantsStream && res.ok && res.headers.get('content-type')?.includes('text/event-stream')) {
      if (!res.body) {
        return NextResponse.json(
          { status: 'failed', error: { code: 'AISIM_INVALID_RESPONSE', message: 'AISimulators returned an empty stream' } },
          { status: 502 },
        )
      }
      return new NextResponse(res.body, {
        status: res.status,
        headers: {
          'Content-Type': res.headers.get('content-type') ?? 'text/event-stream; charset=utf-8',
          'Cache-Control': 'no-cache, no-store',
          'X-Accel-Buffering': 'no',
        },
      })
    }

    const text = await res.text()
    let data: unknown
    try {
      data = JSON.parse(text)
    } catch {
      return NextResponse.json(
        { status: 'failed', error: { code: 'AISIM_INVALID_RESPONSE', message: 'AISimulators returned non-JSON response' } },
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
        { status: 'failed', error: { code: 'AISIM_TIMEOUT', message: 'AISimulators API timed out' } },
        { status: 504 },
      )
    }
    return NextResponse.json(
      { status: 'failed', error: { code: 'AISIM_UNAVAILABLE', message: 'AISimulators API is unreachable' } },
      { status: 502 },
    )
  }
}

function streamRecommendation(
  request: ReturnType<typeof RecommendRequestSchema.parse>,
  requestSignal: AbortSignal,
): NextResponse {
  const encoder = new TextEncoder()
  const abortController = new AbortController()
  if (requestSignal.aborted) abortController.abort()
  let closed = false
  const abort = () => abortController.abort()
  requestSignal.addEventListener('abort', abort, { once: true })
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (event: RecommendProgressEvent) => {
        if (closed) return
        controller.enqueue(encoder.encode(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`))
      }

      const finish = () => {
        if (closed) return
        closed = true
        controller.close()
      }

      try {
        for await (const event of incrementalRecommend(request, abortController.signal)) send(event)
        finish()
      } catch (err: unknown) {
        if (!abortController.signal.aborted) {
          send({
            type: 'completed',
            response: {
              requestId: generateRequestId(),
              status: 'failed',
              error: {
                code: 'INTERNAL_ERROR',
                message: err instanceof Error ? err.message : 'An unexpected error occurred',
              },
            },
          })
        }
        finish()
      } finally {
        requestSignal.removeEventListener('abort', abort)
      }
    },
    cancel() {
      closed = true
      abortController.abort()
      requestSignal.removeEventListener('abort', abort)
    },
  })

  return new NextResponse(stream, {
    status: 200,
    headers: {
      'Cache-Control': 'no-cache, no-store',
      'Content-Type': 'text/event-stream; charset=utf-8',
      'X-Accel-Buffering': 'no',
    },
  })
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const include = req.nextUrl.searchParams.get('include')
    const baseUrl = process.env.AISIMULATORS_GATEWAY_URL ?? ''

    // A /api gateway is another ConfigIQ deployment and already returns the
    // normalized RecommendResponse. Forward it once instead of parsing that
    // normalized response as a raw AISimulators service response a second time.
    if (include || /\/api\/?$/.test(baseUrl)) {
      return proxyToGateway(body, req, include ?? undefined)
    }

    const validated = RecommendRequestSchema.parse(body)
    if (req.headers.get('accept')?.includes('text/event-stream')) {
      return streamRecommendation(validated, req.signal)
    }
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

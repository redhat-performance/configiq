/**
 * Same-origin proxies for the deployment artifact endpoints.
 *
 * The browser never reaches AISimulators directly: the gateway host differs per
 * deployment and may not be publicly resolvable. These handlers forward the
 * request server-side, preserve the status the service chose, and pass a zip
 * response through as bytes so the archive is not re-encoded.
 */

import { NextRequest, NextResponse } from 'next/server'

import { gatewayTimeoutSeconds } from './timeout'
import { recordErrorCode } from '@/lib/otel-metrics'

const DEPLOY_ROUTE = '/api/deploy'
const TARGETS_ROUTE = '/api/deploy/targets'

function notConfigured(route: string): NextResponse {
  recordErrorCode(route, 'AISIM_NOT_CONFIGURED', 503)
  return NextResponse.json(
    { status: 'failed', error: { code: 'AISIM_NOT_CONFIGURED', message: 'AISimulators API URL is not configured' } },
    { status: 503 },
  )
}

function unavailable(route: string, status: 502 | 504): NextResponse {
  const timeout = status === 504
  recordErrorCode(route, timeout ? 'AISIM_TIMEOUT' : 'AISIM_UNAVAILABLE', status)
  return NextResponse.json(
    {
      status: 'failed',
      error: {
        code: timeout ? 'AISIM_TIMEOUT' : 'AISIM_UNAVAILABLE',
        message: timeout ? 'AISimulators API timed out' : 'AISimulators API is unreachable',
      },
    },
    { status },
  )
}

interface UpstreamError {
  status: number
  payload: unknown
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json()
  } catch {
    return null
  }
}

async function upstreamFailure(response: Response, route: string): Promise<UpstreamError> {
  const payload = await readJson(response)
  recordErrorCode(route, 'DEPLOY_FAILED', response.status)
  // 404 (unknown target) and 422 (the evaluated limits cannot be reproduced)
  // are answers the caller can act on, so the service's own status is kept.
  return { status: response.status, payload }
}

/** GET /api/deploy/targets -> AISimulators GET /deploy/targets. */
export async function handleDeployTargets(req: NextRequest): Promise<NextResponse> {
  const baseUrl = process.env.AISIMULATORS_GATEWAY_URL
  if (!baseUrl) return notConfigured(TARGETS_ROUTE)

  try {
    const res = await fetch(`${baseUrl}/deploy/targets`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: AbortSignal.any([req.signal, AbortSignal.timeout(gatewayTimeoutSeconds() * 1000)]),
    })
    if (!res.ok) {
      const { status, payload } = await upstreamFailure(res, TARGETS_ROUTE)
      return NextResponse.json(payload ?? { detail: 'The deployment target catalog could not be read.' }, { status })
    }
    return NextResponse.json(await res.json(), {
      status: 200,
      headers: { 'Cache-Control': 'no-store' },
    })
  } catch (err: unknown) {
    if (err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      if (req.signal.aborted) throw err
      return unavailable(TARGETS_ROUTE, 504)
    }
    return unavailable(TARGETS_ROUTE, 502)
  }
}

/** POST /api/deploy -> AISimulators POST /deploy, JSON or zip. */
export async function handleDeploy(req: NextRequest): Promise<NextResponse> {
  const baseUrl = process.env.AISIMULATORS_GATEWAY_URL
  if (!baseUrl) return notConfigured(DEPLOY_ROUTE)

  let body: unknown
  try {
    body = await req.json()
  } catch {
    recordErrorCode(DEPLOY_ROUTE, 'INVALID_REQUEST', 400)
    return NextResponse.json(
      { status: 'failed', error: { code: 'INVALID_REQUEST', message: 'Invalid JSON body' } },
      { status: 400 },
    )
  }

  const format = new URL(req.url).searchParams.get('format')
  const upstream = format === 'zip'
    ? `${baseUrl}/deploy?format=zip`
    : `${baseUrl}/deploy`

  try {
    const res = await fetch(upstream, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: format === 'zip' ? 'application/zip' : 'application/json',
      },
      body: JSON.stringify(body),
      signal: AbortSignal.any([req.signal, AbortSignal.timeout(gatewayTimeoutSeconds() * 1000)]),
    })

    if (!res.ok) {
      const { status, payload } = await upstreamFailure(res, DEPLOY_ROUTE)
      return NextResponse.json(payload ?? { detail: 'The deployment artifact could not be rendered.' }, { status })
    }

    if (format === 'zip') {
      const contentType = res.headers.get('content-type') ?? ''
      if (!contentType.includes('application/zip')) {
        // A mislabelled body would be saved as a corrupt archive; fail loudly.
        recordErrorCode(DEPLOY_ROUTE, 'AISIM_INVALID_RESPONSE', 502)
        return NextResponse.json(
          { status: 'failed', error: { code: 'AISIM_INVALID_RESPONSE', message: 'The deployment service did not return an archive.' } },
          { status: 502 },
        )
      }
      const headers = new Headers({ 'Content-Type': 'application/zip', 'Cache-Control': 'no-store' })
      const disposition = res.headers.get('content-disposition')
      if (disposition) headers.set('Content-Disposition', disposition)
      return new NextResponse(res.body, { status: 200, headers })
    }

    return NextResponse.json(await res.json(), {
      status: 200,
      headers: { 'Cache-Control': 'no-store' },
    })
  } catch (err: unknown) {
    if (err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      if (req.signal.aborted) throw err
      return unavailable(DEPLOY_ROUTE, 504)
    }
    return unavailable(DEPLOY_ROUTE, 502)
  }
}

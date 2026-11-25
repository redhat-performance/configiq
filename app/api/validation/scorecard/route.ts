// GET /api/validation/scorecard
// Same-origin proxy for the published simulator scorecard artifact. The sweep
// runs in a scheduled job and publishes to a public Hugging Face dataset, so
// this route only reads and validates it; nothing is computed per request.
//
// The browser must call this route rather than the artifact host directly so
// the repository, channel, and caching policy stay server-side.

import { NextResponse } from 'next/server'
import { readPublishedScorecard } from '@/lib/scorecard/published'
import {
  SCORECARD_CHANNELS,
  type ScorecardChannel,
} from '@/lib/scorecard/types'

function parseChannel(value: string | null): ScorecardChannel | null {
  if (!value) return 'dev'
  return (SCORECARD_CHANNELS as readonly string[]).includes(value)
    ? value as ScorecardChannel
    : null
}

function failure(status: number, code: string, message: string) {
  return NextResponse.json(
    { status: 'failed', error: { code, message } },
    { status, headers: { 'Cache-Control': 'no-store', 'Access-Control-Allow-Origin': '*' } },
  )
}

export async function GET(request: Request) {
  const url = new URL(request.url)
  const channel = parseChannel(url.searchParams.get('channel'))
  if (!channel) {
    return failure(400, 'SCORECARD_CHANNEL_INVALID', `channel must be one of ${SCORECARD_CHANNELS.join(', ')}`)
  }
  const includePoints = url.searchParams.get('points') === '1'
  const refresh = url.searchParams.get('refresh') === '1'

  const result = await readPublishedScorecard({ channel, includePoints, refresh })
  if (!result.ok) {
    if (result.error.kind === 'missing') {
      return failure(404, 'SCORECARD_NOT_PUBLISHED', result.error.message)
    }
    if (result.error.kind === 'invalid') {
      return failure(502, 'SCORECARD_INVALID', result.error.message)
    }
    return failure(503, 'SCORECARD_UNAVAILABLE', result.error.message)
  }

  return NextResponse.json(
    {
      status: 'ok',
      channel,
      fetchedAt: result.value.fetchedAt,
      summary: result.value.summary,
      points: includePoints ? result.value.points : undefined,
    },
    {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        // The sweep runs daily; five minutes is short enough to pick up a new
        // snapshot promptly and long enough that a burst of visitors shares it.
        'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=3600',
      },
    },
  )
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

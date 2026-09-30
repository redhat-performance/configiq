import type { RecommendProgressEvent, RecommendResponse } from './recommend'

/**
 * Read the server-sent event response produced by the incremental
 * `/api/recommend` route and return its terminal recommendation.
 *
 * Keeping the parser shared avoids pages quietly falling back to the much
 * slower unbounded JSON recommendation path when AISimulators is in use.
 */
export async function readRecommendStream(
  response: Response,
  onProgress?: (event: Exclude<RecommendProgressEvent, { type: 'completed' }>) => void,
): Promise<RecommendResponse> {
  if (!response.ok) {
    let message = `Recommendation request failed (${response.status})`
    try {
      const body = await response.json() as { error?: { message?: unknown }; detail?: unknown }
      if (typeof body.error?.message === 'string') message = body.error.message
      else if (typeof body.detail === 'string') message = body.detail
    } catch { /* retain the HTTP error */ }
    throw new Error(message)
  }

  // Older ConfigIQ gateways may return already-normalized JSON even when
  // streaming was requested. Keep that response compatible with the SSE path.
  if (response.headers.get('content-type')?.includes('application/json')) {
    const body: unknown = await response.json()
    if (
      body && typeof body === 'object' &&
      ((body as RecommendResponse).status === 'completed' || (body as RecommendResponse).status === 'failed')
    ) {
      return body as RecommendResponse
    }
    throw new Error('Recommendation API returned an invalid JSON response')
  }

  if (!response.body) throw new Error('Recommendation stream unavailable')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let completed: RecommendResponse | null = null

  const handleBlock = (block: string) => {
    const dataLine = block
      .split('\n')
      .map(line => line.trimEnd())
      .find(line => line.startsWith('data:'))
    if (!dataLine) return

    const event = JSON.parse(dataLine.slice(5).trimStart()) as RecommendProgressEvent
    if (event.type === 'completed') completed = event.response
    else onProgress?.(event)
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer = (buffer + decoder.decode(value, { stream: !done })).replaceAll('\r\n', '\n')
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop() ?? ''
    blocks.forEach(handleBlock)
    if (done) break
  }

  if (buffer.trim()) handleBlock(buffer)
  if (!completed) throw new Error('Recommendation stream ended before a result was received')
  return completed
}

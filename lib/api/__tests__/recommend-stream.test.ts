import { describe, expect, it, vi } from 'vitest'
import { readRecommendStream } from '../recommend-stream'

function streamResponse(chunks: string[], status = 200): Response {
  const encoder = new TextEncoder()
  return new Response(new ReadableStream({
    start(controller) {
      chunks.forEach(chunk => controller.enqueue(encoder.encode(chunk)))
      controller.close()
    },
  }), {
    status,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

describe('readRecommendStream', () => {
  it('reads progress and a completed recommendation across chunk boundaries', async () => {
    const onProgress = vi.fn()
    const completed = {
      type: 'completed',
      response: {
        requestId: 'size_123',
        status: 'failed',
        error: { code: 'AISIM_NO_CONFIGURATION', message: 'No configuration' },
      },
    }
    const response = streamResponse([
      'event: window_started\r',
      '\ndata: {"type":"window_started","window":{"minGpus":1,',
      '"maxGpus":1}}\r\n\r\nevent: completed\r\ndata: ',
      `${JSON.stringify(completed)}\r\n\r\n`,
    ])

    await expect(readRecommendStream(response, onProgress)).resolves.toEqual(completed.response)
    expect(onProgress).toHaveBeenCalledWith({
      type: 'window_started',
      window: { minGpus: 1, maxGpus: 1 },
    })
  })

  it('reports a stream that ends without a terminal result', async () => {
    const response = streamResponse([
      'event: search_started\ndata: {"type":"search_started","maxGpus":1024}\n\n',
    ])

    await expect(readRecommendStream(response)).rejects.toThrow(
      'Recommendation stream ended before a result was received',
    )
  })

  it('accepts the normalized JSON response returned by a ConfigIQ proxy', async () => {
    const result = {
      requestId: 'size_123',
      status: 'completed',
      mode: 'agg',
      recommendation: { gpusNeeded: 1 },
    }
    const response = Response.json(result)

    await expect(readRecommendStream(response)).resolves.toMatchObject(result)
  })

  it('surfaces a JSON API error before attempting to parse events', async () => {
    const response = new Response(JSON.stringify({ detail: 'Invalid workload' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    })

    await expect(readRecommendStream(response)).rejects.toThrow('Invalid workload')
  })
})

import { afterEach, describe, expect, it, vi } from 'vitest'
import { NextRequest } from 'next/server'
import { POST } from './route'
import { readRecommendStream } from '@/lib/api/recommend-stream'

const requestBody = { model_path: 'Qwen/Qwen3-8B' }

function request(accept: string, path = '/api/recommend', signal?: AbortSignal): NextRequest {
  return new NextRequest(`http://localhost:3001${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: accept },
    body: JSON.stringify(requestBody),
    signal,
  })
}

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
})

describe('POST /api/recommend through a ConfigIQ gateway', () => {
  it('forwards progress as it arrives without waiting for the terminal event', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://configiq.dev/api')
    const encoder = new TextEncoder()
    let upstreamController: ReadableStreamDefaultController<Uint8Array> | undefined
    const upstream = new ReadableStream<Uint8Array>({
      start(controller) { upstreamController = controller },
    })
    const fetchMock = vi.fn(() => Promise.resolve(new Response(upstream, {
      headers: { 'Content-Type': 'text/event-stream; charset=utf-8' },
    })))
    vi.stubGlobal('fetch', fetchMock)

    const response = await POST(request('text/event-stream'))
    expect(fetchMock).toHaveBeenCalledWith('https://configiq.dev/api/recommend', expect.objectContaining({
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    }))
    expect(response.headers.get('content-type')).toContain('text/event-stream')
    expect(response.headers.get('x-accel-buffering')).toBe('no')

    const onProgress = vi.fn()
    const recommendation = readRecommendStream(response, onProgress)
    upstreamController!.enqueue(encoder.encode('event: search_started\ndata: {"type":"search_started","maxGpus":8}\n\n'))
    await vi.waitFor(() => expect(onProgress).toHaveBeenCalledWith({ type: 'search_started', maxGpus: 8 }))
    upstreamController!.enqueue(encoder.encode('event: completed\ndata: {"type":"completed","response":{"requestId":"size_123","status":"failed","error":{"code":"AISIM_NO_CONFIGURATION","message":"No configuration"}}}\n\n'))
    upstreamController!.close()
    await expect(recommendation).resolves.toMatchObject({
      requestId: 'size_123',
      status: 'failed',
      error: { code: 'AISIM_NO_CONFIGURATION' },
    })
  })

  it('keeps the JSON contract for ordinary callers and older JSON gateways', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://configiq.dev/api')
    const result = { requestId: 'size_123', status: 'completed', recommendation: { gpusNeeded: 1 } }
    const fetchMock = vi.fn(() => Promise.resolve(Response.json(result)))
    vi.stubGlobal('fetch', fetchMock)

    const jsonResponse = await POST(request('application/json', '/api/recommend?include=config,memory'))
    expect(fetchMock).toHaveBeenCalledWith(
      'https://configiq.dev/api/recommend?include=config%2Cmemory',
      expect.objectContaining({ headers: { 'Content-Type': 'application/json', Accept: 'application/json' } }),
    )
    expect(await jsonResponse.json()).toEqual(result)

    const streamRequestResponse = await POST(request('text/event-stream'))
    await expect(readRecommendStream(streamRequestResponse)).resolves.toEqual(result)
  })

  it('stops the upstream request when the client disconnects', async () => {
    vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'https://configiq.dev/api')
    const client = new AbortController()
    const fetchMock = vi.fn((_url: string, _init?: RequestInit) => Promise.resolve(new Response(new ReadableStream({
      start(controller) { controller.enqueue(new TextEncoder().encode('event: search_started\n\n')) },
    }), { headers: { 'Content-Type': 'text/event-stream' } })))
    vi.stubGlobal('fetch', fetchMock)

    await POST(request('text/event-stream', '/api/recommend', client.signal))
    const upstreamSignal = fetchMock.mock.calls[0]?.[1]?.signal
    if (!upstreamSignal) throw new Error('The gateway request must have an abort signal')
    expect(upstreamSignal.aborted).toBe(false)
    client.abort()
    expect(upstreamSignal.aborted).toBe(true)
  })
})

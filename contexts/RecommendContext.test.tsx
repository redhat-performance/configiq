// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/api/recommend-stream', () => ({
  readRecommendStream: vi.fn().mockResolvedValue({
    status: 'failed',
    error: { code: 'AISIM_NO_CONFIGURATION', message: 'test response' },
  }),
}))

vi.mock('@/lib/api/pareto', () => ({
  fetchFrontier: vi.fn().mockResolvedValue({ candidates: [], views: null, sweep: null }),
}))

import { RecommendProvider, useRecommend } from './RecommendContext'
import { readRecommendStream } from '@/lib/api/recommend-stream'
import { fetchFrontier } from '@/lib/api/pareto'

let root: ReturnType<typeof createRoot>
let host: HTMLDivElement
let startSizing: ReturnType<typeof useRecommend>['startSizing']

function Consumer() {
  const currentStartSizing = useRecommend().startSizing
  React.useEffect(() => {
    startSizing = currentStartSizing
  }, [currentStartSizing])
  return null
}

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  // The module mocks are shared across tests; clear call history so each test
  // observes only its own requests.
  vi.clearAllMocks()
  host = document.createElement('div')
  document.body.append(host)
  root = createRoot(host)
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200 }))
})

afterEach(async () => {
  await act(async () => root.unmount())
  host.remove()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

async function render() {
  await act(async () => root.render(<RecommendProvider><Consumer /></RecommendProvider>))
}

describe('RecommendContext request construction', () => {
  it.each(['roce-200g', null])('preserves the explicit BLIS fabric value %j', async fabric => {
    await render()

    await act(async () => {
      startSizing({
        model_path: 'org/model',
        system: 'h200_sxm',
        isl: 2048,
        osl: 128,
        sla_mode: 'enforce',
        ttft: 1000,
        target_concurrency: 8,
        simulator: 'blis',
        fabric,
      })
      await Promise.resolve()
    })

    const fetchMock = vi.mocked(fetch)
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.simulator).toBe('blis')
    expect(body).toHaveProperty('fabric', fabric)
  })

  it('forwards a caller-selected fabric for any simulator', async () => {
    await render()

    await act(async () => {
      startSizing({
        model_path: 'org/model',
        system: 'h200_sxm',
        isl: 2048,
        osl: 128,
        sla_mode: 'enforce',
        ttft: 1000,
        target_concurrency: 8,
        simulator: 'aisimulate',
        fabric: 'roce-200g',
      })
      await Promise.resolve()
    })

    const fetchMock = vi.mocked(fetch)
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.simulator).toBe('aisimulate')
    expect(body).toHaveProperty('fabric', 'roce-200g')
  })

  it('omits a fabric the caller did not select', async () => {
    await render()

    await act(async () => {
      startSizing({
        model_path: 'org/model',
        system: 'h200_sxm',
        isl: 2048,
        osl: 128,
        sla_mode: 'enforce',
        ttft: 1000,
        target_concurrency: 8,
        simulator: 'aisimulate',
      })
      await Promise.resolve()
    })

    const fetchMock = vi.mocked(fetch)
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body).not.toHaveProperty('fabric')
  })

  it('builds a throughput-demand request without latency bounds', async () => {
    await render()

    await act(async () => {
      startSizing({
        model_path: 'org/model',
        system: 'h200_sxm',
        isl: 2048,
        osl: 128,
        sla_mode: 'none',
        target_request_rate: 12.5,
        simulator: 'aisimulate',
      })
      await Promise.resolve()
    })

    const fetchMock = vi.mocked(fetch)
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.objective).toBe('throughput_demand')
    expect(body.sla_mode).toBe('none')
    expect(body.target_request_rate).toBe(12.5)
    expect(body).not.toHaveProperty('target_concurrency')
    expect(body).not.toHaveProperty('ttft')
    expect(body).not.toHaveProperty('tpot')
    expect(body).not.toHaveProperty('request_latency')
  })

  it('sizes the companion scan from the primary result and names the window in API fields', async () => {
    vi.mocked(readRecommendStream).mockResolvedValueOnce({
      status: 'completed',
      recommendation: { gpusNeeded: 4 },
    } as never)
    await render()

    await act(async () => {
      startSizing({
        model_path: 'org/model',
        system: 'h200_sxm',
        isl: 2048,
        osl: 128,
        sla_mode: 'enforce',
        ttft: 1000,
        target_concurrency: 8,
        simulator: 'aisimulate',
      })
      await Promise.resolve()
      await Promise.resolve()
    })

    const frontier = vi.mocked(fetchFrontier)
    expect(frontier).toHaveBeenCalledTimes(1)
    const [request] = frontier.mock.calls[0]
    // 2x the primary GPU count, floored at a 32-GPU window, halved below it.
    expect(request).toMatchObject({
      simulator: 'aisimulate',
      min_candidate_gpus: 2,
      max_candidate_gpus: 32,
    })
    expect(request).not.toHaveProperty('minGpus')
    expect(request).not.toHaveProperty('maxGpus')
  })

  it('never launches a frontier for BLIS and reports why', async () => {
    vi.mocked(readRecommendStream).mockResolvedValueOnce({
      status: 'completed',
      recommendation: { gpusNeeded: 4 },
    } as never)
    await render()

    await act(async () => {
      startSizing({
        model_path: 'org/model',
        system: 'h200_sxm',
        isl: 2048,
        osl: 128,
        sla_mode: 'enforce',
        ttft: 1000,
        target_concurrency: 8,
        simulator: 'blis',
      })
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(vi.mocked(fetchFrontier)).not.toHaveBeenCalled()
  })
})

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

import { RecommendProvider, useRecommend } from './RecommendContext'

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
})

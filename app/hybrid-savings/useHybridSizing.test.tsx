// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { GpuOption } from '@/lib/hooks/useCatalog'
import type { RecommendResponse } from '@/lib/api/recommend'
import { readRecommendStream } from '@/lib/api/recommend-stream'
import { useHybridSizing } from './useHybridSizing'

vi.mock('@/lib/api/recommend-stream', () => ({ readRecommendStream: vi.fn() }))

function gpu(systemId: string): GpuOption {
  return { systemId, label: systemId, vendor: null, architecture: null, vramGb: null,
    bandwidthTbps: null, tflopsBf16: null, tdpWatts: 400, gpusPerNode: null }
}
const result = {
  status: 'completed',
  recommendation: { gpusPerReplica: 1, replicasNeeded: 1 },
  throughput: { tokensPerSecond: 1000 },
  performance: { ttftLatencyMs: 100, tpotMs: 20 },
} as RecommendResponse
const input = {
  signature: 'first', model: 'Qwen/Qwen3-8B', backend: 'vllm',
  averageInputTokens: 2048, averageOutputTokens: 512, targetTtftMs: 5000, targetTpotMs: 60,
  targetConcurrency: 32, prefixTokens: 0,
  preferredCloudProvider: null, gpuOptions: [gpu('l40s')],
}

let root: ReturnType<typeof createRoot>
let hook: ReturnType<typeof useHybridSizing>
let host: HTMLDivElement
const fetchMock = vi.fn<typeof fetch>()
function Harness() {
  const next = useHybridSizing()
  React.useEffect(() => { hook = next })
  return null
}
function deferred() {
  let resolve!: (value: Response) => void
  const promise = new Promise<Response>(done => { resolve = done })
  return { promise, resolve }
}

beforeEach(async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset().mockResolvedValue(new Response())
  vi.mocked(readRecommendStream).mockReset().mockResolvedValue(result)
  host = document.createElement('div')
  document.body.append(host)
  root = createRoot(host)
  await act(async () => root.render(<Harness />))
})
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.unstubAllGlobals() })

describe('Hybrid live sizing orchestration', () => {
  it('collects eligible offers and forwards the selected reference load, prefix and HF config', async () => {
    await act(async () => hook.runSizing({ ...input, targetConcurrency: 64, prefixTokens: 1024, modelConfig: { model_type: 'qwen3' } }))
    const init = fetchMock.mock.calls[0][1]!
    expect(init.headers).toEqual({ 'Content-Type': 'application/json', Accept: 'text/event-stream' })
    expect(JSON.parse(init.body as string)).toMatchObject({
      model_path: input.model, system: 'l40s', backend: 'vllm',
      isl: 2048, osl: 512, target_concurrency: 64, prefix: 1024, top_n: 5,
      model_config: { model_type: 'qwen3' },
    })
    expect(hook.lastSizingSignature).toBe('first')
    expect(hook.progress).toEqual({ completed: 1, total: 1 })
    expect(hook.candidates.some(candidate => candidate.configurationId === 'aws-g6e-xlarge')).toBe(true)
    expect(hook.candidates.some(candidate => candidate.purchasePricePerReplica !== null)).toBe(true)
    expect(hook.isSizing).toBe(false)
  })

  it('keeps successfully sized but unpriced systems and counts failures separately', async () => {
    vi.mocked(readRecommendStream).mockResolvedValueOnce(result).mockResolvedValueOnce({
      status: 'failed', requestId: 'failed', error: { code: 'NO_FIT', message: 'No fit' },
    })
    await act(async () => hook.runSizing({ ...input, gpuOptions: [gpu('unpriced'), gpu('bad')] }))
    expect(hook.candidates).toHaveLength(1)
    expect(hook.candidates[0]).toMatchObject({ systemId: 'unpriced', cloudHourlyCostPerInstance: null, purchasePricePerReplica: null })
    expect(hook.failedSystems).toBe(1)
    expect(hook.progress).toEqual({ completed: 2, total: 2 })
    expect(hook.sizingError).toBeNull()
  })

  it('limits outstanding requests to four and completes the full queue', async () => {
    const pending = Array.from({ length: 6 }, deferred)
    fetchMock.mockImplementation(() => pending[fetchMock.mock.calls.length - 1].promise)
    let run!: Promise<void>
    await act(async () => { run = hook.runSizing({ ...input, gpuOptions: Array.from({ length: 6 }, (_, index) => gpu('unpriced-' + index)) }) })
    expect(fetchMock).toHaveBeenCalledTimes(4)
    await act(async () => { pending[0].resolve(new Response()); pending[1].resolve(new Response()) })
    expect(fetchMock).toHaveBeenCalledTimes(6)
    await act(async () => { for (const request of pending.slice(2)) request.resolve(new Response()); await run })
    expect(hook.progress.completed).toBe(6)
  })

  it('cancels active requests and never publishes late completions', async () => {
    const pending = deferred()
    fetchMock.mockReturnValue(pending.promise)
    let run!: Promise<void>
    await act(async () => { run = hook.runSizing(input) })
    const signal = fetchMock.mock.calls[0][1]!.signal!
    await act(async () => hook.cancel())
    expect(signal.aborted).toBe(true)
    expect(hook.isSizing).toBe(false)
    await act(async () => { pending.resolve(new Response()); await run })
    expect(hook.candidates).toEqual([])
    expect(hook.lastSizingSignature).toBeNull()
    expect(hook.progress.completed).toBe(0)
  })

  it('supersedes an old run without mixing models or progress', async () => {
    const old = deferred()
    fetchMock.mockReturnValueOnce(old.promise).mockResolvedValue(new Response())
    let run!: Promise<void>
    await act(async () => { run = hook.runSizing(input) })
    await act(async () => hook.runSizing({ ...input, signature: 'new-model', model: 'Qwen/Qwen3-32B' }))
    await act(async () => { old.resolve(new Response()); await run })
    expect(hook.lastSizingSignature).toBe('new-model')
    expect(hook.progress.completed).toBe(1)
  })

  it('aborts on unmount and reports API failures without a false success', async () => {
    fetchMock.mockRejectedValue(new Error('offline'))
    await act(async () => hook.runSizing(input))
    expect(hook.failedSystems).toBe(1)
    expect(hook.sizingError).toContain('could not find')
    expect(hook.isSizing).toBe(false)
    const pending = deferred()
    fetchMock.mockReturnValue(pending.promise)
    let run!: Promise<void>
    await act(async () => { run = hook.runSizing(input) })
    const signal = fetchMock.mock.calls.at(-1)![1]!.signal!
    await act(async () => root.unmount())
    expect(signal.aborted).toBe(true)
    pending.resolve(new Response())
    await run
    root = createRoot(host)
  })
})

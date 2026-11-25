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

vi.mock('@/lib/api/pareto', async importOriginal => {
  const actual = await importOriginal<typeof import('@/lib/api/pareto')>()
  return {
    ...actual,
    fetchFrontier: vi.fn().mockResolvedValue({ candidates: [], views: null, sweep: null, debug: null }),
  }
})

import { RecommendProvider, useRecommend } from './RecommendContext'
import { readRecommendStream } from '@/lib/api/recommend-stream'
import { fetchFrontier } from '@/lib/api/pareto'

let root: ReturnType<typeof createRoot>
let host: HTMLDivElement
let startSizing: ReturnType<typeof useRecommend>['startSizing']
let startPareto: ReturnType<typeof useRecommend>['startPareto']
let reset: ReturnType<typeof useRecommend>['reset']
let latest: ReturnType<typeof useRecommend>

function Consumer() {
  const current = useRecommend()
  React.useEffect(() => {
    startSizing = current.startSizing
    startPareto = current.startPareto
    reset = current.reset
    latest = current
  })
  return null
}

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  // The module mocks are shared across tests; restore the default sizing
  // response and clear call history so each test observes only its own run.
  vi.clearAllMocks()
  vi.mocked(readRecommendStream).mockResolvedValue({
    status: 'failed',
    error: { code: 'AISIM_NO_CONFIGURATION', message: 'test response' },
  } as never)
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

/** Let the mocked transport promises and their follow-up state writes settle. */
async function flush(times = 6) {
  for (let index = 0; index < times; index += 1) await Promise.resolve()
}

const SIZING_PARAMS = {
  model_path: 'org/model',
  system: 'h200_sxm',
  isl: 2048,
  osl: 128,
  sla_mode: 'enforce' as const,
  ttft: 1000,
  target_concurrency: 8,
  simulator: 'aisim' as const,
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
        simulator: 'aisim',
        fabric: 'roce-200g',
      })
      await Promise.resolve()
    })

    const fetchMock = vi.mocked(fetch)
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.simulator).toBe('aisim')
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
        simulator: 'aisim',
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
        simulator: 'aisim',
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

  it('builds a concurrency-capacity request without latency or rate targets', async () => {
    await render()

    await act(async () => {
      startSizing({
        model_path: 'org/model',
        system: 'h200_sxm',
        isl: 2048,
        osl: 128,
        sla_mode: 'none',
        objective: 'concurrency_capacity',
        target_concurrency: 64,
        simulator: 'aisim',
      })
      await Promise.resolve()
    })

    const fetchMock = vi.mocked(fetch)
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.objective).toBe('concurrency_capacity')
    expect(body.sla_mode).toBe('none')
    expect(body.target_concurrency).toBe(64)
    expect(body).not.toHaveProperty('target_request_rate')
    expect(body).not.toHaveProperty('ttft')
    expect(body).not.toHaveProperty('tpot')
    expect(body).not.toHaveProperty('request_latency')
  })
})

describe('RecommendContext action modes', () => {
  const completed = (gpusNeeded = 4) => ({
    status: 'completed',
    simulator: 'aisim',
    mode: 'agg',
    recommendation: {
      gpusNeeded,
      gpusPerReplica: gpusNeeded,
      totalGpus: gpusNeeded,
      replicasNeeded: 1,
      tensorParallelSize: gpusNeeded,
      pipelineParallelSize: 1,
      dataParallelSize: 1,
      contextParallelSize: 1,
      moeTensorParallelSize: null,
      moeExpertParallelSize: null,
      batchSize: 32,
    },
    phases: { prefill: null, decode: null, encode: null },
    performance: {
      ttftLatencyMs: 200,
      tpotMs: 20,
      requestLatencyMs: 5000,
      concurrency: 32,
    },
    throughput: {
      tokensPerSecond: 800,
      tokensPerSecondPerGpu: 800 / gpusNeeded,
      tokensPerSecondPerUser: 25,
    },
    memory: { value: 50, unit: 'GB', breakdown: null },
    memoryAvailable: true,
    metadata: {
      modelPath: 'org/model',
      system: 'h200_sxm',
      inputTokens: 2048,
      outputTokens: 128,
      targetTtftMs: 1000,
      durationMs: 1,
    },
    warnings: [],
  } as never)

  it('does not launch a Pareto sweep for a Calculate result', async () => {
    vi.mocked(readRecommendStream).mockResolvedValueOnce(completed())
    await render()

    await act(async () => {
      startSizing(SIZING_PARAMS)
      await flush()
    })

    expect(vi.mocked(fetchFrontier)).not.toHaveBeenCalled()
    expect(latest.runMode).toBe('sizing')
    expect(latest.result).not.toBeNull()
  })

  it('runs the sizing recommendation and the sweep together for a Pareto action', async () => {
    vi.mocked(readRecommendStream).mockResolvedValueOnce(completed())
    await render()

    await act(async () => {
      startPareto(SIZING_PARAMS)
      await flush()
    })

    // The sizing half asks the same workload for its recommended config; the
    // sweep asks for the frontier. Both are kept on one run.
    const sizingBody = JSON.parse(vi.mocked(fetch).mock.calls[0][1]?.body as string)
    expect(sizingBody).toMatchObject({ objective: 'min_gpus', simulator: 'aisim' })
    expect(sizingBody).not.toHaveProperty('mode')

    const frontier = vi.mocked(fetchFrontier)
    expect(frontier).toHaveBeenCalledTimes(1)
    expect(frontier.mock.calls[0][0]).toMatchObject({
      objective: 'pareto',
      simulator: 'aisim',
    })

    expect(latest.runMode).toBe('pareto')
    expect(latest.result).not.toBeNull()
    expect(latest.frontier.status).toBe('ready')
    expect(latest.isLoading).toBe(false)
  })

  it('collates the recommended configuration into the frontier it belongs on', async () => {
    vi.mocked(readRecommendStream).mockResolvedValueOnce(completed())
    vi.mocked(fetchFrontier).mockResolvedValueOnce({
      candidates: [{
        candidate_id: 'candidate-000001',
        status: 'feasible',
        reason_category: null,
        mode: 'agg',
        topology: { deployment_mode: 'agg', tp: 4, pp: 1, attention_dp: 1, replicas: 1 },
        used_gpus: 4,
        metrics: { mean_ttft_ms: 180 },
        objectives: { throughput_per_user: 22, throughput_per_gpu: 200 },
        transfer_delay_modeled: false,
        provenance_sources: ['sdk'],
        simulator: 'aisim',
      }],
      views: { pareto_front: ['candidate-000001'], top_n: ['candidate-000001'] },
      sweep: null,
      debug: null,
    })
    await render()

    await act(async () => {
      startPareto(SIZING_PARAMS)
      await flush()
    })

    if (latest.frontier.status !== 'ready') throw new Error('expected a ready frontier')
    expect(latest.frontier.candidates).toHaveLength(1)
    expect(latest.frontier.candidates[0].candidate_id).toBe('candidate-000001')
    expect(latest.frontier.candidates[0].recommended).toBe(true)
  })

  it('keeps the sweep replay record with the frontier it describes', async () => {
    const debug = {
      endpoint: '/recommend',
      simulator: 'aisimulate',
      execution: 'in_process',
      created_at: 1_730_000_000,
      versions: {},
      selection: null,
      notes: [],
      invocations: [{
        id: 'candidate-1',
        label: 'Candidate c1',
        command: 'aisimulate predict -c c1.yaml',
        execution: 'in_process',
        group: null,
        candidate_id: 'candidate-000001',
        notes: [],
        files: [],
      }],
      script: null,
    }
    vi.mocked(fetchFrontier).mockResolvedValueOnce({
      candidates: [],
      views: null,
      sweep: null,
      debug,
    })
    await render()

    await act(async () => {
      startPareto(SIZING_PARAMS)
      await flush()
    })

    if (latest.frontier.status !== 'ready') throw new Error('expected a ready frontier')
    expect(latest.frontier.replay).toEqual(debug)
  })

  it('sends one Pareto sweep request with no GPU-window bounds', async () => {
    await render()

    await act(async () => {
      startPareto(SIZING_PARAMS)
      await flush()
    })

    const frontier = vi.mocked(fetchFrontier)
    expect(frontier).toHaveBeenCalledTimes(1)
    const [request] = frontier.mock.calls[0]
    expect(request).toMatchObject({
      simulator: 'aisim',
      objective: 'pareto',
      sla_mode: 'enforce',
      ttft: 1000,
    })
    expect(request).not.toHaveProperty('min_candidate_gpus')
    expect(request).not.toHaveProperty('max_candidate_gpus')
    expect(request).not.toHaveProperty('target_concurrency')
    expect(request).not.toHaveProperty('minGpus')
    expect(request).not.toHaveProperty('maxGpus')
    expect(latest.runMode).toBe('pareto')
    expect(latest.frontier.status).toBe('ready')
  })

  it('drops load targets and latency bounds from a no-SLA Pareto request', async () => {
    await render()

    await act(async () => {
      startPareto({
        ...SIZING_PARAMS,
        sla_mode: 'none',
        ttft: undefined,
        target_request_rate: 9,
      })
      await flush()
    })

    const [request] = vi.mocked(fetchFrontier).mock.calls[0]
    expect(request).toMatchObject({ objective: 'pareto', sla_mode: 'none' })
    expect(request).not.toHaveProperty('target_request_rate')
    expect(request).not.toHaveProperty('target_concurrency')
    expect(request).not.toHaveProperty('ttft')
    expect(request).not.toHaveProperty('tpot')
    expect(request).not.toHaveProperty('request_latency')
  })

  it('keeps the caller-selected simulator instead of substituting AISim', async () => {
    await render()

    await act(async () => {
      startPareto({ ...SIZING_PARAMS, simulator: 'blis' })
      await flush()
    })

    const [request] = vi.mocked(fetchFrontier).mock.calls[0]
    expect(request.simulator).toBe('blis')
  })

  it('clears a Pareto frontier when Calculate starts', async () => {
    vi.mocked(readRecommendStream)
      .mockResolvedValueOnce(completed())
      .mockResolvedValueOnce(completed())
    await render()

    await act(async () => {
      startPareto(SIZING_PARAMS)
      await flush()
    })
    expect(latest.frontier.status).toBe('ready')

    await act(async () => {
      startSizing(SIZING_PARAMS)
      await flush()
    })

    expect(latest.runMode).toBe('sizing')
    expect(latest.frontier.status).toBe('idle')
  })

  it('does not carry a previous sizing result into the Pareto recommendation', async () => {
    vi.mocked(readRecommendStream).mockResolvedValueOnce(completed(4))
    await render()

    await act(async () => {
      startSizing(SIZING_PARAMS)
      await flush()
    })
    expect(latest.result).not.toBeNull()

    let resolveStream!: (value: unknown) => void
    vi.mocked(readRecommendStream).mockReturnValueOnce(new Promise(resolve => {
      resolveStream = resolve
    }) as never)

    await act(async () => {
      startPareto(SIZING_PARAMS)
      await flush()
    })
    // The old result is discarded; the Pareto run answers for itself.
    expect(latest.runMode).toBe('pareto')
    expect(latest.result).toBeNull()

    await act(async () => {
      resolveStream(completed(8))
      await flush()
    })
    expect(latest.result?.recommendation.gpusNeeded).toBe(8)
  })

  it('keeps the sizing result when only the sweep fails', async () => {
    vi.mocked(readRecommendStream).mockResolvedValueOnce(completed())
    vi.mocked(fetchFrontier).mockRejectedValueOnce(new Error('sweep timed out'))
    await render()

    await act(async () => {
      startPareto(SIZING_PARAMS)
      await flush()
    })

    expect(latest.frontier).toEqual({ status: 'error', message: 'sweep timed out' })
    expect(latest.error).toBeNull()
    expect(latest.result).not.toBeNull()
  })

  it('keeps the frontier when only the sizing recommendation fails', async () => {
    await render()

    await act(async () => {
      startPareto(SIZING_PARAMS)
      await flush()
    })

    expect(latest.frontier.status).toBe('ready')
    expect(latest.result).toBeNull()
    expect(latest.error).toBe('test response')
  })

  it('discards a primary completion aborted by reset', async () => {
    let resolveStream!: (value: unknown) => void
    vi.mocked(readRecommendStream).mockReturnValueOnce(new Promise(resolve => {
      resolveStream = resolve
    }) as never)
    await render()

    await act(async () => {
      startSizing(SIZING_PARAMS)
      await flush(1)
    })
    await act(async () => { reset() })
    await act(async () => {
      resolveStream({ status: 'completed', recommendation: { gpusNeeded: 4 } })
      await flush()
    })

    expect(latest.runMode).toBeNull()
    expect(latest.result).toBeNull()
    expect(vi.mocked(fetchFrontier)).not.toHaveBeenCalled()
  })
})

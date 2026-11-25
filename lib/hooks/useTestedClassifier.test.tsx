// @vitest-environment happy-dom
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { CanonicalClassifierInput, ClassifierPrediction } from '@/lib/tested-models/types'
import { useTestedClassifier } from './useTestedClassifier'

function input(concurrency: number): CanonicalClassifierInput {
  return {
    effective: {
      model_id: 'org/model',
      system_id: 'h200_sxm',
      backend: 'vllm',
      backend_version: '0.24.0',
      isl: 1024,
      osl: 512,
      max_seq_len: null,
      prefill_max_seq_len: null,
      decode_max_seq_len: null,
      concurrency,
      tp_size: 1,
      pp_size: 1,
      dp_size: 1,
      cp_size: 1,
      moe_tp_size: null,
      moe_ep_size: null,
      shared_prefix_tokens: 0,
      prefix_caching_enabled: null,
      weight_precision: null,
      kv_cache_precision: null,
      moe_quant_mode: null,
      gpu_memory_utilization: null,
      max_num_seqs: null,
      chunked_prefill_enabled: null,
      serving_mode: 'agg',
      replicas: 1,
      prefill: null,
      decode: null,
      encode: null,
    },
    request: {
      request_kind: 'predict',
      target_request_rate: null,
      target_concurrency: null,
      target_latency_ms: null,
      target_ttft_ms: null,
      target_tpot_ms: null,
      database_mode: null,
      top_n: null,
    },
    model_config: null,
  }
}

function prediction(confidence: number): ClassifierPrediction {
  return {
    status: 'within_validated_range',
    confidence,
    pair_id: 'org_model__h200',
    classifier: 'tested-models',
    reasons: [],
  }
}

interface Pending {
  body: string
  respond: (value: ClassifierPrediction) => void
  fail: (error: Error) => void
}

let pending: Pending[]
let root: ReturnType<typeof createRoot>
let host: HTMLDivElement

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  pending = []
  vi.stubGlobal('fetch', vi.fn((_url: string, init?: RequestInit) => new Promise((resolve, reject) => {
    pending.push({
      body: String(init?.body ?? ''),
      respond: value => resolve({ json: async () => value } as Response),
      fail: reject,
    })
  })))
  host = document.createElement('div')
  document.body.append(host)
  root = createRoot(host)
})

afterEach(async () => {
  await act(async () => root.unmount())
  host.remove()
  vi.unstubAllGlobals()
})

function Harness({ value }: { value: CanonicalClassifierInput | null }) {
  const state = useTestedClassifier(value)
  return <output data-testid="state">{JSON.stringify(state)}</output>
}

function state(): ReturnType<typeof useTestedClassifier> {
  return JSON.parse(host.querySelector('[data-testid="state"]')?.textContent ?? 'null')
}

async function render(value: CanonicalClassifierInput | null) {
  await act(async () => root.render(<Harness value={value} />))
}

describe('useTestedClassifier', () => {
  it('reports loading until the response for the current input arrives', async () => {
    await render(input(8))

    expect(state().prediction).toBeNull()
    expect(state().isLoading).toBe(true)
    expect(pending).toHaveLength(1)

    await act(async () => pending[0].respond(prediction(0.9)))

    expect(state().prediction).toEqual(prediction(0.9))
    expect(state().isLoading).toBe(false)
  })

  it('hides a previous response as soon as the input changes', async () => {
    await render(input(8))
    await act(async () => pending[0].respond(prediction(0.9)))
    expect(state().prediction?.confidence).toBe(0.9)

    await render(input(16))

    // The render after the selection change must not show a verdict for the
    // previous configuration while the new request is in flight.
    expect(state().prediction).toBeNull()
    expect(state().isLoading).toBe(true)
    expect(pending).toHaveLength(2)
    expect(pending[1].body).toContain('"concurrency":16')
  })

  it('ignores a response for an input that is no longer selected', async () => {
    await render(input(8))
    await render(input(16))

    await act(async () => pending[0].respond(prediction(0.1)))
    expect(state().prediction).toBeNull()

    await act(async () => pending[1].respond(prediction(0.8)))
    expect(state().prediction?.confidence).toBe(0.8)
  })

  it('reports classification as unavailable when the request fails', async () => {
    await render(input(8))
    await act(async () => pending[0].fail(new Error('offline')))

    expect(state().prediction?.status).toBe('validation_unavailable')
    expect(state().prediction?.confidence).toBeNull()
    expect(state().isLoading).toBe(false)
  })

  it('clears the prediction when the input becomes null', async () => {
    await render(input(8))
    await act(async () => pending[0].respond(prediction(0.9)))

    await render(null)

    expect(state().prediction).toBeNull()
    expect(state().isLoading).toBe(false)
  })
})

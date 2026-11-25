// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { CanonicalClassifierInput, ClassifierPrediction } from '@/lib/tested-models/types'
import { TestedPerformanceFeedback } from './TestedPerformanceFeedback'

vi.mock('next/link', () => ({
  default: ({
    href,
    children,
    ...rest
  }: {
    href: string
    children?: React.ReactNode
  } & Record<string, unknown>) => <a href={href} {...rest}>{children}</a>,
}))

const input: CanonicalClassifierInput = {
  effective: {
    model_id: 'google/gemma-4-31b-it',
    system_id: 'h200_sxm',
    backend: 'vllm',
    backend_version: '0.24.0',
    isl: 2048,
    osl: 512,
    max_seq_len: null,
    prefill_max_seq_len: null,
    decode_max_seq_len: null,
    concurrency: 32,
    tp_size: 4,
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

const prediction: ClassifierPrediction = {
  status: 'within_validated_range',
  confidence: 0.87,
  pair_id: 'google_gemma_4_31b_it__h200',
  classifier: 'tested-models',
  reasons: [],
  dashboard_href: '/tested-models/google_gemma_4_31b_it__h200',
}

let root: ReturnType<typeof createRoot>
let host: HTMLDivElement

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  host = document.createElement('div')
  document.body.append(host)
  root = createRoot(host)
})

afterEach(async () => {
  await act(async () => root.unmount())
  host.remove()
})

async function mount(element: React.ReactElement) {
  await act(async () => root.render(element))
}

function linkUrl(): URL {
  const link = host.querySelector('a')
  expect(link).not.toBeNull()
  return new URL(link!.getAttribute('href') ?? '', 'http://localhost')
}

describe('TestedPerformanceFeedback measured-envelope link', () => {
  it('carries the classifier configuration and the predicted throughput as a snapshot', async () => {
    await mount(
      <TestedPerformanceFeedback
        prediction={prediction}
        predictionInput={input}
        predictionThroughput={1234.5}
      />,
    )

    const url = linkUrl()
    expect(url.pathname).toBe('/tested-models/google_gemma_4_31b_it__h200')
    expect(url.searchParams.get('v')).toBe('1')
    // The chart selection is left to the destination; only the prediction's
    // coordinates are encoded, so an unmeasured ISL/OSL cannot open empty.
    expect(url.searchParams.get('isl')).toBeNull()
    expect(url.searchParams.get('osl')).toBeNull()
    expect(url.searchParams.get('turns')).toBeNull()
    expect(url.searchParams.get('tp')).toBe('4')
    expect(url.searchParams.get('predIsl')).toBe('2048')
    expect(url.searchParams.get('predOsl')).toBe('512')
    expect(url.searchParams.get('predConcurrency')).toBe('32')
    expect(url.searchParams.get('predTp')).toBe('4')
    expect(url.searchParams.get('predThroughput')).toBe('1234.5')
    expect(url.searchParams.get('predReplicas')).toBe('1')
    expect(url.searchParams.get('predStatus')).toBe('within_validated_range')
    expect(url.searchParams.get('predConfidence')).toBe('0.87')
    expect(url.searchParams.get('predBackend')).toBe('vllm')
    expect(url.searchParams.get('predBackendVersion')).toBe('0.24.0')
    expect(url.searchParams.get('predServingMode')).toBe('agg')
    expect(host.textContent).toContain('View your prediction on the measured envelope')
  })

  it('falls back to the plain envelope link without an aggregate throughput', async () => {
    await mount(
      <TestedPerformanceFeedback
        prediction={prediction}
        predictionInput={input}
        predictionThroughput={null}
      />,
    )

    const url = linkUrl()
    expect(url.pathname).toBe('/tested-models/google_gemma_4_31b_it__h200')
    expect(url.searchParams.get('predThroughput')).toBeNull()
    expect(url.searchParams.get('predStatus')).toBeNull()
    expect(host.textContent).toContain('View the measured performance envelope')
  })

  it('does not invent a pair link when the classifier measured nothing', async () => {
    await mount(
      <TestedPerformanceFeedback
        prediction={{ ...prediction, status: 'no_measurements', pair_id: null, dashboard_href: null }}
        modelId="google/gemma-4-31b-it"
        systemId="h200_sxm"
      />,
    )

    const url = linkUrl()
    expect(url.pathname).toBe('/tested-models')
    expect(url.searchParams.get('model')).toBe('google/gemma-4-31b-it')
    expect(url.searchParams.get('system')).toBe('h200_sxm')
    expect(host.textContent).toContain('View tested coverage')
  })
})

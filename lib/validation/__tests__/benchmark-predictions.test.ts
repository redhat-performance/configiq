import { describe, expect, it } from 'vitest'
import {
  measuredPoolTopology,
  perPoolBatch,
  predictionFabric,
} from '../benchmark-predictions'
import type { DatasetConfiguration } from '../benchmark-types'

function configuration(overrides: Partial<DatasetConfiguration> = {}): DatasetConfiguration {
  return {
    measurement_source: null,
    benchmark_version: null,
    backend: 'vllm',
    backend_version: 'v0.31.0',
    precision: null,
    weight_quantization: null,
    kv_quantization: null,
    prefix_caching: null,
    prefix_tokens: null,
    prefix_count: null,
    max_seq_len: null,
    max_num_seqs: null,
    gpu_memory_utilization: null,
    chunked_prefill_enabled: null,
    spec_decoding: null,
    request_type: null,
    turns: null,
    tp_prefill: null,
    tp_decode: null,
    inferencex_topology: null,
    inferencex_curve: null,
    inferencex_offload: null,
    ...overrides,
  }
}

describe('measuredPoolTopology', () => {
  it('uses the decode width for both pools when the run is aggregated', () => {
    const topology = measuredPoolTopology(configuration(), 8)

    expect(topology).toEqual({
      prefillTp: 8,
      decodeTp: 8,
      prefillWorkers: 1,
      decodeWorkers: 1,
      known: false,
    })
  })

  it('reads per-pool widths and worker counts from a disaggregated run', () => {
    const topology = measuredPoolTopology(configuration({
      tp_prefill: '4',
      tp_decode: '8',
      inferencex_topology: 'disaggregated · multi-node · P TP4/EP2 · D TP8 · 12 GPUs · workers P2/D3',
    }), 8)

    expect(topology).toEqual({
      prefillTp: 4,
      decodeTp: 8,
      prefillWorkers: 2,
      decodeWorkers: 3,
      known: true,
    })
  })

  it('withholds comparability when a disaggregated run does not name workers', () => {
    const topology = measuredPoolTopology(configuration({
      tp_prefill: '4',
      tp_decode: '8',
      inferencex_topology: 'disaggregated · multi-node · P TP4 · D TP8',
    }), 8)

    expect(topology.known).toBe(false)
  })
})

describe('predictionFabric', () => {
  it('forwards a fabric only to a disaggregated simulator that consumes one', () => {
    expect(predictionFabric('disagg', true, 'roce-200g')).toBe('roce-200g')
    expect(predictionFabric('disagg', false, 'roce-200g')).toBeUndefined()
    expect(predictionFabric('agg', true, 'roce-200g')).toBeUndefined()
  })
})

describe('perPoolBatch', () => {
  it('divides measured concurrency across pool workers', () => {
    expect(perPoolBatch(25, 2)).toBe(13)
    expect(perPoolBatch(1, 4)).toBe(1)
  })
})

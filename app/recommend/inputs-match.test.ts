import { describe, expect, it } from 'vitest'
import {
  resolveEffectiveFabric,
  submittedInputsMatch,
  type CurrentRecommendInputs,
  type SubmittedRecommendParams,
} from './inputs-match'

const baseParams: SubmittedRecommendParams = {
  model_path: 'Qwen/Qwen3-32B',
  system: 'h200',
  isl: 2048,
  osl: 256,
  sla_mode: 'enforce',
  objective: 'min_gpus',
  ttft: 500,
  tpot: 30,
  target_concurrency: 32,
  prefix: 0,
}

const baseInputs: CurrentRecommendInputs = {
  model: 'Qwen/Qwen3-32B',
  system: 'h200',
  isl: 2048,
  osl: 256,
  objective: 'min_gpus',
  ttft: 500,
  tpot: 30,
  targetConcurrency: 32,
  demandRate: 5,
  requestLatency: null,
  prefix: 0,
  maxSeqLen: null,
  prefillMaxSeqLen: null,
  decodeMaxSeqLen: null,
  effectiveFabric: undefined,
}

describe('submittedInputsMatch', () => {
  it('matches identical decomposed latency inputs', () => {
    expect(submittedInputsMatch(baseParams, baseInputs)).toBe(true)
  })

  it('ignores TTFT and TPOT when the request used an end-to-end bound', () => {
    const params: SubmittedRecommendParams = {
      ...baseParams,
      ttft: undefined,
      tpot: undefined,
      request_latency: 2000,
    }
    const inputs: CurrentRecommendInputs = {
      ...baseInputs,
      ttft: 120,
      tpot: 12,
      requestLatency: 2000,
    }
    expect(submittedInputsMatch(params, inputs)).toBe(true)
  })

  it('marks an end-to-end bound dirty when the visible bound changes', () => {
    const params: SubmittedRecommendParams = {
      ...baseParams,
      ttft: undefined,
      tpot: undefined,
      request_latency: 2000,
    }
    expect(submittedInputsMatch(params, { ...baseInputs, requestLatency: 2500 })).toBe(false)
  })

  it('marks the form dirty when an end-to-end bound replaces TTFT/TPOT', () => {
    const params: SubmittedRecommendParams = {
      ...baseParams,
      ttft: undefined,
      tpot: undefined,
      request_latency: 2000,
    }
    const inputs: CurrentRecommendInputs = {
      ...baseInputs,
      requestLatency: null,
    }
    expect(submittedInputsMatch(params, inputs)).toBe(false)
  })

  it('marks decomposed latency changes dirty', () => {
    expect(submittedInputsMatch(baseParams, { ...baseInputs, ttft: 600 })).toBe(false)
    expect(submittedInputsMatch(baseParams, { ...baseInputs, tpot: 40 })).toBe(false)
  })

  it('defaults omitted TTFT/TPOT and concurrency the same way the request does', () => {
    const params: SubmittedRecommendParams = {
      ...baseParams,
      ttft: 500,
      tpot: undefined,
      target_concurrency: undefined,
    }
    expect(submittedInputsMatch(params, baseInputs)).toBe(true)
  })

  it('normalizes omitted prefix and sequence caps to their form defaults', () => {
    const params: SubmittedRecommendParams = {
      ...baseParams,
      prefix: undefined,
      max_seq_len: undefined,
      prefill_max_seq_len: undefined,
      decode_max_seq_len: undefined,
    }
    expect(submittedInputsMatch(params, baseInputs)).toBe(true)
    expect(submittedInputsMatch(params, { ...baseInputs, maxSeqLen: 8192 })).toBe(false)
  })

  it('compares only the arrival rate for a demand request', () => {
    const params: SubmittedRecommendParams = {
      ...baseParams,
      sla_mode: 'none',
      objective: 'throughput_demand',
      ttft: undefined,
      tpot: undefined,
      target_concurrency: undefined,
      target_request_rate: 5,
    }
    const inputs: CurrentRecommendInputs = {
      ...baseInputs,
      objective: 'throughput_demand',
      ttft: 900,
      tpot: 90,
      demandRate: 5,
    }
    expect(submittedInputsMatch(params, inputs)).toBe(true)
    expect(submittedInputsMatch(params, { ...inputs, demandRate: 6 })).toBe(false)
  })

  it('compares only the concurrency for a capacity request', () => {
    const params: SubmittedRecommendParams = {
      ...baseParams,
      sla_mode: 'none',
      objective: 'concurrency_capacity',
      ttft: undefined,
      tpot: undefined,
      target_concurrency: 64,
    }
    const inputs: CurrentRecommendInputs = {
      ...baseInputs,
      objective: 'concurrency_capacity',
      ttft: 4000,
      tpot: 100,
      targetConcurrency: 64,
    }
    expect(submittedInputsMatch(params, inputs)).toBe(true)
    expect(submittedInputsMatch(params, { ...inputs, targetConcurrency: 80 })).toBe(false)
  })

  it('falls back to the sla_mode objective when none was sent', () => {
    const params: SubmittedRecommendParams = {
      ...baseParams,
      objective: undefined,
      sla_mode: 'none',
      target_request_rate: 5,
    }
    const inputs: CurrentRecommendInputs = {
      ...baseInputs,
      objective: 'throughput_demand',
      demandRate: 5,
    }
    expect(submittedInputsMatch(params, inputs)).toBe(true)
  })

  it('marks a different objective dirty', () => {
    expect(submittedInputsMatch(baseParams, { ...baseInputs, objective: 'throughput_demand' })).toBe(false)
  })

  it('matches an aggregate-only request that omitted the fabric', () => {
    const params: SubmittedRecommendParams = { ...baseParams, fabric: undefined }
    const inputs: CurrentRecommendInputs = {
      ...baseInputs,
      effectiveFabric: resolveEffectiveFabric(true, true, true, 'roce-200g'),
    }
    expect(inputs.effectiveFabric).toBeUndefined()
    expect(submittedInputsMatch(params, inputs)).toBe(true)
  })

  it('marks a fabric change dirty when the request carried one', () => {
    const params: SubmittedRecommendParams = { ...baseParams, fabric: 'roce-200g' }
    expect(submittedInputsMatch(params, { ...baseInputs, effectiveFabric: 'roce-200g' })).toBe(true)
    expect(submittedInputsMatch(params, { ...baseInputs, effectiveFabric: null })).toBe(false)
  })
})

describe('submittedInputsMatch for Pareto sweeps', () => {
  const paretoParams: SubmittedRecommendParams = {
    ...baseParams,
    sla_mode: 'none',
    pareto: true,
    ttft: undefined,
    tpot: undefined,
    target_concurrency: undefined,
    target_request_rate: undefined,
  }

  it('ignores load targets for a no-SLA sweep', () => {
    const inputs: CurrentRecommendInputs = {
      ...baseInputs,
      targetConcurrency: 1,
      demandRate: 99,
    }
    expect(submittedInputsMatch(paretoParams, inputs)).toBe(true)
  })

  it('ignores latency bounds for a no-SLA sweep', () => {
    const inputs: CurrentRecommendInputs = {
      ...baseInputs,
      ttft: 900,
      tpot: 90,
    }
    expect(submittedInputsMatch(paretoParams, inputs)).toBe(true)
  })

  it('still compares the workload shape', () => {
    expect(submittedInputsMatch(paretoParams, { ...baseInputs, isl: 4096 })).toBe(false)
    expect(submittedInputsMatch(paretoParams, { ...baseInputs, model: 'Qwen/Qwen3-8B' })).toBe(false)
    expect(submittedInputsMatch(paretoParams, { ...baseInputs, system: 'h100' })).toBe(false)
  })

  it('compares TTFT and TPOT when the sweep enforced the SLA', () => {
    const params: SubmittedRecommendParams = {
      ...paretoParams,
      sla_mode: 'enforce',
      ttft: 500,
    }
    expect(submittedInputsMatch(params, baseInputs)).toBe(true)
    expect(submittedInputsMatch(params, { ...baseInputs, ttft: 600 })).toBe(false)
    expect(submittedInputsMatch(params, { ...baseInputs, tpot: 40 })).toBe(false)
  })

  it('compares the end-to-end bound when the sweep sent one', () => {
    const params: SubmittedRecommendParams = {
      ...paretoParams,
      sla_mode: 'enforce',
      ttft: undefined,
      tpot: undefined,
      request_latency: 2000,
    }
    const inputs: CurrentRecommendInputs = {
      ...baseInputs,
      ttft: 120,
      tpot: 12,
      requestLatency: 2000,
    }
    expect(submittedInputsMatch(params, inputs)).toBe(true)
    expect(submittedInputsMatch(params, { ...inputs, requestLatency: 2500 })).toBe(false)
  })

  it('marks a switch between decomposed and end-to-end bounds dirty', () => {
    const params: SubmittedRecommendParams = {
      ...paretoParams,
      sla_mode: 'enforce',
      ttft: undefined,
      tpot: undefined,
      request_latency: 2000,
    }
    expect(submittedInputsMatch(params, baseInputs)).toBe(false)
  })

  it('does not require a Pareto flag to compare sizing requests', () => {
    expect(baseParams.pareto).toBeUndefined()
    expect(submittedInputsMatch(baseParams, { ...baseInputs, ttft: 600 })).toBe(false)
  })
})

describe('resolveEffectiveFabric', () => {
  it('omits the fabric when the backend does not require one', () => {
    expect(resolveEffectiveFabric(false, false, true, 'roce-200g')).toBeUndefined()
  })

  it('omits the fabric for aggregate-only goals', () => {
    expect(resolveEffectiveFabric(true, true, true, 'roce-200g')).toBeUndefined()
  })

  it('keeps a selected fabric when disaggregation supports it', () => {
    expect(resolveEffectiveFabric(true, false, true, 'roce-200g')).toBe('roce-200g')
  })

  it('sends null when no fabric was selected or disaggregation needs none', () => {
    expect(resolveEffectiveFabric(true, false, true, null)).toBeNull()
    expect(resolveEffectiveFabric(true, false, false, 'roce-200g')).toBeNull()
  })
})

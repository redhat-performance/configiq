import { describe, expect, it } from 'vitest'

import type { RecommendResult } from '../api/recommend'
import type { Candidate } from '../types/pareto'
import {
  RECOMMENDED_CANDIDATE_ID,
  candidateForRecommendation,
  collateRecommendation,
  isSizingProjection,
  resolveSelection,
  sweepCandidates,
} from './recommendation'

function result(overrides: Partial<RecommendResult> = {}): RecommendResult {
  return {
    requestId: 'size_test',
    status: 'completed',
    simulator: 'aisimulate',
    mode: 'agg',
    recommendation: {
      gpusNeeded: 4,
      gpusPerReplica: 4,
      totalGpus: 4,
      replicasNeeded: 1,
      tensorParallelSize: 4,
      pipelineParallelSize: 1,
      dataParallelSize: 1,
      contextParallelSize: 1,
      moeTensorParallelSize: null,
      moeExpertParallelSize: null,
      batchSize: 64,
    },
    phases: { prefill: null, decode: null, encode: null },
    performance: {
      ttftLatencyMs: 210,
      tpotMs: 24,
      requestLatencyMs: 12_000,
      concurrency: 64,
    },
    throughput: {
      tokensPerSecond: 900,
      tokensPerSecondPerGpu: 225,
      tokensPerSecondPerUser: 14,
    },
    memory: { value: 58, unit: 'GB', breakdown: null },
    memoryAvailable: true,
    engineVersion: '0.14.0',
    precision: { weights: null, kvCache: null },
    metadata: {
      modelPath: 'org/model',
      system: 'h200_sxm',
      inputTokens: 2048,
      outputTokens: 128,
      targetTtftMs: 1000,
      durationMs: 12,
    },
    warnings: [],
    ...overrides,
  }
}

function candidate(overrides: Partial<Candidate> = {}): Candidate {
  return {
    candidate_id: 'candidate-000001',
    status: 'feasible',
    reason_category: null,
    mode: 'agg',
    topology: {
      deployment_mode: 'agg',
      tp: 4,
      pp: 1,
      attention_dp: 1,
      replicas: 1,
    },
    used_gpus: 4,
    metrics: { mean_ttft_ms: 200, mean_tpot_ms: 23 },
    objectives: { throughput_per_user: 12, throughput_per_gpu: 210 },
    transfer_delay_modeled: false,
    provenance_sources: ['sdk'],
    simulator: 'aisimulate',
    ...overrides,
  }
}

describe('candidateForRecommendation', () => {
  it('projects the reported deployment and measurements into the sweep shape', () => {
    const recommended = candidateForRecommendation(result())

    expect(recommended.candidate_id).toBe(RECOMMENDED_CANDIDATE_ID)
    expect(recommended.status).toBe('feasible')
    expect(recommended.mode).toBe('agg')
    expect(recommended.used_gpus).toBe(4)
    expect(recommended.topology).toMatchObject({
      deployment_mode: 'agg',
      tp: 4,
      pp: 1,
      attention_dp: 1,
      replicas: 1,
    })
    expect(recommended.objectives).toEqual({
      throughput_per_user: 14,
      throughput_per_gpu: 225,
    })
    expect(recommended.metrics).toMatchObject({
      ttft_ms: 210,
      tpot_ms: 24,
      request_latency_ms: 12_000,
    })
  })

  it('does not plot a point when the service reported no throughput', () => {
    const recommended = candidateForRecommendation(result({
      throughput: { tokensPerSecond: 900, tokensPerSecondPerGpu: 0, tokensPerSecondPerUser: 0 },
    }))

    expect(recommended.objectives).toBeNull()
  })

  it('maps a disaggregated recommendation onto the role topology', () => {
    const recommended = candidateForRecommendation(result({
      mode: 'disagg',
      recommendation: {
        ...result().recommendation,
        gpusNeeded: 12,
        replicasNeeded: 2,
      },
      phases: {
        prefill: {
          workers: 2,
          gpusPerWorker: 2,
          tensorParallelSize: 2,
          pipelineParallelSize: 1,
          dataParallelSize: 1,
          contextParallelSize: 1,
          moeTensorParallelSize: null,
          moeExpertParallelSize: null,
          batchSize: 32,
          memoryGb: 40,
        },
        decode: {
          workers: 4,
          gpusPerWorker: 2,
          tensorParallelSize: 2,
          pipelineParallelSize: 1,
          dataParallelSize: 1,
          contextParallelSize: 1,
          moeTensorParallelSize: null,
          moeExpertParallelSize: null,
          batchSize: 128,
          memoryGb: 52,
        },
        encode: null,
      },
    }))

    expect(recommended.mode).toBe('disagg')
    expect(recommended.topology).toMatchObject({
      deployment_mode: 'disagg',
      prefill_tp: 2,
      prefill_replicas: 2,
      decode_tp: 2,
      decode_replicas: 4,
    })
  })
})

describe('isSizingProjection', () => {
  it('recognizes the projected row and the matched sweep row', () => {
    expect(isSizingProjection(candidateForRecommendation(result()))).toBe(true)

    const matched = collateRecommendation([candidate()], result())[0]
    expect(isSizingProjection(matched)).toBe(true)
  })

  it('does not treat an evaluated sweep row as a sizing projection', () => {
    expect(isSizingProjection(candidate())).toBe(false)
  })
})

describe('collateRecommendation', () => {
  it('leaves the sweep untouched when the sizing half produced no result', () => {
    const candidates = [candidate()]

    expect(collateRecommendation(candidates, null)).toBe(candidates)
    expect(collateRecommendation(candidates, undefined)).toBe(candidates)
  })

  it('marks an equivalent sweep candidate instead of listing it twice', () => {
    const candidates = [candidate()]

    const collated = collateRecommendation(candidates, result())

    expect(collated).toHaveLength(1)
    expect(collated[0].candidate_id).toBe('candidate-000001')
    expect(collated[0].recommended).toBe(true)
    // The row is the sizing recommendation, so it shows the sizing numbers
    // rather than the sweep's measurements for the same deployment.
    expect(collated[0].metrics).toEqual({
      ttft_ms: 210,
      tpot_ms: 24,
      request_latency_ms: 12_000,
    })
    expect(collated[0].objectives).toEqual({
      throughput_per_user: 14,
      throughput_per_gpu: 225,
    })
    // The matched row keeps the sweep id, so provenance is what marks it as a
    // sizing projection rather than an evaluated candidate.
    expect(isSizingProjection(collated[0])).toBe(true)
  })

  it('matches a sweep row whose expert widths are the same', () => {
    const candidates = [
      candidate({
        topology: {
          deployment_mode: 'agg',
          tp: 4,
          pp: 1,
          attention_dp: 1,
          replicas: 1,
          moe_tp: 1,
          moe_ep: 1,
        },
      }),
    ]

    expect(collateRecommendation(candidates, result())).toHaveLength(1)
  })

  it('does not match a deployment that differs only in expert parallelism', () => {
    const candidates = [
      candidate({
        topology: {
          deployment_mode: 'agg',
          tp: 4,
          pp: 1,
          attention_dp: 1,
          replicas: 1,
          moe_tp: 2,
          moe_ep: 2,
        },
      }),
    ]

    const collated = collateRecommendation(candidates, result())

    expect(collated).toHaveLength(2)
    expect(collated.filter(entry => entry.recommended)).toHaveLength(1)
  })

  it('adds the recommendation when the sweep evaluated no equivalent deployment', () => {
    const candidates = [candidate({ candidate_id: 'candidate-000002', used_gpus: 8 })]

    const collated = collateRecommendation(candidates, result())

    expect(collated).toHaveLength(2)
    const recommended = collated.find(entry => entry.recommended)
    expect(recommended?.candidate_id).toBe(RECOMMENDED_CANDIDATE_ID)
    expect(recommended?.used_gpus).toBe(4)
  })

  it('does not match a different serving mode or GPU count', () => {
    const candidates = [
      candidate({ mode: 'disagg', topology: { deployment_mode: 'disagg' } }),
      candidate({ candidate_id: 'candidate-000003', used_gpus: 8 }),
    ]

    const collated = collateRecommendation(candidates, result())

    expect(collated).toHaveLength(3)
    expect(collated.filter(entry => entry.recommended)).toHaveLength(1)
  })

  it('adds the recommendation when the equivalent sweep row is not feasible', () => {
    const candidates = [candidate({ status: 'timed_out' })]

    const collated = collateRecommendation(candidates, result())

    expect(collated).toHaveLength(2)
    const recommended = collated.find(entry => entry.recommended)
    expect(recommended?.candidate_id).toBe(RECOMMENDED_CANDIDATE_ID)
    expect(recommended?.status).toBe('feasible')
  })
})

describe('resolveSelection', () => {
  const first = candidate({ candidate_id: 'first' })
  const second = candidate({ candidate_id: 'second' })
  const failedRow = candidate({ candidate_id: 'failed', status: 'timed_out' })
  const recommended = candidate({
    candidate_id: RECOMMENDED_CANDIDATE_ID,
    recommended: true,
  })
  const frontier = new Set(['first', 'second'])

  it('keeps a row the user selected when the recommendation arrives', () => {
    expect(
      resolveSelection('first', [first, second, recommended], [first, second, recommended], frontier, true),
    ).toBe('first')
  })

  it('replaces an automatic selection with a late recommendation', () => {
    expect(
      resolveSelection('first', [first, second, recommended], [first, second, recommended], frontier, false),
    ).toBe(RECOMMENDED_CANDIDATE_ID)
  })

  it('keeps an automatic selection when there is no recommendation', () => {
    expect(resolveSelection('first', [first, second], [first, second], frontier, false)).toBe('first')
  })

  it('keeps a failed row the user selected from the disclosure', () => {
    expect(
      resolveSelection('failed', [first, failedRow, recommended], [first, recommended], frontier, true),
    ).toBe('failed')
  })

  it('re-selects when the selected row disappeared', () => {
    expect(resolveSelection('gone', [recommended], [recommended], frontier, true)).toBe(
      RECOMMENDED_CANDIDATE_ID,
    )
  })

  it('falls back to a frontier point, then to any feasible row', () => {
    expect(resolveSelection(null, [first, second], [first, second], new Set(['second']), false)).toBe('second')
    expect(resolveSelection(null, [second], [second], new Set(), false)).toBe('second')
    expect(resolveSelection(null, [], [], new Set(), false)).toBeNull()
  })
})

describe('sweepCandidates', () => {
  it('drops a standalone sizing row from the sweep coverage input', () => {
    const collated = collateRecommendation(
      [
        candidate({ candidate_id: 'candidate-000001', used_gpus: 8 }),
        candidate({ candidate_id: 'candidate-000002', used_gpus: 16 }),
      ],
      result(),
    )

    expect(collated).toHaveLength(3)
    expect(sweepCandidates(collated).map(entry => entry.candidate_id)).toEqual([
      'candidate-000001',
      'candidate-000002',
    ])
  })

  it('keeps a matched row because the sweep evaluated it', () => {
    const collated = collateRecommendation([candidate()], result())

    expect(collated).toHaveLength(1)
    expect(sweepCandidates(collated)).toHaveLength(1)
    expect(sweepCandidates(collated)[0].candidate_id).toBe('candidate-000001')
  })
})

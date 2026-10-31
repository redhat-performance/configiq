import { describe, it, expect } from 'vitest'
import { computeFrontier, pickRecommendation } from './frontier'
import type { Candidate } from '../types/pareto'

function makeCandidate(overrides: Partial<Candidate> = {}): Candidate {
  return {
    candidate_id: 'candidate-000001',
    status: 'feasible',
    reason_category: null,
    mode: 'agg',
    topology: { deployment_mode: 'agg' },
    used_gpus: 8,
    metrics: {},
    objectives: { throughput_per_user: 20, throughput_per_gpu: 40 },
    transfer_delay_modeled: false,
    provenance_sources: [],
    simulator: 'aisimulate',
    cost_per_m_tokens: null,
    ...overrides,
  }
}

function makeFeasible(overrides: Partial<Candidate> = {}): Candidate {
  return makeCandidate({ status: 'feasible', ...overrides })
}

function makeInfeasible(overrides: Partial<Candidate> = {}): Candidate {
  return makeCandidate({ status: 'infeasible', objectives: null, ...overrides })
}

describe('computeFrontier', () => {
  it('returns empty array for empty input', () => {
    expect(computeFrontier([])).toEqual([])
  })

  it('excludes infeasible candidates', () => {
    const candidates = [
      makeFeasible({ candidate_id: 'a' }),
      makeInfeasible({ candidate_id: 'b' }),
    ]
    expect(computeFrontier(candidates).map((c) => c.candidate_id)).toEqual(['a'])
  })

  it('removes dominated points', () => {
    // c1 dominates c2: both objectives strictly higher
    const c1 = makeFeasible({
      candidate_id: 'c1',
      objectives: { throughput_per_user: 100, throughput_per_gpu: 50 },
    })
    const c2 = makeFeasible({
      candidate_id: 'c2',
      objectives: { throughput_per_user: 80, throughput_per_gpu: 40 },
    })
    const frontier = computeFrontier([c1, c2])
    expect(frontier).toHaveLength(1)
    expect(frontier[0]).toBe(c1)
  })

  it('keeps both points when neither dominates the other', () => {
    // trade-off: c1 better on x, c2 better on y
    const c1 = makeFeasible({
      candidate_id: 'c1',
      objectives: { throughput_per_user: 100, throughput_per_gpu: 30 },
    })
    const c2 = makeFeasible({
      candidate_id: 'c2',
      objectives: { throughput_per_user: 50, throughput_per_gpu: 60 },
    })
    expect(computeFrontier([c1, c2])).toHaveLength(2)
  })

  it('returns empty when all candidates are infeasible', () => {
    expect(computeFrontier([makeInfeasible(), makeInfeasible()])).toEqual([])
  })

  it('excludes candidates with null objectives', () => {
    const c = makeFeasible({ objectives: null })
    expect(computeFrontier([c])).toEqual([])
  })
})

describe('pickRecommendation', () => {
  it('returns fewest GPUs clearing SLA', () => {
    const sla = 25
    const f1 = makeFeasible({
      used_gpus: 4,
      objectives: { throughput_per_user: 30, throughput_per_gpu: 60 },
    })
    const f2 = makeFeasible({
      used_gpus: 2,
      objectives: { throughput_per_user: 28, throughput_per_gpu: 40 },
    })
    const f3 = makeFeasible({
      used_gpus: 8,
      objectives: { throughput_per_user: 10, throughput_per_gpu: 20 },
    })
    expect(pickRecommendation([f1, f2, f3], sla)).toBe(f2)
  })

  it('breaks ties by cost ascending', () => {
    const sla = 25
    const f1 = makeFeasible({
      used_gpus: 4,
      cost_per_m_tokens: 2.0,
      objectives: { throughput_per_user: 30, throughput_per_gpu: 50 },
    })
    const f2 = makeFeasible({
      used_gpus: 4,
      cost_per_m_tokens: 1.5,
      objectives: { throughput_per_user: 31, throughput_per_gpu: 55 },
    })
    expect(pickRecommendation([f1, f2], sla)).toBe(f2)
  })

  it('returns null when no candidate clears SLA', () => {
    const c = makeFeasible({ objectives: { throughput_per_user: 5, throughput_per_gpu: 20 } })
    expect(pickRecommendation([c], 25)).toBeNull()
  })

  it('returns null for empty frontier', () => {
    expect(pickRecommendation([], 20)).toBeNull()
  })

  it('treats null used_gpus as Infinity for sorting', () => {
    const f1 = makeFeasible({
      used_gpus: null,
      objectives: { throughput_per_user: 30, throughput_per_gpu: 60 },
    })
    const f2 = makeFeasible({
      used_gpus: 4,
      objectives: { throughput_per_user: 28, throughput_per_gpu: 40 },
    })
    expect(pickRecommendation([f1, f2], 25)).toBe(f2)
  })
})

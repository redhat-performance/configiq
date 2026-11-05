import { describe, it, expect } from 'vitest'
import { computeFrontier } from './frontier'
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

import { describe, it, expect } from 'vitest'
import { computeFrontier, displayedFrontier } from './frontier'
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

describe('displayedFrontier', () => {
  const high = { throughput_per_user: 14, throughput_per_gpu: 225 }
  const low = { throughput_per_user: 12, throughput_per_gpu: 210 }

  it('uses the server view unchanged for sweep objectives', () => {
    const a = makeFeasible({ candidate_id: 'a', objectives: high })
    const b = makeFeasible({ candidate_id: 'b', objectives: low })

    // The server's verdict is kept even though the local computation agrees.
    expect(displayedFrontier([a, b], ['a'])).toEqual(new Set(['a']))
  })

  it('keeps an empty server view', () => {
    expect(displayedFrontier([makeFeasible()], [])).toEqual(new Set())
  })

  it('falls back to the displayed objectives when no view was reported', () => {
    const a = makeFeasible({
      candidate_id: 'a',
      objectives: { throughput_per_user: 100, throughput_per_gpu: 30 },
    })
    const b = makeFeasible({
      candidate_id: 'b',
      objectives: { throughput_per_user: 50, throughput_per_gpu: 60 },
    })

    expect(displayedFrontier([a, b], null)).toEqual(new Set(['a', 'b']))
  })

  it('recomputes once a sizing projection displays objectives the sweep did not measure', () => {
    // The server saw this row with the sweep's lower numbers, so it was not
    // frontier then. The displayed sizing numbers dominate the other row, so
    // the label has to follow what the panel actually shows.
    const matched = makeFeasible({
      candidate_id: 'candidate-000001',
      objectives: high,
      provenance_sources: ['sizing recommendation'],
      recommended: true,
    })
    const other = makeFeasible({ candidate_id: 'candidate-000002', objectives: low })

    expect(displayedFrontier([matched, other], ['candidate-000002'])).toEqual(
      new Set(['candidate-000001']),
    )
  })

  it('leaves the server view alone when the projection has no objectives', () => {
    const projected = makeFeasible({
      candidate_id: 'sizing-recommendation',
      objectives: null,
      provenance_sources: ['sizing recommendation'],
    })
    const other = makeFeasible({ candidate_id: 'candidate-000002' })

    expect(displayedFrontier([other, projected], ['candidate-000002'])).toEqual(
      new Set(['candidate-000002']),
    )
  })
})

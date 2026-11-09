import type { Candidate, FrontierSweep } from '@/lib/types/pareto'

const NON_FEASIBLE_STATUSES = [
  'timed_out',
  'infeasible',
  'failed',
  'unsupported',
  'resource_limited',
] as const

/**
 * Describe how much of the search space a Pareto sweep covered.
 *
 * An enumerated sweep evaluates a known grid, so it can be called exhaustive;
 * a sampled sweep cannot. The caption must not confuse the two, and a spent
 * budget always makes the result non-exhaustive.
 */
export function describeSweepCoverage(
  sweep: Pick<FrontierSweep, 'algorithm' | 'evaluated' | 'budgetExhausted'> | null | undefined,
  candidates: Pick<Candidate, 'status'>[],
): string {
  const evaluated = sweep?.evaluated ?? candidates.length
  const enumerated = sweep?.algorithm === 'enumerate'
  const budgetExhausted = sweep?.budgetExhausted === true
  const feasible = candidates.filter(candidate => candidate.status === 'feasible').length

  const parts = [
    enumerated ? `${evaluated} evaluated` : `${evaluated} sampled`,
    `${feasible} feasible`,
  ]
  for (const status of NON_FEASIBLE_STATUSES) {
    const count = candidates.filter(candidate => candidate.status === status).length
    if (count > 0) parts.push(`${count} ${status.replace('_', ' ')}`)
  }

  const coverage = enumerated
    ? budgetExhausted
      ? 'the trial budget stopped the enumeration before every grid point ran.'
      : 'exhaustive over the enumerated grid.'
    : budgetExhausted
      ? 'not exhaustive: the trial budget was reached.'
      : 'not exhaustive.'

  return `${parts.join(' · ')} — ${coverage}`
}

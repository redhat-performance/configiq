import type { Candidate } from '../types/pareto'

/**
 * Returns the non-dominated subset of feasible candidates.
 * Both objectives are maximized: throughput_per_user (x) and throughput_per_gpu (y).
 */
export function computeFrontier(candidates: Candidate[]): Candidate[] {
  const feasible = candidates.filter(
    (c) => c.status === 'feasible' && c.objectives !== null
  )
  if (feasible.length === 0) return []

  return feasible.filter((a) => {
    const ax = a.objectives!.throughput_per_user
    const ay = a.objectives!.throughput_per_gpu
    // a is dominated if some other candidate b is >= in both and > in at least one
    return !feasible.some((b) => {
      if (b === a) return false
      const bx = b.objectives!.throughput_per_user
      const by = b.objectives!.throughput_per_gpu
      return bx >= ax && by >= ay && (bx > ax || by > ay)
    })
  })
}

/**
 * Picks the recommended candidate from a frontier:
 * - Must clear slaTokPerSec (throughput_per_user >= sla)
 * - Fewest GPUs first; ties broken by cost_per_m_tokens ascending (nulls last)
 * Returns null if no candidate clears the SLA.
 */
export function pickRecommendation(
  frontier: Candidate[],
  slaTokPerSec: number
): Candidate | null {
  const clearing = frontier.filter(
    (c) => c.objectives !== null && c.objectives.throughput_per_user >= slaTokPerSec
  )
  if (clearing.length === 0) return null

  return clearing.sort((a, b) => {
    const gpuDiff = (a.used_gpus ?? Infinity) - (b.used_gpus ?? Infinity)
    if (gpuDiff !== 0) return gpuDiff
    const ac = a.cost_per_m_tokens ?? Infinity
    const bc = b.cost_per_m_tokens ?? Infinity
    return ac - bc
  })[0]
}

import type { Candidate } from '../types/pareto'

/**
 * Returns the non-dominated subset of feasible candidates.
 * Both objectives are maximized: throughput_per_user (x) and throughput_per_gpu (y).
 *
 * This is a client-side fallback for when the simulator does not return a
 * `pareto_front` view. Prefer the simulator's view when it exists.
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

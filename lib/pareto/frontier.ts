import type { Candidate } from '../types/pareto'
import { isSizingProjection } from './recommendation'

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

/**
 * The default-axis frontier membership to display.
 *
 * The server's view is computed from the sweep's measurements, so it is used
 * unchanged while every plotted objective came from that sweep. Collation gives
 * a row the objectives the sizing response reported instead, which the sweep
 * never measured; keeping the server's verdict would then label a point using
 * numbers the chart no longer shows. Once a sizing projection carries
 * objectives, membership is recomputed over the displayed values so the chart,
 * table, and labels agree.
 *
 * `serverParetoFront` is the server's answer after it evaluated the sweep: an
 * empty set means nothing was non-dominated and is kept, while `null` means no
 * view was reported and the local computation is the fallback.
 */
export function displayedFrontier(
  candidates: Candidate[],
  serverParetoFront: readonly string[] | null,
): Set<string> {
  const hasProjectedObjectives = candidates.some(
    (candidate) => isSizingProjection(candidate) && candidate.objectives !== null,
  )
  if (serverParetoFront && !hasProjectedObjectives) {
    return new Set(serverParetoFront)
  }
  return new Set(computeFrontier(candidates).map((candidate) => candidate.candidate_id))
}

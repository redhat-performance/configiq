import type { Candidate, CandidateViews, FrontierSweep } from '../types/pareto'
import { formatApiError, parseFrontierBody } from '../pareto/parse'

export interface FrontierResponse {
  candidates: Candidate[]
  views: CandidateViews | null
  sweep: FrontierSweep | null
}

/**
 * Request the Pareto candidate ledger through the same-origin /api/recommend
 * proxy. The request carries the same workload shape as the sizing request and
 * only differs in `objective`.
 *
 * This is always AISimulate-only. BLIS does not provide a frontier, and the
 * service rejects the combination rather than substituting another simulator.
 */
export async function fetchFrontier(
  request: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<FrontierResponse> {
  const slaMode = request.sla_mode === 'none' ? 'none' : 'enforce'
  const payload: Record<string, unknown> = {
    ...request,
    simulator: 'aisimulate',
    objective: 'pareto',
    sla_mode: slaMode,
  }
  // A sweep drives its own load; the primary request's target is not a bound.
  delete payload.target_request_rate
  delete payload.target_concurrency
  if (slaMode === 'none') {
    delete payload.ttft
    delete payload.tpot
    delete payload.request_latency
  } else if (payload.request_latency != null) {
    delete payload.ttft
    delete payload.tpot
  }

  const res = await fetch('/api/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(payload),
    signal,
  })

  const body: unknown = await res.json().catch(() => null)

  if (!res.ok) {
    throw new Error(formatApiError(body, `Pareto sweep failed (${res.status})`))
  }

  const parsed = parseFrontierBody(body)
  if (parsed === null) {
    throw new Error('The Pareto response did not identify AISimulate as the simulator.')
  }
  return parsed
}

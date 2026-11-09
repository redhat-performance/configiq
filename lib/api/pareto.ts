import type { Candidate, CandidateViews, FrontierSweep } from '../types/pareto'
import { formatApiError, parseFrontierBody } from '../pareto/parse'

export interface FrontierResponse {
  candidates: Candidate[]
  views: CandidateViews | null
  sweep: FrontierSweep | null
}

/** A failed Pareto request, carrying the status and body for the Debug panel. */
export class FrontierRequestError extends Error {
  readonly status: number
  readonly body: unknown

  constructor(message: string, status: number, body: unknown) {
    super(message)
    this.name = 'FrontierRequestError'
    this.status = status
    this.body = body
  }
}

/**
 * Build the exact payload sent by the standalone Pareto action.
 *
 * A sweep drives its own load, so the sizing request's target rate and
 * concurrency are not bounds and must not be sent. The caller's simulator is
 * preserved: Pareto is capability-gated in the UI, and a request for a
 * simulator that cannot provide a frontier must surface the API's explicit
 * rejection rather than be redirected to another simulator.
 */
export function buildFrontierRequest(request: Record<string, unknown>): Record<string, unknown> {
  const slaMode = request.sla_mode === 'none' ? 'none' : 'enforce'
  const payload: Record<string, unknown> = {
    ...request,
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
  return payload
}

/**
 * Request the Pareto candidate ledger through the same-origin /api/recommend
 * proxy. The request carries the same workload shape as the sizing request and
 * only differs in `objective`.
 *
 * Whether a simulator can sweep a frontier is its own capability. The caller
 * keeps the selected simulator so an unsupported request fails explicitly; the
 * response must confirm the same simulator and is never redirected to another.
 */
export async function fetchFrontier(
  request: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<FrontierResponse> {
  const payload = buildFrontierRequest(request)

  const res = await fetch('/api/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(payload),
    signal,
  })

  const body: unknown = await res.json().catch(() => null)

  if (!res.ok) {
    throw new FrontierRequestError(
      formatApiError(body, `Pareto sweep failed (${res.status})`),
      res.status,
      body,
    )
  }

  const parsed = parseFrontierBody(body)
  if (parsed === null) {
    throw new Error('The Pareto response did not identify the simulator that produced it.')
  }
  const requested = typeof payload.simulator === 'string' ? payload.simulator : null
  if (requested != null && parsed.simulator !== requested) {
    throw new Error(
      `Requested ${requested}, but the API confirmed ${parsed.simulator}. No fallback was accepted.`,
    )
  }
  return parsed
}

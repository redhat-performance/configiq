import type { Candidate, CandidateViews, ParetoRequest } from '../types/pareto'

export interface FrontierResponse {
  candidates: Candidate[]
  views: CandidateViews | null
}

/**
 * Runs a bounded aisimulate Pareto sweep via the /api/design proxy and returns
 * the full candidate ledger plus the SDK-computed views. One request, no stream.
 */
export async function fetchFrontier(
  request: ParetoRequest,
  signal?: AbortSignal
): Promise<FrontierResponse> {
  const res = await fetch('/api/design', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  })

  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail?.error ?? detail?.detail ?? `sweep failed (${res.status})`)
  }

  const data = await res.json()
  const candidates: Candidate[] = (data.candidates ?? []).map((c: Candidate) => ({
    ...c,
    cost_per_m_tokens: null,
  }))
  return { candidates, views: data.views ?? null }
}

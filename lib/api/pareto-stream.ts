import type { Candidate, ParetoRequest, ParetoStreamDoneEvent } from '../types/pareto'

/**
 * Opens an SSE connection to /api/design/stream, calling onCandidate for each
 * candidate event and onDone when the sweep finishes.
 *
 * Returns a cleanup function that closes the connection.
 */
export function streamCandidates(
  request: ParetoRequest,
  onCandidate: (c: Candidate) => void,
  onDone: (views: ParetoStreamDoneEvent['views']) => void,
  onError?: (err: Error) => void
): () => void {
  const params = new URLSearchParams({
    model_path: request.model_path,
    system: request.system,
    isl: String(request.isl),
    osl: String(request.osl),
    ttft: String(request.ttft),
    tpot: String(request.tpot),
    backend: request.backend,
    kv_transfer_bandwidth_gb_per_second: String(request.kv_transfer_bandwidth_gb_per_second),
    database_mode: request.database_mode,
    ...(request.min_candidate_gpus != null ? { min_candidate_gpus: String(request.min_candidate_gpus) } : {}),
    ...(request.max_candidate_gpus != null ? { max_candidate_gpus: String(request.max_candidate_gpus) } : {}),
  })

  const eventSource = new EventSource(`/api/design/stream?${params}`)

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.done === true) {
        onDone(data.views)
        eventSource.close()
      } else {
        onCandidate({ ...data, cost_per_m_tokens: null } as Candidate)
      }
    } catch {
      // malformed event — skip
    }
  }

  eventSource.onerror = (event) => {
    eventSource.close()
    onError?.(new Error('SSE connection error'))
  }

  return () => eventSource.close()
}

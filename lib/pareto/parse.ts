import type {
  Candidate,
  CandidateObjectives,
  CandidateStatus,
  CandidateTopology,
  CandidateViews,
  FrontierSweep,
} from '../types/pareto'

const CANDIDATE_STATUSES: ReadonlySet<string> = new Set([
  'feasible',
  'infeasible',
  'unsupported',
  'timed_out',
  'failed',
  'resource_limited',
])

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function parseObjectives(value: unknown): CandidateObjectives | null {
  if (!isRecord(value)) return null
  if (
    typeof value.throughput_per_user !== 'number' ||
    typeof value.throughput_per_gpu !== 'number'
  ) {
    return null
  }
  const objectives: CandidateObjectives = {
    throughput_per_user: value.throughput_per_user,
    throughput_per_gpu: value.throughput_per_gpu,
  }
  for (const [key, raw] of Object.entries(value)) {
    if (key !== 'throughput_per_user' && key !== 'throughput_per_gpu' && typeof raw === 'number') {
      objectives[key] = raw
    }
  }
  return objectives
}

function parseMetrics(value: unknown): Record<string, number | null> {
  if (!isRecord(value)) return {}
  const metrics: Record<string, number | null> = {}
  for (const [key, raw] of Object.entries(value)) {
    if (typeof raw === 'number' || raw === null) metrics[key] = raw
  }
  return metrics
}

/**
 * Parse one ledger record without inventing values. A record whose status is
 * unknown is dropped rather than relabelled: the UI must not present a made-up
 * state as simulator output.
 */
export function parseCandidate(value: unknown): Candidate | null {
  if (!isRecord(value)) return null
  const status = value.status
  if (typeof status !== 'string' || !CANDIDATE_STATUSES.has(status)) return null
  if (typeof value.candidate_id !== 'string' || value.candidate_id.length === 0) return null

  const topology: CandidateTopology = isRecord(value.topology)
    ? (value.topology as CandidateTopology)
    : {}
  const mode = typeof value.mode === 'string' && value.mode.length > 0
    ? value.mode
    : typeof topology.deployment_mode === 'string'
      ? topology.deployment_mode
      : 'agg'

  return {
    candidate_id: value.candidate_id,
    status: status as CandidateStatus,
    reason_category: typeof value.reason_category === 'string' ? value.reason_category : null,
    mode,
    topology,
    used_gpus: typeof value.used_gpus === 'number' ? value.used_gpus : null,
    metrics: parseMetrics(value.metrics),
    objectives: parseObjectives(value.objectives),
    transfer_delay_modeled: value.transfer_delay_modeled === true,
    provenance_sources: Array.isArray(value.provenance_sources)
      ? value.provenance_sources.filter((source): source is string => typeof source === 'string')
      : [],
    simulator: 'aisimulate',
  }
}

export function parseViews(value: unknown): CandidateViews | null {
  if (!isRecord(value)) return null
  const ids = (raw: unknown): string[] =>
    Array.isArray(raw) ? raw.filter((id): id is string => typeof id === 'string') : []
  return {
    pareto_front: ids(value.pareto_front),
    top_n: ids(value.top_n),
  }
}

export function parseSweep(value: unknown): FrontierSweep | null {
  if (!isRecord(value)) return null
  const number = (raw: unknown): number | null => (typeof raw === 'number' ? raw : null)
  return {
    maxTrials: number(value.max_trials),
    evaluated: number(value.evaluated),
    budgetExhausted: value.budget_exhausted === true,
    strictSla: typeof value.strict_sla === 'boolean' ? value.strict_sla : null,
    algorithm: typeof value.algorithm === 'string' ? value.algorithm : null,
    seed: number(value.seed),
    windowMinGpus: number(value.window_min_gpus),
    windowMaxGpus: number(value.window_max_gpus),
  }
}

/** Parse the candidate ledger, views, and sweep facts from a Pareto response. */
export function parseFrontierBody(
  value: unknown
): { candidates: Candidate[]; views: CandidateViews | null; sweep: FrontierSweep | null } | null {
  if (!isRecord(value) || value.simulator !== 'aisimulate') return null
  return {
    candidates: Array.isArray(value.candidates)
      ? value.candidates
          .map(parseCandidate)
          .filter((candidate): candidate is Candidate => candidate !== null)
      : [],
    views: parseViews(value.views),
    sweep: parseSweep(value.sweep),
  }
}

/**
 * Turn a FastAPI validation `detail` array (or the app's error envelope) into
 * readable text. Without this the UI shows "[object Object]" and hides which
 * field was rejected.
 */
export function formatApiError(
  body: unknown,
  fallback: string,
): string {
  if (isRecord(body)) {
    const error = body.error
    if (isRecord(error)) {
      if (typeof error.message === 'string' && error.message.length > 0) return error.message
      const details = error.details
      if (Array.isArray(details) && details.length > 0) return formatValidationDetails(details, fallback)
    }
    const detail = body.detail
    if (typeof detail === 'string' && detail.length > 0) return detail
    if (Array.isArray(detail) && detail.length > 0) return formatValidationDetails(detail, fallback)
  }
  return fallback
}

function formatValidationDetails(details: unknown[], fallback: string): string {
  const parts = details.map((entry) => {
    if (!isRecord(entry)) return null
    const message = typeof entry.msg === 'string'
      ? entry.msg
      : typeof entry.message === 'string'
        ? entry.message
        : null
    if (message === null) return null
    const location = Array.isArray(entry.loc)
      ? entry.loc.filter((part): part is string | number =>
          typeof part === 'string' || typeof part === 'number').join('.')
      : ''
    return location ? `${location}: ${message}` : message
  }).filter((part): part is string => part !== null)
  return parts.length > 0 ? parts.join('; ') : fallback
}

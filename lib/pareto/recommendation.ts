import type { RecommendResult } from '../api/recommend'
import type { Candidate, CandidateTopology } from '../types/pareto'

/**
 * Identifier of the row built from a sizing recommendation when the sweep
 * evaluated no equivalent candidate. Service identifiers are shaped like
 * `candidate-000001` or `agg-tp1-c1`, so this marker cannot collide with one.
 */
export const RECOMMENDED_CANDIDATE_ID = 'sizing-recommendation'

/** Provenance marker on a row whose numbers came from the sizing result. */
const SIZING_PROVENANCE = 'sizing recommendation'

/**
 * True when a row describes the sizing recommendation rather than a sweep
 * measurement. A matched row keeps the sweep's candidate id, so provenance is
 * the only reliable signal: the id of a projected row is not.
 */
export function isSizingProjection(candidate: Candidate): boolean {
  return candidate.provenance_sources.includes(SIZING_PROVENANCE)
}

/**
 * The sweep's own rows, without a standalone sizing projection. A collated
 * recommendation that matched a sweep row keeps that row's id and stays in the
 * list; only the added row was never a sweep attempt, so coverage counts must
 * not include it.
 */
export function sweepCandidates(candidates: Candidate[]): Candidate[] {
  return candidates.filter(
    candidate => candidate.candidate_id !== RECOMMENDED_CANDIDATE_ID,
  )
}

function dimension(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

/** Parallelism is absent from some records; an omitted width means one. */
function parallel(value: unknown): number {
  const width = dimension(value)
  return width != null && width > 0 ? width : 1
}

/**
 * The deployment the sizing result reports, in the topology vocabulary the
 * sweep uses. Every value is copied from the result; a phase the service did
 * not report stays null rather than being filled in.
 */
function topologyFor(result: RecommendResult): CandidateTopology {
  if (result.mode === 'disagg') {
    const prefill = result.phases.prefill
    const decode = result.phases.decode
    return {
      deployment_mode: 'disagg',
      prefill_tp: prefill?.tensorParallelSize ?? null,
      prefill_pp: prefill?.pipelineParallelSize ?? null,
      prefill_attention_dp: prefill?.dataParallelSize ?? null,
      prefill_moe_tp: prefill?.moeTensorParallelSize ?? null,
      prefill_moe_ep: prefill?.moeExpertParallelSize ?? null,
      prefill_replicas: prefill?.workers ?? null,
      decode_tp: decode?.tensorParallelSize ?? null,
      decode_pp: decode?.pipelineParallelSize ?? null,
      decode_attention_dp: decode?.dataParallelSize ?? null,
      decode_moe_tp: decode?.moeTensorParallelSize ?? null,
      decode_moe_ep: decode?.moeExpertParallelSize ?? null,
      decode_replicas: decode?.workers ?? null,
    }
  }
  const settings = result.recommendation
  return {
    deployment_mode: 'agg',
    tp: settings.tensorParallelSize,
    pp: settings.pipelineParallelSize,
    attention_dp: settings.dataParallelSize,
    moe_tp: settings.moeTensorParallelSize,
    moe_ep: settings.moeExpertParallelSize,
    replicas: settings.replicasNeeded,
  }
}

/**
 * Express the sizing recommendation as a frontier row.
 *
 * This projects the values the sizing response reported into the sweep's shape.
 * It does not predict anything: throughput, latency, GPU count, and topology
 * all come from the result. Objectives stay null when the service reported no
 * per-user or per-GPU throughput, and the row is then listed without a point
 * rather than plotted at zero.
 */
export function candidateForRecommendation(result: RecommendResult): Candidate {
  const perUser = result.throughput.tokensPerSecondPerUser
  const perGpu = result.throughput.tokensPerSecondPerGpu
  const objectives = perUser > 0 && perGpu > 0
    ? { throughput_per_user: perUser, throughput_per_gpu: perGpu }
    : null
  return {
    candidate_id: RECOMMENDED_CANDIDATE_ID,
    status: 'feasible',
    reason_category: null,
    mode: result.mode,
    topology: topologyFor(result),
    used_gpus: result.recommendation.gpusNeeded,
    metrics: {
      ttft_ms: result.performance.ttftLatencyMs,
      tpot_ms: result.performance.tpotMs,
      request_latency_ms: result.performance.requestLatencyMs,
    },
    objectives,
    // The sizing response does not report whether it modeled a P/D transfer.
    // The detail view says that instead of presenting false as a measurement.
    transfer_delay_modeled: false,
    provenance_sources: [SIZING_PROVENANCE],
    simulator: result.simulator,
  }
}

/**
 * A comparable identity for a deployment: its serving mode, GPU count, and
 * parallelism widths, including the expert widths that distinguish MoE
 * deployments. Two rows with the same key describe the same deployment, so the
 * sizing recommendation is marked on the sweep's row instead of being listed a
 * second time.
 */
function deploymentKey(candidate: Candidate): string {
  const topology = candidate.topology
  if (candidate.mode === 'disagg') {
    return JSON.stringify([
      'disagg',
      candidate.used_gpus ?? null,
      parallel(topology.prefill_tp),
      parallel(topology.prefill_pp),
      parallel(topology.prefill_attention_dp),
      parallel(topology.prefill_replicas),
      parallel(topology.prefill_moe_tp),
      parallel(topology.prefill_moe_ep),
      parallel(topology.decode_tp),
      parallel(topology.decode_pp),
      parallel(topology.decode_attention_dp),
      parallel(topology.decode_replicas),
      parallel(topology.decode_moe_tp),
      parallel(topology.decode_moe_ep),
    ])
  }
  return JSON.stringify([
    'agg',
    candidate.used_gpus ?? null,
    parallel(topology.tp),
    parallel(topology.pp),
    parallel(topology.attention_dp),
    parallel(topology.replicas),
    parallel(topology.moe_tp),
    parallel(topology.moe_ep),
  ])
}

/**
 * Add the sizing recommendation to a sweep's candidate list.
 *
 * When the sweep already evaluated the same deployment, that row is marked and
 * kept once. Otherwise the recommendation is appended so it can be plotted and
 * listed beside the sweep's candidates. A run without a sizing result (the
 * sizing half failed) leaves the sweep untouched.
 */
export function collateRecommendation(
  candidates: Candidate[],
  result: RecommendResult | null | undefined,
): Candidate[] {
  if (result == null) return candidates
  const recommended = candidateForRecommendation(result)
  const key = deploymentKey(recommended)
  // Only a feasible sweep row can stand in for the recommendation. A failed or
  // timed-out row with the same deployment would otherwise hide the
  // recommendation behind the "failed attempts" disclosure.
  const match = candidates.findIndex(
    candidate => candidate.status === 'feasible' && deploymentKey(candidate) === key,
  )
  if (match >= 0) {
    const collated = candidates.slice()
    // The row is shown as the sizing recommendation, so it displays the sizing
    // response's numbers. Keeping the sweep's measurements here would show
    // throughput and latency that the recommendation did not report.
    collated[match] = {
      ...recommended,
      // Keep the sweep row's identity: the server's Pareto view and the row
      // selection refer to this candidate id.
      candidate_id: collated[match].candidate_id,
      status: collated[match].status,
      recommended: true,
    }
    return collated
  }
  return [...candidates, { ...recommended, recommended: true }]
}

/**
 * Which candidate the detail view should show after the candidate list changes.
 *
 * The panel auto-selects the recommendation so the row the user just asked for
 * is inspected first. A recommendation that arrives after the sweep has already
 * picked a row must replace that automatic pick, but it must never replace a
 * row the user selected — including a failed row picked from the disclosure,
 * which no fallback would ever choose.
 */
export function resolveSelection(
  selectedId: string | null,
  candidates: Candidate[],
  feasible: Candidate[],
  frontierIds: ReadonlySet<string>,
  explicit: boolean,
): string | null {
  const recommendedId =
    feasible.find(candidate => candidate.recommended)?.candidate_id ?? null
  const present =
    selectedId != null &&
    candidates.some(candidate => candidate.candidate_id === selectedId)
  if (present && (explicit || recommendedId == null || selectedId === recommendedId)) {
    return selectedId
  }
  return (
    recommendedId ??
    [...frontierIds].find(id =>
      feasible.some(candidate => candidate.candidate_id === id),
    ) ??
    feasible[0]?.candidate_id ??
    null
  )
}

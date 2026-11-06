// Pareto frontier types — mirrors CandidateResponse / CandidateViews from the
// AISimulators REST API. Values are copied from the aisimulate SDK record; the
// browser must not synthesize topology or performance data.

export type CandidateStatus =
  | 'feasible'
  | 'infeasible'
  | 'unsupported'
  | 'timed_out'
  | 'failed'
  | 'resource_limited'

export type SimulatorId = 'aisimulate'

/**
 * Flat topology dict emitted by the aisimulate SDK (CandidateProvenance.topology).
 * Keys are exactly what the SDK produces: agg uses tp, pp, attention_dp, moe_tp,
 * moe_ep; disagg uses the prefill_ and decode_ prefixed keys plus replicas.
 * No nested objects.
 */
export interface CandidateTopology {
  deployment_mode?: string
  // agg
  tp?: number | null
  pp?: number | null
  attention_dp?: number | null
  moe_tp?: number | null
  moe_ep?: number | null
  replicas?: number | null
  strategy?: string | null
  // disagg — prefill role
  prefill_tp?: number | null
  prefill_pp?: number | null
  prefill_attention_dp?: number | null
  prefill_moe_tp?: number | null
  prefill_moe_ep?: number | null
  prefill_replicas?: number | null
  prefill_strategy?: string | null
  // disagg — decode role
  decode_tp?: number | null
  decode_pp?: number | null
  decode_attention_dp?: number | null
  decode_moe_tp?: number | null
  decode_moe_ep?: number | null
  decode_replicas?: number | null
  decode_strategy?: string | null
  [key: string]: unknown
}

/** Objectives from the aisimulate SDK — keyed by target name, both maximized. */
export interface CandidateObjectives {
  throughput_per_user: number    // x-axis: tok/s/user (the SLA axis)
  throughput_per_gpu: number     // y-axis: efficiency axis
  [key: string]: number
}

export interface Candidate {
  candidate_id: string
  status: CandidateStatus
  reason_category: string | null
  mode: string
  topology: CandidateTopology
  used_gpus: number | null
  metrics: Record<string, number | null>
  objectives: CandidateObjectives | null
  /**
   * True only when the response was produced by a sweep that explicitly
   * modeled P/D KV transfer and the candidate has replay metrics. False does
   * not mean the candidate is aggregated; check `mode`.
   */
  transfer_delay_modeled: boolean
  provenance_sources: string[]
  simulator: SimulatorId
}

export interface CandidateViews {
  pareto_front: string[]
  top_n: string[]
}

/** Run-control facts from the sweep that produced a frontier. */
export interface FrontierSweep {
  maxTrials: number | null
  evaluated: number | null
  budgetExhausted: boolean
  strictSla: boolean | null
  algorithm: string | null
  seed: number | null
  windowMinGpus: number | null
  windowMaxGpus: number | null
}

/** State of the Pareto companion sweep that runs alongside a sizing request. */
export type FrontierState =
  | { status: 'idle' }
  | { status: 'loading' }
  | {
      status: 'ready'
      candidates: Candidate[]
      views: CandidateViews | null
      sweep: FrontierSweep | null
    }
  | { status: 'unavailable'; reason: string }
  | { status: 'error'; message: string }

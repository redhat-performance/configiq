// Pareto frontier types — mirrors CandidateResponse / CandidateViews from the FastAPI backend.

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
  tp?: number | null
  pp?: number | null
  attention_dp?: number | null
  moe_tp?: number | null
  moe_ep?: number | null
  prefill_tp?: number | null
  prefill_pp?: number | null
  prefill_replicas?: number | null
  decode_tp?: number | null
  decode_pp?: number | null
  decode_replicas?: number | null
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
  transfer_delay_modeled: boolean
  provenance_sources: string[]
  simulator: SimulatorId
  /** Populated client-side from aicostings; not from the backend. */
  cost_per_m_tokens: number | null
}

export interface CandidateViews {
  pareto_front: string[]
  top_n: string[]
}

export interface ParetoRequest {
  model_path: string
  system: string
  isl: number
  osl: number
  ttft: number
  tpot: number
  backend: string
  kv_transfer_bandwidth_gb_per_second: number
  min_candidate_gpus?: number
  max_candidate_gpus?: number
  database_mode: string
}


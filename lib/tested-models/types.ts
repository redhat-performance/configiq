/**
 * Canonical input contract for tested-model performance classification.
 *
 * Values in the effective configuration section describe the configuration
 * that AISimulators selected or evaluated. Request-only values remain in the
 * request context section so recommendation targets are not confused with
 * measured serving parameters.
 */

export type ClassifierRequestKind = 'recommend' | 'predict'

export type ClassifierServingMode = 'agg' | 'disagg'

export interface ClassifierPhaseInput {
  tp_size: number
  pp_size: number
  dp_size: number
  cp_size: number
  moe_tp_size: number | null
  moe_ep_size: number | null
  workers: number
  batch_size: number | null
}

export interface ClassifierRequestContext {
  request_kind: ClassifierRequestKind
  target_request_rate: number | null
  target_concurrency: number | null
  target_latency_ms: number | null
  target_ttft_ms: number | null
  target_tpot_ms: number | null
  database_mode: 'HYBRID' | 'SILICON' | 'EMPIRICAL' | 'SOL' | null
  top_n: number | null
}

export interface ClassifierEffectiveConfiguration {
  model_id: string
  system_id: string
  backend: string
  backend_version: string | null

  isl: number
  osl: number
  max_seq_len: number | null
  prefill_max_seq_len: number | null
  decode_max_seq_len: number | null

  concurrency: number
  tp_size: number
  pp_size: number
  dp_size: number
  cp_size: number
  moe_tp_size: number | null
  moe_ep_size: number | null

  shared_prefix_tokens: number
  prefix_caching_enabled: boolean | null
  weight_precision: string | null
  kv_cache_precision: string | null
  dtype?: string | null
  gemm_quant_mode?: string | null
  kvcache_quant_mode?: string | null
  moe_quant_mode: string | null
  gpu_memory_utilization: number | null
  max_num_seqs: number | null
  chunked_prefill_enabled: boolean | null

  serving_mode: ClassifierServingMode
  replicas: number | null
  prefill: ClassifierPhaseInput | null
  decode: ClassifierPhaseInput | null
  encode: ClassifierPhaseInput | null
}

export interface CanonicalClassifierInput {
  effective: ClassifierEffectiveConfiguration
  request: ClassifierRequestContext

  /** Optional source metadata used for reproducibility, not necessarily features. */
  model_config: Record<string, unknown> | null
}

export type ClassifierStatus =
  | 'within_validated_range'
  | 'near_validated_boundary'
  | 'outside_validated_range'
  | 'measured_no_envelope'
  | 'no_measurements'
  | 'validation_unavailable'

export interface ClassifierPrediction {
  status: ClassifierStatus
  confidence: number | null
  pair_id: string | null
  classifier: string | null
  reasons: string[]
  /** Canonical link to the measured graph for this pair, when one exists. */
  dashboard_href?: string | null
  /** Canonical link to the measured boundary section. */
  boundary_href?: string | null
}

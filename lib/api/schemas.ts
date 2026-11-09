// API Request/Response Schemas
// Zod schemas for validation

import { z } from 'zod'

// ═══ GPU CATALOG QUERY SCHEMA ═══

export const GpuCatalogQuerySchema = z.object({
  min_memory: z.string().transform(val => val ? parseFloat(val) : undefined).optional(),
  max_price: z.string().transform(val => val ? parseFloat(val) : undefined).optional(),
  vendor: z.string().optional(),
  sort: z.enum(['memory', 'price', 'performance']).default('memory')
})

export type GpuCatalogQuery = z.infer<typeof GpuCatalogQuerySchema>

// ═══ MODEL CATALOG QUERY SCHEMA ═══

export const ModelCatalogQuerySchema = z.object({
  q: z.string().optional(),  // Search query
  vendor: z.string().optional(),
  min_params: z.string().transform(val => val ? parseInt(val, 10) : undefined).optional(),
  max_params: z.string().transform(val => val ? parseInt(val, 10) : undefined).optional(),
  limit: z.string().optional().default('50').transform(val => parseInt(val, 10))
})

export type ModelCatalogQuery = z.infer<typeof ModelCatalogQuerySchema>

// ═══ RECOMMEND REQUEST SCHEMA ═══

export const RecommendRequestSchema = z.object({
  model_path: z.string().min(1, 'model_path is required'),
  system: z.string().min(1, 'system is required'),
  backend: z.string().default('vllm'),
  backend_version: z.string().nullish(),
  isl: z.number().int().positive('isl must be a positive integer'),
  osl: z.number().int().positive('osl must be a positive integer'),
  max_seq_len: z.number().int().positive('max_seq_len must be a positive integer').nullish(),
  prefill_max_seq_len: z.number().int().positive('prefill_max_seq_len must be a positive integer').nullish(),
  decode_max_seq_len: z.number().int().positive('decode_max_seq_len must be a positive integer').nullish(),
  ttft: z.number().positive('ttft must be a positive number (milliseconds)').nullish(),
  tpot: z.number().positive('tpot must be a positive number (milliseconds)').nullish(),
  target_request_rate: z.number().positive().nullish(),
  target_concurrency: z.number().positive().nullish(),
  request_latency: z.number().positive().nullish(),
  prefix: z.number().int().min(0).default(0),
  database_mode: z.enum(['HYBRID', 'SILICON', 'EMPIRICAL', 'SOL']).default('HYBRID'),
  top_n: z.number().int().min(1).max(20).optional(),
  // HF config.json for models AISimulators can't resolve from its catalog (nullish = resolve from HF).
  model_config: z.record(z.string(), z.unknown()).nullish(),
  // An open string discovered from GET /simulators. The schema does not list
  // simulator IDs: a simulator is added or retired by the service, and an
  // unknown value must reach the service for an explicit unsupported response
  // rather than be rejected here against a stale union.
  simulator: z.string().min(1).default('aisimulate'),
  // Whether a simulator consumes a fabric is its own capability; the service
  // rejects one it does not support instead of this schema matching on its ID.
  fabric: z.string().min(1).nullish(),
  /**
   * Internal client hint describing how this proxy should bound the request:
   * `single` also sends a `search_budget_seconds` for a simulator that owns one
   * bounded search, `windowed` relies on the service's own GPU window. Both
   * make one request; the hint is never forwarded upstream as a window.
   */
  search_strategy: z.enum(['windowed', 'single']).default('windowed'),
  /**
   * Optimization objective. `pareto` returns a candidate ledger and the
   * simulator-computed frontier views; `throughput_demand` finds the fewest
   * GPUs that sustain `target_request_rate` with no latency target;
   * `concurrency_capacity` finds the fewest GPUs that hold
   * `target_concurrency` sequences at the per-sequence context length; the
   * normal `min_gpus` path sizes subject to the latency SLA. The selected
   * simulator advertises which objectives it supports; an unsupported request
   * is rejected by the service rather than answered by another simulator.
   */
  objective: z.enum(['min_gpus', 'throughput_demand', 'pareto', 'concurrency_capacity']).default('min_gpus'),
  /**
   * Whether latency targets constrain the result. `enforce` requires a latency
   * bound and is required by `min_gpus`; `none` rejects any latency bound, is
   * required by `throughput_demand` and `concurrency_capacity`, and is optional
   * for `pareto`.
   */
  sla_mode: z.enum(['enforce', 'none']).default('enforce'),
  /**
   * Optional P/D KV transfer bandwidth for a Pareto sweep. Omit to apply
   * AISimulate's default (no modeled P/D transfer delay).
   */
  kv_transfer_bandwidth_gb_per_second: z.number().positive().nullish(),
}).strict().superRefine((data, ctx) => {
  const hasLatency = data.request_latency != null
  const hasTokenLatency = data.ttft != null || data.tpot != null

  // P/D transfer bandwidth is a Pareto sweep input. Accepting it elsewhere
  // would imply a topology the search does not model.
  if (data.objective !== 'pareto' && data.kv_transfer_bandwidth_gb_per_second != null) {
    ctx.addIssue({
      code: 'custom',
      message: "kv_transfer_bandwidth_gb_per_second is only supported with objective='pareto'",
    })
  }

  if (data.objective === 'pareto') {
    // A pareto sweep drives load internally, so it needs no target.
    if (data.target_request_rate != null || data.target_concurrency != null) {
      ctx.addIssue({
        code: 'custom',
        message: 'A pareto sweep does not accept target_request_rate or target_concurrency',
      })
    }
    if (data.sla_mode === 'none' && (hasLatency || hasTokenLatency)) {
      ctx.addIssue({
        code: 'custom',
        message: 'sla_mode=none does not accept ttft, tpot, or request_latency',
      })
    }
    if (data.sla_mode === 'enforce') {
      if (hasLatency && hasTokenLatency) {
        ctx.addIssue({
          code: 'custom',
          message: 'request_latency is mutually exclusive with ttft and tpot',
        })
      }
      if (!hasLatency && data.ttft == null) {
        ctx.addIssue({
          code: 'custom',
          message: 'sla_mode=enforce requires request_latency or a ttft target',
        })
      }
    }
    return
  }

  if (data.objective === 'throughput_demand') {
    if (data.sla_mode !== 'none') {
      ctx.addIssue({ code: 'custom', message: 'throughput_demand requires sla_mode=none' })
    }
    if (data.target_request_rate == null || data.target_concurrency != null) {
      ctx.addIssue({
        code: 'custom',
        message: 'throughput_demand requires target_request_rate and no target_concurrency',
      })
    }
    if (hasLatency || hasTokenLatency) {
      ctx.addIssue({
        code: 'custom',
        message: 'throughput_demand does not accept ttft, tpot, or request_latency',
      })
    }
    return
  }

  if (data.objective === 'concurrency_capacity') {
    // Capacity is a memory-residency question: a whole number of sequences
    // must fit at the per-sequence context length. It has no latency target
    // and no arrival rate, so both are rejected rather than reinterpreted.
    if (data.sla_mode !== 'none') {
      ctx.addIssue({ code: 'custom', message: 'concurrency_capacity requires sla_mode=none' })
    }
    if (data.target_concurrency == null || data.target_request_rate != null) {
      ctx.addIssue({
        code: 'custom',
        message: 'concurrency_capacity requires target_concurrency and no target_request_rate',
      })
    } else if (!Number.isInteger(data.target_concurrency)) {
      ctx.addIssue({
        code: 'custom',
        message: 'concurrency_capacity requires a whole number of concurrent sequences',
      })
    }
    if (hasLatency || hasTokenLatency) {
      ctx.addIssue({
        code: 'custom',
        message: 'concurrency_capacity does not accept ttft, tpot, or request_latency',
      })
    }
    return
  }

  // min_gpus: exactly one load target and an enforced latency SLA.
  if (data.sla_mode !== 'enforce') {
    ctx.addIssue({ code: 'custom', message: 'min_gpus requires sla_mode=enforce' })
  }
  if ((data.target_request_rate != null) === (data.target_concurrency != null)) {
    ctx.addIssue({
      code: 'custom',
      message: 'Exactly one of target_request_rate or target_concurrency must be provided',
    })
  }
  if (hasLatency && hasTokenLatency) {
    ctx.addIssue({ code: 'custom', message: 'request_latency is mutually exclusive with ttft and tpot' })
  }
  if (!hasLatency && data.ttft == null) {
    ctx.addIssue({
      code: 'custom',
      message: 'sla_mode=enforce requires request_latency or a ttft target',
    })
  }
})

/**
 * Caller-facing request shape. Fields with schema defaults (`objective`,
 * `sla_mode`, `database_mode`) stay optional at the call site and are filled
 * by `RecommendRequestSchema.parse` at the API boundary. `top_n` is optional
 * because the service chooses a goal-specific default when it is omitted.
 */
export type RecommendRequest = z.input<typeof RecommendRequestSchema>

// ═══ KV CACHE CALCULATOR REQUEST SCHEMA ═══

export const KvCacheCalcRequestSchema = z.object({
  model_path: z.string().min(1, 'model_path is required'),
  system: z.string().min(1, 'system is required'),
  backend: z.string().default('vllm'),
  backend_version: z.string().nullish(),
  max_num_tokens: z.number().int().positive().default(8192),
  max_batch_size: z.number().int().positive().default(128),
  tp_size: z.number().int().positive().default(1),
  pp_size: z.number().int().positive().default(1),
  moe_tp_size: z.number().int().positive().nullish(),
  moe_ep_size: z.number().int().positive().nullish(),
  memory_fraction_kind: z.enum(['of_total', 'of_free']).default('of_total'),
  memory_fraction_value: z.number().min(0).max(1).default(1.0),
  // HF config.json for models AISimulators can't resolve from its catalog (nullish = resolve from HF).
  model_config: z.record(z.string(), z.unknown()).nullish(),
}).strict()

export type KvCacheCalcRequest = z.infer<typeof KvCacheCalcRequestSchema>

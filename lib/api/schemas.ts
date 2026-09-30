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
  ttft: z.number().positive('ttft must be a positive number (milliseconds)'),
  tpot: z.number().positive('tpot must be a positive number (milliseconds)').default(30),
  target_request_rate: z.number().positive().nullish(),
  target_concurrency: z.number().positive().nullish(),
  request_latency: z.number().positive().nullish(),
  prefix: z.number().int().min(0).default(0),
  database_mode: z.enum(['HYBRID', 'SILICON', 'EMPIRICAL', 'SOL']).default('HYBRID'),
  top_n: z.number().int().min(1).max(20).default(5),
  // HF config.json for models AISimulators can't resolve from its catalog (nullish = resolve from HF).
  model_config: z.record(z.string(), z.unknown()).nullish(),
}).strict().refine(
  data => (data.target_request_rate != null) !== (data.target_concurrency != null),
  { message: 'Exactly one of target_request_rate or target_concurrency must be provided' }
)

export type RecommendRequest = z.infer<typeof RecommendRequestSchema>

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

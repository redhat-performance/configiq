/**
 * The engine limits an answer was evaluated with, as returned by
 * `include=engine`.
 *
 * These are reported facts, not a request echo: a limit the caller left out is
 * still published with whatever the simulation resolved, and a limit nothing
 * could state is published as `not-modeled` rather than omitted. `POST /deploy`
 * renders exactly this section, so the deployment is the measured
 * configuration instead of a re-derivation of it.
 */

/** Why an engine limit holds the value it holds. */
export type EngineSource = 'evaluated' | 'requested' | 'backend-default' | 'not-modeled'

export interface EngineField {
  /** The limit, or null when the run states no value for it. */
  value: boolean | number | string | null
  source: EngineSource
}

/** Limits for one serving role (an aggregated deployment, or one P/D pool). */
export interface EngineRole {
  replicas: EngineField
  tensor_parallel_size: EngineField
  pipeline_parallel_size: EngineField
  data_parallel_size: EngineField
  context_parallel_size: EngineField
  decode_context_parallel_size: EngineField
  moe_tensor_parallel_size: EngineField
  moe_expert_parallel_size: EngineField
  max_model_len: EngineField
  max_num_seqs: EngineField
  max_num_batched_tokens: EngineField
  block_size: EngineField
  gpu_memory_utilization: EngineField
  enable_prefix_caching: EngineField
  enable_chunked_prefill: EngineField
  kv_cache_dtype: EngineField
  quantization: EngineField
}

/** The exact engine limits one answer evaluated. */
export interface EngineSection {
  model: string
  backend: string
  backend_version: string | null
  mode: 'agg' | 'disagg'
  agg: EngineRole | null
  prefill: EngineRole | null
  decode: EngineRole | null
}

const SOURCES = new Set<EngineSource>([
  'evaluated',
  'requested',
  'backend-default',
  'not-modeled',
])

const ROLE_FIELDS = [
  'replicas',
  'tensor_parallel_size',
  'pipeline_parallel_size',
  'data_parallel_size',
  'context_parallel_size',
  'decode_context_parallel_size',
  'moe_tensor_parallel_size',
  'moe_expert_parallel_size',
  'max_model_len',
  'max_num_seqs',
  'max_num_batched_tokens',
  'block_size',
  'gpu_memory_utilization',
  'enable_prefix_caching',
  'enable_chunked_prefill',
  'kv_cache_dtype',
  'quantization',
] as const satisfies readonly (keyof EngineRole)[]

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

/**
 * One reported limit.
 *
 * A missing or unrecognized entry becomes `not-modeled` rather than an invented
 * number: the safe direction, because a deployment that needs the field then
 * refuses instead of rendering a value nobody stated.
 */
function parseEngineField(value: unknown): EngineField {
  if (!isRecord(value) || !SOURCES.has(value.source as EngineSource)) {
    return { value: null, source: 'not-modeled' }
  }
  const raw = value.value
  const valid = raw === null
    || typeof raw === 'boolean'
    || typeof raw === 'number'
    || typeof raw === 'string'
  return valid
    ? { value: raw, source: value.source as EngineSource }
    : { value: null, source: 'not-modeled' }
}

function parseEngineRole(value: unknown): EngineRole | null {
  if (!isRecord(value)) return null
  const role = {} as EngineRole
  for (const field of ROLE_FIELDS) {
    role[field] = parseEngineField(value[field])
  }
  return role
}

/**
 * Validate an engine section from an API response.
 *
 * Returns null when the value is not a section at all, so a caller disables the
 * deploy action instead of forwarding a malformed section to the renderer.
 */
export function parseEngineSection(value: unknown): EngineSection | null {
  if (!isRecord(value)) return null
  const { model, backend, backend_version: backendVersion, mode } = value
  if (typeof model !== 'string' || model.length === 0) return null
  if (typeof backend !== 'string' || backend.length === 0) return null
  if (mode !== 'agg' && mode !== 'disagg') return null
  return {
    model,
    backend,
    backend_version: typeof backendVersion === 'string' && backendVersion.length > 0
      ? backendVersion
      : null,
    mode,
    agg: parseEngineRole(value.agg),
    prefill: parseEngineRole(value.prefill),
    decode: parseEngineRole(value.decode),
  }
}

/**
 * The roles a section reports, in display order.
 *
 * A P/D section has no aggregated role and an aggregated section has no pools;
 * callers render exactly the roles they were given.
 */
export function engineRoles(section: EngineSection): Array<{ role: 'agg' | 'prefill' | 'decode'; label: string; limits: EngineRole }> {
  const rows: Array<{ role: 'agg' | 'prefill' | 'decode'; label: string; limits: EngineRole }> = []
  if (section.agg) rows.push({ role: 'agg', label: 'Aggregated', limits: section.agg })
  if (section.prefill) rows.push({ role: 'prefill', label: 'Prefill', limits: section.prefill })
  if (section.decode) rows.push({ role: 'decode', label: 'Decode', limits: section.decode })
  return rows
}

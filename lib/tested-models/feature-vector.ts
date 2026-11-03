import type {
  CanonicalClassifierInput,
  ClassifierEffectiveConfiguration,
} from './types'
import { canonicalQuantization, type QuantizationField } from './quantization'

export interface FeatureSchema {
  feature_names: string[]
  numeric_medians: Record<string, number>
}

const RUNTIME_FAMILY_LABEL: Record<string, string> = {
  vllm: 'vLLM',
  sglang: 'sglang',
  'trt-llm': 'trt-llm',
  rhaiis: 'rhaiis',
  nim: 'nim',
  aic: 'aic',
  tgi: 'tgi',
  lmdeploy: 'lmdeploy',
}

const RUNTIME_TOKEN_FAMILY: Record<string, string> = {
  vllm: 'vllm',
  sglang: 'sglang',
  'tensorrt-llm': 'trt-llm',
  'trt-llm': 'trt-llm',
  rhaiis: 'rhaiis',
  nim: 'nim',
  aic: 'aic',
  tgi: 'tgi',
  lmdeploy: 'lmdeploy',
}

const RUNTIME_TOKEN_PATTERN = new RegExp(
  `\\b(${Object.keys(RUNTIME_TOKEN_FAMILY)
    .sort((left, right) => right.length - left.length)
    .map(token => token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .join('|')})-(\\d+)(?:\\.(\\d+))?`,
  'gi',
)

const BARE_VERSION_PATTERN = /^\s*(\d+)(?:\.(\d+))?(?:[-._+]\S*)?\s*$/

/** Canonical runtime family for an already-resolved backend name. */
export function backendFamily(value: string | null | undefined): string {
  const text = (value ?? '').trim().toLowerCase()
  return text in RUNTIME_FAMILY_LABEL ? text : 'unknown'
}

/**
 * Only the major.minor family is a feature; patch and suffix detail stays in
 * coverage provenance. Parsing is bounded to avoid regex backtracking.
 */
function versionFamily(major: string, minor?: string): string {
  return minor ? `${major}.${minor}` : major
}

/**
 * Pool a raw version label into a backend-scoped major.minor family.
 *
 * Training rows carry labels such as `vLLM-0.24.0` while the runtime sends the
 * catalog version `0.24.0`. Both must map to the same feature category, and
 * families from different runtimes must never collide.
 */
export function normalizeBackendVersion(
  backend: string | null | undefined,
  version: string | null | undefined,
): string {
  const text = (version ?? '').trim()
  if (!text || ['nan', 'none', 'unknown', 'current'].includes(text.toLowerCase())) return 'unknown'
  const family = backendFamily(backend)
  const matches = [...text.matchAll(RUNTIME_TOKEN_PATTERN)]
  if (matches.length > 0) {
    for (const match of matches) {
      const resolved = RUNTIME_TOKEN_FAMILY[match[1].toLowerCase()]
      if (resolved === family) return `${RUNTIME_FAMILY_LABEL[resolved]}-${versionFamily(match[2], match[3])}`
    }
    const resolved = RUNTIME_TOKEN_FAMILY[matches[0][1].toLowerCase()]
    return `${RUNTIME_FAMILY_LABEL[resolved]}-${versionFamily(matches[0][2], matches[0][3])}`
  }
  const bare = BARE_VERSION_PATTERN.exec(text)
  if (bare && family !== 'unknown') return `${RUNTIME_FAMILY_LABEL[family]}-${versionFamily(bare[1], bare[2])}`
  return 'unknown'
}

function systemFamily(systemId: string): string {
  const normalized = systemId.toLowerCase()
  if (normalized.startsWith('h200')) return 'H200'
  if (normalized.startsWith('b200')) return 'B200'
  if (normalized.startsWith('b300')) return 'B300'
  if (normalized.startsWith('mi300x')) return 'MI300X'
  if (normalized.startsWith('mi355x')) return 'MI355X'
  return systemId
}

function modelConfigNumber(config: Record<string, unknown> | null, keys: string[]): number | null {
  for (const key of keys) {
    const value = config?.[key]
    if (typeof value === 'number' && Number.isFinite(value)) return value
    const nested = config?.text_config
    if (nested && typeof nested === 'object') {
      const nestedValue = (nested as Record<string, unknown>)[key]
      if (typeof nestedValue === 'number' && Number.isFinite(nestedValue)) return nestedValue
    }
  }
  return null
}

function dtypeValue(effective: ClassifierEffectiveConfiguration): string {
  const value = effective.dtype ?? effective.weight_precision
  if (!value) return 'unknown'
  const normalized = value.toLowerCase()
  if (normalized === 'fp16' || normalized === 'float16' || normalized === 'bfloat16' || normalized === 'bf16') return 'float16'
  if (normalized === 'fp8') return 'fp8'
  return value
}

export function buildFeatureValues(input: CanonicalClassifierInput): Record<string, string | number> {
  const effective = input.effective
  const modelConfig = input.model_config
  const tp = effective.tp_size || 1
  const pp = effective.pp_size || 1
  const dp = effective.dp_size || 1
  const ep = effective.moe_ep_size || 1
  const cp = effective.cp_size || 1
  const isl = effective.isl || 0
  const osl = effective.osl || 0
  const concurrency = effective.concurrency || 0
  const sharedPrefix = effective.shared_prefix_tokens || 0
  const unknown = Number.NaN
  const parallelism = tp * pp * dp * ep * cp
  const tokenBudget = (isl + osl) * Math.max(concurrency, 1)

  return {
    model_id: effective.model_id,
    system_id: effective.system_id,
    backend: effective.backend,
    version: normalizeBackendVersion(effective.backend, effective.backend_version),
    accelerator: systemFamily(effective.system_id),
    request_type: 'unknown',
    precision: dtypeValue(effective),
    weight_quantization: canonicalQuantization('weight_quantization', effective.gemm_quant_mode),
    kv_quantization: canonicalQuantization('kv_quantization', effective.kvcache_quant_mode),
    tp,
    pp,
    dp,
    ep,
    cp,
    isl,
    osl,
    concurrency,
    prefix_tokens: sharedPrefix,
    prefix_count: unknown,
    prefix_caching: effective.prefix_caching_enabled == null ? unknown : Number(effective.prefix_caching_enabled),
    shared_prefix: sharedPrefix,
    moe_num_experts: modelConfigNumber(modelConfig, ['num_experts', 'num_local_experts']) ?? unknown,
    moe_top_k: modelConfigNumber(modelConfig, ['num_experts_per_tok', 'num_selected_experts']) ?? unknown,
    token_budget: tokenBudget,
    parallelism,
    memory_pressure: tokenBudget / Math.max(parallelism, 1),
    prefix_pressure: sharedPrefix * Math.max(concurrency, 1),
    sequence_pressure: (isl + osl) / Math.max(parallelism, 1),
  }
}

export function buildFeatureVector(input: CanonicalClassifierInput, schema: FeatureSchema): number[] {
  const values = buildFeatureValues(input)
  const originalQuantization: Record<QuantizationField, string> = {
    weight_quantization: input.effective.gemm_quant_mode ?? '',
    kv_quantization: input.effective.kvcache_quant_mode ?? '',
  }
  const selected = new Map<string, string | null>()
  for (const field of ['weight_quantization', 'kv_quantization'] as const) {
    selected.set(field, selectQuantizationCategory(field, originalQuantization[field], schema.feature_names))
  }
  return schema.feature_names.map((name) => {
    const separator = name.indexOf('=')
    if (separator >= 0) {
      const field = name.slice(0, separator)
      const expected = name.slice(separator + 1)
      const category = selected.get(field)
      if (category !== undefined) return category === expected ? 1 : 0
      return String(values[field] ?? '') === expected ? 1 : 0
    }
    const value = values[name]
    return typeof value === 'number' && Number.isFinite(value)
      ? value
      : schema.numeric_medians[name] ?? 0
  })
}

/**
 * Pick exactly one schema category for a quantization field. Legacy schemas can
 * contain both an alias and its canonical spelling, so an exact match wins and
 * a canonical match is only used when no exact category exists. Returning a
 * single category preserves the one-hot encoding the classifier was trained on.
 */
function selectQuantizationCategory(field: QuantizationField, actual: string, featureNames: string[]): string | null {
  const prefix = `${field}=`
  const categories = featureNames
    .filter(name => name.startsWith(prefix))
    .map(name => name.slice(prefix.length))
  if (categories.length === 0) return null
  if (categories.includes(actual)) return actual
  const canonical = canonicalQuantization(field, actual)
  if (categories.includes(canonical)) return canonical
  return categories.find(category => canonicalQuantization(field, category) === canonical) ?? null
}

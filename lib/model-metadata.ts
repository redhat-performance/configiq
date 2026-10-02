import type { ModelSpec } from '@/lib/hooks/useCatalog'

/**
 * Derived, searchable facts about a model.
 *
 * Two sources feed this:
 *   1. The checkpoint ID string — vendor, family, parameter count, quantization.
 *      Free, synchronous, available for every model including ones the catalog
 *      has never seen.
 *   2. ModelSpec from GET /models?include=specs — expert counts, context length,
 *      architecture. Only present for catalog models.
 *
 * Everything is best-effort: a field is null when neither source can supply it.
 * Nothing here performs I/O, so it is safe to call per row while rendering.
 */

/** Canonical quantization labels, in detection priority order. */
const QUANT_FORMATS = [
  'NVFP4',
  'MXFP4',
  'FP8',
  'FP4',
  'INT8',
  'INT4',
  'AWQ',
  'GPTQ',
  'GGUF',
  'BF16',
  'FP16',
] as const

export type QuantFormat = (typeof QUANT_FORMATS)[number] | string

/** vLLM-style weight/activation precision tokens, e.g. w4a16, w8a8. */
const WEIGHT_ACT_RE = /^w\d+a\d+$/i

/**
 * Trailing tokens that describe *how* a checkpoint was quantized rather than
 * which model it is. Stripped when deriving the family so that Qwen3-32B,
 * Qwen3-32B-FP8 and Qwen3-32B-FP8-Static-PerTensor share one family.
 *
 * Deliberately excludes instruct/chat/base — those are different models, not
 * different precisions of the same model, and must not collapse together.
 */
const PRECISION_NOISE = new Set([
  'dynamic',
  'static',
  'pertensor',
  'perchannel',
  'pertoken',
  'quantized',
  'quant',
  'bnb',
  // Granularity and symmetry qualifiers that trail a format tag, e.g.
  // NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block.
  'block',
  'blockwise',
  'channelwise',
  'groupwise',
  'tensorwise',
  'sym',
  'asym',
])

/**
 * Models whose checkpoint name does not encode a parameter count. Hand-kept;
 * a model missing from here simply reports null parameters rather than a wrong
 * number. Values are total (not active) parameters in billions.
 */
const MODEL_PARAMETER_OVERRIDES: Array<[RegExp, number]> = [
  [/qwen3\.8-2\.4t/i, 2_400],
  [/llama-4-maverick/i, 401.6],
  [/llama-4-scout/i, 108.6],
  [/minimax-m2\.5/i, 228.7],
  [/minimax-m2\.7/i, 228.7],
  [/minimax-m3/i, 427],
  [/mimo-v2-flash/i, 309.8],
  [/^deepseek-r1(?:-\d{4})?(?:-(?:base|chat|instruct|bf16|fp8))?$/i, 684.5],
  [/^deepseek-v3(?:\.1|\.2)?(?:-\d{4})?(?:-(?:base|chat|instruct|bf16|fp8))?$/i, 685.4],
  [/deepseek-v4-flash/i, 290.9],
  [/deepseek-v4-pro/i, 1_598.8],
  [/kimi-k3/i, 2_779.9],
  [/kimi-k2(?:\.[5-7])?/i, 1_026.9],
  [/glm-5(?:\.[1-3])?/i, 753.9],
  [/step-3\.7-flash/i, 201.4],
]

/** Orthogonal type facets — a model can be both vision and MoE. */
export type ModelTypeFacet = 'text' | 'vision' | 'moe'

export interface ModelFacets {
  id: string
  /** Hugging Face org prefix, or '' for a bare checkpoint name. */
  vendor: string
  /** Checkpoint name with trailing quantization/precision tokens removed. */
  family: string
  /** Total parameters in billions, or null when unknown. */
  paramsB: number | null
  /** Active parameters per token for MoE checkpoints (the A22B in 235B-A22B). */
  activeParamsB: number | null
  /** Canonical quantization label, or null for an unquantized checkpoint. */
  quantization: QuantFormat | null
  /** 'text' or 'vision' (mutually exclusive) plus 'moe' when applicable. */
  types: ModelTypeFacet[]
  contextLength: number | null
  isTested: boolean
  inCatalog: boolean
  isHuggingFace: boolean
}

export function normalizeModelId(value: string): string {
  return value.trim().toLowerCase()
}

/** The part after the org prefix, e.g. 'Qwen/Qwen3-32B' → 'Qwen3-32B'. */
export function checkpointName(modelId: string): string {
  return modelId.split('/').pop() ?? modelId
}

export function modelVendor(modelId: string): string {
  const slash = modelId.indexOf('/')
  return slash > 0 ? modelId.slice(0, slash) : ''
}

function tokenize(checkpoint: string): string[] {
  return checkpoint.split(/[-_.]+/).filter(Boolean)
}

/**
 * Whether a trailing token belongs to a precision suffix.
 *
 * A bare format tag (FP8, W8A8) always qualifies. The granularity and symmetry
 * qualifiers only qualify when the checkpoint names a format somewhere, so that
 * an ordinary name ending in a word like "Block" is not mistaken for a
 * quantization suffix and merged into the wrong family.
 */
function isPrecisionToken(token: string, hasFormat: boolean): boolean {
  const upper = token.toUpperCase()
  return (QUANT_FORMATS as readonly string[]).includes(upper) ||
    WEIGHT_ACT_RE.test(token) ||
    (hasFormat && PRECISION_NOISE.has(token.toLowerCase()))
}

/**
 * Canonical quantization label for a checkpoint, or null when it is unquantized.
 *
 * Matching is token-based rather than substring-based so that a version number
 * such as Llama-4 cannot be mistaken for an FP4 precision tag. When a checkpoint
 * names more than one format (GLM-5.2-NVFP4-FP8) the highest-priority match
 * wins, which is the weight precision.
 */
export function modelQuantization(modelId: string): QuantFormat | null {
  const tokens = tokenize(checkpointName(modelId))
  const upper = new Set(tokens.map(t => t.toUpperCase()))
  for (const format of QUANT_FORMATS) {
    if (upper.has(format)) return format
  }
  const weightAct = tokens.find(t => WEIGHT_ACT_RE.test(t))
  return weightAct ? weightAct.toUpperCase() : null
}

/**
 * Checkpoint name with trailing precision tokens stripped, so that every
 * quantized variant of a model resolves to the same family.
 */
export function modelFamily(modelId: string): string {
  let family = checkpointName(modelId)
  const hasFormat = modelQuantization(modelId) !== null
  // Strip one trailing segment at a time so multi-token suffixes such as
  // -FP8-Static-PerTensor collapse fully, but stop at the first real token.
  for (;;) {
    const match = family.match(/^(.*)[-_]([^-_]+)$/)
    if (!match || !isPrecisionToken(match[2], hasFormat)) break
    family = match[1]
  }
  return family
}

export function modelParameterBillions(modelId: string): number | null {
  const checkpoint = checkpointName(modelId)
  const override = MODEL_PARAMETER_OVERRIDES.find(([pattern]) => pattern.test(checkpoint))
  if (override) return override[1]

  const match = checkpoint.match(/(?:^|[-_])(\d+(?:\.\d+)?)([bm])(?:[-_]|$)/i)
  if (!match) return null
  const value = Number(match[1])
  if (!Number.isFinite(value)) return null
  return match[2].toLowerCase() === 'm' ? value / 1_000 : value
}

/**
 * Active parameters per token for MoE checkpoints that advertise them, e.g.
 * Qwen3-235B-A22B → 22. Null for dense models and for MoE checkpoints that
 * do not encode the active count.
 */
export function modelActiveParameterBillions(modelId: string): number | null {
  const match = checkpointName(modelId).match(/(?:^|[-_])a(\d+(?:\.\d+)?)b(?:[-_]|$)/i)
  if (!match) return null
  const value = Number(match[1])
  return Number.isFinite(value) ? value : null
}

export function modelSizeLabel(modelId: string): string {
  const billions = modelParameterBillions(modelId)
  if (billions === null) return 'Parameters unavailable'
  if (billions >= 1_000) {
    return `${Number((billions / 1_000).toFixed(2))}T parameters`
  }
  return `${Number(billions.toFixed(1))}B parameters`
}

export function modelTierLabel(modelId: string): 'Small model' | 'Medium model' | 'Large model' {
  const billions = modelParameterBillions(modelId)
  if (billions !== null && billions <= 12) return 'Small model'
  if (billions !== null && billions <= 50) return 'Medium model'
  return 'Large model'
}

function isMoe(modelId: string, spec: ModelSpec | undefined): boolean {
  if (spec?.num_experts && spec.num_experts > 1) return true
  const checkpoint = checkpointName(modelId)
  // Active-parameter notation (235B-A22B) and expert-count notation (17B-128E)
  // both only appear on MoE checkpoints.
  return /(?:^|[-_])a\d+(?:\.\d+)?b(?:[-_]|$)/i.test(checkpoint) ||
    /(?:^|[-_])\d+e(?:[-_]|$)/i.test(checkpoint)
}

function isVision(modelId: string, spec: ModelSpec | undefined): boolean {
  if (spec?.architecture?.toLowerCase().includes('conditionalgeneration')) return true
  const checkpoint = checkpointName(modelId)
  return /(?:^|[-_])(vl|vision|llava|multimodal)(?:[-_]|$)/i.test(checkpoint)
}

export function modelTypes(modelId: string, spec?: ModelSpec): ModelTypeFacet[] {
  const types: ModelTypeFacet[] = [isVision(modelId, spec) ? 'vision' : 'text']
  if (isMoe(modelId, spec)) types.push('moe')
  return types
}

/**
 * Display label matching the three-way classification the hybrid savings page
 * has always shown. Kept as a single string because it is rendered inline as
 * prose; use modelTypes() when you need the facets separately.
 */
export function modelTypeLabel(spec: ModelSpec | undefined): string {
  if (spec?.num_experts && spec.num_experts > 1) return 'Mixture of experts'
  if (spec?.architecture?.toLowerCase().includes('conditionalgeneration')) return 'Multimodal'
  return 'Dense model'
}

export function isModelListedAsTested(modelId: string, testedModels: string[]): boolean {
  const normalizedModelId = normalizeModelId(modelId)
  return testedModels.some(candidate => normalizeModelId(candidate) === normalizedModelId)
}

export interface ModelFacetSources {
  spec?: ModelSpec
  testedModels?: string[]
  catalogModels?: Set<string> | string[]
  huggingFaceModels?: string[]
}

/** Build the full facet set for one model. Pure; no I/O. */
export function modelFacets(modelId: string, sources: ModelFacetSources = {}): ModelFacets {
  const { spec, testedModels = [], catalogModels = [], huggingFaceModels = [] } = sources
  const catalog = catalogModels instanceof Set ? catalogModels : new Set(catalogModels)
  return {
    id: modelId,
    vendor: modelVendor(modelId),
    family: modelFamily(modelId),
    paramsB: modelParameterBillions(modelId),
    activeParamsB: modelActiveParameterBillions(modelId),
    quantization: modelQuantization(modelId),
    types: modelTypes(modelId, spec),
    contextLength: spec?.context_length ?? null,
    isTested: isModelListedAsTested(modelId, testedModels),
    inCatalog: catalog.has(modelId),
    isHuggingFace: huggingFaceModels.includes(modelId),
  }
}

import type { FrontierModel } from '@/lib/hooks/useCostings'
import type { HostedPrice } from './calc'

export interface HostedPricingResolution {
  selected: FrontierModel | null
  matches: FrontierModel[]
  /** Cheapest exact-checkpoint offer per provider and pricing source, in cost order. */
  providerMatches: FrontierModel[]
}

function normalizeModelId(value: string): string {
  return value.trim().toLowerCase()
}

function slug(value: string): string {
  return value.toLowerCase().replace(/_/g, '-')
}

function stripServingFormat(value: string): string {
  return value
    .replace(/-(?:nvfp4|bf16)$/i, '')
    .replace(/-fp8(?:-static-pertensor)?$/i, '')
}

/**
 * Equivalent hosted identities for an AISimulators checkpoint. Serving
 * format suffixes change the self-managed artifact, not the model family the
 * hosted API bills for. Every non-format alias below is explicit so similarly
 * named distilled or fine-tuned checkpoints are never inferred by suffix.
 */
export function hostedPricingAliases(modelId: string): string[] {
  const normalized = normalizeModelId(modelId)
  const separator = normalized.indexOf('/')
  if (separator < 0) return [normalized]

  const owner = normalized.slice(0, separator)
  const name = normalized.slice(separator + 1)
  const base = stripServingFormat(name)
  const candidates = new Set<string>([normalized, `${owner}/${base}`])
  const add = (candidate: string) => candidates.add(normalizeModelId(candidate))

  if (owner === 'minimaxai') add(`minimax/${base}`)

  if (owner === 'deepseek-ai') {
    add(`deepseek/${base === 'deepseek-v3.1' ? 'deepseek-chat-v3.1' : slug(base)}`)
  }

  if (owner === 'xiaomimimo') {
    const mimoId = slug(base)
    add(`xiaomi/${mimoId}`)
    add(`openrouter/xiaomi/${mimoId}`)
    add(`novita/xiaomimimo/${mimoId}`)
  }

  if (owner === 'qwen') {
    if (slug(base) === 'qwen3-coder-480b-a35b-instruct') {
      add('novita/qwen/qwen3-coder-480b-a35b-instruct')
      add('deepinfra/qwen/qwen3-coder-480b-a35b-instruct')
    }
    if (slug(base) === 'qwen3-vl-32b-thinking') {
      add('dashscope/qwen3-vl-32b-thinking')
    }
  }

  if (owner === 'zai-org') add(`z-ai/${base}`)
  if (owner === 'stepfun-ai') add(`stepfun/${base}`)
  if (owner === 'sgl-project' && base.startsWith('deepseek-')) add(`deepseek/${base}`)

  if (owner === 'meta-llama') {
    // Provider feeds sometimes omit the redundant "Meta-" prefix, but a base
    // model and its Instruct variant are different checkpoints. Only normalize
    // the prefix when the selected checkpoint is already explicitly Instruct.
    if (base.startsWith('meta-llama-3.1-') && base.endsWith('-instruct')) {
      add(`meta-llama/${base.replace(/^meta-/, '')}`)
    }
    if (base.startsWith('llama-4-maverick-')) add('meta-llama/llama-4-maverick')
    if (base.startsWith('llama-4-scout-')) add('meta-llama/llama-4-scout')
  }

  if (owner === 'mistralai' && base === 'mistral-medium-3.5-128b') {
    add('mistralai/mistral-medium-3-5')
  }

  if (owner === 'nvidia') {
    if (base.startsWith('deepseek-')) {
      add(`deepseek/${base === 'deepseek-v3.1' ? 'deepseek-chat-v3.1' : base}`)
    }
    if (base.startsWith('glm-')) add(`z-ai/${base}`)
    if (base.startsWith('gemma-')) add(`google/${base}`)
    if (base.startsWith('kimi-')) add(`moonshotai/${base}`)
    if (base.startsWith('llama-3.1-')) add(`meta-llama/${base}`)
    if (base.startsWith('minimax-')) add(`minimax/${base}`)
    if (base.startsWith('qwen')) add(`qwen/${base}`)

    if (base.startsWith('nvidia-nemotron-')) {
      add(`nvidia/${base.replace(/^nvidia-/, '')}`)
    }
    if (
      slug(base) === 'llama-3-3-nemotron-super-49b-v1' ||
      slug(base) === 'llama-3.3-nemotron-super-49b-v1'
    ) {
      add('nebius/nvidia/llama-3.3-nemotron-super-49b-v1')
    }
  }

  return [...candidates]
}

function finalModelSegment(value: string): string {
  return normalizeModelId(value).split('/').pop() ?? ''
}

function normalizedModelName(value: string): string {
  const normalized = normalizeModelId(value)
    .split(':')
    .pop()!
    .trim()
    .replace(/[\s_]+/g, '-')
  return normalized.split('/').pop() ?? normalized
}

/**
 * Provider feeds use several id layouts for the same checkpoint. Accept an
 * exact canonical id, a provider-prefixed canonical id, or a record whose
 * own model name is the exact requested checkpoint slug. The name check is
 * important: final-segment matching alone incorrectly treated checkpoints
 * such as `deepseek-r1-0528-distill-qwen3-8b` as Qwen3-8B offers.
 */
function isExactCheckpointOffer(aliases: string[], price: FrontierModel): boolean {
  const checkpoint = price.hugging_face_id?.trim()
  // Explicit metadata outranks slugs/display names. Never rescue a conflicting
  // canonical checkpoint with legacy name matching (base, distill, version).
  if (checkpoint) return hostedPricingAliases(checkpoint).some(alias => aliases.includes(alias))

  const candidate = normalizeModelId(price.id)
  const candidateParts = candidate.split('/')
  const candidateName = normalizedModelName(price.name)

  return aliases.some(alias => {
    const aliasParts = alias.split('/')
    const aliasSegment = aliasParts.at(-1) ?? ''
    return candidate === alias ||
      (aliasParts.length >= 2 && candidateParts.slice(-aliasParts.length).join('/') === alias) ||
      (finalModelSegment(candidate) === aliasSegment && candidateName === aliasSegment.replace(/_/g, '-'))
  })
}

function hasUsablePrice(model: FrontierModel): boolean {
  return Number.isFinite(model.price_per_m_input) &&
    model.price_per_m_input >= 0 &&
    Number.isFinite(model.price_per_m_output) &&
    model.price_per_m_output >= 0
}

export interface HostedRequestShape {
  averageInputTokens: number
  averageOutputTokens: number
}

/** Standard synchronous, uncached text tariff. No guessed cache-hit discounts. */
export function hostedPriceForWorkload(
  model: FrontierModel,
  shape?: HostedRequestShape,
): HostedPrice | null {
  if (!hasUsablePrice(model) || model.billing?.unsupported.length) return null
  const price: HostedPrice = {
    modelId: model.id, label: model.name,
    inputPerMillion: model.price_per_m_input,
    outputPerMillion: model.price_per_m_output,
    requestPerRequest: model.billing?.request_price ?? 0,
  }
  if (!shape) return price.requestPerRequest || model.billing?.overrides.length ? null : price
  const { averageInputTokens: input, averageOutputTokens: output } = shape
  if (![input, output].every(value => Number.isSafeInteger(value) && value > 0)) return null
  // Legacy context_window has different source semantics. Prefer explicit
  // limits; LiteLLM's legacy value describes input, not total context.
  const maxInput = model.max_input_tokens ?? (model.source === 'litellm' ? model.context_window : null)
  const maxContext = model.max_context_tokens ?? (model.source !== 'litellm' ? model.context_window : null)
  if ((maxInput != null && input > maxInput) ||
      (model.max_output_tokens != null && output > model.max_output_tokens) ||
      (maxContext != null && input + output > maxContext)) return null
  for (const tier of model.billing?.overrides ?? []) {
    // OpenRouter's threshold is strictly greater than, not greater-or-equal.
    if (input <= tier.min_prompt_tokens) continue
    price.inputPerMillion = tier.input_per_million ?? price.inputPerMillion
    price.outputPerMillion = tier.output_per_million ?? price.outputPerMillion
    price.requestPerRequest = tier.request_price ?? price.requestPerRequest
  }
  return price
}

/** An author namespace is not a serving endpoint identity. */
export function hostedProviderIdentity(model: FrontierModel): string {
  return model.serving_provider?.trim() || (model.source === 'openrouter' ? 'OpenRouter' : model.provider.trim())
}

/** Usage charges, including request fees; excludes fixed fees and operations. */
export function hostedTokenUsageCost(
  model: FrontierModel,
  monthlyInputTokens: number,
  monthlyOutputTokens: number,
  shape?: HostedRequestShape,
): number {
  const inputTokens = Number.isFinite(monthlyInputTokens) ? Math.max(monthlyInputTokens, 0) : 0
  const outputTokens = Number.isFinite(monthlyOutputTokens) ? Math.max(monthlyOutputTokens, 0) : 0
  const price = hostedPriceForWorkload(model, shape)
  if (!price) return Number.POSITIVE_INFINITY
  const requests = shape ? Math.max(inputTokens / shape.averageInputTokens, outputTokens / shape.averageOutputTokens) : 0
  return inputTokens / 1_000_000 * price.inputPerMillion +
    outputTokens / 1_000_000 * price.outputPerMillion + requests * (price.requestPerRequest ?? 0)
}

/**
 * Resolve every hosted offer for the exact checkpoint, then choose the
 * least-cost offer for the workload's actual input/output mix.
 *
 * Provider feeds often prefix the Hugging Face id (for example,
 * `deepinfra/deepseek-ai/DeepSeek-R1`) or expose the checkpoint slug under a
 * provider namespace. Serving-format and known instruction aliases are
 * explicit; distilled or fine-tuned checkpoints are never inferred by suffix.
 */
export function resolveHostedPricing(
  modelId: string,
  prices: FrontierModel[],
  monthlyInputTokens: number,
  monthlyOutputTokens: number,
  shape?: HostedRequestShape,
): HostedPricingResolution {
  const requestedSegment = finalModelSegment(modelId)
  if (!requestedSegment) {
    return { selected: null, matches: [], providerMatches: [] }
  }

  const aliases = hostedPricingAliases(modelId)
  const requestedId = normalizeModelId(modelId)
  const matches = prices.filter(price =>
    hostedPriceForWorkload(price, shape) !== null && isExactCheckpointOffer(aliases, price),
  )

  const ranked = [...matches].sort((left, right) => {
    const costDifference = hostedTokenUsageCost(left, monthlyInputTokens, monthlyOutputTokens, shape) -
      hostedTokenUsageCost(right, monthlyInputTokens, monthlyOutputTokens, shape)
    if (Math.abs(costDifference) > Number.EPSILON) return costDifference

    const exactDifference =
      Number(normalizeModelId(right.id) === requestedId) -
      Number(normalizeModelId(left.id) === requestedId)
    if (exactDifference !== 0) return exactDifference

    return left.id.localeCompare(right.id)
  })
  const seenProviders = new Set<string>()
  const providerMatches = ranked.filter(offer => {
    const provider = `${hostedProviderIdentity(offer).toLowerCase()}::${offer.source ?? ''}`
    if (seenProviders.has(provider)) return false
    seenProviders.add(provider)
    return true
  })
  const selected = providerMatches[0] ?? null

  return {
    selected,
    matches: ranked,
    providerMatches,
  }
}

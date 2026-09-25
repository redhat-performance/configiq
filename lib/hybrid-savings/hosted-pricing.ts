import type { FrontierModel } from '@/lib/hooks/useCostings'

export interface HostedPricingResolution {
  selected: FrontierModel | null
  matches: FrontierModel[]
  /** Cheapest exact-checkpoint offer from each distinct provider, in cost order. */
  providerMatches: FrontierModel[]
  selectedMonthlyUsageCost: number | null
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
function isExactCheckpointOffer(modelId: string, price: FrontierModel): boolean {
  const candidate = normalizeModelId(price.id)
  const candidateParts = candidate.split('/')
  const candidateName = normalizedModelName(price.name)

  return hostedPricingAliases(modelId).some(alias => {
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

function usageCost(
  model: FrontierModel,
  monthlyInputTokens: number,
  monthlyOutputTokens: number,
): number {
  const inputTokens = Number.isFinite(monthlyInputTokens) ? Math.max(monthlyInputTokens, 0) : 0
  const outputTokens = Number.isFinite(monthlyOutputTokens) ? Math.max(monthlyOutputTokens, 0) : 0
  return inputTokens / 1_000_000 * model.price_per_m_input +
    outputTokens / 1_000_000 * model.price_per_m_output
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
): HostedPricingResolution {
  const requestedSegment = finalModelSegment(modelId)
  if (!requestedSegment) {
    return { selected: null, matches: [], providerMatches: [], selectedMonthlyUsageCost: null }
  }

  const matches = prices.filter(price =>
    hasUsablePrice(price) && isExactCheckpointOffer(modelId, price),
  )

  const ranked = [...matches].sort((left, right) => {
    const costDifference = usageCost(left, monthlyInputTokens, monthlyOutputTokens) -
      usageCost(right, monthlyInputTokens, monthlyOutputTokens)
    if (Math.abs(costDifference) > Number.EPSILON) return costDifference

    const exactDifference =
      Number(normalizeModelId(right.id) === normalizeModelId(modelId)) -
      Number(normalizeModelId(left.id) === normalizeModelId(modelId))
    if (exactDifference !== 0) return exactDifference

    return left.id.localeCompare(right.id)
  })
  const seenProviders = new Set<string>()
  const providerMatches = ranked.filter(offer => {
    const provider = offer.provider.trim().toLowerCase()
    if (seenProviders.has(provider)) return false
    seenProviders.add(provider)
    return true
  })
  const selected = providerMatches[0] ?? null

  return {
    selected,
    matches: ranked,
    providerMatches,
    selectedMonthlyUsageCost: selected
      ? usageCost(selected, monthlyInputTokens, monthlyOutputTokens)
      : null,
  }
}

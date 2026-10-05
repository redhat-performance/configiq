import type { ComboBoxItem } from '@/components/ModelComboBox/ModelComboBox'
import type { FrontierModel } from '@/lib/hooks/useCostings'

export const SAME_API_MODEL = '__same_model__'

/** The feed has no modality/rate-condition fields. Keep special-rate and
 * non-text identities out of the ordinary text-token comparison. */
export function isStandardTokenOffer(offer: FrontierModel): boolean {
  return Number.isFinite(offer.price_per_m_input) && offer.price_per_m_input >= 0 &&
    Number.isFinite(offer.price_per_m_output) && offer.price_per_m_output >= 0 &&
    !/(?:[:@]|batch|image|audio|realtime|transcrib|embedding|moderation|\btts\b|search|long-context)/i.test(offer.id)
}

/** Explicit model families, not the feed's heuristic "frontier" tier.
 * Normalize provider-prefixed records without conflating distinct versions. */
export function apiModelIdentity(offer: FrontierModel): { id: string; group: string } | null {
  const name = offer.id.toLowerCase().split('/').at(-1) ?? ''
  if (/^gpt-|^o[134](?:-|$)/.test(name)) return { id: `openai/${name}`, group: 'OpenAI models' }
  if (/^claude-/.test(name)) return { id: `anthropic/${name}`, group: 'Anthropic models' }
  if (/^gemini-/.test(name)) return { id: `google/${name}`, group: 'Google models' }
  return null
}

export function buildApiModelItems(offers: FrontierModel[], selectedModel: string): ComboBoxItem[] {
  const identities = new Map<string, ComboBoxItem>()
  for (const offer of offers) {
    if (!isStandardTokenOffer(offer)) continue
    const identity = apiModelIdentity(offer)
    if (identity) identities.set(identity.id, {
      value: identity.id, label: identity.id.split('/').at(-1)!, group: identity.group,
    })
  }
  return [
    { value: SAME_API_MODEL, label: selectedModel || 'Select a model above', group: '' },
    ...[...identities.values()].sort((a, b) => a.value.localeCompare(b.value)),
  ]
}

export function hostedPricingSource(offer: FrontierModel): string {
  if (offer.source === 'openrouter') return 'OpenRouter catalogue'
  if (offer.source === 'litellm') return 'LiteLLM catalogue'
  if (offer.source === 'override') return 'Curated catalogue override'
  return 'aicostings catalogue'
}

export function hostedOfferLabel(offer: FrontierModel): string {
  // OpenRouter namespaces identify model authors, not necessarily billing hosts.
  if (offer.source === 'openrouter') return `OpenRouter · ${offer.provider}`
  if (offer.source === 'litellm') return `${offer.provider} · LiteLLM catalogue`
  const labels: Record<string, string> = { openrouter: 'OpenRouter', ovhcloud: 'OVHcloud', deepinfra: 'DeepInfra', togetherai: 'Together AI' }
  return labels[offer.provider.trim().toLowerCase()] ?? offer.provider
}

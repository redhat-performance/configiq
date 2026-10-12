import { describe, expect, it } from 'vitest'
import type { FrontierModel } from '@/lib/hooks/useCostings'
import { apiModelIdentity, buildApiModelItems, hostedOfferLabel, hostedPricingSource, isStandardTokenOffer, SAME_API_MODEL } from './api-models'

const offer = (id: string, changes: Partial<FrontierModel> = {}): FrontierModel => ({
  id, name: id, provider: 'OpenAI', tier: 'balanced', price_per_m_input: 1,
  price_per_m_output: 2, context_window: 128000, updated_at: null, source: 'openrouter', ...changes,
})

describe('live hosted API alternatives', () => {
  it('groups and deduplicates actual priced models without relying on the tier heuristic', () => {
    const items = buildApiModelItems([
      offer('openai/gpt-5'), offer('azure/openai/gpt-5', { source: 'litellm' }),
      offer('claude-opus-4.5'), offer('google/gemini-2.5-pro'), offer('qwen/qwen3-8b'),
    ], 'Qwen/Qwen3-8B')
    expect(items[0]).toEqual({ value: SAME_API_MODEL, label: 'Qwen/Qwen3-8B', group: '' })
    expect(items.map(item => item.value)).toEqual([SAME_API_MODEL, 'anthropic/claude-opus-4.5', 'google/gemini-2.5-pro', 'openai/gpt-5'])
    expect(items.find(item => item.value === 'openai/gpt-5')?.group).toBe('OpenAI models')
    expect(apiModelIdentity(offer('openai/o3'))?.id).toBe('openai/o3')
  })

  it.each(['openai/gpt-5:batch', 'openai/gpt-5:free', 'google/gemini-3-pro-image', 'openai/gpt-4o-audio', 'openai/gpt-realtime', 'openai/gpt-search', 'openai/text-embedding'])('excludes special-rate and non-text records: %s', id => {
    expect(isStandardTokenOffer(offer(id))).toBe(false)
    expect(buildApiModelItems([offer(id)], 'Qwen/Qwen3-8B')).toHaveLength(1)
  })

  it('rejects missing, negative and non-finite rates', () => {
    for (const rate of [undefined, NaN, Infinity, -1]) {
      expect(buildApiModelItems([offer('openai/gpt-5', { price_per_m_input: rate as number })], 'Qwen/Qwen3-8B')).toHaveLength(1)
    }
  })

  it('distinguishes a model author from a pricing source or billing host', () => {
    expect(hostedOfferLabel(offer('anthropic/claude-opus', { provider: 'Anthropic' }))).toBe('OpenRouter · Anthropic')
    expect(hostedPricingSource(offer('openai/gpt-5'))).toBe('OpenRouter catalogue')
    expect(hostedOfferLabel(offer('openai/gpt-5', { source: 'litellm' }))).toBe('OpenAI · LiteLLM catalogue')
    expect(hostedOfferLabel(offer('qwen/qwen3-8b', { source: undefined, provider: 'ovhcloud' }))).toBe('OVHcloud')
  })
})

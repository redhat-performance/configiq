import { describe, expect, it } from 'vitest'
import type { FrontierModel } from '@/lib/hooks/useCostings'
import { resolveHostedPricing } from './hosted-pricing'

function pricedModel(
  id: string,
  provider: string,
  input: number,
  output: number,
): FrontierModel {
  return {
    id,
    name: id,
    provider,
    tier: 'balanced',
    price_per_m_input: input,
    price_per_m_output: output,
    context_window: null,
    updated_at: null,
  }
}

describe('hosted pricing resolution', () => {
  it('selects the cheapest provider for the actual token mix', () => {
    const prices = [
      pricedModel('deepinfra/deepseek-ai/DeepSeek-R1', 'DeepInfra', 0.4, 2),
      pricedModel('openrouter/deepseek/deepseek-r1', 'OpenRouter', 0.5, 1),
      pricedModel('deepseek-ai/DeepSeek-R1', 'DeepSeek', 0.8, 0.8),
    ]

    const inputHeavy = resolveHostedPricing('deepseek-ai/DeepSeek-R1', prices, 20_000_000, 1_000_000)
    const outputHeavy = resolveHostedPricing('deepseek-ai/DeepSeek-R1', prices, 1_000_000, 20_000_000)

    expect(inputHeavy.selected?.provider).toBe('DeepInfra')
    expect(outputHeavy.selected?.provider).toBe('DeepSeek')
    expect(inputHeavy.matches).toHaveLength(3)
  })

  it('keeps only the cheapest offer per provider for the compact selector', () => {
    const prices = [
      pricedModel('route-a/owner/model-a', 'Provider A', 1, 2),
      pricedModel('route-b/owner/model-a', 'Provider A', 2, 3),
      pricedModel('route-c/owner/model-a', 'Provider B', 1.5, 2),
      pricedModel('route-d/owner/model-a', 'Provider C', 3, 3),
    ]

    const result = resolveHostedPricing('owner/model-a', prices, 2_000_000, 1_000_000)

    expect(result.matches).toHaveLength(4)
    expect(result.providerMatches.map(offer => offer.provider)).toEqual([
      'Provider A',
      'Provider B',
      'Provider C',
    ])
    expect(result.providerMatches[0].id).toBe('route-a/owner/model-a')
  })

  it('does not treat a derived checkpoint ending in the same slug as the requested model', () => {
    const prices = [
      pricedModel('llamagate/qwen3-8b', 'Llamagate', 0.04, 0.14),
      pricedModel('openrouter/qwen/qwen3-8b', 'OpenRouter', 0.117, 0.455),
      pricedModel('novita/deepseek/deepseek-r1-0528-qwen3-8b', 'Novita', 0.01, 0.01),
    ]

    const result = resolveHostedPricing('Qwen/Qwen3-8B', prices, 2_000_000_000, 500_000_000)

    expect(result.selected?.provider).toBe('Llamagate')
    expect(result.matches.map(match => match.provider)).toEqual(['Llamagate', 'OpenRouter'])
  })

  it('uses the same hosted model price for a serving-format checkpoint', () => {
    const prices = [pricedModel('qwen/qwen3-32b', 'Qwen', 0.08, 0.28)]
    const result = resolveHostedPricing(
      'Qwen/Qwen3-32B-FP8-Static-PerTensor',
      prices,
      20_000_000,
      5_000_000,
    )

    expect(result.selected?.provider).toBe('Qwen')
    expect(result.matches).toHaveLength(1)
  })

  it('does not silently use an instruction-tuned offer for a base checkpoint', () => {
    const prices = [
      pricedModel('meta-llama/llama-3.1-8b-instruct', 'Hosted provider', 0.1, 0.2),
      pricedModel('hyperbolic/meta-llama/meta-llama-3.1-405b-instruct', 'Hosted provider', 0.2, 0.5),
      pricedModel('google/gemma-4-26b-a4b-it', 'Hosted provider', 0.2, 0.4),
    ]

    expect(resolveHostedPricing(
      'meta-llama/Meta-Llama-3.1-8B', prices, 1_000_000, 1_000_000,
    ).selected).toBeNull()
    expect(resolveHostedPricing(
      'google/gemma-4-26b-a4b', prices, 1_000_000, 1_000_000,
    ).selected).toBeNull()
    expect(resolveHostedPricing(
      'meta-llama/Meta-Llama-3.1-405B', prices, 1_000_000, 1_000_000,
    ).selected).toBeNull()
    expect(resolveHostedPricing(
      'nvidia/gemma-4-26b-a4b', prices, 1_000_000, 1_000_000,
    ).selected).toBeNull()
  })

  it('resolves explicit serving-format and instruction aliases', () => {
    const prices = [
      pricedModel('meta-llama/llama-3.1-70b-instruct', 'Meta', 0.4, 0.4),
      pricedModel('qwen/qwen3-235b-a22b', 'Qwen', 0.2, 0.6),
    ]

    expect(resolveHostedPricing(
      'nvidia/Llama-3.1-70B-Instruct-FP8', prices, 1_000_000, 1_000_000,
    ).selected?.provider).toBe('Meta')
    expect(resolveHostedPricing(
      'nvidia/Qwen3-235B-A22B-NVFP4', prices, 1_000_000, 1_000_000,
    ).selected?.provider).toBe('Qwen')
  })

  it('ignores invalid price records', () => {
    const prices = [
      pricedModel('provider/model-a', 'Invalid', Number.NaN, 1),
      pricedModel('other/model-a', 'Valid', 1, 2),
    ]
    const result = resolveHostedPricing('owner/model-a', prices, 1_000_000, 1_000_000)

    expect(result.selected?.provider).toBe('Valid')
    expect(result.matches).toHaveLength(1)
  })
})

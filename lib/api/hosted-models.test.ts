import { describe, expect, it } from 'vitest'
import { parseHostedModels } from './hosted-models'

const model = { id: 'owner/model', name: 'Model', provider: 'Provider', tier: 'balanced',
  price_per_m_input: 1, price_per_m_output: 2 }

describe('hosted pricing API contract', () => {
  it('accepts older catalogues without inventing billing conditions or limits', () => {
    expect(parseHostedModels({ models: [model] }).models[0]).toEqual(model)
  })
  it('preserves normalized billing, limits and distinct provider identity', () => {
    const detailed = { ...model, source: 'openrouter', model_author: 'Author', serving_provider: null,
      max_context_tokens: 200000, max_output_tokens: 32000,
      billing: { currency: 'USD', request_price: 0.01, input_cache_read_per_million: 0.1,
        overrides: [{ min_prompt_tokens: 128000, output_per_million: 4 }], unsupported: [] } }
    expect(parseHostedModels({ models: [detailed] }).models[0]).toEqual(detailed)
  })
  it.each([
    { price_per_m_input: -1 }, { price_per_m_output: Infinity }, { price_per_m_input: true },
    { max_context_tokens: 0 }, { max_output_tokens: 1.5 },
    { billing: { currency: 'EUR' } }, { billing: { currency: 'USD', request_price: -1 } },
    { billing: { currency: 'USD', overrides: [{ min_prompt_tokens: 1, utc_start: '12:00' }] } },
  ])('rejects invalid or unhandled prices instead of silently dropping their terms: %j', patch => {
    expect(() => parseHostedModels({ models: [{ ...model, ...patch }] })).toThrow()
  })
  it('rejects malformed envelopes', () => {
    expect(() => parseHostedModels({ models: {} })).toThrow()
    expect(() => parseHostedModels({})).toThrow()
  })
})

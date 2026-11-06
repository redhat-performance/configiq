import { describe, expect, it } from 'vitest'
import {
  costPerMillionOutputTokens,
  formatCostPerMillionOutputTokens,
  outputTokensAtVolume,
} from './token-economics'

describe('costPerMillionOutputTokens', () => {
  it('divides the selected monthly cost by the requested output volume', () => {
    expect(costPerMillionOutputTokens(8_817, 500_000_000)).toBeCloseTo(17.634, 6)
    expect(costPerMillionOutputTokens(0, 1)).toBe(0)
    expect(costPerMillionOutputTokens(1, 1_000_000)).toBe(1)
  })

  it('returns null for undefined metric inputs', () => {
    expect(costPerMillionOutputTokens(1_000, 0)).toBeNull()
    expect(costPerMillionOutputTokens(1_000, -5)).toBeNull()
    expect(costPerMillionOutputTokens(-1, 1_000_000)).toBeNull()
    expect(costPerMillionOutputTokens(Number.NaN, 1_000_000)).toBeNull()
    expect(costPerMillionOutputTokens(Number.POSITIVE_INFINITY, 1_000_000)).toBeNull()
    expect(costPerMillionOutputTokens(1_000, Number.NaN)).toBeNull()
    expect(costPerMillionOutputTokens(Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY)).toBeNull()
  })
})

describe('outputTokensAtVolume', () => {
  it('applies the output share of the billed-token sample', () => {
    expect(outputTokensAtVolume(2_500_000_000, 0.2)).toBe(500_000_000)
    expect(outputTokensAtVolume(2_500_000_000, 0)).toBe(0)
    expect(outputTokensAtVolume(2_500_000_000, 1)).toBe(2_500_000_000)
  })

  it('clamps invalid shares and volumes', () => {
    expect(outputTokensAtVolume(1_000, 1.5)).toBe(1_000)
    expect(outputTokensAtVolume(1_000, -0.5)).toBe(0)
    expect(outputTokensAtVolume(-1_000, 0.5)).toBe(0)
    expect(outputTokensAtVolume(Number.NaN, 0.5)).toBe(0)
    expect(outputTokensAtVolume(1_000, Number.NaN)).toBe(0)
  })
})

describe('formatCostPerMillionOutputTokens', () => {
  it('formats card display values', () => {
    expect(formatCostPerMillionOutputTokens(17.634)).toBe('$17.63')
    expect(formatCostPerMillionOutputTokens(0)).toBe('$0.00')
    expect(formatCostPerMillionOutputTokens(0.004)).toBe('<$0.01')
    expect(formatCostPerMillionOutputTokens(null)).toBe('Not available')
  })
})

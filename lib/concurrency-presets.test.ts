import { describe, expect, it } from 'vitest'
import { matchConcurrencyPresetKey, type ConcurrencyPreset } from './concurrency-presets'

const presets: ConcurrencyPreset[] = [
  { key: 'xs', label: 'X-Small (2)', concurrency: 2 },
  { key: 'sm', label: 'Small (10)', concurrency: 10 },
  { key: 'md', label: 'Medium (100)', concurrency: 100 },
]

describe('matchConcurrencyPresetKey', () => {
  it('matches a typed value to its t-shirt preset', () => {
    expect(matchConcurrencyPresetKey(presets, '10')).toBe('sm')
    expect(matchConcurrencyPresetKey(presets, 100)).toBe('md')
  })

  it('tolerates surrounding whitespace and leading zeros', () => {
    expect(matchConcurrencyPresetKey(presets, ' 010 ')).toBe('sm')
  })

  it('returns no selection for a value outside the presets', () => {
    expect(matchConcurrencyPresetKey(presets, '5')).toBe('')
    expect(matchConcurrencyPresetKey(presets, '101')).toBe('')
  })

  it('returns no selection for empty or invalid input', () => {
    expect(matchConcurrencyPresetKey(presets, '')).toBe('')
    expect(matchConcurrencyPresetKey(presets, 'ten')).toBe('')
    expect(matchConcurrencyPresetKey(presets, 0)).toBe('')
    expect(matchConcurrencyPresetKey(presets, Number.NaN)).toBe('')
  })

  it('returns no selection when the catalog has no presets', () => {
    expect(matchConcurrencyPresetKey([], '10')).toBe('')
  })
})

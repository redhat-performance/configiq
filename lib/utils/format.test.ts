import { describe, expect, it } from 'vitest'

import { formatLatencyBound } from './format'

describe('formatLatencyBound', () => {
  it('renders whole and fractional seconds without trailing zeros', () => {
    expect(formatLatencyBound(1000)).toBe('1')
    expect(formatLatencyBound(2500)).toBe('2.5')
    expect(formatLatencyBound(250)).toBe('0.25')
  })

  it('preserves sub-millisecond target precision', () => {
    expect(formatLatencyBound(1.4)).toBe('0.0014')
  })
})

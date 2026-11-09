import { describe, expect, it } from 'vitest'

import { formatCandidateId } from './format'

describe('formatCandidateId', () => {
  it('collapses AISimulate six-digit sequences to three digits', () => {
    expect(formatCandidateId('candidate-000001')).toBe('candidate-001')
    expect(formatCandidateId('candidate-000042')).toBe('candidate-042')
    expect(formatCandidateId('candidate-000999')).toBe('candidate-999')
  })

  it('keeps numbers above three digits intact', () => {
    expect(formatCandidateId('candidate-001234')).toBe('candidate-1234')
  })

  it('leaves non-AISimulate identifiers unchanged', () => {
    expect(formatCandidateId('agg-tp1-c1')).toBe('agg-tp1-c1')
  })
})

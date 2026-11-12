import { describe, expect, it } from 'vitest'
import { statusForModel } from './model-status'

describe('statusForModel', () => {
  it('keeps a status for the model it was computed for', () => {
    expect(statusForModel({ model: 'org/model-a', status: 'catalog' }, 'org/model-a')).toBe('catalog')
  })

  it('does not carry a validated status over to a different model', () => {
    expect(statusForModel({ model: 'org/model-a', status: 'fetched' }, 'org/model-b')).toBe('idle')
  })

  it('treats an unchecked initial model as idle', () => {
    expect(statusForModel({ model: '', status: 'idle' }, 'org/model-a')).toBe('idle')
  })
})

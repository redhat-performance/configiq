import { describe, expect, it } from 'vitest'
import { mapBackendOption } from './useCatalog'

describe('mapBackendOption', () => {
  it('maps backend memory defaults from the REST catalog', () => {
    expect(mapBackendOption({
      id: 'vllm',
      versions: ['0.10.0'],
      memory_fraction: 0.92,
      memory_fraction_kind: 'of_free',
    })).toMatchObject({
      id: 'vllm',
      memoryFraction: 0.92,
      memoryFractionKind: 'of_free',
    })
  })

  it('uses safe defaults when optional memory metadata is absent or invalid', () => {
    expect(mapBackendOption({ id: 'vllm', versions: [], memory_fraction_kind: 'invalid' })).toMatchObject({
      memoryFraction: null,
      memoryFractionKind: 'of_total',
    })
  })

  it('ignores malformed backend entries', () => {
    expect(mapBackendOption({ id: 'vllm' })).toBeNull()
  })
})

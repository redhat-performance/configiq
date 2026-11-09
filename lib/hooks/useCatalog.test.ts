import { describe, expect, it } from 'vitest'
import { hasDefaultMemoryFractionForVersion, mapBackendOption, systemMemoryFraction } from './useCatalog'

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
      disaggregationRequiresFabric: false,
    })
  })

  it('reports whether a P/D topology needs a fabric', () => {
    expect(mapBackendOption({
      id: 'vllm',
      versions: ['0.31.0'],
      disaggregation_requires_fabric: true,
    })?.disaggregationRequiresFabric).toBe(true)
  })

  it('ignores malformed backend entries', () => {
    expect(mapBackendOption({ id: 'vllm' })).toBeNull()
  })

  it('maps per-system defaults so a system-specific fraction is available', () => {
    const backend = mapBackendOption({
      id: 'vllm',
      versions: ['0.14.0', '0.31.0'],
      default_version: 'current',
      memory_fraction: 0.92,
      system_defaults: {
        a100_sxm: { version: '0.14.0', memory_fraction: 0.9 },
        h200_sxm: { version: '0.24.0', memory_fraction: 0.92 },
        broken: { version: 42 },
      },
    })

    expect(backend?.defaultVersion).toBe('current')
    expect(backend?.systemDefaults).toEqual({
      a100_sxm: { version: '0.14.0', memoryFraction: 0.9 },
      h200_sxm: { version: '0.24.0', memoryFraction: 0.92 },
      broken: { version: null, memoryFraction: null },
    })
  })

  it('defaults systemDefaults to empty when the API omits it', () => {
    expect(mapBackendOption({ id: 'vllm', versions: ['0.10.0'] })?.systemDefaults).toEqual({})
  })
})

describe('systemMemoryFraction', () => {
  const backend = mapBackendOption({
    id: 'vllm',
    versions: ['0.14.0', '0.24.0'],
    memory_fraction: 0.92,
    system_defaults: {
      a100_sxm: { version: '0.14.0', memory_fraction: 0.9 },
      h200_sxm: { version: '0.24.0', memory_fraction: null },
    },
  })

  it('prefers the selected system default over the fleet figure', () => {
    expect(systemMemoryFraction(backend ?? undefined, 'a100_sxm')).toBe(0.9)
  })

  it('falls back to the fleet figure when the system has no usable default', () => {
    expect(systemMemoryFraction(backend ?? undefined, 'h200_sxm')).toBe(0.92)
    expect(systemMemoryFraction(backend ?? undefined, 'b60')).toBe(0.92)
  })

  it('returns null when the backend is unknown', () => {
    expect(systemMemoryFraction(undefined, 'a100_sxm')).toBeNull()
  })
})

describe('hasDefaultMemoryFractionForVersion', () => {
  const backend = mapBackendOption({
    id: 'vllm',
    versions: ['0.14.0', '0.24.0'],
    default_version: 'current',
    system_defaults: {
      a100_sxm: { version: '0.14.0', memory_fraction: 0.9 },
    },
  })

  it('accepts an omitted pin, current alias, or matching system version', () => {
    expect(hasDefaultMemoryFractionForVersion(backend ?? undefined, 'a100_sxm', '')).toBe(true)
    expect(hasDefaultMemoryFractionForVersion(backend ?? undefined, 'a100_sxm', 'current')).toBe(true)
    expect(hasDefaultMemoryFractionForVersion(backend ?? undefined, 'a100_sxm', '0.14.0')).toBe(true)
  })

  it('requires an explicit fraction for another version or missing per-system data', () => {
    expect(hasDefaultMemoryFractionForVersion(backend ?? undefined, 'a100_sxm', '0.24.0')).toBe(false)
    expect(hasDefaultMemoryFractionForVersion(backend ?? undefined, 'h100_sxm', '0.14.0')).toBe(false)
    expect(hasDefaultMemoryFractionForVersion(undefined, 'a100_sxm', '0.14.0')).toBe(false)
  })
})

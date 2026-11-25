import { describe, expect, it } from 'vitest'
import type { GpuOption } from '@/lib/hooks/useCatalog'
import { findCatalogGpu, matchAccelerator } from '../gpu-catalog'

function gpu(overrides: Partial<GpuOption> = {}): GpuOption {
  return {
    systemId: 'h200_sxm',
    label: 'NVIDIA H200',
    vendor: 'nvidia',
    architecture: 'hopper',
    vramGb: 141,
    bandwidthTbps: 4.8,
    tflopsBf16: 989,
    tdpWatts: 700,
    gpusPerNode: 8,
    ...overrides,
  }
}

const GPUS = [
  gpu(),
  gpu({ systemId: 'b200_sxm', label: 'NVIDIA B200', vramGb: 192 }),
  gpu({ systemId: 'gb200', label: 'NVIDIA GB200', vramGb: 192 }),
]

describe('findCatalogGpu', () => {
  it('matches the catalog spelling exactly and case-insensitively', () => {
    expect(findCatalogGpu('NVIDIA H200', GPUS)?.systemId).toBe('h200_sxm')
    expect(findCatalogGpu('nvidia h200', GPUS)?.systemId).toBe('h200_sxm')
  })

  it('matches a shorter measured label to the catalog system', () => {
    expect(findCatalogGpu('H200', GPUS)?.systemId).toBe('h200_sxm')
    expect(findCatalogGpu('B200', GPUS)?.systemId).toBe('b200_sxm')
  })

  it('does not match a different part whose name contains the label', () => {
    expect(findCatalogGpu('B200', [gpu({ systemId: 'gb200', label: 'NVIDIA GB200' })])).toBeNull()
  })

  it('returns null for an empty or unknown label', () => {
    expect(findCatalogGpu('', GPUS)).toBeNull()
    expect(findCatalogGpu('   ', GPUS)).toBeNull()
    expect(findCatalogGpu('TPU v7', GPUS)).toBeNull()
  })
})

describe('matchAccelerator', () => {
  it('keeps an exact label', () => {
    expect(matchAccelerator(['H200', 'B200'], 'b200')).toBe('B200')
  })

  it('keeps the measured spelling when the parts are the same', () => {
    expect(matchAccelerator(['H200', 'B200'], 'NVIDIA H200')).toBe('H200')
  })

  it('falls back to the first label when nothing matches', () => {
    expect(matchAccelerator(['H200', 'B200'], 'TPU v7')).toBe('H200')
    expect(matchAccelerator(['H200'], '')).toBe('H200')
  })

  it('returns an empty string when there is nothing to choose', () => {
    expect(matchAccelerator([], 'H200')).toBe('')
  })
})

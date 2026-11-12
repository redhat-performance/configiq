import { describe, expect, it } from 'vitest'
import {
  filterRows,
  getConfigurationOptions,
  getFilterOptions,
  normalizeBackendVersion,
} from '../benchmark-dataset'
import type { DatasetRow, BenchmarkFilters } from '../benchmark-types'

const filters: BenchmarkFilters = {
  accelerator: 'H200',
  model: 'example/model',
  backend: 'vllm',
  backendVersion: 'v0.24.0',
  isl: '8192',
  osl: '1024',
  tp: '4',
}

const internal: DatasetRow = {
  accelerator: 'H200', model: 'example/model', version: 'GT release 2',
  backend: 'vllm', backend_version: 'vLLM-0.24.0', precision: 'fp4',
  isl: 8192, osl: 1024, tp: 4, concurrency: 8,
}

const inferencex: DatasetRow = {
  ...internal, benchmark_source: 'inferencex', version: 'InferenceX FP4',
  backend_version: 'v0.24.0', inferencex_topology: 'aggregated · single-node',
}

describe('shared measured backend filters', () => {
  it('finds Red Hat and InferenceX under the same vLLM release without merging their provenance', () => {
    const rows = [internal, inferencex]
    expect(getFilterOptions(rows, { ...filters, backendVersion: '' }).backendVersion).toEqual(['v0.24.0'])
    expect(filterRows(rows, filters)).toHaveLength(2)

    const configurations = getConfigurationOptions(filterRows(rows, filters))
    expect(configurations).toHaveLength(2)
    expect(configurations.map(option => option.label)).toEqual([
      expect.stringContaining('InferenceX'),
      expect.stringContaining('Red Hat'),
    ])
    expect(configurations.find(option => option.values.measurement_source === 'Red Hat')?.values.benchmark_version)
      .toBe('GT release 2')
  })

  it('does not equate unknown or nightly tags with a numbered release', () => {
    expect(normalizeBackendVersion('vllm', null)).toBe('Version not reported')
    expect(normalizeBackendVersion('vllm', 'nightly-abc123')).toBe('nightly-abc123')
    expect(filterRows([{ ...internal, backend_version: 'nightly-abc123' }], filters)).toEqual([])
  })

  it('matches a bare release pin to a vendor build of that release', () => {
    const vendor: DatasetRow = {
      ...inferencex,
      backend: 'sglang',
      backend_version: 'v0.5.14-rocm720-mi35x-20260706',
    }
    const pin: BenchmarkFilters = { ...filters, backend: 'sglang', backendVersion: '0.5.14' }

    expect(filterRows([vendor], pin)).toHaveLength(1)
    expect(filterRows([{ ...vendor, backend_version: 'v0.5.21-rocm720-mi35x' }], pin)).toEqual([])
  })

  it('keeps two vendor builds of the same release distinct', () => {
    const buildA: DatasetRow = {
      ...inferencex,
      backend: 'sglang',
      backend_version: 'v0.5.14-rocm720-mi35x',
    }
    const buildB: DatasetRow = { ...buildA, backend_version: 'v0.5.14-cuda130' }
    const pin: BenchmarkFilters = {
      ...filters,
      backend: 'sglang',
      backendVersion: 'v0.5.14-rocm720-mi35x',
    }

    expect(filterRows([buildA, buildB], pin)).toHaveLength(1)
  })

  it('does not widen a specific build pin to a bare release row', () => {
    const bare: DatasetRow = {
      ...inferencex,
      backend: 'sglang',
      backend_version: 'v0.5.14',
    }
    const pin: BenchmarkFilters = {
      ...filters,
      backend: 'sglang',
      backendVersion: 'v0.5.14-rocm720-mi35x',
    }

    expect(filterRows([bare], pin)).toEqual([])
  })
})


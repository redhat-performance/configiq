import { describe, expect, it } from 'vitest'
import {
  filterRows,
  getConfigurationOptions,
  getFilterOptions,
  normalizeBackendVersion,
} from './ground-truth-dataset'
import type { DatasetRow, GroundTruthFilters } from './ground-truth-types'

const filters: GroundTruthFilters = {
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
  it('finds GT and InferenceX under the same vLLM release without merging their provenance', () => {
    const rows = [internal, inferencex]
    expect(getFilterOptions(rows, { ...filters, backendVersion: '' }).backendVersion).toEqual(['v0.24.0'])
    expect(filterRows(rows, filters)).toHaveLength(2)

    const configurations = getConfigurationOptions(filterRows(rows, filters))
    expect(configurations).toHaveLength(2)
    expect(configurations.map(option => option.label)).toEqual([
      expect.stringContaining('InferenceX'),
      expect.stringContaining('Internal GT'),
    ])
    expect(configurations.find(option => option.values.measurement_source === 'Internal GT')?.values.benchmark_version)
      .toBe('GT release 2')
  })

  it('does not equate unknown or nightly tags with a numbered release', () => {
    expect(normalizeBackendVersion('vllm', null)).toBe('Version not reported')
    expect(normalizeBackendVersion('vllm', 'nightly-abc123')).toBe('nightly-abc123')
    expect(filterRows([{ ...internal, backend_version: 'nightly-abc123' }], filters)).toEqual([])
  })
})


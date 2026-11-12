import { describe, expect, it } from 'vitest'
import { acceleratorNamesMatch, resolveBenchmarkFilters } from '../benchmark-resolve'
import type { BenchmarkFilters, DatasetRow } from '../benchmark-types'

const filters: BenchmarkFilters = {
  accelerator: 'H200',
  model: 'example/model',
  backend: 'vllm',
  backendVersion: 'v0.24.0',
  isl: '8192',
  osl: '1024',
  tp: '1',
}

function row(overrides: Partial<DatasetRow> = {}): DatasetRow {
  return {
    accelerator: 'H200',
    model: 'example/model',
    backend: 'vllm',
    backend_version: 'vLLM-0.24.0',
    isl: 8192,
    osl: 1024,
    tp: 1,
    concurrency: 8,
    ...overrides,
  }
}

describe('benchmark accelerator matching', () => {
  it('matches display prefixes and vendor prefixes', () => {
    expect(acceleratorNamesMatch('NVIDIA H200', 'H200')).toBe(true)
    expect(acceleratorNamesMatch('AMD MI300X', 'MI300X')).toBe(true)
  })

  it('does not match a shorter name inside a longer model name', () => {
    expect(acceleratorNamesMatch('B200', 'GB200')).toBe(false)
    expect(acceleratorNamesMatch('NVIDIA GB200', 'B200')).toBe(false)
  })
})

describe('resolveBenchmarkFilters', () => {
  it('canonicalizes a Settings release pin to the measured label', () => {
    const resolution = resolveBenchmarkFilters([row()], { ...filters, backendVersion: '0.24.0' })

    expect(resolution.unavailable).toBeNull()
    expect(resolution.filters.backendVersion).toBe('v0.24.0')
  })

  it('canonicalizes non-vLLM release pins to the measured label', () => {
    const sglang = resolveBenchmarkFilters([
      row({ backend: 'sglang', backend_version: 'v0.5.0' }),
    ], { ...filters, backend: 'sglang', backendVersion: '0.5.0' })

    expect(sglang.unavailable).toBeNull()
    expect(sglang.filters.backendVersion).toBe('v0.5.0')

    const tensorrt = resolveBenchmarkFilters([
      row({ backend: 'tensorrt-llm', backend_version: 'v1.3.0rc29' }),
    ], { ...filters, backend: 'tensorrt-llm', backendVersion: '1.3.0rc29' })

    expect(tensorrt.unavailable).toBeNull()
    expect(tensorrt.filters.backendVersion).toBe('v1.3.0rc29')
  })

  it('still reports a different non-vLLM release as unavailable', () => {
    const resolution = resolveBenchmarkFilters([
      row({ backend: 'sglang', backend_version: 'v0.5.1' }),
    ], { ...filters, backend: 'sglang', backendVersion: '0.5.0' })

    expect(resolution.unavailable?.field).toBe('backendVersion')
  })

  it('matches a bare catalog pin to a measured vendor build', () => {
    const resolution = resolveBenchmarkFilters([
      row({ backend: 'sglang', backend_version: 'v0.5.14-rocm720-mi35x-20260706' }),
    ], { ...filters, backend: 'sglang', backendVersion: '0.5.14' })

    expect(resolution.unavailable).toBeNull()
    expect(resolution.filters.backendVersion).toBe('v0.5.14')
  })

  it('still reports a vendor build of a different release as unavailable', () => {
    const resolution = resolveBenchmarkFilters([
      row({ backend: 'sglang', backend_version: 'v0.5.21-rocm720-mi35x' }),
    ], { ...filters, backend: 'sglang', backendVersion: '0.5.14' })

    expect(resolution.unavailable?.field).toBe('backendVersion')
  })

  it('repairs the accelerator to a GPU that supports the pinned release', () => {
    const resolution = resolveBenchmarkFilters([
      row(),
      row({ accelerator: 'B200', backend_version: 'vLLM-0.25.0' }),
    ], {
      ...filters,
      accelerator: 'H100',
      backendVersion: '0.25.0',
    })

    expect(resolution.unavailable).toBeNull()
    expect(resolution.filters.accelerator).toBe('B200')
    expect(resolution.filters.backendVersion).toBe('v0.25.0')
  })

  it('still reports a release that no GPU measured', () => {
    const resolution = resolveBenchmarkFilters([row()], { ...filters, backendVersion: '9.9.9' })

    expect(resolution.filters.backendVersion).toBe('v9.9.9')
    expect(resolution.unavailable).toEqual(expect.objectContaining({ field: 'backendVersion' }))
  })
})

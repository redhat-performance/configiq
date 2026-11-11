import { NextRequest } from 'next/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as dataset from '@/lib/validation/benchmark-dataset'
import type { BenchmarkFilters, DatasetRow } from '@/lib/validation/benchmark-types'
import { GET } from './route'

vi.mock('@/lib/validation/benchmark-dataset', async importOriginal => {
  const actual = await importOriginal<typeof import('@/lib/validation/benchmark-dataset')>()
  return {
    ...actual,
    getBenchmarkDataset: vi.fn(),
    getInferenceXDataset: vi.fn(),
  }
})

const filters: BenchmarkFilters = {
  accelerator: 'H200',
  model: 'example/model',
  backend: 'vllm',
  backendVersion: 'v0.24.0',
  isl: '8192',
  osl: '1024',
  tp: '1',
}

const internal: DatasetRow = {
  accelerator: 'H200', model: 'example/model', version: 'Benchmark release',
  backend: 'vllm', backend_version: 'vLLM-0.24.0', precision: 'fp4',
  isl: 8192, osl: 1024, tp: 1, concurrency: 8,
  ttft_median: 100, tpot_median: 10, request_latency_median: 1,
  'output_tok/sec': 100,
}

const inferencex: DatasetRow = {
  ...internal,
  benchmark_source: 'inferencex', version: 'InferenceX FP4',
  backend_version: 'v0.24.0', source_run_url: 'https://example.com/inferencex-run',
  measurement_date: '2026-10-20', inferencex_topology: 'aggregated · single-node',
  ttft_median: 200,
}

function requestFor(row: DatasetRow): NextRequest {
  const configuration = dataset.getConfigurationOptions([row])[0]
  const params = new URLSearchParams({ mode: 'data', configuration: configuration.key, ...filters })
  return new NextRequest(`https://configiq.dev/api/validation/benchmark?${params}`)
}

function resolveRequest(overrides: Partial<BenchmarkFilters>): NextRequest {
  const params = new URLSearchParams({ mode: 'resolve', ...filters, ...overrides })
  return new NextRequest(`https://configiq.dev/api/validation/benchmark?${params}`)
}

beforeEach(() => {
  vi.mocked(dataset.getBenchmarkDataset).mockResolvedValue({
    rows: [internal],
    source: {
      id: 'internal', url: 'https://example.com/bench', revision: null,
      lastModified: null, loadedAt: '2026-10-20', rowCount: 1,
    },
    expiresAt: Infinity,
  })
  vi.mocked(dataset.getInferenceXDataset).mockResolvedValue({
    rows: [inferencex],
    source: { url: 'https://inferencex.semianalysis.com/api', retrievedAt: '2026-10-20', rowCount: 1 },
    expiresAt: Infinity,
  })
})

describe('GET /api/validation/benchmark', () => {
  it('returns only Red Hat measurements for a Red Hat configuration', async () => {
    const response = await GET(requestFor(internal))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.matchedRows).toBe(1)
    expect(body.benchmarkPoints).toHaveLength(1)
    expect(body.inferencexPoints).toEqual([])
  })

  it('returns only InferenceX measurements for an InferenceX configuration', async () => {
    const response = await GET(requestFor(inferencex))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.matchedRows).toBe(1)
    expect(body.benchmarkPoints).toEqual([])
    expect(body.inferencexPoints).toHaveLength(1)
    expect(body.inferencexRunUrl).toBe('https://example.com/inferencex-run')
    expect(body.inferencexRuns).toEqual([
      { url: 'https://example.com/inferencex-run', date: '2026-10-20' },
    ])
  })

  it('lists every distinct InferenceX run behind an averaged point', async () => {
    const secondRun: DatasetRow = {
      ...inferencex,
      concurrency: 16,
      source_run_url: 'https://example.com/inferencex-run-2',
      measurement_date: '2026-10-21',
    }
    const repeatedFirstRun: DatasetRow = {
      ...inferencex,
      concurrency: 32,
    }
    vi.mocked(dataset.getInferenceXDataset).mockResolvedValueOnce({
      rows: [inferencex, secondRun, repeatedFirstRun],
      source: { url: 'https://inferencex.semianalysis.com/api', retrievedAt: '2026-10-20', rowCount: 3 },
      expiresAt: Infinity,
    })

    const response = await GET(requestFor(inferencex))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.inferencexPoints).toHaveLength(3)
    expect(body.inferencexRuns).toEqual([
      { url: 'https://example.com/inferencex-run', date: '2026-10-20' },
      { url: 'https://example.com/inferencex-run-2', date: '2026-10-21' },
    ])
  })

  it('keeps Red Hat measurements available when the private InferenceX dataset cannot be read', async () => {
    vi.mocked(dataset.getInferenceXDataset).mockRejectedValueOnce(new Error('Private dataset unavailable'))
    const response = await GET(requestFor(internal))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.benchmarkPoints).toHaveLength(1)
    expect(body.inferencexPoints).toEqual([])
    expect(body.inferencexError).toBe('Private dataset unavailable')
  })

  it('keeps InferenceX measurements available when Red Hat data cannot be read', async () => {
    vi.mocked(dataset.getBenchmarkDataset).mockRejectedValueOnce(new Error('Internal dataset unavailable'))
    const response = await GET(requestFor(inferencex))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.benchmarkPoints).toEqual([])
    expect(body.inferencexPoints).toHaveLength(1)
    expect(body.benchmarkError).toBe('Internal dataset unavailable')
  })

  it('repairs an accelerator the model has no rows for instead of clearing the run', async () => {
    vi.mocked(dataset.getBenchmarkDataset).mockResolvedValueOnce({
      rows: [
        internal,
        { ...internal, accelerator: 'B200', isl: 2048, concurrency: 16 },
      ],
      source: {
        id: 'internal', url: 'https://example.com/bench', revision: null,
        lastModified: null, loadedAt: '2026-10-20', rowCount: 2,
      },
      expiresAt: Infinity,
    })

    const response = await GET(resolveRequest({ accelerator: 'MI300X' }))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.adjustments).toEqual([
      expect.objectContaining({ field: 'accelerator', from: 'MI300X' }),
    ])
    expect(['H200', 'B200']).toContain(body.filters.accelerator)
    expect(body.unavailable).toBeNull()
  })

  it('repairs an unmeasured token length to the nearest measured value', async () => {
    vi.mocked(dataset.getBenchmarkDataset).mockResolvedValueOnce({
      rows: [
        internal,
        { ...internal, isl: 2048, concurrency: 16 },
      ],
      source: {
        id: 'internal', url: 'https://example.com/bench', revision: null,
        lastModified: null, loadedAt: '2026-10-20', rowCount: 2,
      },
      expiresAt: Infinity,
    })

    const response = await GET(resolveRequest({ isl: '4096' }))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.adjustments).toEqual([
      expect.objectContaining({ field: 'isl', from: '4096', to: '2048' }),
    ])
    expect(body.filters.isl).toBe('2048')
    expect(body.filters.osl).toBe('1024')
  })

  it('never rewrites the model and reports why it cannot be resolved', async () => {
    const response = await GET(resolveRequest({ model: 'missing/model' }))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.filters.model).toBe('missing/model')
    expect(body.adjustments).toEqual([])
    expect(body.unavailable).toEqual(expect.objectContaining({ field: 'model' }))
    expect(body.configurations).toEqual([])
  })

  it('reports a version mismatch instead of substituting a measured version', async () => {
    const response = await GET(resolveRequest({ backendVersion: 'v9.9.9' }))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.filters.backendVersion).toBe('v9.9.9')
    expect(body.adjustments).toEqual([])
    expect(body.unavailable).toEqual(expect.objectContaining({ field: 'backendVersion' }))
  })

  it('accepts a Settings release pin written without the measured prefix', async () => {
    const response = await GET(resolveRequest({ backendVersion: '0.24.0' }))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.unavailable).toBeNull()
    expect(body.filters.backendVersion).toBe('v0.24.0')
    expect(body.configurations.length).toBeGreaterThan(0)
  })

  it('offers a measured vendor build for a bare catalog release pin', async () => {
    const vendor: DatasetRow = {
      ...inferencex,
      backend: 'sglang',
      backend_version: 'v0.5.14-rocm720-mi35x-20260706',
    }
    vi.mocked(dataset.getInferenceXDataset).mockResolvedValueOnce({
      rows: [vendor],
      source: { url: 'https://inferencex.semianalysis.com/api', retrievedAt: '2026-10-20', rowCount: 1 },
      expiresAt: Infinity,
    })

    const response = await GET(resolveRequest({ backend: 'sglang', backendVersion: '0.5.14' }))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.unavailable).toBeNull()
    expect(body.filters.backendVersion).toBe('v0.5.14')
    expect(body.configurations).toHaveLength(1)
    expect(body.configurations[0].values.backend_version).toBe('v0.5.14-rocm720-mi35x-20260706')
  })

  it('returns vendor-tagged rows for a bare catalog release pin', async () => {
    const vendor: DatasetRow = {
      ...inferencex,
      backend: 'sglang',
      backend_version: 'v0.5.14-rocm720-mi35x-20260706',
    }
    vi.mocked(dataset.getInferenceXDataset).mockResolvedValueOnce({
      rows: [vendor],
      source: { url: 'https://inferencex.semianalysis.com/api', retrievedAt: '2026-10-20', rowCount: 1 },
      expiresAt: Infinity,
    })
    const configuration = dataset.getConfigurationOptions([vendor])[0]
    const params = new URLSearchParams({
      mode: 'data',
      configuration: configuration.key,
      ...filters,
      backend: 'sglang',
      backendVersion: 'v0.5.14',
    })

    const response = await GET(new NextRequest(`https://configiq.dev/api/validation/benchmark?${params}`))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.matchedRows).toBe(1)
    expect(body.benchmarkPoints).toEqual([])
    expect(body.inferencexPoints).toHaveLength(1)
  })

  it('repairs several invalid workload dimensions against each other instead of giving up', async () => {
    vi.mocked(dataset.getBenchmarkDataset).mockResolvedValueOnce({
      rows: [
        internal,
        { ...internal, isl: 2048, osl: 512, concurrency: 16 },
      ],
      source: {
        id: 'internal', url: 'https://example.com/bench', revision: null,
        lastModified: null, loadedAt: '2026-10-20', rowCount: 2,
      },
      expiresAt: Infinity,
    })

    const response = await GET(resolveRequest({ isl: '4096', osl: '4096' }))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.unavailable).toBeNull()
    expect(body.filters.isl).toBe('2048')
    expect(body.filters.osl).toBe('512')
    expect(body.adjustments).toEqual([
      expect.objectContaining({ field: 'isl', from: '4096', to: '2048' }),
      expect.objectContaining({ field: 'osl', from: '4096', to: '512' }),
    ])
  })

  it('checks the version against the selected backend', async () => {
    vi.mocked(dataset.getBenchmarkDataset).mockResolvedValueOnce({
      rows: [
        internal,
        { ...internal, backend: 'tensorrt-llm', backend_version: '1.0.0', concurrency: 16 },
      ],
      source: {
        id: 'internal', url: 'https://example.com/bench', revision: null,
        lastModified: null, loadedAt: '2026-10-20', rowCount: 2,
      },
      expiresAt: Infinity,
    })

    const response = await GET(resolveRequest({ backend: 'tensorrt-llm', backendVersion: 'v0.24.0' }))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.filters.backendVersion).toBe('v0.24.0')
    expect(body.adjustments).toEqual([])
    expect(body.unavailable).toEqual(expect.objectContaining({ field: 'backendVersion' }))
  })

  it('falls back to the newest measured version when Settings leaves it blank', async () => {
    const response = await GET(resolveRequest({ backendVersion: '' }))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.unavailable).toBeNull()
    expect(body.filters.backendVersion).toBe('v0.24.0')
    expect(body.versionFallback).toEqual(expect.objectContaining({ value: 'v0.24.0' }))
  })

  it('offers every model and only that model’s measured accelerators', async () => {
    const response = await GET(resolveRequest({}))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.options.model).toEqual(['example/model'])
    expect(body.options.accelerator).toEqual(['H200'])
    expect(body.filters).toEqual(filters)
  })
})

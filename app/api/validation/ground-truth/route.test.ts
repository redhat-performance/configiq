import { NextRequest } from 'next/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as dataset from '@/lib/validation/ground-truth-dataset'
import type { DatasetRow, GroundTruthFilters } from '@/lib/validation/ground-truth-types'
import { GET } from './route'

vi.mock('@/lib/validation/ground-truth-dataset', async importOriginal => {
  const actual = await importOriginal<typeof import('@/lib/validation/ground-truth-dataset')>()
  return {
    ...actual,
    getGroundTruthDataset: vi.fn(),
    getInferenceXDataset: vi.fn(),
  }
})

const filters: GroundTruthFilters = {
  accelerator: 'H200',
  model: 'example/model',
  backend: 'vllm',
  backendVersion: 'v0.24.0',
  isl: '8192',
  osl: '1024',
  tp: '1',
}

const internal: DatasetRow = {
  accelerator: 'H200', model: 'example/model', version: 'GT release',
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
  return new NextRequest(`https://configiq.dev/api/validation/ground-truth?${params}`)
}

beforeEach(() => {
  vi.mocked(dataset.getGroundTruthDataset).mockResolvedValue({
    rows: [internal],
    source: {
      id: 'internal', url: 'https://example.com/gt', revision: null,
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

describe('GET /api/validation/ground-truth', () => {
  it('returns only internal GT measurements for an internal configuration', async () => {
    const response = await GET(requestFor(internal))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.matchedRows).toBe(1)
    expect(body.groundTruthPoints).toHaveLength(1)
    expect(body.inferencexPoints).toEqual([])
  })

  it('returns only InferenceX measurements for an InferenceX configuration', async () => {
    const response = await GET(requestFor(inferencex))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.matchedRows).toBe(1)
    expect(body.groundTruthPoints).toEqual([])
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

  it('keeps internal GT available when the private InferenceX dataset cannot be read', async () => {
    vi.mocked(dataset.getInferenceXDataset).mockRejectedValueOnce(new Error('Private dataset unavailable'))
    const response = await GET(requestFor(internal))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.groundTruthPoints).toHaveLength(1)
    expect(body.inferencexPoints).toEqual([])
    expect(body.inferencexError).toBe('Private dataset unavailable')
  })

  it('keeps InferenceX measurements available when internal GT cannot be read', async () => {
    vi.mocked(dataset.getGroundTruthDataset).mockRejectedValueOnce(new Error('Internal dataset unavailable'))
    const response = await GET(requestFor(inferencex))
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.groundTruthPoints).toEqual([])
    expect(body.inferencexPoints).toHaveLength(1)
    expect(body.groundTruthError).toBe('Internal dataset unavailable')
  })
})

import { afterEach, describe, expect, it, vi } from 'vitest'
import { aggregateGroundTruthRows, getFilterOptions } from '../ground-truth-dataset'
import type { DatasetRow, GroundTruthFilters } from '../ground-truth-types'

vi.mock('hyparquet', () => ({
  parquetReadObjects: vi.fn(async () => [
    {
      accelerator: 'H200',
      model: 'org/model',
      version: 'vLLM-0.10.2',
      isl: 2048,
      osl: 128,
      tp: 4,
      concurrency: 1,
      'output_tok/sec': 100,
    },
  ]),
}))

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

const EMPTY_FILTERS: GroundTruthFilters = {
  accelerator: '', model: '', backend: '', backendVersion: '', isl: '', osl: '', tp: '',
}

function row(values: DatasetRow): DatasetRow {
  return {
    accelerator: 'H200',
    model: 'org/model',
    version: 'vLLM-0.10.2',
    ...values,
  }
}

describe('ground-truth dataset integer columns', () => {
  it('exposes INT64 values returned as bigint in filter options', () => {
    const options = getFilterOptions(
      [row({ isl: 2048n, osl: 128n, tp: 4n })],
      EMPTY_FILTERS,
    )

    expect(options.isl).toEqual(['2048'])
    expect(options.osl).toEqual(['128'])
    expect(options.tp).toEqual(['4'])
  })

  it('groups rows whose concurrency is a bigint', () => {
    const points = aggregateGroundTruthRows([
      row({ concurrency: 1n, 'output_tok/sec': 100, ttft_median: 1000, tpot_median: 100, request_latency_median: 2 }),
      row({ concurrency: 2n, 'output_tok/sec': 200, ttft_median: 2000, tpot_median: 200, request_latency_median: 3 }),
    ])

    expect(points).toEqual([
      {
        concurrency: 1,
        ttftSeconds: 1,
        tpotSeconds: 0.1,
        outputTokensPerSecond: 100,
        outputTokensPerSecondPerUser: 100,
        requestLatencySeconds: 2,
        sampleCount: 1,
      },
      {
        concurrency: 2,
        ttftSeconds: 2,
        tpotSeconds: 0.2,
        outputTokensPerSecond: 200,
        outputTokensPerSecondPerUser: 100,
        requestLatencySeconds: 3,
        sampleCount: 1,
      },
    ])
  })

  it('skips a bigint outside the safe integer range instead of losing precision', () => {
    const unsafe = BigInt(Number.MAX_SAFE_INTEGER) + 2n
    expect(aggregateGroundTruthRows([
      row({ concurrency: unsafe, 'output_tok/sec': 100 }),
    ])).toEqual([])
  })
})

function jsonResponse(body: unknown) {
  return {
    ok: true,
    status: 200,
    headers: new Headers(),
    json: async () => body,
  }
}

function datasetFetch() {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    if (url.includes('/parquet?dataset=')) {
      return jsonResponse({
        parquet_files: [{ url: 'https://huggingface.co/datasets/x.parquet', filename: 'x.parquet', size: 1 }],
      })
    }
    if (url.includes('/api/datasets/')) return jsonResponse({ sha: 'abc' })
    return { ok: true, status: 200, headers: new Headers(), arrayBuffer: async () => new ArrayBuffer(0) }
  })
}

function failedFetch() {
  return vi.fn(async () => ({ ok: false, status: 404, headers: new Headers() }))
}

describe('ground-truth dataset cache', () => {
  it('serves the cached dataset to every concurrent caller when a reload fails', async () => {
    vi.resetModules()
    const now = Date.now()
    vi.spyOn(Date, 'now').mockReturnValue(now)
    vi.stubGlobal('fetch', datasetFetch())
    const mod = await import('../ground-truth-dataset')

    const first = await mod.getGroundTruthDataset()
    expect(first.rows).toHaveLength(1)

    vi.mocked(Date.now).mockReturnValue(now + 31_000)
    vi.stubGlobal('fetch', failedFetch())

    const results = await Promise.all([mod.getGroundTruthDataset(true), mod.getGroundTruthDataset(true)])
    expect(results).toEqual([first, first])
  })

  it('propagates a reload failure when no cached dataset exists', async () => {
    vi.resetModules()
    vi.stubGlobal('fetch', failedFetch())
    const mod = await import('../ground-truth-dataset')

    await expect(mod.getGroundTruthDataset()).rejects.toThrow('Hugging Face')
  })
})

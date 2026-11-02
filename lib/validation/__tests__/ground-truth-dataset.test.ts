import { describe, expect, it } from 'vitest'
import { aggregateGroundTruthRows, getFilterOptions } from '../ground-truth-dataset'
import type { DatasetRow, GroundTruthFilters } from '../ground-truth-types'

const EMPTY_FILTERS: GroundTruthFilters = {
  accelerator: '', model: '', version: '', isl: '', osl: '', tp: '',
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

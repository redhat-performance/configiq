import { describe, expect, it } from 'vitest'

import { comparisonCsv } from './csv'
import { COMPARISON_SCHEMA_VERSION, type ComparisonEntry } from './types'

function entry(overrides: Partial<ComparisonEntry> = {}): ComparisonEntry {
  return {
    id: 'entry-1',
    savedAt: '2026-11-21T00:00:00.000Z',
    name: 'Llama · H200',
    link: '/predict?model=llama&system=h200',
    model: 'meta-llama/Llama-3.1-70B',
    system: 'h200',
    systemLabel: 'H200',
    simulator: 'aisimulate',
    engineVersion: '0.14.0',
    backend: 'vllm',
    backendVersion: '0.9.0',
    mode: 'agg',
    workload: {
      isl: 2048,
      osl: 128,
      concurrency: 32,
      weightPrecision: 'FP8',
      kvPrecision: 'FP16',
    },
    topology: { tp: 8, pp: 1, replicas: 1, prefill: null, decode: null },
    figures: {
      gpus: 8,
      ttftMs: 180,
      tpotMs: 22.5,
      requestLatencyMs: 3060,
      tokensPerSecond: 1600,
      tokensPerSecondPerGpu: 200,
      tokensPerSecondPerUser: 50,
      weightMemoryGb: 140,
      kvCachePerRequestMb: 2500,
      kvCacheUsedGb: 80,
      totalVramGb: 640,
      cloudMonthlyUsd: 14_600,
      cloudFiveYearUsd: 876_000,
      selfHostedMonthlyUsd: 4_000,
      selfHostedFiveYearUsd: 240_000,
    },
    ...overrides,
  }
}

function parse(text: string): string[][] {
  return text.trimEnd().split('\n').map(line => line.split(','))
}

describe('comparisonCsv', () => {
  it('writes one metric per row and one result per column', () => {
    const csv = comparisonCsv({
      schemaVersion: COMPARISON_SCHEMA_VERSION,
      entries: [entry(), entry({ id: 'b', name: 'Llama · H100', link: '/predict?n=2', figures: { ...entry().figures, gpus: 16 } })],
    })
    const rows = parse(csv)

    expect(rows[0]).toEqual(['Metric', 'Llama · H200', 'Llama · H100'])
    const gpus = rows.find(row => row[0] === 'GPUs')
    expect(gpus).toEqual(['GPUs', '8', '16'])
  })

  it('writes raw figures, not the table formatting', () => {
    const csv = comparisonCsv({ schemaVersion: COMPARISON_SCHEMA_VERSION, entries: [entry()] })
    const rows = parse(csv)

    expect(rows.find(row => row[0] === 'Cloud cost / month')?.[1]).toBe('14600')
    expect(rows.find(row => row[0] === 'TTFT (ms)')?.[1]).toBe('180')
  })

  it('leaves an unreported metric empty', () => {
    const csv = comparisonCsv({
      schemaVersion: COMPARISON_SCHEMA_VERSION,
      entries: [entry({ figures: { ...entry().figures, ttftMs: null } })],
    })

    expect(parse(csv).find(row => row[0] === 'TTFT (ms)')?.[1]).toBe('')
  })

  it('quotes a name that contains a comma', () => {
    const csv = comparisonCsv({
      schemaVersion: COMPARISON_SCHEMA_VERSION,
      entries: [entry({ name: 'Llama, H200' })],
    })

    expect(csv.split('\n')[0]).toBe('Metric,"Llama, H200"')
  })

  it('neutralises a name a spreadsheet would evaluate as a formula', () => {
    const csv = comparisonCsv({
      schemaVersion: COMPARISON_SCHEMA_VERSION,
      entries: [entry({ name: '=HYPERLINK("http://example.test")' })],
    })

    expect(csv.split('\n')[0]).toBe('Metric,"\'=HYPERLINK(""http://example.test"")"')
  })

  it('neutralises a link that starts with a formula character', () => {
    const csv = comparisonCsv({
      schemaVersion: COMPARISON_SCHEMA_VERSION,
      entries: [entry({ link: '@SUM(1)' })],
    })

    expect(csv.split('\n')[2]).toBe("Link,'@SUM(1)")
  })

  it('writes a negative figure as a number, not as text', () => {
    const csv = comparisonCsv({
      schemaVersion: COMPARISON_SCHEMA_VERSION,
      entries: [entry({ figures: { ...entry().figures, cloudMonthlyUsd: -1200 } })],
    })

    expect(parse(csv).find(row => row[0] === 'Cloud cost / month')?.[1]).toBe('-1200')
  })

  it('ends with a newline', () => {
    expect(comparisonCsv({ schemaVersion: COMPARISON_SCHEMA_VERSION, entries: [] }).endsWith('\n')).toBe(true)
  })
})

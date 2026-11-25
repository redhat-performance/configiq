import { describe, expect, it } from 'vitest'

import { backfillComparisonCosts, mergeCostFigures, missingCost, type ComparisonCostSources } from './backfill'
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
    workload: { isl: 2048, osl: 128, concurrency: 32, weightPrecision: 'FP8', kvPrecision: 'FP16' },
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
      cloudMonthlyUsd: null,
      cloudFiveYearUsd: null,
      selfHostedMonthlyUsd: null,
      selfHostedFiveYearUsd: null,
    },
    ...overrides,
  }
}

function sources(overrides: Partial<ComparisonCostSources> = {}): ComparisonCostSources {
  return {
    gpuCloudRates: new Map([['h200', {
      'aws.us-east-1': {
        on_demand: 2.5,
        reserved_1yr: null,
        reserved_3yr: null,
        spot_median: null,
        rate_basis: 'gpu_hour',
        gpus_per_instance: 1,
      },
    }]]),
    gpuHardwareCosts: new Map([['h200', {
      new_usd: 20_000,
      new_usd_low: null,
      new_usd_high: null,
      indicative: true,
      source_label: null,
      source_url: null,
      source_date: null,
    }]]),
    preferredProvider: null,
    ...overrides,
  }
}

function doc(entries: ComparisonEntry[]) {
  return { schemaVersion: COMPARISON_SCHEMA_VERSION, entries }
}

describe('missingCost', () => {
  it('reports a snapshot with any money figure still missing', () => {
    expect(missingCost(entry())).toBe(true)
    expect(missingCost(entry({ figures: { ...entry().figures, cloudMonthlyUsd: 1 } }))).toBe(true)
    expect(missingCost(entry({ figures: { ...entry().figures, cloudMonthlyUsd: 0, cloudFiveYearUsd: 0,
      selfHostedMonthlyUsd: 0, selfHostedFiveYearUsd: 0 } }))).toBe(false)
  })
})

describe('backfillComparisonCosts', () => {
  it('fills the four figures from the published rate and hardware price', () => {
    const result = backfillComparisonCosts(doc([entry()]), sources())

    expect(result.filled).toBe(1)
    const figures = result.document.entries[0].figures
    expect(figures.cloudMonthlyUsd).toBe(8 * 2.5 * 730)
    expect(figures.cloudFiveYearUsd).toBe(8 * 2.5 * 730 * 60)
    expect(figures.selfHostedMonthlyUsd).toBeCloseTo((8 * 20_000) / 60)
    expect(figures.selfHostedFiveYearUsd).toBe(160_000)
    // Measured figures are the run's own and stay untouched.
    expect(figures.ttftMs).toBe(180)
  })

  it('keeps figures the result page already computed', () => {
    const saved = entry({ figures: { ...entry().figures,
      cloudMonthlyUsd: 1_000,
      cloudFiveYearUsd: 60_000,
      selfHostedMonthlyUsd: 500,
      selfHostedFiveYearUsd: 30_000,
    } })

    const result = backfillComparisonCosts(doc([saved]), sources())

    expect(result.filled).toBe(0)
    expect(result.document.entries[0]).toBe(saved)
  })

  it('fills only the gaps of a partly costed snapshot', () => {
    const saved = entry({ figures: { ...entry().figures, cloudMonthlyUsd: 1_000 } })

    const figures = backfillComparisonCosts(doc([saved]), sources()).document.entries[0].figures

    expect(figures.cloudMonthlyUsd).toBe(1_000)
    expect(figures.cloudFiveYearUsd).toBe(60_000)
    expect(figures.selfHostedMonthlyUsd).toBeCloseTo((8 * 20_000) / 60)
  })

  it('derives a missing monthly cloud cost from the saved five-year total', () => {
    const saved = entry({ figures: { ...entry().figures, cloudFiveYearUsd: 6_000 } })

    const figures = backfillComparisonCosts(doc([saved]), sources()).document.entries[0].figures

    // The pair must agree, so the catalogue rate does not override the saved
    // five-year figure even though a rate is published.
    expect(figures.cloudMonthlyUsd).toBe(6_000 / 60)
    expect(figures.cloudFiveYearUsd).toBe(6_000)
  })

  it('leaves a figure null when no catalogue price is published', () => {
    const result = backfillComparisonCosts(doc([entry()]), sources({
      gpuCloudRates: new Map(),
      gpuHardwareCosts: new Map(),
    }))

    expect(result.filled).toBe(0)
    expect(result.document.entries[0].figures.cloudMonthlyUsd).toBeNull()
    expect(result.document.entries[0].figures.selfHostedMonthlyUsd).toBeNull()
  })

  it('resolves the rate the way the result pages do', () => {
    const rates = {
      'aws.us-east-1': { on_demand: 4, reserved_1yr: null, reserved_3yr: null, spot_median: null,
        rate_basis: 'gpu_hour' as const, gpus_per_instance: 1 },
      'gcp.us-central1': { on_demand: 2, reserved_1yr: null, reserved_3yr: null, spot_median: null,
        rate_basis: 'gpu_hour' as const, gpus_per_instance: 1 },
    }
    const result = backfillComparisonCosts(doc([entry()]), sources({
      gpuCloudRates: new Map([['h200', rates]]),
      preferredProvider: 'aws.us-east-1',
    }))

    expect(result.document.entries[0].figures.cloudMonthlyUsd).toBe(8 * 4 * 730)
  })

  it('leaves the document alone when nothing is missing', () => {
    const document = doc([])

    const result = backfillComparisonCosts(document, sources())

    expect(result.filled).toBe(0)
    expect(result.document).toBe(document)
  })
})

describe('mergeCostFigures', () => {
  it('copies completed figures onto the set that is in storage now', () => {
    const stored = entry({ id: 'a' })
    const filled = entry({ id: 'a', figures: { ...entry().figures, cloudMonthlyUsd: 14_600 } })

    const merged = mergeCostFigures(doc([stored]), doc([filled]))

    expect(merged.entries[0].figures.cloudMonthlyUsd).toBe(14_600)
    // The current entry keeps its own identity and measured figures.
    expect(merged.entries[0].link).toBe(stored.link)
    expect(merged.entries[0].figures.ttftMs).toBe(stored.figures.ttftMs)
  })

  it('keeps a result another tab saved after this page loaded', () => {
    const added = entry({ id: 'b', link: '/predict?n=2' })

    const merged = mergeCostFigures(doc([entry({ id: 'a' }), added]), doc([entry({ id: 'a' })]))

    expect(merged.entries.map(saved => saved.id)).toEqual(['a', 'b'])
    expect(merged.entries[1]).toBe(added)
  })

  it('does not restore a result another tab removed', () => {
    const merged = mergeCostFigures(
      doc([entry({ id: 'b' })]),
      doc([entry({ id: 'a' }), entry({ id: 'b' })]),
    )

    expect(merged.entries.map(saved => saved.id)).toEqual(['b'])
  })

  it('leaves the document alone when the current set has nothing to complete', () => {
    const current = doc([entry({ id: 'a' })])

    expect(mergeCostFigures(current, doc([]))).toBe(current)
  })
})

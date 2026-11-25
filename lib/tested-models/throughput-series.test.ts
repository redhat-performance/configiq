import { describe, expect, it } from 'vitest'
import {
  comparePredictionToMeasured,
  interpolateThroughput,
  listWorkloads,
  matchPredictionWorkload,
  selectDefaultWorkload,
  selectSeries,
  workloadKey,
  type PredictionSnapshot,
  type ThroughputCurve,
  type ThroughputPoint,
} from './throughput-series'

function point(concurrency: number, throughput: number, observations = 1): ThroughputPoint {
  return {
    concurrency,
    measuredConcurrency: concurrency,
    throughput,
    throughputMin: throughput,
    throughputMax: throughput,
    latency: null,
    observations,
  }
}

function curve(
  overrides: Partial<ThroughputCurve> & Pick<ThroughputCurve, 'points'>,
): ThroughputCurve {
  return {
    isl: 1000,
    osl: 1000,
    turns: 1,
    sweepId: 'sweep-a',
    backend: 'vllm',
    backendVersions: ['vLLM-0.24.0'],
    configLabel: 'vLLM-0.24.0',
    ...overrides,
    tp: overrides.tp ?? 1,
  }
}

const prediction: PredictionSnapshot = {
  isl: 1000,
  osl: 1000,
  concurrency: 20,
  tp: 1,
  throughput: 1300,
  status: 'within_validated_range',
  confidence: 0.9,
  backend: 'vllm',
  backendVersion: 'vLLM-0.24.0',
  servingMode: 'agg',
}

describe('listWorkloads', () => {
  it('collapses curves into one option per workload with coverage', () => {
    const curves = [
      curve({ tp: 1, points: [point(1, 100), point(10, 900)] }),
      curve({ tp: 2, points: [point(1, 200)] }),
      curve({ isl: 8000, osl: 1000, tp: 1, points: [point(1, 50)] }),
    ]

    const options = listWorkloads(curves)

    expect(options.map(option => [option.isl, option.osl, option.turns])).toEqual([
      [1000, 1000, 1],
      [8000, 1000, 1],
    ])
    expect(options[0].levels).toBe(2)
    expect(options[0].observations).toBe(3)
    expect(options[0].tps).toEqual([1, 2])
  })

  it('sorts single-turn workloads before multi-turn ones', () => {
    const curves = [
      curve({ isl: 128, osl: 128, turns: 5, points: [point(1, 10)] }),
      curve({ isl: 1000, osl: 1000, turns: 1, points: [point(1, 100)] }),
    ]

    expect(listWorkloads(curves).map(option => option.turns)).toEqual([1, 5])
  })
})

describe('selectDefaultWorkload', () => {
  it('prefers a single-turn workload with the most concurrency levels', () => {
    const curves = [
      curve({ isl: 128, osl: 128, turns: 5, points: [point(1, 10), point(2, 20), point(4, 30)] }),
      curve({ isl: 1000, osl: 1000, turns: 1, points: [point(1, 100), point(10, 900)] }),
    ]

    expect(selectDefaultWorkload(curves)?.key).toBe(workloadKey({ isl: 1000, osl: 1000, turns: 1 }))
  })

  it('returns null when the payload has no curves', () => {
    expect(selectDefaultWorkload([])).toBeNull()
  })
})

describe('selectSeries', () => {
  it('draws one line per TP from the sweep that covers the most TPs', () => {
    const curves = [
      curve({ tp: 1, sweepId: 'primary', points: [point(1, 100), point(10, 900)] }),
      curve({ tp: 2, sweepId: 'primary', points: [point(1, 200), point(10, 1800)] }),
      // A second sweep with more levels but only one TP.
      curve({ tp: 1, sweepId: 'other', points: [point(1, 110), point(10, 950), point(20, 990)] }),
    ]

    const series = selectSeries(curves, { isl: 1000, osl: 1000, turns: 1 })

    expect(series.primarySweepId).toBe('primary')
    expect(series.lines.map(line => [line.tp, line.curve.sweepId])).toEqual([
      [1, 'primary'],
      [2, 'primary'],
    ])
    expect(series.mixedConfigurations).toEqual([])
  })

  it('keeps a TP on the primary sweep even when its own sweep has more levels', () => {
    const curves = [
      curve({ tp: 1, sweepId: 'primary', points: [point(1, 100), point(10, 900)] }),
      curve({ tp: 2, sweepId: 'primary', points: [point(1, 200)] }),
      curve({ tp: 2, sweepId: 'tp2-only', points: [point(1, 210), point(10, 1500)] }),
    ]

    const series = selectSeries(curves, { isl: 1000, osl: 1000, turns: 1 })

    expect(series.lines.find(line => line.tp === 2)?.curve.sweepId).toBe('primary')
    expect(series.mixedConfigurations).toEqual([])
  })

  it('reports a TP that only exists in another sweep', () => {
    const curves = [
      curve({ tp: 1, sweepId: 'primary', points: [point(1, 100)] }),
      curve({ tp: 2, sweepId: 'tp2-only', points: [point(1, 210), point(10, 1500)] }),
    ]

    const series = selectSeries(curves, { isl: 1000, osl: 1000, turns: 1 })

    // No sweep covers both TPs, so the richer sweep sets the reference and the
    // other TP is reported as a mixed configuration.
    expect(series.primarySweepId).toBe('tp2-only')
    expect(series.mixedConfigurations).toEqual([1])
  })

  it('returns an empty series for an unmeasured workload', () => {
    const series = selectSeries(
      [curve({ tp: 1, points: [point(1, 100)] })],
      { isl: 8000, osl: 1000, turns: 1 },
    )

    expect(series.lines).toEqual([])
    expect(series.primarySweepId).toBeNull()
  })
})

describe('matchPredictionWorkload', () => {
  it('matches only a single-turn workload with the same ISL and OSL', () => {
    const curves = [
      curve({ isl: 1000, osl: 1000, turns: 5, points: [point(1, 10)] }),
      curve({ isl: 1000, osl: 1000, turns: 1, points: [point(1, 100)] }),
    ]

    expect(matchPredictionWorkload(curves, { isl: 1000, osl: 1000 })?.turns).toBe(1)
    expect(matchPredictionWorkload(curves, { isl: 1000, osl: 2048 })).toBeNull()
  })
})

describe('interpolateThroughput', () => {
  it('interpolates between the bracketing levels', () => {
    const result = interpolateThroughput(
      curve({ tp: 1, points: [point(1, 100), point(10, 1000)] }),
      5.5,
    )

    expect(result?.measured).toBe(550)
    expect(result?.exact).toBe(false)
    expect(result?.clamped).toBeNull()
  })

  it('reports an exact measured level without calling it clamped', () => {
    const single = curve({ tp: 1, points: [point(10, 500), point(20, 900)] })

    expect(interpolateThroughput(single, 10)).toMatchObject({ measured: 500, exact: true, clamped: null })
    expect(interpolateThroughput(single, 20)).toMatchObject({ measured: 900, exact: true, clamped: null })
    expect(interpolateThroughput(single, 15)).toMatchObject({ measured: 700, exact: false, clamped: null })
  })

  it('clamps below and above the measured range', () => {
    const single = curve({ tp: 1, points: [point(10, 500), point(20, 900)] })

    expect(interpolateThroughput(single, 1)).toMatchObject({ measured: 500, exact: false, clamped: 'below' })
    expect(interpolateThroughput(single, 40)).toMatchObject({ measured: 900, exact: false, clamped: 'above' })
  })

  it('returns null for a curve without points', () => {
    expect(interpolateThroughput(curve({ tp: 1, points: [] }), 10)).toBeNull()
  })
})

describe('comparePredictionToMeasured', () => {
  it('places a prediction on the selected TP line and reports the delta', () => {
    const series = selectSeries(
      [curve({ tp: 1, points: [point(1, 100), point(10, 1000), point(40, 1400)] })],
      { isl: 1000, osl: 1000, turns: 1 },
    )

    const comparison = comparePredictionToMeasured(series, prediction)

    expect(comparison.tpMeasured).toBe(true)
    expect(comparison.at?.measured).toBe(1133.3333333333333)
    expect(comparison.deltaPct).toBeCloseTo(14.7, 1)
    expect(comparison.checks.filter(check => check.ok).map(check => check.id)).toContain('concurrency')
    expect(comparison.checks.find(check => check.id === 'delta')?.ok).toBe(true)
  })

  it('explains a missing TP instead of borrowing another line', () => {
    const series = selectSeries(
      [curve({ tp: 2, points: [point(1, 200), point(10, 1800)] })],
      { isl: 1000, osl: 1000, turns: 1 },
    )

    const comparison = comparePredictionToMeasured(series, prediction)

    expect(comparison.tpMeasured).toBe(false)
    expect(comparison.curve).toBeNull()
    expect(comparison.checks[0]).toMatchObject({ id: 'tp', ok: false })
    expect(comparison.checks[0].detail).toContain('TP 2')
  })

  it('flags a prediction above the observed maximum', () => {
    const series = selectSeries(
      [curve({ tp: 1, points: [point(1, 100), point(40, 900)] })],
      { isl: 1000, osl: 1000, turns: 1 },
    )

    const comparison = comparePredictionToMeasured(series, { ...prediction, throughput: 5000 })

    expect(comparison.checks.find(check => check.id === 'throughput')?.ok).toBe(false)
    expect(comparison.checks.find(check => check.id === 'delta')?.ok).toBe(false)
  })

  it('flags a concurrency outside the measured levels', () => {
    const series = selectSeries(
      [curve({ tp: 1, points: [point(1, 100), point(40, 900)] })],
      { isl: 1000, osl: 1000, turns: 1 },
    )

    const comparison = comparePredictionToMeasured(series, { ...prediction, concurrency: 200 })

    expect(comparison.checks.find(check => check.id === 'concurrency')?.ok).toBe(false)
    expect(comparison.at?.clamped).toBe('above')
  })

  it('flags a backend family that was never measured', () => {
    const series = selectSeries(
      [curve({ tp: 1, points: [point(1, 100), point(40, 900)] })],
      { isl: 1000, osl: 1000, turns: 1 },
    )

    const comparison = comparePredictionToMeasured(series, {
      ...prediction,
      backend: 'sglang',
      backendVersion: '0.5.18',
    })

    expect(comparison.checks.find(check => check.id === 'version')?.ok).toBe(false)
  })

  it('flags a disaggregated prediction against aggregate measurements', () => {
    const series = selectSeries(
      [curve({ tp: 1, points: [point(1, 100), point(40, 900)] })],
      { isl: 1000, osl: 1000, turns: 1 },
    )

    const comparison = comparePredictionToMeasured(series, { ...prediction, servingMode: 'disagg' })

    expect(comparison.checks.find(check => check.id === 'serving-mode')?.ok).toBe(false)
  })
})

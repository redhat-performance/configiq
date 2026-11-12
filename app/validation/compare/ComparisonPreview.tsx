'use client'

import { useEffect, useRef, useState } from 'react'
import {
  Chart,
  ChartAxis,
  ChartGroup,
  ChartLine,
  ChartScatter,
  ChartVoronoiContainer,
} from '@patternfly/react-charts/victory'
import type { BenchmarkPoint } from '@/lib/validation/benchmark-types'
import styles from './validation-compare.module.css'

export interface EstimatePoint {
  concurrency: number
  ttftSeconds: number | null
  tpotSeconds: number | null
  outputTokensPerSecond: number | null
  outputTokensPerSecondPerUser: number | null
  requestLatencySeconds: number | null
}

/** One simulator's predictions across the measured concurrency levels. */
export interface PredictionSeries {
  simulator: string
  label: string
  /** Engine release the response reported, when it named one. */
  version: string | null
  /** Whether the run used the benchmark's measured engine release. */
  comparable: boolean
  /** True when the user changed prediction assumptions from the measured run. */
  overridden: boolean
  points: EstimatePoint[]
}

type MetricKey = Exclude<keyof EstimatePoint, 'concurrency'>

interface MetricDefinition {
  key: MetricKey
  title: string
  unit: string
  description: string
  digits: number
}

const metrics: MetricDefinition[] = [
  { key: 'ttftSeconds', title: 'Time to first token', unit: 'seconds', description: 'Measured median vs. predicted mean TTFT', digits: 3 },
  { key: 'tpotSeconds', title: 'Time per output token', unit: 'seconds', description: 'Measured median vs. predicted mean TPOT', digits: 4 },
  { key: 'outputTokensPerSecond', title: 'Output throughput', unit: 'tokens/s', description: 'Measured aggregate vs. predicted aggregate throughput', digits: 1 },
  { key: 'outputTokensPerSecondPerUser', title: 'Throughput per user', unit: 'tokens/s/user', description: 'Measured vs. predicted throughput per user', digits: 1 },
  { key: 'requestLatencySeconds', title: 'Request latency', unit: 'seconds', description: 'Measured median vs. predicted request latency', digits: 2 },
]

export const BENCHMARK_COLORS = {
  redHat: '#c9190b',
  inferencex: '#ec7a08',
} as const

/** Brand colors for the simulators the catalog can return. */
export const SIMULATOR_COLORS: Record<string, string> = {
  // NVIDIA green
  aisimulate: '#76b900',
  // IBM blue
  blis: '#0f62fe',
}

/** Stable line colors, used for simulators without a brand color. */
export const PREDICTION_COLORS = ['#009596', '#5752d1', '#3e8635', '#a18f00'] as const

/** A simulator's brand color, falling back to the shared palette by position. */
export function predictionColor(simulator: string, index: number): string {
  return SIMULATOR_COLORS[simulator] ?? PREDICTION_COLORS[index % PREDICTION_COLORS.length]
}

function formatValue(value: number | null, digits: number): string {
  return value === null ? '—' : value.toLocaleString('en-US', { maximumFractionDigits: digits })
}

/**
 * Track the rendered chart width so Victory never draws a 600px canvas into a
 * narrower card. The fallback width is used until the first measurement.
 */
function useChartWidth() {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(600)
  useEffect(() => {
    const element = ref.current
    if (!element) return
    const measure = () => {
      const next = Math.floor(element.clientWidth)
      if (next > 0) setWidth(next)
    }
    measure()
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', measure)
      return () => window.removeEventListener('resize', measure)
    }
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  return { ref, width }
}

function difference(value: number | null, baseline: number | null): string {
  if (value === null || baseline === null || baseline === 0) return '—'
  const delta = ((value - baseline) / Math.abs(baseline)) * 100
  return `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}%`
}

interface SeriesDatum {
  x: number
  y: number
  series: string
}

function toData(points: Array<{ concurrency: number } & Partial<Record<MetricKey, number | null>>>, key: MetricKey, label: string): SeriesDatum[] {
  return points.flatMap(point => {
    const value = point[key]
    return value == null || !Number.isFinite(value) ? [] : [{ x: point.concurrency, y: value, series: label }]
  })
}

/** Horizontal space Victory reserves around the x scale (left 76, right 20). */
const CHART_HORIZONTAL_PADDING = 96
/** Approximate width of one 11.5px Red Hat Text digit, in pixels. */
const TICK_CHAR_WIDTH = 6.6
/** Space to leave between adjacent tick labels, in pixels. */
const TICK_LABEL_PADDING = 10
/** Keep the first label clear of the y-axis, in pixels. */
const CONCURRENCY_EDGE_GAP = 6

/**
 * Choose concurrency ticks that leave room for their labels.
 *
 * Concurrency is a linear axis, so low measured levels (1, 5, 10, ...) sit a
 * fraction of a pixel apart while high levels are far apart. A fixed tick count
 * cannot prevent their labels colliding, so measure the plot and drop a tick
 * only when its label would touch the previous one. The maximum level is always
 * labelled.
 */
export function selectConcurrencyTicks(values: number[], chartWidth: number): number[] {
  if (values.length <= 2) return values
  const domainMax = Math.max(...values) * 1.04
  const plotWidth = Math.max(80, chartWidth - CHART_HORIZONTAL_PADDING)
  const scale = plotWidth / domainMax
  const labelWidth = (value: number) => value.toLocaleString('en-US').length * TICK_CHAR_WIDTH
  const fitsAfter = (previous: number, value: number) =>
    (value - previous) * scale >= (labelWidth(previous) + labelWidth(value)) / 2 + TICK_LABEL_PADDING
  const ticks: number[] = []
  for (const value of values) {
    if (ticks.length === 0) {
      if (value * scale >= CONCURRENCY_EDGE_GAP) ticks.push(value)
    } else if (fitsAfter(ticks[ticks.length - 1], value)) {
      ticks.push(value)
    }
  }
  const last = values[values.length - 1]
  if (ticks[ticks.length - 1] !== last) {
    if (ticks.length > 0 && !fitsAfter(ticks[ticks.length - 1], last)) ticks.pop()
    ticks.push(last)
  }
  return ticks
}

function MetricTable({
  metric,
  benchmark,
  inferencex,
  series,
}: {
  metric: MetricDefinition
  benchmark: BenchmarkPoint[]
  inferencex: BenchmarkPoint[]
  series: PredictionSeries[]
}) {
  const concurrencyValues = [...new Set([
    ...benchmark.map(point => point.concurrency),
    ...inferencex.map(point => point.concurrency),
    ...series.flatMap(entry => entry.points.map(point => point.concurrency)),
  ])].sort((left, right) => left - right)
  const benchmarkBy = new Map(benchmark.map(point => [point.concurrency, point]))
  const inferencexBy = new Map(inferencex.map(point => [point.concurrency, point]))
  const seriesBy = series.map(entry => ({ entry, points: new Map(entry.points.map(point => [point.concurrency, point])) }))
  const hasInferencex = inferencex.some(point => point[metric.key] !== null)

  if (concurrencyValues.length === 0) {
    return <p className={styles.tableEmpty}>No measured concurrency levels for this metric.</p>
  }

  return (
    <div className={styles.tableScroll}>
      <table className={styles.metricTable}>
        <caption>
          {metric.title} ({metric.unit}) at each measured concurrency level
        </caption>
        <thead>
          <tr>
            <th scope="col">Concurrency</th>
            <th scope="col">Red Hat</th>
            {hasInferencex && <th scope="col">InferenceX</th>}
            {seriesBy.map(({ entry }) => (
              <th scope="col" key={entry.simulator}>{entry.label}</th>
            ))}
            {seriesBy.map(({ entry }) => (
              <th scope="col" key={`${entry.simulator}-delta`}>Δ {entry.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {concurrencyValues.map(concurrency => {
            const redHat = benchmarkBy.get(concurrency)
            const external = inferencexBy.get(concurrency)
            const redHatValue = redHat ? redHat[metric.key] : null
            const inferencexValue = external ? external[metric.key] : null
            const baseline = redHatValue ?? inferencexValue
            return (
              <tr key={concurrency}>
                <th scope="row">{concurrency.toLocaleString('en-US')}</th>
                <td>{formatValue(redHatValue, metric.digits)}</td>
                {hasInferencex && <td>{formatValue(inferencexValue, metric.digits)}</td>}
                {seriesBy.map(({ entry, points }) => (
                  <td key={entry.simulator}>{formatValue(points.get(concurrency)?.[metric.key] ?? null, metric.digits)}</td>
                ))}
                {seriesBy.map(({ entry, points }) => {
                  const value = points.get(concurrency)?.[metric.key] ?? null
                  return (
                    <td key={`${entry.simulator}-delta`}>
                      {entry.comparable ? difference(value, baseline) : 'not comparable'}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function ChartCard({
  metric,
  benchmark,
  inferencex,
  series,
  mounted,
}: {
  metric: MetricDefinition
  benchmark: BenchmarkPoint[]
  inferencex: BenchmarkPoint[]
  series: PredictionSeries[]
  mounted: boolean
}) {
  const [flipped, setFlipped] = useState(false)
  const frontRef = useRef<HTMLDivElement>(null)
  const backRef = useRef<HTMLDivElement>(null)
  const { ref: canvasRef, width: chartWidth } = useChartWidth()
  const tableId = `benchmark-metric-table-${metric.key}`
  useEffect(() => {
    // Only the visible face may receive focus or be read by assistive tech.
    if (frontRef.current) frontRef.current.inert = flipped
    if (backRef.current) backRef.current.inert = !flipped
  }, [flipped])
  const allValues = [
    ...benchmark.map(point => point[metric.key]),
    ...inferencex.map(point => point[metric.key]),
    ...series.flatMap(entry => entry.points.map(point => point[metric.key])),
  ].filter((value): value is number => value !== null)
  const max = allValues.length ? Math.max(...allValues) : 1
  const concurrencyValues = [...new Set([
    ...benchmark.map(point => point.concurrency),
    ...inferencex.map(point => point.concurrency),
    ...series.flatMap(entry => entry.points.map(point => point.concurrency)),
  ])].sort((left, right) => left - right)
  const tickValues = selectConcurrencyTicks(concurrencyValues, Math.max(240, chartWidth))
  const benchmarkData = toData(benchmark, metric.key, 'Red Hat benchmark')
  const inferencexData = toData(inferencex, metric.key, 'InferenceX benchmark')
  const predictionData = series.map(entry => ({ entry, data: toData(entry.points, metric.key, entry.label) }))
  const seriesNames = [
    ...(benchmarkData.length > 0 ? ['red-hat-line'] : []),
    ...(inferencexData.length > 0 ? ['inferencex-line'] : []),
    ...predictionData.filter(entry => entry.data.length > 0).map(entry => `${entry.entry.simulator}-line`),
  ]

  return (
    <div
      className={`${styles.flipCard} ${flipped ? styles.flipped : ''}`}
      data-flip-card={metric.key}
      onClick={event => {
        const target = event.target as HTMLElement
        // Keep tables and their controls interactive without flipping back.
        if (target.closest('table, button, a, input, select, textarea')) return
        setFlipped(value => !value)
      }}
    >
      <div ref={frontRef} className={`${styles.flipFace} ${styles.flipFront}`} aria-hidden={flipped}>
        <div className={styles.chartHead}>
          <div>
            <h3>{metric.title}</h3>
            <p>{metric.description}</p>
          </div>
          <span className={styles.unit}>{metric.unit}</span>
        </div>
        <div className={styles.chartCanvas} ref={canvasRef}>
          {mounted && concurrencyValues.length > 0 ? (
            <Chart
              ariaTitle={`${metric.title} by concurrency`}
              ariaDesc={`${['Red Hat benchmark', inferencexData.length > 0 ? 'InferenceX benchmark' : null, ...series.map(entry => entry.label)].filter(Boolean).join(', ')} ${metric.title.toLowerCase()} compared at measured concurrency levels`}
              domain={{ x: [0, Math.max(...concurrencyValues) * 1.04], y: [0, max * 1.14] }}
              height={278}
              width={Math.max(240, chartWidth)}
              padding={{ top: 18, right: 20, bottom: 54, left: 76 }}
              containerComponent={
                <ChartVoronoiContainer
                  radius={28}
                  mouseFollowTooltips={false}
                  voronoiBlacklist={seriesNames}
                  labels={({ datum }: { datum: { series: string; x: number; y: number | null } }) =>
                    `${datum.series} · concurrency ${datum.x}: ${formatValue(Number(datum.y), metric.digits)} ${metric.unit}`}
                  constrainToVisibleArea
                />
              }
            >
              <ChartAxis
                label="Concurrency"
                tickValues={tickValues}
                showGrid
                style={{
                  axis: { stroke: '#8a8d90', strokeWidth: 1 },
                  axisLabel: { padding: 34, fontSize: 12, fontWeight: 500, fill: '#3c3f42' },
                  grid: { stroke: '#f0f0f0', strokeWidth: 1 },
                  ticks: { stroke: '#b8bbbe' },
                  tickLabels: { fontSize: 11.5, fill: '#3c3f42' },
                }}
              />
              <ChartAxis
                dependentAxis
                tickFormat={(value: number) => Number(value).toLocaleString('en-US', { maximumFractionDigits: metric.digits })}
                showGrid
                style={{
                  axis: { stroke: '#8a8d90', strokeWidth: 1 },
                  grid: { stroke: '#e4e4e4', strokeWidth: 1 },
                  ticks: { stroke: '#b8bbbe' },
                  tickLabels: { fontSize: 11.5, fill: '#3c3f42' },
                }}
              />
              <ChartGroup>
                {benchmarkData.length > 0 && <ChartLine
                  data={benchmarkData}
                  name="red-hat-line"
                  style={{ data: { stroke: BENCHMARK_COLORS.redHat, strokeWidth: 3, strokeLinecap: 'round', strokeLinejoin: 'round' } }}
                />}
                {inferencexData.length > 0 && <ChartLine
                  data={inferencexData}
                  name="inferencex-line"
                  style={{ data: { stroke: BENCHMARK_COLORS.inferencex, strokeWidth: 3, strokeLinecap: 'round', strokeLinejoin: 'round' } }}
                />}
                {predictionData.map((entry, index) => entry.data.length > 0 && (
                  <ChartLine
                    key={entry.entry.simulator}
                    data={entry.data}
                    name={`${entry.entry.simulator}-line`}
                    style={{ data: { stroke: predictionColor(entry.entry.simulator, index), strokeWidth: 3, strokeLinecap: 'round', strokeLinejoin: 'round' } }}
                  />
                ))}
              </ChartGroup>
              <ChartGroup>
                {benchmarkData.length > 0 && <ChartScatter
                  data={benchmarkData}
                  name="red-hat-points"
                  size={({ active }) => active ? 6 : 4}
                  style={{ data: { fill: '#ffffff', stroke: BENCHMARK_COLORS.redHat, strokeWidth: 2 } }}
                />}
                {inferencexData.length > 0 && <ChartScatter
                  data={inferencexData}
                  name="inferencex-points"
                  size={({ active }) => active ? 6 : 4}
                  style={{ data: { fill: '#ffffff', stroke: BENCHMARK_COLORS.inferencex, strokeWidth: 2 } }}
                />}
                {predictionData.map((entry, index) => entry.data.length > 0 && (
                  <ChartScatter
                    key={entry.entry.simulator}
                    data={entry.data}
                    name={`${entry.entry.simulator}-points`}
                    size={({ active }) => active ? 6 : 4}
                    style={{ data: { fill: '#ffffff', stroke: predictionColor(entry.entry.simulator, index), strokeWidth: 2 } }}
                  />
                ))}
              </ChartGroup>
            </Chart>
          ) : <div className={styles.chartPlaceholder} aria-hidden="true" />}
        </div>
        <p className={styles.chartFoot}>
          {concurrencyValues.length} measured concurrency levels · use the table button for exact values
        </p>
        <button
          type="button"
          className={`${styles.flipHint} ${styles.flipButton}`}
          aria-expanded={flipped}
          aria-controls={tableId}
          onClick={() => setFlipped(value => !value)}
        >
          <span aria-hidden="true">↻ </span>table
        </button>
      </div>
      <div
        ref={backRef}
        id={tableId}
        className={`${styles.flipFace} ${styles.flipBack}`}
        aria-hidden={!flipped}
      >
        <div className={styles.backHead}>
          <div>
            <h3>{metric.title}</h3>
            <p>{metric.unit}</p>
          </div>
          <button
            type="button"
            className={`${styles.backClose} ${styles.flipButton}`}
            aria-expanded={flipped}
            aria-controls={tableId}
            onClick={() => setFlipped(false)}
          >
            <span aria-hidden="true">↻ </span>chart
          </button>
        </div>
        <MetricTable metric={metric} benchmark={benchmark} inferencex={inferencex} series={series} />
      </div>
    </div>
  )
}

export default function ComparisonPreview({
  benchmark,
  inferencex,
  series,
}: {
  benchmark: BenchmarkPoint[]
  inferencex: BenchmarkPoint[]
  series: PredictionSeries[]
}) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])

  return (
    <div className={styles.chartGrid}>
      {metrics.map(metric => (
        <ChartCard
          key={metric.key}
          metric={metric}
          benchmark={benchmark}
          inferencex={inferencex}
          series={series}
          mounted={mounted}
        />
      ))}
    </div>
  )
}

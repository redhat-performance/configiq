'use client'

import { useEffect, useState } from 'react'
import {
  Chart,
  ChartAxis,
  ChartGroup,
  ChartLine,
  ChartScatter,
  ChartVoronoiContainer,
} from '@patternfly/react-charts/victory'
import type { GroundTruthPoint } from '@/lib/validation/ground-truth-types'
import styles from './validation-compare.module.css'

export interface EstimatePoint {
  concurrency: number
  ttftSeconds: number | null
  tpotSeconds: number | null
  outputTokensPerSecond: number | null
  outputTokensPerSecondPerUser: number | null
  requestLatencySeconds: number | null
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
  { key: 'ttftSeconds', title: 'Time to first token', unit: 's', description: 'Measured median vs. predicted mean TTFT', digits: 3 },
  { key: 'tpotSeconds', title: 'Time per output token', unit: 's', description: 'Measured median vs. predicted mean TPOT', digits: 4 },
  { key: 'outputTokensPerSecond', title: 'Output throughput', unit: 'tokens/s', description: 'Measured aggregate vs. predicted aggregate throughput', digits: 1 },
  { key: 'outputTokensPerSecondPerUser', title: 'Throughput per user', unit: 'tokens/s/user', description: 'Measured vs. predicted throughput per user', digits: 1 },
  { key: 'requestLatencySeconds', title: 'Request latency', unit: 's', description: 'Measured median vs. predicted request latency', digits: 2 },
]

const chartColors = { groundTruth: '#0066cc', estimate: '#009596' }

function numberValue(point: GroundTruthPoint | EstimatePoint, key: MetricKey): number | null {
  return point[key]
}

function formatValue(value: number | null, digits: number): string {
  return value === null ? '—' : value.toLocaleString('en-US', { maximumFractionDigits: digits })
}

function formatComparison(measured: number | null, estimate: number | null, digits: number): string {
  if (measured === null) return '—'
  if (estimate === null) return `${formatValue(measured, digits)} · no estimate`
  const difference = measured === 0 ? null : ((estimate - measured) / Math.abs(measured)) * 100
  const delta = difference === null ? '' : ` · ${difference >= 0 ? '+' : ''}${difference.toFixed(1)}%`
  return `${formatValue(measured, digits)} → ${formatValue(estimate, digits)}${delta}`
}

function ChartCard({
  metric,
  groundTruth,
  estimates,
  mounted,
}: {
  metric: MetricDefinition
  groundTruth: GroundTruthPoint[]
  estimates: EstimatePoint[]
  mounted: boolean
}) {
  const values = [
    ...groundTruth.map(point => numberValue(point, metric.key)),
    ...estimates.map(point => numberValue(point, metric.key)),
  ].filter((value): value is number => value !== null && Number.isFinite(value))
  const max = values.length ? Math.max(...values) : 1
  const concurrencyValues = [...new Set(groundTruth.map(point => point.concurrency))]
  const tickValues = concurrencyValues.length > 10
    ? concurrencyValues.filter((_, index) => index === 0 || index === concurrencyValues.length - 1 || index % Math.ceil(concurrencyValues.length / 8) === 0)
    : concurrencyValues
  const groundTruthData = groundTruth
    .map(point => ({ x: point.concurrency, y: numberValue(point, metric.key), series: 'Ground truth' }))
    .filter((point): point is { x: number; y: number; series: 'Ground truth' } => point.y !== null)
  const estimateData = estimates
    .map(point => ({ x: point.concurrency, y: numberValue(point, metric.key), series: 'AISimulators' }))
    .filter((point): point is { x: number; y: number; series: 'AISimulators' } => point.y !== null)

  return (
    <article className={styles.chartCard}>
      <div className={styles.chartHead}>
        <div>
          <h3>{metric.title}</h3>
          <p>{metric.description}</p>
        </div>
        <span className={styles.unit}>{metric.unit}</span>
      </div>
      <div className={styles.chartCanvas}>
        {mounted && groundTruth.length > 0 ? (
          <Chart
            ariaTitle={`${metric.title} by concurrency`}
            ariaDesc={`Ground truth and AISimulators ${metric.title.toLowerCase()} compared at measured concurrency levels`}
            domain={{ x: [0, Math.max(...concurrencyValues) * 1.04], y: [0, max * 1.14] }}
            height={278}
            width={600}
            padding={{ top: 18, right: 20, bottom: 54, left: 76 }}
            containerComponent={
              <ChartVoronoiContainer
                radius={28}
                mouseFollowTooltips={false}
                voronoiBlacklist={['ground-truth-line', 'aisimulators-line']}
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
              <ChartLine
                data={groundTruthData}
                name="ground-truth-line"
                style={{ data: { stroke: chartColors.groundTruth, strokeWidth: 3, strokeLinecap: 'round', strokeLinejoin: 'round' } }}
              />
              <ChartLine
                data={estimateData}
                name="aisimulators-line"
                style={{ data: { stroke: chartColors.estimate, strokeWidth: 3, strokeLinecap: 'round', strokeLinejoin: 'round' } }}
              />
            </ChartGroup>
            <ChartGroup>
              <ChartScatter
                data={groundTruthData}
                name="ground-truth-points"
                size={({ active }) => active ? 6 : 4}
                style={{ data: { fill: '#ffffff', stroke: chartColors.groundTruth, strokeWidth: 2 } }}
              />
              <ChartScatter
                data={estimateData}
                name="aisimulators-points"
                size={({ active }) => active ? 6 : 4}
                style={{ data: { fill: '#ffffff', stroke: chartColors.estimate, strokeWidth: 2 } }}
              />
            </ChartGroup>
          </Chart>
        ) : <div className={styles.chartPlaceholder} aria-hidden="true" />}
      </div>
      <p className={styles.chartFoot}>{groundTruth.length} measured concurrency levels</p>
    </article>
  )
}

export default function ComparisonPreview({
  groundTruth,
  estimates,
}: {
  groundTruth: GroundTruthPoint[]
  estimates: EstimatePoint[]
}) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])
  const estimatesByConcurrency = new Map(estimates.map(point => [point.concurrency, point]))

  return (
    <>
      <div className={styles.chartGrid}>
        {metrics.map(metric => (
          <ChartCard
            key={metric.key}
            metric={metric}
            groundTruth={groundTruth}
            estimates={estimates}
            mounted={mounted}
          />
        ))}
      </div>
      {groundTruth.length > 0 && (
        <section className={styles.details} aria-labelledby="comparison-details-heading">
          <div className={styles.detailsHead}>
            <h3 id="comparison-details-heading">Results by concurrency</h3>
            <p>Each cell shows ground truth → AISimulators and the relative difference.</p>
          </div>
          <div className={styles.tableScroll}>
            <table className={styles.detailsTable}>
              <thead>
                <tr>
                  <th scope="col">Concurrency</th>
                  {metrics.map(metric => <th scope="col" key={metric.key}>{metric.title} ({metric.unit})</th>)}
                </tr>
              </thead>
              <tbody>
                {groundTruth.map(point => {
                  const estimate = estimatesByConcurrency.get(point.concurrency)
                  return (
                    <tr key={point.concurrency}>
                      <th scope="row">{point.concurrency.toLocaleString('en-US')}</th>
                      {metrics.map(metric => (
                        <td key={metric.key}>
                          {formatComparison(numberValue(point, metric.key), estimate ? numberValue(estimate, metric.key) : null, metric.digits)}
                        </td>
                      ))}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  )
}

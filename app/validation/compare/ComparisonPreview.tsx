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

const chartColors = { groundTruth: '#0066cc', inferencex: '#ec7a08', estimate: '#009596' }

function numberValue(point: GroundTruthPoint | EstimatePoint, key: MetricKey): number | null {
  return point[key]
}

function formatValue(value: number | null, digits: number): string {
  return value === null ? '—' : value.toLocaleString('en-US', { maximumFractionDigits: digits })
}

function formatComparison(internal: number | null, inferencex: number | null, estimate: number | null, digits: number, showDifference: boolean, predictionLabel: string): string {
  const values = [
    internal === null ? null : `GT ${formatValue(internal, digits)}`,
    inferencex === null ? null : `InferenceX ${formatValue(inferencex, digits)}`,
    estimate === null ? null : `${predictionLabel} ${formatValue(estimate, digits)}`,
  ].filter((value): value is string => value !== null)
  const baseline = internal ?? inferencex
  if (showDifference && baseline !== null && baseline !== 0 && estimate !== null) {
    const difference = ((estimate - baseline) / Math.abs(baseline)) * 100
    values.push(`Δ vs ${internal === null ? 'InferenceX' : 'GT'} ${difference >= 0 ? '+' : ''}${difference.toFixed(1)}%`)
  }
  return values.join(' · ') || '—'
}

function ChartCard({
  metric,
  groundTruth,
  inferencex,
  estimates,
  mounted,
  predictionLabel,
}: {
  metric: MetricDefinition
  groundTruth: GroundTruthPoint[]
  inferencex: GroundTruthPoint[]
  estimates: EstimatePoint[]
  mounted: boolean
  predictionLabel: string
}) {
  const values = [
    ...groundTruth.map(point => numberValue(point, metric.key)),
    ...inferencex.map(point => numberValue(point, metric.key)),
    ...estimates.map(point => numberValue(point, metric.key)),
  ].filter((value): value is number => value !== null && Number.isFinite(value))
  const max = values.length ? Math.max(...values) : 1
  const concurrencyValues = [...new Set([...groundTruth, ...inferencex].map(point => point.concurrency))].sort((left, right) => left - right)
  const tickValues = concurrencyValues.length > 10
    ? concurrencyValues.filter((_, index) => index === 0 || index === concurrencyValues.length - 1 || index % Math.ceil(concurrencyValues.length / 8) === 0)
    : concurrencyValues
  const groundTruthData = groundTruth
    .map(point => ({ x: point.concurrency, y: numberValue(point, metric.key), series: 'Internal GT' }))
    .filter((point): point is { x: number; y: number; series: string } => point.y !== null)
  const inferencexData = inferencex
    .map(point => ({ x: point.concurrency, y: numberValue(point, metric.key), series: 'InferenceX' }))
    .filter((point): point is { x: number; y: number; series: string } => point.y !== null)
  const estimateData = estimates
    .map(point => ({ x: point.concurrency, y: numberValue(point, metric.key), series: predictionLabel }))
    .filter((point): point is { x: number; y: number; series: string } => point.y !== null)

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
        {mounted && concurrencyValues.length > 0 ? (
          <Chart
            ariaTitle={`${metric.title} by concurrency`}
            ariaDesc={`Internal GT, InferenceX, and ${predictionLabel} ${metric.title.toLowerCase()} compared at measured concurrency levels`}
            domain={{ x: [0, Math.max(...concurrencyValues) * 1.04], y: [0, max * 1.14] }}
            height={278}
            width={600}
            padding={{ top: 18, right: 20, bottom: 54, left: 76 }}
            containerComponent={
              <ChartVoronoiContainer
                radius={28}
                mouseFollowTooltips={false}
                voronoiBlacklist={['ground-truth-line', 'inferencex-line', 'aisimulators-line']}
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
              {groundTruthData.length > 0 && <ChartLine
                data={groundTruthData}
                name="ground-truth-line"
                style={{ data: { stroke: chartColors.groundTruth, strokeWidth: 3, strokeLinecap: 'round', strokeLinejoin: 'round' } }}
              />}
              {inferencexData.length > 0 && <ChartLine
                data={inferencexData}
                name="inferencex-line"
                style={{ data: { stroke: chartColors.inferencex, strokeWidth: 3, strokeLinecap: 'round', strokeLinejoin: 'round' } }}
              />}
              {estimateData.length > 0 && <ChartLine
                data={estimateData}
                name="aisimulators-line"
                style={{ data: { stroke: chartColors.estimate, strokeWidth: 3, strokeLinecap: 'round', strokeLinejoin: 'round' } }}
              />}
            </ChartGroup>
            <ChartGroup>
              {groundTruthData.length > 0 && <ChartScatter
                data={groundTruthData}
                name="ground-truth-points"
                size={({ active }) => active ? 6 : 4}
                style={{ data: { fill: '#ffffff', stroke: chartColors.groundTruth, strokeWidth: 2 } }}
              />}
              {inferencexData.length > 0 && <ChartScatter
                data={inferencexData}
                name="inferencex-points"
                size={({ active }) => active ? 6 : 4}
                style={{ data: { fill: '#ffffff', stroke: chartColors.inferencex, strokeWidth: 2 } }}
              />}
              {estimateData.length > 0 && <ChartScatter
                data={estimateData}
                name="aisimulators-points"
                size={({ active }) => active ? 6 : 4}
                style={{ data: { fill: '#ffffff', stroke: chartColors.estimate, strokeWidth: 2 } }}
              />}
            </ChartGroup>
          </Chart>
        ) : <div className={styles.chartPlaceholder} aria-hidden="true" />}
      </div>
      <p className={styles.chartFoot}>{concurrencyValues.length} measured concurrency levels</p>
    </article>
  )
}

export default function ComparisonPreview({
  groundTruth,
  inferencex,
  estimates,
  predictionLabel,
  showDifference,
}: {
  groundTruth: GroundTruthPoint[]
  inferencex: GroundTruthPoint[]
  estimates: EstimatePoint[]
  predictionLabel: string
  showDifference: boolean
}) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])
  const estimatesByConcurrency = new Map(estimates.map(point => [point.concurrency, point]))
  const groundTruthByConcurrency = new Map(groundTruth.map(point => [point.concurrency, point]))
  const inferencexByConcurrency = new Map(inferencex.map(point => [point.concurrency, point]))
  const concurrencyValues = [...new Set([...groundTruth, ...inferencex].map(point => point.concurrency))].sort((left, right) => left - right)

  return (
    <>
      <div className={styles.chartGrid}>
        {metrics.map(metric => (
          <ChartCard
            key={metric.key}
            metric={metric}
            groundTruth={groundTruth}
            inferencex={inferencex}
            estimates={estimates}
            mounted={mounted}
            predictionLabel={predictionLabel}
          />
        ))}
      </div>
      {concurrencyValues.length > 0 && (
        <section className={styles.details} aria-labelledby="comparison-details-heading">
          <div className={styles.detailsHead}>
            <h3 id="comparison-details-heading">Results by concurrency</h3>
            <p>{showDifference
              ? `Each cell shows the selected measured source and ${predictionLabel} separately. The difference uses the selected measurement.`
              : `Each cell shows the selected measured source and ${predictionLabel} separately. Percentage differences are hidden because the prediction backend or version differs from the measured run.`}</p>
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
                {concurrencyValues.map(concurrency => {
                  const internal = groundTruthByConcurrency.get(concurrency)
                  const external = inferencexByConcurrency.get(concurrency)
                  const estimate = estimatesByConcurrency.get(concurrency)
                  return (
                    <tr key={concurrency}>
                      <th scope="row">{concurrency.toLocaleString('en-US')}</th>
                      {metrics.map(metric => (
                        <td key={metric.key}>
                          {formatComparison(internal ? numberValue(internal, metric.key) : null,
                            external ? numberValue(external, metric.key) : null,
                            estimate ? numberValue(estimate, metric.key) : null, metric.digits, showDifference, predictionLabel)}
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

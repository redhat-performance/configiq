'use client'

import * as React from 'react'
import { DescriptionList, DescriptionListDescription, DescriptionListGroup, DescriptionListTerm, Title } from '@patternfly/react-core'
import {
  Chart,
  ChartAxis,
  ChartGroup,
  ChartLine,
  ChartScatter,
  ChartTooltip,
  ChartVoronoiContainer,
} from '@patternfly/react-charts/victory'
import type { InfrastructurePath } from '@/lib/hybrid-savings/calc'
import type { FixedDeploymentSample } from '@/lib/hybrid-savings/fixed-deployment'
import { costPerMillionOutputTokens } from '@/lib/hybrid-savings/token-economics'
import { JargonPopover } from '@/components/ui/JargonPopover'
import { ChartPlotHeader } from './ChartPlotHeader'
import styles from './hybrid-savings.module.css'

interface FixedDeploymentChartProps {
  samples: FixedDeploymentSample[]
  currentCapacityUsedPct: number | null
  path: InfrastructurePath
  hardwareLabel?: string
  hostedModelLabel?: string
  infrastructureModelLabel?: string
}

function compactNumber(value: number): string {
  return new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(value)
}

function compactCurrency(value: number): string {
  return `$${compactNumber(value)}`
}

function preciseCurrency(value: number): string {
  return `$${new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(value)}`
}

/**
 * Fixed-deployment view: the deployment chosen for the current workload,
 * frozen, and re-priced across the demand range it could serve. The hosted
 * API is priced against identical demand, and unit cost falls only as the
 * fixed deployment fills up.
 */
export default function FixedDeploymentChart({
  samples,
  currentCapacityUsedPct,
  path,
  hardwareLabel,
  hostedModelLabel,
  infrastructureModelLabel,
}: FixedDeploymentChartProps) {
  const canvasRef = React.useRef<HTMLDivElement>(null)
  const [width, setWidth] = React.useState(1060)
  React.useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0) setWidth(entry.contentRect.width)
    })
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [])
  if (samples.length < 2) return null

  const hardwareName = path === 'rented' ? 'rented' : 'purchased'
  // Start at 10% usage (or the current lower usage), not an undefined zero.
  // Fit every displayed sample rather than cutting off the early curve.
  const startThreshold = currentCapacityUsedPct !== null && currentCapacityUsedPct > 0
    ? Math.min(10, currentCapacityUsedPct) : 10
  const displayedSamples = samples.filter(sample => sample.capacityUsedPct >= startThreshold)
  if (displayedSamples.length === 0) return null
  const minimumPct = Math.min(...displayedSamples.map(sample => sample.capacityUsedPct))
  const asSeries = (key: 'hosted' | 'hardware') =>
    displayedSamples
      .map((sample): { x: number; y: number; name: string } | null => {
        const option = key === 'hosted' ? sample.hosted : sample.hardware
        const y = option ? costPerMillionOutputTokens(option.monthlyCost, sample.outputTokens) : null
        // Zero output has no defined unit cost.
        return y === null ? null : { x: sample.capacityUsedPct, y, name: key }
      })
      .filter((point): point is { x: number; y: number; name: string } => point !== null)

  const hostedSeries = asSeries('hosted')
  const hardwareSeries = asSeries('hardware')

  const maximumCost = Math.max(0.01, ...[...hostedSeries, ...hardwareSeries].map(point => point.y)) * 1.08

  const currentLine = currentCapacityUsedPct !== null && currentCapacityUsedPct >= minimumPct && currentCapacityUsedPct <= 100
    ? [{ x: currentCapacityUsedPct, y: 0 }, { x: currentCapacityUsedPct, y: maximumCost }]
    : []
  const currentLinePosition = (
    (68 + ((currentCapacityUsedPct ?? 0) - minimumPct) / (100 - minimumPct || 1) * (width - 68 - 24)) / width * 100
  )
  const currentLabelTransform = currentLinePosition < 18
    ? 'translateX(0)'
    : currentLinePosition > 82
      ? 'translateX(-100%)'
      : 'translateX(-50%)'
  const axisCurrency = (tick: number) => tick >= 1000 ? compactCurrency(Number(tick)) : preciseCurrency(Number(tick))
  const seriesLabel = (name: string) => name === 'hosted'
    ? 'Hosted API'
    : path === 'rented' ? 'Rented (fixed)' : 'Purchased (fixed)'

  return (
    <div className={styles.chartPanel}>
      <div className={styles.chartHeader}>
        <div className={styles.chartHeading}>
          <Title headingLevel="h3" size="lg">Cost as hardware usage increases</Title>
          <div className={styles.chartExplanation}>
            Higher usage spreads fixed costs across more output.
            <JargonPopover term="Hardware usage"
              explanation="The same deployment is kept fixed while input and output demand scale together. Capacity is modeled for this request shape and concurrency, not measured GPU utilization or a hardware maximum. Savings still depend on prices and rental billing." />
          </div>
        </div>
        <DescriptionList isCompact className={styles.chartDeployment} aria-label="Fixed hardware">
          <DescriptionListGroup>
            <DescriptionListTerm>Fixed hardware</DescriptionListTerm>
            <DescriptionListDescription>{hardwareLabel ?? `${path === 'rented' ? 'Rented' : 'Purchased'} deployment`}</DescriptionListDescription>
          </DescriptionListGroup>
        </DescriptionList>
      </div>
      {hostedModelLabel && <p className={styles.calculationNote}>Hosted API: {hostedModelLabel} · Rented / purchased: {infrastructureModelLabel}. Same token volumes; model quality and capability may differ.</p>}
      <ChartPlotHeader path={path} />
      <div className={styles.chartCanvas} ref={canvasRef}>
        {currentLine.length > 0 && (
          <span
            className={styles.currentWorkloadLabel}
            style={{ left: `${currentLinePosition}%`, transform: currentLabelTransform }}
          >
            Your usage · {(currentCapacityUsedPct ?? 0).toFixed(1)}%
          </span>
        )}
        <Chart
          ariaDesc={`Hosted API and the frozen ${hardwareName} deployment cost per million requested output tokens by percentage of planning capacity used`}
          ariaTitle="Fixed deployment comparison"
          containerComponent={
            <ChartVoronoiContainer
              constrainToVisibleArea
              voronoiBlacklist={['current-workload']}
              labelComponent={<ChartTooltip style={{ fontFamily: 'var(--sans)', fontSize: 12 }} />}
              labels={({ datum }: { datum: { x?: number; y?: number; name?: string } }) =>
                `${seriesLabel(String(datum.name))}\n${Number(datum.x).toFixed(0)}% capacity used\n${preciseCurrency(Number(datum.y))} / 1M output`}
            />
          }
          domain={{ x: [minimumPct, 100], y: [0, maximumCost] }}
          height={360}
          padding={{ top: 40, right: 24, bottom: 58, left: 68 }}
          width={width}
        >
          <ChartAxis
            label="Planning capacity used (%)"
            crossAxis={false}
            tickValues={Array.from({ length: width < 600 ? 3 : 5 }, (_, index) => minimumPct + index * (100 - minimumPct) / (width < 600 ? 2 : 4))}
            tickFormat={(tick: number) => `${Number(tick).toFixed(0)}%`}
            style={{
              axisLabel: { padding: 40, fontSize: 12, fontFamily: 'var(--sans)' },
              tickLabels: { fontSize: 11.5, fontFamily: 'var(--mono)' },
            }}
          />
          <ChartAxis
            dependentAxis
            crossAxis={false}
            tickFormat={axisCurrency}
            style={{
              grid: { stroke: '#d2d2d2', strokeDasharray: '3,4' },
              tickLabels: { fontSize: 11.5, fontFamily: 'var(--mono)' },
            }}
          />
          <ChartGroup>
            <ChartLine
              data={hostedSeries}
              style={{ data: { stroke: '#258466', strokeWidth: 3 } }}
            />
            <ChartLine
              data={hardwareSeries}
              style={{ data: { stroke: path === 'rented' ? '#ee0000' : '#db8b1d', strokeWidth: 3 } }}
            />
            {currentLine.length > 0 && (
              <ChartLine
                data={currentLine}
                name="current-workload"
                style={{ data: { stroke: '#151515', strokeDasharray: '5,5', strokeWidth: 1 } }}
              />
            )}
            {currentCapacityUsedPct !== null && currentCapacityUsedPct >= 0 && currentCapacityUsedPct <= 100 && (() => {
              const sample = samples.find(entry => entry.capacityUsedPct === currentCapacityUsedPct)
              const option = sample?.hardware ?? null
              const y = option ? costPerMillionOutputTokens(option.monthlyCost, sample?.outputTokens ?? 0) : null
              return y === null ? null : (
                <ChartScatter
                  data={[{ x: currentCapacityUsedPct, y, name: 'hardware' }]}
                  size={5}
                  style={{ data: {
                    fill: path === 'rented' ? '#ee0000' : '#db8b1d',
                    stroke: '#ffffff', strokeWidth: 2,
                  } }}
                />
              )
            })()}
          </ChartGroup>
        </Chart>
      </div>
    </div>
  )
}

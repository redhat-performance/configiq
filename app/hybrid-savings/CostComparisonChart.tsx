'use client'

import * as React from 'react'
import {
  Alert,
  DescriptionList,
  DescriptionListDescription,
  DescriptionListGroup,
  DescriptionListTerm,
  Title,
} from '@patternfly/react-core'
import {
  Chart,
  ChartAxis,
  ChartGroup,
  ChartLine,
  ChartScatter,
  ChartTooltip,
  ChartVoronoiContainer,
} from '@patternfly/react-charts/victory'
import { HYBRID_PLANNING_HORIZON_TOKENS, type CostPoint } from '@/lib/hybrid-savings/calc'
import {
  costPerMillionOutputTokens,
  outputTokensAtVolume,
} from '@/lib/hybrid-savings/token-economics'
import styles from './hybrid-savings.module.css'
import { ChartPlotHeader } from './ChartPlotHeader'

interface CostComparisonChartProps {
  points: CostPoint[]
  currentTokens: number
  outputShare: number
  rentedBreakEvenTokens: number | null
  ownedBreakEvenTokens: number | null
  rentedLowestCostTokens: number | null
  ownedLowestCostTokens: number | null
  transitionsVerified: boolean
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

export default function CostComparisonChart({
  points,
  currentTokens,
  outputShare,
  rentedBreakEvenTokens,
  ownedBreakEvenTokens,
  rentedLowestCostTokens,
  ownedLowestCostTokens,
  transitionsVerified,
  hostedModelLabel,
  infrastructureModelLabel,
}: CostComparisonChartProps) {
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
  if (points.length < 2 || !Number.isFinite(outputShare) || outputShare <= 0) return null

  const maximumTokens = points.reduce((maximum, point) => Math.max(maximum, point.tokens), 1)
  const transitions = [rentedBreakEvenTokens, ownedBreakEvenTokens, rentedLowestCostTokens, ownedLowestCostTokens]
  // Unit cost is undefined at zero and unbounded near it. Start at a quarter
  // of the earliest workload/crossover anchor, using an actual engine sample.
  // Every cost in the explicitly displayed range then fits without clipping.
  const anchors = [currentTokens, maximumTokens, ...transitions]
    .filter((tokens): tokens is number => tokens !== null && Number.isFinite(tokens) && tokens > 0 && tokens <= maximumTokens)
  const startThreshold = Math.min(...anchors) / 4
  const startPoints = points.filter(point => point.tokens >= startThreshold)
  if (startPoints.length === 0) return null
  const minimumTokens = Math.min(...startPoints.map(point => point.tokens))
  const displayedPoints = points.filter(point => point.tokens >= minimumTokens)
  const outputVolume = (tokens: number) => outputTokensAtVolume(tokens, outputShare)
  const workloadSplitLabel = (tokens: number) => {
    const outputTokens = outputVolume(tokens)
    return `${compactNumber(tokens - outputTokens)} input + ${compactNumber(outputTokens)} output`
  }
  const seriesValue = (point: CostPoint, key: 'hosted' | 'rented' | 'owned') =>
    point[key] === null ? null : costPerMillionOutputTokens(point[key], outputVolume(point.tokens))
  const asSeries = (key: 'hosted' | 'rented' | 'owned') =>
    displayedPoints
      .map((point): { x: number; y: number; name: string } | null => {
        const y = seriesValue(point, key)
        // Zero output has no defined unit cost; skip it rather than showing
        // a false zero or an infinite value.
        return y === null ? null : { x: point.tokens, y, name: key }
      })
      .filter((point): point is { x: number; y: number; name: string } => point !== null)

  const hostedSeries = asSeries('hosted')
  const rentedSeries = asSeries('rented')
  const ownedSeries = asSeries('owned')

  const maximumCost = Math.max(0.01, ...[...hostedSeries, ...rentedSeries, ...ownedSeries].map(point => point.y)) * 1.08
  const currentLine = currentTokens >= minimumTokens && currentTokens <= maximumTokens
    ? [{ x: currentTokens, y: 0 }, { x: currentTokens, y: maximumCost }]
    : []
  const currentLinePosition = (
    (68 + (currentTokens - minimumTokens) / (maximumTokens - minimumTokens || 1) * (width - 68 - 24)) / width * 100
  )
  const currentLabelTransform = currentLinePosition < 18
    ? 'translateX(0)'
    : currentLinePosition > 82
      ? 'translateX(-100%)'
      : 'translateX(-50%)'
  const markerPoint = (
    key: 'rented' | 'owned',
    tokens: number | null,
  ): Array<{ x: number; y: number; name: string }> => {
    if (tokens === null) return []
    const point = points.find(candidate => candidate.tokens === tokens)
    const cost = point ? seriesValue(point, key) : null
    return cost === null ? [] : [{ x: tokens, y: cost, name: key }]
  }
  const searchHorizon = `${compactNumber(HYBRID_PLANNING_HORIZON_TOKENS)} total tokens/month`
  const hasSeries = (key: 'hosted' | 'rented' | 'owned') =>
    points.some(point => point.tokens > 0 && Number.isFinite(seriesValue(point, key)))
  const noHardwareCrossover = transitionsVerified &&
    (['hosted', 'rented', 'owned'] as const).every(hasSeries) &&
    [rentedBreakEvenTokens, ownedBreakEvenTokens, rentedLowestCostTokens, ownedLowestCostTokens]
      .every(tokens => tokens === null)
  const transitionText = (key: 'rented' | 'owned', tokens: number | null) => {
    if (!hasSeries(key)) return 'Unavailable'
    return !transitionsVerified ? 'Not verified' : tokens === null
      ? `Not within ${searchHorizon}`
      : `≈ ${compactNumber(tokens)} total tokens/month`
  }
  const currentPoint = points.find(point => point.tokens === currentTokens)
  const currentMarkers = currentPoint && currentLine.length > 0
    ? (['hosted', 'rented', 'owned'] as const).flatMap(name => {
      const y = seriesValue(currentPoint, name)
      return y === null ? [] : [{ x: currentTokens, y, name }]
    }) : []
  const rentedLowestMarker = markerPoint('rented', rentedLowestCostTokens)
  const ownedLowestMarker = markerPoint('owned', ownedLowestCostTokens)
  const rentedBreakEvenMarker = markerPoint('rented', rentedBreakEvenTokens)
  const ownedBreakEvenMarker = markerPoint('owned', ownedBreakEvenTokens)
  const axisCurrency = (tick: number) => tick >= 1000 ? compactCurrency(Number(tick)) : preciseCurrency(Number(tick))
  const seriesLabel = (name: string) => name === 'hosted'
    ? 'Hosted API'
    : name === 'rented' ? 'Rented' : 'Purchased'

  return (
    <div className={styles.chartPanel}>
      <div className={styles.chartHeader}>
        <div className={styles.chartHeading}>
          <Title headingLevel="h3" size="lg">Cost as demand grows</Title>
          <p className={styles.chartExplanation}>Input/output mix stays fixed. Steps show added hardware.</p>
        </div>
        <DescriptionList isCompact columnModifier={{ default: '3Col' }} className={styles.chartWorkload} aria-label="Current monthly workload in tokens">
          <DescriptionListGroup>
            <DescriptionListTerm>Monthly total</DescriptionListTerm>
            <DescriptionListDescription>{compactNumber(currentTokens)}</DescriptionListDescription>
          </DescriptionListGroup>
          <DescriptionListGroup>
            <DescriptionListTerm>Input</DescriptionListTerm>
            <DescriptionListDescription>{compactNumber(currentTokens - outputVolume(currentTokens))}</DescriptionListDescription>
          </DescriptionListGroup>
          <DescriptionListGroup>
            <DescriptionListTerm>Output</DescriptionListTerm>
            <DescriptionListDescription>{compactNumber(outputVolume(currentTokens))}</DescriptionListDescription>
          </DescriptionListGroup>
        </DescriptionList>
      </div>
      {hostedModelLabel && <p className={styles.calculationNote}>Hosted API: {hostedModelLabel} · Rented / purchased: {infrastructureModelLabel}. Same token volumes; model quality and capability may differ.</p>}
      {noHardwareCrossover ? (
        <Alert isInline isPlain variant="info" component="p" className={styles.chartInsight}
          title={`Hosted API stays cheaper up to ${searchHorizon} under these assumptions.`} />
      ) : <div className={styles.chartMilestones} aria-label="Modeled cost milestones">
        <div className={styles.breakEvenMilestone}>
          <strong>Cheaper than hosted API</strong>
          <span className={styles.rentedMilestone}>Rented {transitionText('rented', rentedBreakEvenTokens)}</span>
          <span className={styles.ownedMilestone}>Purchased {transitionText('owned', ownedBreakEvenTokens)}</span>
        </div>
        <div>
          <strong>Cheapest of all options</strong>
          <span className={styles.rentedMilestone}>Rented {transitionText('rented', rentedLowestCostTokens)}</span>
          <span className={styles.ownedMilestone}>Purchased {transitionText('owned', ownedLowestCostTokens)}</span>
        </div>
      </div>}
      {!transitionsVerified && (
        <p className={styles.currentMarker}>Crossover points could not be verified. Current-workload costs remain available.</p>
      )}
      <ChartPlotHeader />
      <div className={styles.chartCanvas} ref={canvasRef}>
        {currentLine.length > 0 && (
          <span
            className={styles.currentWorkloadLabel}
            style={{ left: `${currentLinePosition}%`, transform: currentLabelTransform }}
          >
            Your total · {compactNumber(currentTokens)}
          </span>
        )}
        <Chart
          ariaDesc="Hosted API, rented infrastructure and purchased hardware cost per million requested output tokens by total monthly input and output volume, with the input/output mix held constant"
          ariaTitle="Hybrid cost comparison"
          containerComponent={
            <ChartVoronoiContainer
              constrainToVisibleArea
              voronoiBlacklist={['current-workload']}
              labelComponent={<ChartTooltip style={{ fontFamily: 'var(--sans)', fontSize: 12 }} />}
              labels={({ datum }: { datum: { x?: number; y?: number; name?: string } }) =>
                `${seriesLabel(String(datum.name))}\n${compactNumber(Number(datum.x))} total tokens/month\n${workloadSplitLabel(Number(datum.x))}\n${preciseCurrency(Number(datum.y))} / 1M output`}
            />
          }
          domain={{ x: [minimumTokens, maximumTokens], y: [0, maximumCost] }}
          height={360}
          padding={{ top: 40, right: 24, bottom: 58, left: 68 }}
          width={width}
        >
          <ChartAxis
            label={width < 600 ? 'Total tokens per month' : 'Total tokens per month (input + output)'}
            crossAxis={false}
            tickValues={Array.from({ length: width < 600 ? 3 : 5 }, (_, index) => minimumTokens + index * (maximumTokens - minimumTokens) / (width < 600 ? 2 : 4))}
            tickFormat={(tick: number) => compactNumber(Number(tick))}
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
              data={rentedSeries}
              style={{ data: { stroke: '#ee0000', strokeWidth: 3 } }}
            />
            <ChartLine
              data={ownedSeries}
              style={{ data: { stroke: '#db8b1d', strokeWidth: 3 } }}
            />
            {currentLine.length > 0 && (
              <ChartLine
                data={currentLine}
                name="current-workload"
                style={{ data: { stroke: '#151515', strokeDasharray: '5,5', strokeWidth: 1 } }}
              />
            )}
            {currentMarkers.map(point => (
              <ChartScatter
                key={point.name}
                data={[point]}
                size={5}
                style={{ data: {
                  fill: point.name === 'hosted' ? '#258466' : point.name === 'rented' ? '#ee0000' : '#db8b1d',
                  stroke: '#ffffff', strokeWidth: 2,
                } }}
              />
            ))}
            {rentedLowestMarker.length > 0 && (
              <ChartScatter
                data={rentedLowestMarker}
                size={5}
                style={{ data: { fill: '#ee0000', stroke: '#ffffff', strokeWidth: 2 } }}
              />
            )}
            {ownedLowestMarker.length > 0 && (
              <ChartScatter
                data={ownedLowestMarker}
                size={5}
                style={{ data: { fill: '#db8b1d', stroke: '#ffffff', strokeWidth: 2 } }}
              />
            )}
            {rentedBreakEvenMarker.length > 0 && (
              <ChartScatter
                data={rentedBreakEvenMarker}
                size={7}
                style={{ data: { fill: 'transparent', stroke: '#ee0000', strokeWidth: 2.5 } }}
              />
            )}
            {ownedBreakEvenMarker.length > 0 && (
              <ChartScatter
                data={ownedBreakEvenMarker}
                size={7}
                style={{ data: { fill: 'transparent', stroke: '#db8b1d', strokeWidth: 2.5 } }}
              />
            )}
          </ChartGroup>
        </Chart>
      </div>
    </div>
  )
}

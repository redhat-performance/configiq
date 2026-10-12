'use client'

import * as React from 'react'
import {
  Chart,
  ChartAxis,
  ChartGroup,
  ChartLine,
  ChartScatter,
  ChartTooltip,
  ChartVoronoiContainer,
} from '@patternfly/react-charts/victory'
import { HYBRID_PLANNING_HORIZON_TOKENS, type CostLens, type CostPoint } from '@/lib/hybrid-savings/calc'
import styles from './hybrid-savings.module.css'

interface CostComparisonChartProps {
  points: CostPoint[]
  currentTokens: number
  costLens: CostLens
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

export default function CostComparisonChart({
  points,
  currentTokens,
  costLens,
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
  if (points.length < 2) return null

  const maximumTokens = points.reduce((maximum, point) => Math.max(maximum, point.tokens), 1)
  const maximumCost = points.reduce((maximum, point) => Math.max(
    maximum, point.hosted ?? 0, point.rented ?? 0, point.owned ?? 0,
  ), 1) * 1.08
  const asSeries = (key: 'hosted' | 'rented' | 'owned') =>
    points
      .filter(point => point[key] !== null)
      .map(point => ({ x: point.tokens, y: point[key] as number, name: key }))
  const currentLine = currentTokens > 0 && currentTokens <= maximumTokens
    ? [{ x: currentTokens, y: 0 }, { x: currentTokens, y: maximumCost }]
    : []
  const currentLinePosition = (
    (82 + (currentTokens / maximumTokens) * (width - 82 - 24)) / width * 100
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
    const cost = point?.[key]
    return cost == null ? [] : [{ x: tokens, y: cost, name: key }]
  }
  const transitionText = (tokens: number | null) =>
    !transitionsVerified ? 'Not verified' : tokens === null
      ? maximumTokens > HYBRID_PLANNING_HORIZON_TOKENS
        ? `Not reached by ${compactNumber(HYBRID_PLANNING_HORIZON_TOKENS)}`
        : 'Not reached'
      : `≈ ${compactNumber(tokens)} tokens/month`
  const rentedLowestMarker = markerPoint('rented', rentedLowestCostTokens)
  const ownedLowestMarker = markerPoint('owned', ownedLowestCostTokens)
  const rentedBreakEvenMarker = markerPoint('rented', rentedBreakEvenTokens)
  const ownedBreakEvenMarker = markerPoint('owned', ownedBreakEvenTokens)

  return (
    <div className={styles.chartPanel}>
      <div className={styles.chartHeader}>
        <div>
          <h3>Cost from 0 to {compactNumber(maximumTokens)} billed tokens/month</h3>
          <p>
            Auto-scaled around this workload&apos;s crossovers · starts at zero ·{' '}
            {costLens === 'fully-loaded' ? 'monthly full TCO' : 'monthly marginal cost'}
          </p>
        </div>
        <div className={styles.chartLegend} aria-label="Chart legend">
          <span><i className={styles.hostedSwatch} />Hosted API</span>
          <span><i className={styles.rentedSwatch} />Rented</span>
          <span><i className={styles.ownedSwatch} />Purchased</span>
        </div>
      </div>
      {hostedModelLabel && <p className={styles.calculationNote}>Hosted API: {hostedModelLabel} · Rented / purchased: {infrastructureModelLabel}</p>}
      <div className={styles.chartMilestones} aria-label="Modeled cost milestones">
        <div className={styles.breakEvenMilestone}>
          <strong>Cheaper than hosted API</strong>
          <span className={styles.rentedMilestone}>Rented {transitionText(rentedBreakEvenTokens)}</span>
          <span className={styles.ownedMilestone}>Purchased {transitionText(ownedBreakEvenTokens)}</span>
        </div>
        <div>
          <strong>Lowest-cost eligible option</strong>
          <span className={styles.rentedMilestone}>Rented {transitionText(rentedLowestCostTokens)}</span>
          <span className={styles.ownedMilestone}>Purchased {transitionText(ownedLowestCostTokens)}</span>
        </div>
      </div>
      {!transitionsVerified && (
        <p className={styles.currentMarker}>This workload is too complex to verify transitions across the full range. Current monthly costs remain available; narrow the inputs to check crossover points.</p>
      )}
      <div className={styles.chartCanvas} ref={canvasRef}>
        {currentLine.length > 0 && (
          <span
            className={styles.currentWorkloadLabel}
            style={{ left: `${currentLinePosition}%`, transform: currentLabelTransform }}
          >
            Current · {compactNumber(currentTokens)}
          </span>
        )}
        <Chart
          ariaDesc="Monthly hosted API, rented infrastructure and purchased hardware cost by billed token volume"
          ariaTitle="Hybrid cost comparison"
          containerComponent={
            <ChartVoronoiContainer
              constrainToVisibleArea
              voronoiBlacklist={['current-workload']}
              labelComponent={<ChartTooltip style={{ fontFamily: 'var(--sans)', fontSize: 12 }} />}
              labels={({ datum }: { datum: { x?: number; y?: number } }) =>
                `${compactNumber(Number(datum.x))} tokens/month: ${compactCurrency(Number(datum.y))}`
              }
            />
          }
          domain={{ x: [0, maximumTokens], y: [0, maximumCost] }}
          height={360}
          padding={{ top: 40, right: 24, bottom: 58, left: 82 }}
          width={width}
        >
          <ChartAxis
            label="Billed tokens per month"
            crossAxis={false}
            tickValues={Array.from({ length: width < 600 ? 3 : 5 }, (_, index) => index * maximumTokens / (width < 600 ? 2 : 4))}
            tickFormat={(tick: number) => compactNumber(Number(tick))}
            style={{
              axisLabel: { padding: 40, fontSize: 12, fontFamily: 'var(--sans)' },
              tickLabels: { fontSize: 11.5, fontFamily: 'var(--mono)' },
            }}
          />
          <ChartAxis
            dependentAxis
            crossAxis={false}
            label="Monthly cost (USD)"
            tickFormat={(tick: number) => compactCurrency(Number(tick))}
            style={{
              axisLabel: { padding: 58, fontSize: 12, fontFamily: 'var(--sans)' },
              grid: { stroke: '#d2d2d2', strokeDasharray: '3,4' },
              tickLabels: { fontSize: 11.5, fontFamily: 'var(--mono)' },
            }}
          />
          <ChartGroup>
            <ChartLine
              data={asSeries('hosted')}
              style={{ data: { stroke: '#258466', strokeWidth: 3 } }}
            />
            <ChartLine
              data={asSeries('rented')}
              style={{ data: { stroke: '#ee0000', strokeWidth: 3 } }}
            />
            <ChartLine
              data={asSeries('owned')}
              style={{ data: { stroke: '#db8b1d', strokeWidth: 3 } }}
            />
            {currentLine.length > 0 && (
              <ChartLine
                data={currentLine}
                name="current-workload"
                style={{ data: { stroke: '#151515', strokeDasharray: '5,5', strokeWidth: 1 } }}
              />
            )}
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

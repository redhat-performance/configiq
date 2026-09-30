'use client'

import {
  Chart,
  ChartAxis,
  ChartGroup,
  ChartLine,
  ChartScatter,
  ChartVoronoiContainer,
} from '@patternfly/react-charts/victory'
import type { CostLens, CostPoint } from '@/lib/hybrid-savings/calc'
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
}: CostComparisonChartProps) {
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
    (82 + (currentTokens / maximumTokens) * (1060 - 82 - 32)) / 1060 * 100
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
    !transitionsVerified ? 'Not verified' : tokens === null ? 'Not reached' : `≈ ${compactNumber(tokens)} tokens/month`
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
      <div className={styles.chartCanvas}>
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
              labels={({ datum }: { datum: { x?: number; y?: number } }) =>
                `${compactNumber(Number(datum.x))} tokens/month: ${compactCurrency(Number(datum.y))}`
              }
            />
          }
          domain={{ x: [0, maximumTokens], y: [0, maximumCost] }}
          height={360}
          padding={{ top: 24, right: 32, bottom: 58, left: 82 }}
          width={1060}
        >
          <ChartAxis
            label="Billed tokens per month"
            tickFormat={(tick: number) => compactNumber(Number(tick))}
            style={{
              axisLabel: { padding: 40, fontSize: 12 },
              tickLabels: { fontSize: 11.5 },
            }}
          />
          <ChartAxis
            dependentAxis
            label="Monthly cost (USD)"
            tickFormat={(tick: number) => compactCurrency(Number(tick))}
            style={{
              axisLabel: { padding: 58, fontSize: 12 },
              grid: { stroke: '#d2d2d2', strokeDasharray: '3,4' },
              tickLabels: { fontSize: 11.5 },
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
      {currentLine.length > 0 && (
        <p className={styles.currentMarker}>
          Dashed line: current workload ({compactNumber(currentTokens)} tokens/month)
        </p>
      )}
      <p className={styles.currentMarker}>
        Scale-to-zero rented cost follows modeled GPU runtime. Purchased hardware, active-window rental and always-on rental increase in whole deployment steps.
      </p>
    </div>
  )
}

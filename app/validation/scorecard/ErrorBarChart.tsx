'use client';

import * as React from 'react';
import {
  Chart,
  ChartAxis,
  ChartBar,
  ChartGroup,
} from '@patternfly/react-charts/victory';
import type { ChartRow } from './scorecard-view';
import styles from './scorecard.module.css';

/**
 * Grouped horizontal bars: one bar per simulator for each measured
 * configuration, ranked worst first.
 *
 * This draws published cells, never a second aggregation. Bars that would be
 * unreadable at a glance are left to the table behind the flip; the labels at
 * each bar end keep the visible numbers exact.
 */

export interface ErrorSeries {
  simulator: string;
  label: string;
  color: string;
}

interface ErrorBarChartProps {
  rows: ChartRow[];
  series: ErrorSeries[];
  /** "MAPE", "MdAPE", "p90", or "Bias" — what the bar length means. */
  statisticLabel: string;
  /** Formats a fraction as the page's percentage, keeping a bias sign. */
  formatValue: (value: number) => string;
}

const ROW_HEIGHT = 34;

/** Exported so the flip card can size itself to the chart it contains. */
export const CHART_ROW_HEIGHT = ROW_HEIGHT;

function truncateLabel(value: string, maxChars: number): string {
  if (value.length <= maxChars) return value;
  return `${value.slice(0, Math.max(1, maxChars - 1))}…`;
}

const AXIS_STYLE = {
  axis: { stroke: '#8a8d90', strokeWidth: 1 },
  grid: { stroke: '#f0f0f0', strokeWidth: 1 },
  ticks: { stroke: '#b8bbbe' },
  tickLabels: { fontSize: 11.5, fill: '#3c3f42', fontFamily: 'var(--mono)' },
} as const;

/** Track the rendered width so the SVG never draws wider than its card. */
function useChartWidth(fallback = 640) {
  const ref = React.useRef<HTMLDivElement>(null);
  const [width, setWidth] = React.useState(fallback);
  React.useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const measure = () => {
      const next = Math.floor(element.clientWidth);
      if (next > 0) setWidth(next);
    };
    measure();
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', measure);
      return () => window.removeEventListener('resize', measure);
    }
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return { ref, width };
}

export default function ErrorBarChart({
  rows,
  series,
  statisticLabel,
  formatValue,
}: ErrorBarChartProps) {
  const { ref, width } = useChartWidth();
  const chartWidth = Math.max(240, width);
  const height = rows.length * ROW_HEIGHT + 56;
  const labelWidth = Math.min(380, Math.max(120, Math.round(chartWidth * 0.44)));
  // Category labels are category keys; only the tick text is shortened, so a
  // narrow card keeps the bars and loses the tail of a long model name.
  const tickChars = Math.max(10, Math.floor((labelWidth - 12) / 6.9));
  const values = rows.flatMap(row => [...row.values.values()]);
  const highest = Math.max(0, ...values);
  const lowest = Math.min(0, ...values);
  const headroom = (highest - lowest) * 0.12 || 0.01;

  // One category slot per (configuration, simulator) pair. Victory's group
  // offset is expressed in data units on a category axis, which collapses the
  // two bars in a pair; distinct slots keep them apart and leave each value
  // label its own line, on a phone as well as a wide card.
  const categories: string[] = [];
  const labelTicks: string[] = [];
  const tickText = new Map<string, string>();
  for (const row of rows) {
    series.forEach((entry, index) => {
      const key = `${row.key}\u0000${entry.simulator}`;
      categories.push(key);
      if (index === 0) {
        tickText.set(key, truncateLabel(row.label, tickChars));
        labelTicks.push(key);
      }
    });
  }
  const dataFor = (simulator: string) => rows.flatMap(row => {
    const value = row.values.get(simulator);
    return value === undefined
      ? []
      : [{ x: `${row.key}\u0000${simulator}`, y: value }];
  });

  return (
    <div className={styles.chartCanvas} ref={ref}>
      {width > 0 && rows.length > 0 && (
        <Chart
          ariaTitle={`${statisticLabel} by measured configuration`}
          ariaDesc={`Grouped bars comparing ${series.map(entry => entry.label).join(' and ')} `
            + `on ${statisticLabel} for ${rows.length} measured configurations, worst first.`}
          height={height}
          width={chartWidth}
          domain={{ y: [lowest - (lowest < 0 ? headroom : 0), highest + headroom] }}
          padding={{ top: 4, bottom: 34, left: labelWidth, right: 84 }}
        >
          <ChartAxis
            tickValues={labelTicks}
            tickFormat={(value: string) => tickText.get(value) ?? ''}
            style={AXIS_STYLE}
          />
          <ChartAxis
            dependentAxis
            showGrid
            tickFormat={(value: number) => (chartWidth < 480 ? '' : formatValue(value))}
            style={AXIS_STYLE}
          />
          <ChartGroup>
            {series.map(entry => (
              <ChartBar
                key={entry.simulator}
                horizontal
                name={entry.simulator}
                categories={{ x: categories }}
                data={dataFor(entry.simulator)}
                barWidth={12}
                labels={({ datum }: { datum: { y: number } }) => formatValue(datum.y)}
                style={{
                  data: { fill: entry.color },
                  labels: {
                    fill: '#3c3f42',
                    fontFamily: 'var(--mono)',
                    fontSize: 11.5,
                  },
                }}
              />
            ))}
          </ChartGroup>
        </Chart>
      )}
    </div>
  );
}

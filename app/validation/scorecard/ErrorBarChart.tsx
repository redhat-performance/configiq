'use client';

import * as React from 'react';
import {
  Chart,
  ChartAxis,
  ChartBar,
  ChartGroup,
  ChartLabel,
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

/**
 * Vertical padding the category labels are centred against.
 *
 * Kept beside the height formula because the label offset is derived from the
 * plot area: if one changes without the other, labels drift off their bars.
 */
const CHART_TOP_PADDING = 4;
const CHART_BOTTOM_PADDING = 34;

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
  const height = rows.length * ROW_HEIGHT + CHART_TOP_PADDING + CHART_BOTTOM_PADDING + 18;
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

  // Each configuration's label is a tick on its first simulator's slot, the
  // only position a categorical axis can give it. Move it to the mean slot of
  // the series that actually predicted this row, so two bars get a label
  // between them and a row only one simulator answered gets a label on that
  // bar. The plot height and the number of slots give the slot height without
  // measuring the rendered SVG.
  const plotHeight = height - CHART_TOP_PADDING - CHART_BOTTOM_PADDING;
  const slotHeight = categories.length > 0 ? plotHeight / categories.length : 0;
  const labelOffsets = rows.map(row => {
    const slots = series
      .map((entry, index) => (row.values.has(entry.simulator) ? index : null))
      .filter((index): index is number => index !== null);
    const meanSlot = slots.length > 0
      ? slots.reduce((total, index) => total + index, 0) / slots.length
      : 0;
    return -meanSlot * slotHeight;
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
          padding={{
            top: CHART_TOP_PADDING,
            bottom: CHART_BOTTOM_PADDING,
            left: labelWidth,
            right: 84,
          }}
        >
          <ChartAxis
            tickValues={labelTicks}
            tickFormat={(value: string) => tickText.get(value) ?? ''}
            tickLabelComponent={
              <ChartLabel
                dy={(props: { index?: string | number }) => labelOffsets[Number(props.index)] ?? 0}
              />
            }
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

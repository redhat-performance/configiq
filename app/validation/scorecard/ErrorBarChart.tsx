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

/**
 * Vertical padding the category labels are centred against.
 *
 * Kept beside the height formula because the label offset is derived from the
 * plot area: if one changes without the other, labels drift off their bars.
 */
const CHART_TOP_PADDING = 4;
const CHART_BOTTOM_PADDING = 34;

/** Light wash on alternating configurations, so a pair of bars reads as one row. */
const ROW_BAND_FILL = '#f5f5f5';

interface RowBandsProps {
  rowCount: number;
  /** Every row reserves one category point per series, including an empty one. */
  seriesCount: number;
  /** Full chart width, so the wash also ties the category label to its bars. */
  fullWidth: number;
  /** Supplied by Chart: the plot height, padding removed. */
  height?: number;
}

/**
 * Alternating row bands behind the plot, leaving even rows on the card background.
 *
 * Chart's background coordinates are the chart's scale ranges, which are
 * transposed for the horizontal bars this chart draws, so only the height is
 * used. Victory places categories on a point scale from the plot's bottom edge
 * to its top, so a band's edge is halfway between the last point of one row and
 * the first point of the next. That boundary is exactly where a reader has to
 * choose which of two adjacent bars belongs to which label.
 */
function RowBands({ rowCount, seriesCount, fullWidth, height = 0 }: RowBandsProps) {
  const categoryCount = rowCount * seriesCount;
  if (rowCount === 0 || height === 0 || categoryCount < 2) return null;
  const pitch = height / (categoryCount - 1);
  const plotTop = CHART_TOP_PADDING;
  const plotBottom = plotTop + height;
  return (
    <g aria-hidden="true">
      {Array.from({ length: rowCount }, (_, index) => {
        if (index % 2 === 0) return null;
        const first = index * seriesCount;
        const last = first + seriesCount - 1;
        const top = Math.max(plotTop, plotBottom - (last + 0.5) * pitch);
        const bottom = Math.min(plotBottom, plotBottom - (first - 0.5) * pitch);
        return (
          <rect
            key={index}
            x={0}
            y={top}
            width={fullWidth}
            height={bottom - top}
            fill={ROW_BAND_FILL}
          />
        );
      })}
    </g>
  );
}

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
  //
  // Category points are laid out from the plot's bottom edge upwards, so the
  // ranked list is reversed: the worst configuration, first in `rows`, lands
  // at the top of the chart where a reader starts.
  const displayedRows = [...rows].reverse();
  const categories: string[] = [];
  const labelTicks: string[] = [];
  const tickText = new Map<string, string>();
  for (const row of displayedRows) {
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

  // Each configuration's label is a tick on its first simulator's point, the
  // only position a categorical axis can give it. Move it to the mean point of
  // the series that actually predicted this row, so two bars get a label
  // between them and a row only one simulator answered gets a label on that
  // bar. Victory spreads category points from the plot's bottom edge to its
  // top, so the pitch is one gap fewer than the category count; using the
  // count would drift the label off its bars as the list grows.
  const plotHeight = height - CHART_TOP_PADDING - CHART_BOTTOM_PADDING;
  const pointPitch = categories.length > 1 ? plotHeight / (categories.length - 1) : 0;
  const labelOffsets = displayedRows.map(row => {
    const slots = series
      .map((entry, index) => (row.values.has(entry.simulator) ? index : null))
      .filter((index): index is number => index !== null);
    const meanSlot = slots.length > 0
      ? slots.reduce((total, index) => total + index, 0) / slots.length
      : 0;
    return -meanSlot * pointPitch;
  });

  return (
    <div className={styles.chartCanvas} ref={ref}>
      {width > 0 && rows.length > 0 && (
        <div className={styles.chartSizer} style={{ height }}>
          <Chart
            ariaTitle={`${statisticLabel} by measured configuration`}
            ariaDesc={`Grouped bars comparing ${series.map(entry => entry.label).join(' and ')} `
              + `on ${statisticLabel} for ${rows.length} measured configurations, worst first.`}
            backgroundComponent={
              <RowBands
                rowCount={rows.length}
                seriesCount={series.length}
                fullWidth={chartWidth}
              />
            }
            height={height}
            width={chartWidth}
            style={{ background: { fill: 'transparent' } }}
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
        </div>
      )}
    </div>
  );
}

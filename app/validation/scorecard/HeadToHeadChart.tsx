'use client';

import * as React from 'react';
import {
  formatPercent,
  type HeadToHeadCategory,
  type HeadToHeadSummary,
} from './scorecard-view';
import styles from './scorecard.module.css';

/**
 * Win share by model and by GPU, with the range across configurations.
 *
 * A category's split bar is the mean win share of its measured configurations;
 * the candle spans the lowest and highest configuration share. The candle is
 * what keeps a mean from overclaiming: a category at 50% built from one
 * configuration and one built from twelve configurations that range from 0 to
 * 100% look the same without it. A category with one configuration has no
 * range to draw, so it says so instead.
 */

export interface HeadToHeadSeries {
  id: string;
  label: string;
  color: string;
}

interface HeadToHeadChartProps {
  title: string;
  summary: HeadToHeadSummary;
  first: HeadToHeadSeries;
  second: HeadToHeadSeries;
  /** Formats a gap in percentage points of APE, keeping its sign. */
  formatGap: (pct: number) => string;
}

function formatCount(value: number): string {
  return value.toLocaleString('en-US');
}

export default function HeadToHeadChart({
  title,
  summary,
  first,
  second,
  formatGap,
}: HeadToHeadChartProps) {
  return (
    <div className={styles.h2hPanel}>
      <h4 className={styles.h2hPanelTitle}>{title}</h4>
      {summary.categories.length > 0 ? (
        <ul className={styles.h2hList}>
          {summary.categories.map(category => (
            <HeadToHeadRow
              key={category.key}
              category={category}
              first={first}
              second={second}
              formatGap={formatGap}
            />
          ))}
        </ul>
      ) : (
        <p className={styles.emptyRows}>No pair compared a configuration in this view.</p>
      )}
      <p className={styles.h2hCaption}>
        {formatCount(summary.configurations)} configurations
        {' · '}
        {formatCount(summary.paired)} paired levels
        {' · '}
        median gap {formatGap(summary.medianGapPct)}
        {summary.ties > 0 ? ` · ${formatCount(summary.ties)} ties` : ''}
      </p>
    </div>
  );
}

function HeadToHeadRow({
  category,
  first,
  second,
  formatGap,
}: {
  category: HeadToHeadCategory;
  first: HeadToHeadSeries;
  second: HeadToHeadSeries;
  formatGap: (pct: number) => string;
}) {
  const hasCandle = category.minShare !== null && category.maxShare !== null;
  const firstWidth = category.firstShare * 100;
  const secondWidth = category.secondShare * 100;
  const description = [
    `${category.label}:`,
    `${first.label} won ${formatPercent(category.firstShare)} of paired levels`,
    `${second.label} won ${formatPercent(category.secondShare)}`,
    `${formatCount(category.configurations)} ${category.configurations === 1 ? 'configuration' : 'configurations'}`,
    `${formatCount(category.paired)} paired levels`,
    category.ties > 0 ? `${formatCount(category.ties)} ties` : null,
    `median gap ${formatGap(category.medianGapPct)}`,
    hasCandle
      ? `range ${formatPercent(category.minShare!)} to ${formatPercent(category.maxShare!)}`
      : 'single configuration, no range',
  ].filter(Boolean).join('; ');

  return (
    <li className={styles.h2hRow}>
      <div className={styles.h2hLabel}>
        <span className={styles.h2hName} title={category.label}>{category.label}</span>
        {category.configurations === 1 && (
          <span className={styles.h2hThin}>1 configuration</span>
        )}
      </div>
      <div className={styles.h2hBarArea}>
        <div className={styles.h2hTrack} role="img" aria-label={description}>
          {firstWidth > 0 && (
            <span
              className={styles.h2hSegment}
              style={{ width: `${firstWidth}%`, background: first.color }}
            />
          )}
          {secondWidth > 0 && (
            <span
              className={styles.h2hSegment}
              style={{ width: `${secondWidth}%`, background: second.color }}
            />
          )}
          {hasCandle && (
            <span
              className={styles.h2hCandle}
              style={{
                left: `${category.minShare! * 100}%`,
                width: `${(category.maxShare! - category.minShare!) * 100}%`,
              }}
              aria-hidden="true"
            />
          )}
        </div>
        <span className={styles.h2hValue}>{formatPercent(category.firstShare)}</span>
      </div>
    </li>
  );
}

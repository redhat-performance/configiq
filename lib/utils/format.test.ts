import { describe, expect, it } from 'vitest'

import { formatDatasetDate, formatLatencyBound } from './format'

describe('formatLatencyBound', () => {
  it('renders whole and fractional seconds without trailing zeros', () => {
    expect(formatLatencyBound(1000)).toBe('1')
    expect(formatLatencyBound(2500)).toBe('2.5')
    expect(formatLatencyBound(250)).toBe('0.25')
  })

  it('preserves sub-millisecond target precision', () => {
    expect(formatLatencyBound(1.4)).toBe('0.0014')
  })
})

describe('formatDatasetDate', () => {
  it('spells out the UTC calendar date', () => {
    // 19:22 UTC is the next calendar day east of UTC. A dataset timestamp is
    // not a local event, so the date must not move with the reader's zone.
    expect(formatDatasetDate('2026-11-11T19:22:57.864Z')).toBe('11 Nov 2026');
    expect(formatDatasetDate('2026-11-11T00:00:00.000Z')).toBe('11 Nov 2026');
    expect(formatDatasetDate('2026-11-11T23:59:59.000Z')).toBe('11 Nov 2026');
  });

  it('names the month so day-first and month-first readers agree', () => {
    // 5 December reads as both 5/12 and 12/5; "5 Dec 2026" does not.
    expect(formatDatasetDate('2026-12-05T12:00:00.000Z')).toBe('5 Dec 2026');
  });

  it('returns null for a missing or unparseable value', () => {
    expect(formatDatasetDate(null)).toBeNull();
    expect(formatDatasetDate(undefined)).toBeNull();
    expect(formatDatasetDate('')).toBeNull();
    expect(formatDatasetDate('not a date')).toBeNull();
  });
});

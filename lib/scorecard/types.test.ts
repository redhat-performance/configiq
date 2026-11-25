import { describe, expect, it } from 'vitest';
import {
  scorecardPointsSchema,
  scorecardSummarySchema,
} from './types';
import { point, summary } from './__fixtures__/points';

describe('scorecardSummarySchema', () => {
  it('accepts a complete artifact', () => {
    expect(scorecardSummarySchema.safeParse(summary()).success).toBe(true);
  });

  it('accepts every point status, including ineligible', () => {
    for (const status of ['supported', 'unsupported', 'inconclusive', 'ineligible'] as const) {
      const parsed = scorecardPointsSchema.safeParse({
        schemaVersion: 1,
        channel: 'dev',
        generatedAt: '2026-11-20T10:20:00.000Z',
        points: [point({ status, predicted: status === 'supported' ? point().predicted : null })],
      });
      expect(parsed.success, status).toBe(true);
    }
  });

  it('rejects an unknown schema version', () => {
    expect(scorecardSummarySchema.safeParse({ ...summary(), schemaVersion: 2 }).success).toBe(false);
  });

  it('rejects an unknown channel', () => {
    expect(scorecardSummarySchema.safeParse({ ...summary(), channel: 'staging' }).success).toBe(false);
  });

  it('requires the ineligible coverage bucket', () => {
    const parsed = scorecardSummarySchema.safeParse({
      ...summary(),
      coverage: [{
        simulator: 'aisim',
        source: 'redhat',
        model: 'org/model',
        accelerator: 'H200',
        releaseMatch: 'any',
        supported: 1,
        unsupported: 0,
        inconclusive: 0,
        total: 1,
      }],
    });
    expect(parsed.success).toBe(false);
  });

  it('rejects a negative count', () => {
    const parsed = scorecardSummarySchema.safeParse({
      ...summary(),
      totals: [{
        simulator: 'aisim',
        source: 'all',
        metric: 'ttft',
        releaseMatch: 'any',
        statistic: 'mape',
        value: 0.1,
        configurations: -1,
        points: 1,
      }],
    });
    expect(parsed.success).toBe(false);
  });

  it('rejects a head-to-head pair with one element', () => {
    const parsed = scorecardSummarySchema.safeParse({
      ...summary(),
      headToHead: [{
        source: 'redhat',
        model: 'org/model',
        accelerator: 'H200',
        workload: '1024/128',
        metric: 'ttft',
        pair: ['aisim'],
        paired: 1,
        firstWins: 1,
        secondWins: 0,
        ties: 0,
        medianDeltaPct: 0,
      }],
    });
    expect(parsed.success).toBe(false);
  });
});

describe('scorecardPointsSchema', () => {
  it('accepts a point without a prediction', () => {
    const parsed = scorecardPointsSchema.safeParse({
      schemaVersion: 1,
      channel: 'prod',
      generatedAt: '2026-11-20T10:20:00.000Z',
      points: [point({ status: 'unsupported', reason: 'no such release', predicted: null })],
    });
    expect(parsed.success).toBe(true);
  });

  it('rejects an empty identity', () => {
    const parsed = scorecardPointsSchema.safeParse({
      schemaVersion: 1,
      channel: 'dev',
      generatedAt: '2026-11-20T10:20:00.000Z',
      points: [point({ identity: '' })],
    });
    expect(parsed.success).toBe(false);
  });
});

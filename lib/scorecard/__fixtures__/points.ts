/** Test-only builders: never imported by production components. */
import type { ScorecardPoint, ScorecardSummary } from '../types';

/**
 * One supported point, close enough to be useful in every arithmetic test.
 *
 * Predicted TTFT is 10% high and predicted TPOT is 10% low, so a test can
 * assert both a magnitude and a sign without recomputing them.
 */
export function point(overrides: Partial<ScorecardPoint> = {}): ScorecardPoint {
  return {
    identity: 'point-1',
    source: 'redhat',
    model: 'org/model',
    accelerator: 'H200',
    isl: 1024,
    osl: 128,
    tp: 4,
    concurrency: 8,
    workload: '1024/128',
    configurationKey: 'configuration-1',
    measuredRelease: '0.31.0',
    releaseMatch: false,
    servingMode: 'agg',
    simulator: 'aisimulate',
    status: 'supported',
    reason: null,
    measured: { ttftMs: 100, tpotMs: 10, outputTokensPerSecond: 100 },
    predicted: { ttftMs: 110, tpotMs: 9, requestLatencyMs: 1250, outputTokensPerSecond: 90 },
    engineVersion: '0.31.0',
    ...overrides,
  };
}

/** A summary with valid provenance and no aggregate rows. */
export function summary(overrides: Partial<ScorecardSummary> = {}): ScorecardSummary {
  return {
    schemaVersion: 1,
    channel: 'dev',
    generatedAt: '2026-11-20T10:20:00.000Z',
    gatewayUrl: 'https://aisimulators.example.test',
    durationSeconds: 120,
    attemptedPoints: 4,
    answeredPoints: 3,
    cachedPoints: 0,
    datasets: [{
      source: 'redhat',
      id: 'redhat-performance/configiq-performance-upstream-data',
      revision: 'abc123',
      rowCount: 10,
    }],
    simulators: [{
      id: 'aisimulate',
      label: 'AISimulate',
      version: '0.14.0',
      revision: 'deadbeef',
      available: true,
    }],
    assumptions: {
      ppSize: 1,
      dpSize: 1,
      moeWidths: 'simulator-resolved',
      fromMeasuredRow: ['model', 'accelerator', 'isl', 'osl', 'tp', 'concurrency'],
    },
    incomplete: [],
    cells: [],
    coverage: [],
    totals: [],
    headToHead: [],
    ...overrides,
  };
}

import { normalizeSimulatorId, normalizeSimulatorLabel } from '@/lib/simulator-ids';
import type { ScorecardPoint, ScorecardSummary } from './types';

/**
 * Bring a published snapshot onto the simulator ids and labels this release
 * uses.
 *
 * The sweep is published on its own schedule, so a snapshot can outlive a
 * simulator rename. The artefact is data, not a capability contract: reading
 * an old id as the current one here keeps an already-published scorecard
 * readable without teaching the service to accept the retired id again.
 * Values are untouched — only identity and its display label are rewritten.
 */
export function normalizeScorecardSummary(summary: ScorecardSummary): ScorecardSummary {
  return {
    ...summary,
    simulators: summary.simulators.map(simulator => ({
      ...simulator,
      id: normalizeSimulatorId(simulator.id),
      label: normalizeSimulatorLabel(simulator.id, simulator.label),
    })),
    cells: summary.cells.map(cell => ({
      ...cell,
      simulator: normalizeSimulatorId(cell.simulator),
    })),
    coverage: summary.coverage.map(row => ({
      ...row,
      simulator: normalizeSimulatorId(row.simulator),
    })),
    totals: summary.totals.map(row => ({
      ...row,
      simulator: normalizeSimulatorId(row.simulator),
    })),
    headToHead: summary.headToHead.map(row => ({
      ...row,
      pair: [
        normalizeSimulatorId(row.pair[0]),
        normalizeSimulatorId(row.pair[1]),
      ],
    })),
  };
}

/** The same migration for the per-point document. */
export function normalizeScorecardPoints(points: ScorecardPoint[]): ScorecardPoint[] {
  return points.map(point => ({
    ...point,
    simulator: normalizeSimulatorId(point.simulator),
  }));
}

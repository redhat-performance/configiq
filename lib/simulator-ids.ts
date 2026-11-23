/**
 * Simulator ids this app has published, and the id that replaced them.
 *
 * The service renamed its analytical simulator from `aisimulate` to `aisim`.
 * A profile saved before the rename, and a link shared before it, still carry
 * the old id. Reading it as the new one is a one-time data migration, not a
 * capability rule: nothing here decides what a simulator accepts.
 */
const LEGACY_SIMULATOR_IDS: Record<string, string> = {
  aisimulate: 'aisim',
};

/** The current id for a simulator, accepting ids from earlier releases. */
export function normalizeSimulatorId(id: string): string {
  return LEGACY_SIMULATOR_IDS[id] ?? id;
}

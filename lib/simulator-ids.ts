/**
 * Simulator ids this app has published, and the id that replaced them.
 *
 * The service renamed its analytical simulator from `aisimulate` to `aisim`.
 * A profile saved before the rename, and a link shared before it, still carry
 * the old id. Reading it as the new one is a one-time data migration, not a
 * capability rule: nothing here decides what a simulator accepts.
 *
 * This is a Map, not an object literal, so an external value such as
 * `__proto__` is looked up as a string and never resolves an inherited member.
 */
const LEGACY_SIMULATOR_IDS = new Map<string, string>([
  ['aisimulate', 'aisim'],
]);

/** The current id for a simulator, accepting ids from earlier releases. */
export function normalizeSimulatorId(id: string): string {
  return LEGACY_SIMULATOR_IDS.get(id) ?? id;
}

/**
 * The label that went with a retired id.
 *
 * A published snapshot records the label the catalogue had when the sweep ran,
 * so an artefact from before the rename still says the old name. Map that old
 * id to the current label when rendering; a current id keeps whatever label
 * its snapshot carries.
 */
const LEGACY_SIMULATOR_LABELS = new Map<string, string>([
  ['aisimulate', 'AISim'],
]);

/** A simulator's display label, accepting labels from earlier releases. */
export function normalizeSimulatorLabel(id: string, label: string): string {
  return LEGACY_SIMULATOR_LABELS.get(id) ?? label;
}

/**
 * Retired id → current id, for migrating values this app stored under the old
 * id. A caller copies a retired key's value to the current key and drops it.
 */
export function simulatorIdMigrations(): ReadonlyArray<readonly [string, string]> {
  return [...LEGACY_SIMULATOR_IDS.entries()];
}

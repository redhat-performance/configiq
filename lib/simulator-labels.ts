/**
 * Display names for simulators in the UI.
 *
 * Simulator identity and labels come from the catalog. This only shortens a
 * name for display; it never changes request values, capability lookup, or the
 * simulator reported in a response.
 */
const SIMULATOR_DISPLAY_LABELS: Record<string, string> = {
  aisimulate: 'AISim',
}

/** The label to show for a simulator, falling back to the catalog's label. */
export function simulatorDisplayLabel(id: string, fallback: string): string {
  return SIMULATOR_DISPLAY_LABELS[id] ?? fallback
}

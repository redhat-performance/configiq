/**
 * Simulator colors, shared by every page that charts one simulator against
 * another.
 *
 * A brand color is keyed by simulator id, not by position: a reader who learns
 * that BLIS is blue on Benchmark comparison must see the same blue on the
 * scorecard. Simulators without a brand entry fall back to a stable palette by
 * position, so a new simulator still gets a distinct line.
 *
 * The `aisimulate` keys are the pre-rename id. They stay so a snapshot or a
 * cached response published under the old id still paints in the right color.
 */

/** Brand colors for the simulators the catalog can return. */
export const SIMULATOR_COLORS: Record<string, string> = {
  // NVIDIA green
  aisim: '#76b900',
  aisimulate: '#76b900',
  // IBM blue
  blis: '#0f62fe',
};

/** Stable line colors, used for simulators without a brand color. */
export const PREDICTION_COLORS = ['#009596', '#5752d1', '#3e8635', '#a18f00'] as const;

/** A simulator's brand color, falling back to the shared palette by position. */
export function predictionColor(simulator: string, index: number): string {
  return SIMULATOR_COLORS[simulator] ?? PREDICTION_COLORS[index % PREDICTION_COLORS.length];
}

/**
 * Text-safe brand shades.
 *
 * Headings sit on the striped table header (#e8e8e8). AISim green (#76b900) and
 * BLIS blue (#0f62fe, 4.08:1) both miss the 4.5:1 normal-text threshold there at
 * 11.5px, so each series keeps its hue in a darker shade that clears it.
 */
export const SIMULATOR_TEXT_COLORS: Record<string, string> = {
  aisim: '#4a6e00',
  aisimulate: '#4a6e00',
  blis: '#0043ce',
};

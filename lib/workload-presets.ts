export interface WorkloadPreset {
  key: string
  label: string
  isl: number
  osl: number
}

export const DEFAULT_WORKLOAD: WorkloadPreset = {
  key: 'default',
  label: 'Default',
  isl: 2048,
  osl: 128
}

/**
 * The workload shapes a page offers: the built-in default first, then the
 * deployment's configured presets.
 */
export function workloadPresetOptions(presets: readonly WorkloadPreset[]): WorkloadPreset[] {
  return [DEFAULT_WORKLOAD, ...presets]
}

/**
 * The preset that names this input/output shape, or null when the reader has a
 * custom one.
 *
 * Selection follows the values, as it does for concurrency presets: clicking a
 * preset writes the lengths, and a shared view that restores those lengths must
 * also restore the selection. Keeping the choice in separate state would let a
 * link apply the numbers while the chip still claimed "Default".
 */
export function matchWorkloadPreset(
  presets: readonly WorkloadPreset[],
  isl: number,
  osl: number,
): WorkloadPreset | null {
  return presets.find(preset => preset.isl === isl && preset.osl === osl) ?? null
}

/**
 * The preset whose tokens-per-request total matches, for a page that models
 * only the combined context budget. Such a page cannot tell two shapes with the
 * same total apart, so the first match is the honest answer.
 */
export function matchWorkloadPresetByTokens(
  presets: readonly WorkloadPreset[],
  tokens: number,
): WorkloadPreset | null {
  return presets.find(preset => preset.isl + preset.osl === tokens) ?? null
}

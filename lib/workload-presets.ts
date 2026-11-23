export interface WorkloadPreset {
  key: string
  label: string
  isl: number
  osl: number
  /**
   * Cached prefix tokens shared by every request, a subset of `isl`.
   * Omitted means the preset has no reuse assumption.
   */
  prefix?: number
}

export const DEFAULT_WORKLOAD: WorkloadPreset = {
  key: 'default',
  label: 'Default',
  isl: 2048,
  osl: 128,
  // A system prompt is cached on every request, so the default shape still
  // reflects real usage (#236). Clear the field for a strictly cold view.
  prefix: 200
}

/**
 * The workload shapes a page offers: the built-in default first, then the
 * deployment's configured presets.
 */
export function workloadPresetOptions(presets: readonly WorkloadPreset[]): WorkloadPreset[] {
  return [DEFAULT_WORKLOAD, ...presets]
}

/**
 * The preset that names this input/output shape and prefix, or null when the
 * reader has a custom one.
 *
 * Selection follows the values, as it does for concurrency presets: clicking a
 * preset writes the lengths and prefix, and a shared view that restores those
 * values must also restore the selection. Keeping the choice in separate state
 * would let a link apply the numbers while the chip still claimed "Default".
 * Prefix is part of the identity because two views with the same lengths but a
 * different reuse assumption are different workloads.
 */
export function matchWorkloadPreset(
  presets: readonly WorkloadPreset[],
  isl: number,
  osl: number,
  prefix: number,
): WorkloadPreset | null {
  return presets.find(preset =>
    preset.isl === isl &&
    preset.osl === osl &&
    (preset.prefix ?? 0) === prefix,
  ) ?? null
}

/**
 * The preset whose tokens-per-request total matches, for a page that models
 * only the combined context budget. Such a page cannot tell two shapes with the
 * same total apart, and it has no prefix control, so the first match is the
 * honest answer.
 */
export function matchWorkloadPresetByTokens(
  presets: readonly WorkloadPreset[],
  tokens: number,
): WorkloadPreset | null {
  return presets.find(preset => preset.isl + preset.osl === tokens) ?? null
}

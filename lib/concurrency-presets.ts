export interface ConcurrencyPreset {
  key: string
  label: string
  concurrency: number
}

/**
 * The t-shirt preset that matches a concurrency value, or an empty string when
 * it matches none.
 *
 * Accepts the raw text-field value so a manually typed "10" selects the same
 * preset as clicking "Small (10)", and changing it to "5" clears the selection.
 */
export function matchConcurrencyPresetKey(
  presets: readonly ConcurrencyPreset[],
  value: string | number,
): string {
  const parsed = typeof value === 'number' ? value : Number.parseInt(value, 10)
  if (!Number.isSafeInteger(parsed) || parsed < 1) return ''
  return presets.find(preset => preset.concurrency === parsed)?.key ?? ''
}

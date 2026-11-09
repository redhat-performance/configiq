/**
 * Compact chip labels shared by the model ComboBox and the tested-model pages.
 * Keeping one implementation prevents the filter rows and result cards from
 * drifting apart.
 */

/** Compact parameter count for a chip: 0.6B, 32B, 2.4T. Matches modelSizeLabel. */
export function paramsChip(paramsB: number | null): string | null {
  if (paramsB === null) return null
  if (paramsB >= 1_000) return `${Number((paramsB / 1_000).toFixed(2))}T`
  if (paramsB >= 0.1) return `${Number(paramsB.toFixed(1))}B`
  return `${Math.round(paramsB * 1_000)}M`
}

/** Compact context window for a chip: 128K, 1M. */
export function contextChip(contextLength: number | null): string | null {
  if (!contextLength) return null
  if (contextLength >= 1_048_576) return `${Number((contextLength / 1_048_576).toFixed(1))}M ctx`
  if (contextLength >= 1_024) return `${Math.round(contextLength / 1_024)}K ctx`
  return `${contextLength} ctx`
}

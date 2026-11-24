/** Helpers shared by the result-to-snapshot builders. */

/** A stable local identifier. Not meaningful outside this browser. */
export function newComparisonId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

/** A reported number, or null when the result did not give a finite one. */
export function finite(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

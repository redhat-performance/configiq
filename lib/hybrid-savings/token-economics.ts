/**
 * Output-token economics for Hybrid Savings.
 *
 * The primary comparison metric is cost per million *requested output*
 * tokens: the selected monthly cost — which includes input processing and
 * every other item in the selected cost view — divided by the requested
 * output volume. It is not the provider's output-token tariff, and the same
 * denominator is applied to hosted, rented and purchased options.
 */

const MILLION = 1_000_000
const usd = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' })
const decimal = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 })

/**
 * Selected monthly cost per million requested output tokens.
 *
 * Returns null when the metric is undefined: non-finite inputs, negative
 * cost, non-positive output or a non-finite result. A genuine zero cost with
 * positive output returns 0.
 */
export function costPerMillionOutputTokens(
  monthlyCost: number,
  outputTokens: number,
): number | null {
  if (!Number.isFinite(monthlyCost) || !Number.isFinite(outputTokens)) return null
  if (monthlyCost < 0 || outputTokens <= 0) return null
  const value = monthlyCost * MILLION / outputTokens
  return Number.isFinite(value) ? value : null
}

/** Output volume carried by a billed-token sample of the current mix. */
export function outputTokensAtVolume(volume: number, outputShare: number): number {
  if (!Number.isFinite(volume) || !Number.isFinite(outputShare)) return 0
  const share = Math.min(Math.max(outputShare, 0), 1)
  return Math.max(volume, 0) * share
}

/** Card display for the output-unit metric; undefined renders as text. */
export function formatCostPerMillionOutputTokens(value: number | null): string {
  return value === null ? 'Not available' : formatHybridUsd(value)
}

/** Keep tiny nonzero estimates visibly distinct from zero. */
export function formatHybridUsd(value: number): string {
  return value > 0 && value < 0.01 ? '<$0.01' : usd.format(value)
}

export function formatPlanningCapacityUse(percent: number): string {
  return percent > 0 && percent < 0.1 ? '<0.1%' : `${decimal.format(percent)}%`
}

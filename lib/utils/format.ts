/**
 * General formatting utilities.
 */

/** Format a number to a fixed number of decimal places, stripping trailing zeros */
export function formatNumber(value: number, decimals = 2): string {
  return parseFloat(value.toFixed(decimals)).toString();
}

/** Format a GB value with appropriate unit */
export function formatMemory(gb: number): string {
  if (gb >= 1000) return `${formatNumber(gb / 1000, 1)} TB`;
  return `${formatNumber(gb, 1)} GB`;
}

/** Render a millisecond latency bound in seconds, dropping a trailing `.000`. */
export function formatLatencyBound(ms: number): string {
  const sec = (ms / 1000).toFixed(3)
  return sec.endsWith('000') ? (ms / 1000).toFixed(0) : parseFloat(sec).toString()
}

/** Format a USD cost */
export function formatCost(usd: number): string {
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(2)}`;
}

/** Format tokens per second */
export function formatThroughput(tps: number): string {
  if (tps >= 1000) return `${formatNumber(tps / 1000, 1)}k tok/s`;
  return `${formatNumber(tps, 0)} tok/s`;
}

/** Format a byte count using binary units (GiB, MiB, KiB) */
export function formatBytes(bytes: number): string {
  if (bytes >= 1_073_741_824) return `${formatNumber(bytes / 1_073_741_824, 2)} GiB`;
  if (bytes >= 1_048_576) return `${formatNumber(bytes / 1_048_576, 2)} MiB`;
  if (bytes >= 1024) return `${formatNumber(bytes / 1024, 2)} KiB`;
  return `${bytes} B`;
}

/** Average hours per month (365 days × 24 hrs ÷ 12 months). Used for GPU cost calculations. */
export const HOURS_PER_MONTH = 730;

/** Months in 3-year hardware amortisation period. */
export const AMORT_MONTHS_3YR = 36;

/** Months in 5-year hardware lifecycle period (for 5-year TCO calculations). */
export const AMORT_MONTHS_5YR = 60;

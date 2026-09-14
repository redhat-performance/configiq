// Single source of truth for the AIConfigurator request timeout.
//
// The default must stay in sync with nginx's proxy_read_timeout for the webapp
// (see configiq-deploy/deploy.sh) — nginx must outlast this so a slow request
// surfaces the app's AIC_TIMEOUT rather than a bare 504.

/** Default AIConfigurator timeout (seconds) when the env var is unset. */
export const DEFAULT_AIC_TIMEOUT_SECONDS = 90

/**
 * Resolve the configured AIConfigurator timeout in seconds, honoring the
 * AICONFIGURATOR_TIMEOUT_SECONDS env var and falling back to `defaultSeconds`.
 * Only a positive integer is accepted; anything else (negative, zero,
 * non-integer, or unparseable) uses the fallback — a negative value would make
 * AbortSignal.timeout() throw and fail every request. `defaultSeconds` lets
 * callers with a different baseline (e.g. the catalog fetch's 30s) share this
 * validation. Server-side only (reads process.env).
 */
export function aicTimeoutSeconds(defaultSeconds: number = DEFAULT_AIC_TIMEOUT_SECONDS): number {
  // Number() (not parseInt) so partial values like "1.5" or "90seconds" are
  // rejected rather than truncated; only a whole positive number is honored.
  const parsed = Number(process.env.AICONFIGURATOR_TIMEOUT_SECONDS)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : defaultSeconds
}

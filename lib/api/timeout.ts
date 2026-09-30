// Single source of truth for the AISimulators request timeout.
//
// The default must stay below nginx's webapp proxy_read_timeout (120s; see
// configiq-deploy/deploy.sh) so a slow request surfaces the app's
// AISIM_TIMEOUT rather than a bare nginx 504.

/** Default AISimulators timeout (seconds) when the env var is unset. */
export const DEFAULT_GATEWAY_TIMEOUT_SECONDS = 90

/**
 * Resolve the configured AISimulators timeout in seconds, honoring the
 * AISIMULATORS_TIMEOUT_SECONDS env var and falling back to `defaultSeconds`.
 * Only a positive integer is accepted; anything else (negative, zero,
 * non-integer, or unparseable) uses the fallback — a negative value would make
 * AbortSignal.timeout() throw and fail every request. `defaultSeconds` lets
 * callers with a different baseline (e.g. the catalog fetch's 30s) share this
 * validation. Server-side only (reads process.env).
 */
export function gatewayTimeoutSeconds(defaultSeconds: number = DEFAULT_GATEWAY_TIMEOUT_SECONDS): number {
  // Number() (not parseInt) so partial values like "1.5" or "90seconds" are
  // rejected rather than truncated; only a whole positive number is honored.
  const parsed = Number(process.env.AISIMULATORS_TIMEOUT_SECONDS)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : defaultSeconds
}

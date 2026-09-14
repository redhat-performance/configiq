// Single source of truth for the AIConfigurator request timeout.
//
// The default must stay in sync with nginx's proxy_read_timeout for the webapp
// (see configiq-deploy/deploy.sh) — nginx must outlast this so a slow request
// surfaces the app's AIC_TIMEOUT rather than a bare 504.

/** Default AIConfigurator timeout (seconds) when the env var is unset. */
export const DEFAULT_AIC_TIMEOUT_SECONDS = 90

/**
 * Resolve the configured AIConfigurator timeout in seconds, honoring the
 * AICONFIGURATOR_TIMEOUT_SECONDS env var and falling back to the default.
 * Server-side only (reads process.env).
 */
export function aicTimeoutSeconds(): number {
  return (
    parseInt(process.env.AICONFIGURATOR_TIMEOUT_SECONDS || '', 10) ||
    DEFAULT_AIC_TIMEOUT_SECONDS
  )
}

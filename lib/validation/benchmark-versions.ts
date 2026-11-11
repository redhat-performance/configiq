export const UNREPORTED_VERSION = 'Version not reported'

/** Version fields arrive as strings, numbers, or bigints from Parquet. */
function normalizeVersionValue(value: unknown): string {
  if (value == null) return ''
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : ''
  if (typeof value === 'bigint') return value.toString()
  return ''
}

/**
 * Canonical release label for a backend version.
 *
 * Settings offers catalog literals (`0.31.0`, `0.5.21`), while measured rows
 * keep the release label as reported (`v0.31.0`, `v1.3.0rc29`). Canonicalizing
 * every backend, not just vLLM, keeps an explicit non-vLLM pin from being
 * reported as missing. Suffixes are preserved so distinct builds and releases
 * stay distinct, and labels without a release number (nightly tags) pass
 * through unchanged.
 */
export function normalizeBackendVersion(_backend: unknown, value: unknown): string {
  const version = normalizeVersionValue(value)
  if (!version) return UNREPORTED_VERSION
  const release = /^(?:(?:vllm|sglang|tensorrt-llm|trtllm|trt)[-\s])?v?(\d+\.\d+\.\d+(?:[-+]?[a-z][\w.-]*)?)$/i.exec(version)
  if (release) return `v${release[1]}`
  return version
}

export interface BackendVersionIdentity {
  /** Canonical display label, for example `v0.24.0` or `v0.5.14-rocm720`. */
  label: string
  /** Release core without the `v` prefix, for example `0.24.0` or `1.3.0rc29`. */
  core: string | null
  /** True when the label names only the release, with no build suffix. */
  bare: boolean
}

/**
 * Split a normalized label into its release core and build suffix.
 *
 * `v0.5.14-rocm720-mi35x` has core `0.5.14` and is not bare;
 * `v1.3.0rc29` has core `1.3.0rc29` and is bare. Reusing the same split for
 * both sides of a comparison keeps catalog pins and measured image tags
 * consistent.
 */
export function backendVersionIdentity(value: unknown): BackendVersionIdentity {
  const label = normalizeBackendVersion(null, value)
  if (label === UNREPORTED_VERSION) return { label, core: null, bare: false }
  const match = /^v?(\d+\.\d+\.\d+(?:[a-z]+\d*)?)(.*)$/i.exec(label)
  if (!match) return { label, core: null, bare: false }
  return {
    label,
    core: match[1].toLocaleLowerCase(),
    bare: match[2] === '',
  }
}

/**
 * Whether a requested release includes a measured release label.
 *
 * A bare pin (`0.5.14`) matches a measured vendor build of the same release
 * (`v0.5.14-rocm720-mi35x`), which is how a Settings catalog literal compares
 * with an InferenceX image tag. A pin to a specific build only matches that
 * build, and different releases never match.
 */
export function backendVersionsMatch(requested: unknown, measured: unknown): boolean {
  const wanted = backendVersionIdentity(requested)
  const actual = backendVersionIdentity(measured)
  if (wanted.label === actual.label) return true
  return wanted.bare && wanted.core !== null && wanted.core === actual.core
}

/**
 * Whether two release labels can be compared as the same release.
 *
 * This is the symmetric form used for provenance: a simulator reports the
 * catalog release (`0.5.14`) while the measured run may name a vendor build
 * (`v0.5.14-rocm720-mi35x`). Distinct builds of one release still stay distinct.
 */
export function backendVersionsComparable(left: unknown, right: unknown): boolean {
  const first = backendVersionIdentity(left)
  const second = backendVersionIdentity(right)
  if (first.label === second.label) return true
  return (first.bare || second.bare) && first.core !== null && first.core === second.core
}

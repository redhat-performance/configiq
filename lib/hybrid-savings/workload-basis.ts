import { DEFAULT_WORKLOAD, type WorkloadPreset } from '@/lib/workload-presets'
import type { HybridWorkload } from './calc'

/**
 * Request-mix analysis for Hybrid Savings.
 *
 * Monthly input/output totals and the representative request shape are two
 * independent inputs. Sizing conservatively follows the larger side's
 * implied request count (see workloadMix in calc.ts); these helpers make
 * that basis visible and offer a deterministic alignment that keeps the
 * output length and both monthly totals unchanged.
 */

/** Relative request-count mismatch at or below this is treated as aligned. */
export const REQUEST_MIX_ALIGNMENT_TOLERANCE = 0.005

export interface RequestMixAnalysis {
  monthlyInputTokens: number
  monthlyOutputTokens: number
  averageInputTokens: number
  averageOutputTokens: number
  /** Both monthly volumes are positive, so a two-sided mix exists. */
  twoSided: boolean
  /** Monthly input tokens divided by average input length; null when not computable. */
  inputImpliedRequests: number | null
  /** Monthly output tokens divided by average output length; null when not computable. */
  outputImpliedRequests: number | null
  /** The larger implied request count used for sizing; null when neither side is computable. */
  sizingRequests: number | null
  /** |Ni - No| / max(Ni, No); null when the mix is not two-sided. */
  relativeMismatch: number | null
  /** Within tolerance; only meaningful when twoSided. */
  aligned: boolean
}

/** Hybrid demand aligned with the shared default request shape. */
export const DEFAULT_HYBRID_WORKLOAD: HybridWorkload = {
  monthlyInputTokens: 2_400_000_000,
  monthlyOutputTokens: 150_000_000,
  averageInputTokens: DEFAULT_WORKLOAD.isl,
  averageOutputTokens: DEFAULT_WORKLOAD.osl,
  activeHoursPerMonth: 730,
  peakToAverage: 1,
}

function nonNegativeFinite(value: number): number {
  return Number.isFinite(value) && value > 0 ? value : 0
}

/**
 * Implied request counts and the conservative sizing basis for the current
 * workload. Values mirror the engine's clamping in workloadFacts.
 */
export function analyzeRequestMix(workload: HybridWorkload): RequestMixAnalysis {
  const monthlyInputTokens = nonNegativeFinite(workload.monthlyInputTokens)
  const monthlyOutputTokens = nonNegativeFinite(workload.monthlyOutputTokens)
  const averageInputTokens = Number.isFinite(workload.averageInputTokens) && workload.averageInputTokens > 0
    ? workload.averageInputTokens
    : 0
  const averageOutputTokens = Number.isFinite(workload.averageOutputTokens) && workload.averageOutputTokens > 0
    ? workload.averageOutputTokens
    : 0

  const inputImpliedRequests = averageInputTokens > 0 && monthlyInputTokens > 0
    ? monthlyInputTokens / averageInputTokens
    : null
  const outputImpliedRequests = averageOutputTokens > 0 && monthlyOutputTokens > 0
    ? monthlyOutputTokens / averageOutputTokens
    : null
  const sizingRequests = inputImpliedRequests === null && outputImpliedRequests === null
    ? null
    : Math.max(inputImpliedRequests ?? 0, outputImpliedRequests ?? 0)
  const relativeMismatch = inputImpliedRequests !== null && outputImpliedRequests !== null
    ? Math.abs(inputImpliedRequests - outputImpliedRequests) /
      Math.max(inputImpliedRequests, outputImpliedRequests)
    : null

  return {
    monthlyInputTokens,
    monthlyOutputTokens,
    averageInputTokens,
    averageOutputTokens,
    twoSided: monthlyInputTokens > 0 && monthlyOutputTokens > 0,
    inputImpliedRequests,
    outputImpliedRequests,
    sizingRequests,
    relativeMismatch,
    aligned: relativeMismatch !== null && relativeMismatch <= REQUEST_MIX_ALIGNMENT_TOLERANCE,
  }
}

export interface RequestMixAlignment {
  /** Deterministic proposal for averageInputTokens; null when not applicable. */
  proposedInputTokens: number | null
  /** Why no proposal is available; null when a proposal is offered. */
  reason: string | null
}

/**
 * Deterministic alignment proposal: the average input length that makes the
 * input-implied request count match the output-implied request count,
 * keeping the output length and both monthly totals unchanged.
 *
 * A returned proposal has already been validated: it is a safe integer of at
 * least one token, not shorter than the cached prefix, and it brings the
 * mix within tolerance. When alignment is not possible the reason explains
 * why.
 */
export function proposeRequestMixAlignment(
  workload: HybridWorkload,
  prefixTokens: number,
): RequestMixAlignment {
  const analysis = analyzeRequestMix(workload)
  if (!analysis.twoSided) {
    return {
      proposedInputTokens: null,
      reason: 'Enter positive monthly input and output volumes to align the request mix.',
    }
  }
  if (analysis.aligned) {
    return { proposedInputTokens: null, reason: null }
  }

  const raw = workload.averageOutputTokens * workload.monthlyInputTokens / workload.monthlyOutputTokens
  const proposed = Math.round(raw)
  if (!Number.isFinite(raw)) {
    return { proposedInputTokens: null, reason: 'The proposed average input length is not a finite number.' }
  }
  if (!Number.isSafeInteger(proposed)) {
    return { proposedInputTokens: null, reason: 'The proposed average input length is not a safe integer.' }
  }
  if (proposed < 1) {
    return { proposedInputTokens: null, reason: 'The proposed average input length is below one token.' }
  }
  if (proposed < prefixTokens) {
    return {
      proposedInputTokens: null,
      reason: `The proposed average input length (${proposed.toLocaleString()}) is shorter than the cached prefix (${prefixTokens.toLocaleString()}).`,
    }
  }

  const revisedInputRequests = analysis.monthlyInputTokens / proposed
  const denominator = Math.max(revisedInputRequests, analysis.outputImpliedRequests ?? 0)
  const revisedMismatch = denominator > 0
    ? Math.abs(revisedInputRequests - (analysis.outputImpliedRequests ?? 0)) / denominator
    : 0
  if (revisedMismatch > REQUEST_MIX_ALIGNMENT_TOLERANCE) {
    return { proposedInputTokens: null, reason: 'Rounding prevents aligning this mix within tolerance.' }
  }

  return { proposedInputTokens: proposed, reason: null }
}

/**
 * Presets control only input and output lengths, independently of sizing targets.
 */
export function matchRequestShapePreset(
  fields: Pick<HybridWorkload, 'averageInputTokens' | 'averageOutputTokens'>,
  presets: readonly WorkloadPreset[],
): WorkloadPreset | null {
  return presets.find(preset =>
    preset.isl === fields.averageInputTokens &&
    preset.osl === fields.averageOutputTokens,
  ) ?? null
}

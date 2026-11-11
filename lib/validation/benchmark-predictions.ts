import type { DatasetConfiguration } from './benchmark-types'

export type ServingMode = 'agg' | 'disagg'

export interface MeasuredPoolTopology {
  prefillTp: number
  decodeTp: number
  prefillWorkers: number
  decodeWorkers: number
  /** True when the measured run named both pools' tensor and worker widths. */
  known: boolean
}

function positiveInteger(value: string | null | undefined): number | null {
  const parsed = Number(value)
  return value && Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null
}

/**
 * Prefill/decode shape recorded by the measured run.
 *
 * InferenceX disaggregated rows record prefill and decode TP plus worker
 * counts; aggregated and Red Hat rows do not. Predictions use the recorded
 * pool widths so a disaggregated comparison does not silently run a different
 * topology, and `known` gates deltas when the measured shape is unavailable.
 */
export function measuredPoolTopology(
  configuration: DatasetConfiguration | null | undefined,
  decodeTpFallback: number,
): MeasuredPoolTopology {
  const fallback = Number.isFinite(decodeTpFallback) && decodeTpFallback > 0 ? decodeTpFallback : 1
  const decodeTp = positiveInteger(configuration?.tp_decode) ?? fallback
  const prefillTp = positiveInteger(configuration?.tp_prefill) ?? decodeTp
  const workers = /workers\s+P(\d+)\s*\/\s*D(\d+)/i.exec(configuration?.inferencex_topology ?? '')
  return {
    prefillTp,
    decodeTp,
    prefillWorkers: workers ? Math.max(1, Number(workers[1])) : 1,
    decodeWorkers: workers ? Math.max(1, Number(workers[2])) : 1,
    known: Boolean(workers) && decodeTp > 0 && prefillTp > 0,
  }
}

/**
 * Fabric to send with a prediction.
 *
 * A fabric is a disaggregated-serving input, and only simulators whose catalog
 * entry declares fabric support consume it. Forwarding one to a simulator that
 * rejects it would drop the whole prediction.
 */
export function predictionFabric(
  mode: ServingMode,
  requiresFabric: boolean,
  fabric: string,
): string | undefined {
  return mode === 'disagg' && requiresFabric && fabric ? fabric : undefined
}

/** Concurrency each pool worker receives for a measured total concurrency. */
export function perPoolBatch(concurrency: number, workers: number): number {
  return Math.max(1, Math.ceil(concurrency / Math.max(1, workers)))
}

/** Load multiplier the comparison page applies at ordinary concurrency. */
const COMPARISON_REQUESTS_PER_LOAD_UNIT = 10

/**
 * Ceiling on the simulated requests one comparison point may ask for.
 *
 * A fixed multiplier makes a high-concurrency level request thousands of
 * simulated requests at once, which risks a simulator's request cap dropping
 * the point. Bounding the product keeps every measured level plottable and
 * gives each point a comparable run size.
 */
const MAX_COMPARISON_REQUESTS_PER_POINT = 4096

/**
 * Requests-per-load-unit to send for one measured concurrency level.
 *
 * Scales the multiplier down as concurrency rises, but never below one request
 * per labeled concurrency slot. Above `MAX_COMPARISON_REQUESTS_PER_POINT` the
 * request count therefore equals concurrency; if that exceeds a simulator's own
 * limit, its explicit rejection is reported instead of plotting metrics for a
 * smaller run against the higher concurrency label.
 */
export function comparisonRequestsPerLoadUnit(concurrency: number): number {
  const level = Number.isFinite(concurrency) && concurrency > 0 ? concurrency : 1
  return Math.max(1, Math.min(COMPARISON_REQUESTS_PER_LOAD_UNIT, MAX_COMPARISON_REQUESTS_PER_POINT / level))
}

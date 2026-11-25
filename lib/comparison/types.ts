import type { EngineSection } from '../api/engine'

/**
 * The comparison set: result snapshots a reader chose to keep.
 *
 * An entry is a measured result, not a request. It carries the figures the
 * result page showed, the identity needed to read them, and the shared-view
 * link that produced it. Replay artifacts stay on the result page: they are
 * large, tied to a container layout, and do not answer which configuration is
 * better for a workload.
 */

/** Bump only when a stored field changes meaning or is removed. */
export const COMPARISON_SCHEMA_VERSION = 1 as const

/** Browser-local storage key. Nothing is sent to a ConfigIQ service. */
export const COMPARISON_STORAGE_KEY = 'gc-comparison-set'

/** At most this many entries; a save that would exceed it is refused. */
export const MAX_COMPARISON_ENTRIES = 100

/** At most this many UTF-8 bytes; a save that would exceed it is refused. */
export const MAX_COMPARISON_BYTES = 1_000_000

export type ComparisonMode = 'agg' | 'disagg'

/**
 * What the sizing run was asked to achieve.
 *
 * Predict entries have no goal: their workload is the whole request. A sizing
 * entry needs the target beside its figures, because two runs that differ only
 * by their SLO can produce very different deployments.
 */
export interface ComparisonSizingGoal {
  kind: 'latency' | 'throughput' | 'concurrency'
  /** End-to-end request bound, when the run used one instead of TTFT/TPOT. */
  targetRequestLatencyMs: number | null
  targetTtftMs: number | null
  targetTpotMs: number | null
  /** Arrival rate a throughput-demand run was asked to sustain. */
  targetRequestRate: number | null
  /** Resident sequences a concurrency-capacity run was asked to hold. */
  targetConcurrency: number | null
}

export interface ComparisonWorkload {
  isl: number
  osl: number
  concurrency: number
  weightPrecision: string
  kvPrecision: string
}

/** Parallelism the simulator reported for the saved run. */
export interface ComparisonTopology {
  /** Aggregated deployment only. */
  tp: number | null
  pp: number | null
  replicas: number | null
  /** Disaggregated deployment only. */
  prefill: { tp: number; pp: number; workers: number } | null
  decode: { tp: number; pp: number; workers: number } | null
}

/** The measured figures a result page displayed. Null means "not reported". */
export interface ComparisonFigures {
  gpus: number
  ttftMs: number | null
  tpotMs: number | null
  requestLatencyMs: number | null
  tokensPerSecond: number | null
  tokensPerSecondPerGpu: number | null
  tokensPerSecondPerUser: number | null
  weightMemoryGb: number | null
  kvCachePerRequestMb: number | null
  kvCacheUsedGb: number | null
  totalVramGb: number | null
  cloudMonthlyUsd: number | null
  cloudFiveYearUsd: number | null
  selfHostedMonthlyUsd: number | null
  selfHostedFiveYearUsd: number | null
}

/** The engine limits a saved run evaluated, so an artifact can be rendered later. */
export interface ComparisonDeployment {
  /** Exact limits the run was evaluated with. */
  engine: EngineSection
  /**
   * The model's `config.json` when the run supplied one. An artifact rendered
   * from this snapshot needs the same facts the prediction did.
   */
  modelConfigData: Record<string, unknown> | null
}

export interface ComparisonEntry {
  /** Stable local identifier. Not meaningful outside this browser. */
  id: string
  /** When this snapshot was saved or last refreshed. */
  savedAt: string
  /** Display name, derived from the model and GPU system. */
  name: string
  /** Shared-view link of the page that produced the snapshot. */
  link: string
  model: string
  /** Catalog system identifier, as sent to the API. */
  system: string
  /** Display label of the GPU system. */
  systemLabel: string
  simulator: string
  engineVersion: string | null
  backend: string | null
  backendVersion: string | null
  /** Sizing target, when the snapshot came from a recommendation. */
  goal?: ComparisonSizingGoal | null
  mode: ComparisonMode
  workload: ComparisonWorkload
  topology: ComparisonTopology
  figures: ComparisonFigures
  /**
   * The evaluated limits this snapshot can render a deployment from.
   *
   * Optional because a snapshot saved before deployment artifacts existed has
   * none; the comparison offers a deploy action only for entries that carry it.
   */
  deployment?: ComparisonDeployment | null
}

/** The versioned document stored under `gc-comparison-set`. */
export interface ComparisonDocument {
  schemaVersion: typeof COMPARISON_SCHEMA_VERSION
  entries: ComparisonEntry[]
}

export function emptyComparisonDocument(): ComparisonDocument {
  return { schemaVersion: COMPARISON_SCHEMA_VERSION, entries: [] }
}

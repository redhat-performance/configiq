import type { InferenceConfigResult } from '../gpu-math/inference-config/types'
import { comparisonCost } from './cost'
import { finite, newComparisonId } from './entry-common'
import type {
  ComparisonEntry,
  ComparisonFigures,
  ComparisonTopology,
  ComparisonWorkload,
} from './types'

export interface PredictEntryInput {
  /** Shared-view link of the result being saved. */
  link: string
  model: string
  /** Catalog system identifier, as sent to the API. */
  system: string
  systemLabel: string
  backend: string | null
  backendVersion: string | null
  /** Workload the result was calculated with. */
  workload: Omit<ComparisonWorkload, 'weightPrecision' | 'kvPrecision'> & {
    weightPrecision: string
    kvPrecision: string
  }
  /** Total GPUs the result requires, as the page reported them. */
  gpus: number
  cloudRatePerHour: number | null
  hardwareCostUsd: number | null
  result: InferenceConfigResult
  /** The model's config.json when the evaluation supplied one. */
  deploymentModelConfigData?: Record<string, unknown> | null
  /** Overrides for tests; the page uses the current time and a fresh id. */
  now?: Date
  id?: string
}

function topologyFor(result: InferenceConfigResult): ComparisonTopology {
  const disagg = result.mode === 'disagg' ? result.disagg ?? null : null
  if (disagg != null) {
    return {
      tp: null,
      pp: null,
      replicas: null,
      prefill: {
        tp: disagg.prefill.tp_size,
        pp: disagg.prefill.pp_size,
        workers: disagg.prefill.workers,
      },
      decode: {
        tp: disagg.decode.tp_size,
        pp: disagg.decode.pp_size,
        workers: disagg.decode.workers,
      },
    }
  }
  return {
    tp: result.memory_analysis.tp_size,
    pp: result.parallelism_strategy.pp_size || 1,
    replicas: result.memory_analysis.replicas,
    prefill: null,
    decode: null,
  }
}

/**
 * Project a performance result into a comparison snapshot.
 *
 * Every value is copied from the result the page displayed. A metric the
 * simulator did not report stays null and is shown as unavailable; nothing is
 * estimated here. Per-GPU throughput is the one derived figure, and it divides
 * the reported total by the GPU count the page reported for the same run.
 */
export function buildPredictEntry({
  link,
  model,
  system,
  systemLabel,
  backend,
  backendVersion,
  workload,
  gpus,
  cloudRatePerHour,
  hardwareCostUsd,
  result,
  deploymentModelConfigData = null,
  now = new Date(),
  id = newComparisonId(),
}: PredictEntryInput): ComparisonEntry {
  const performance = result.performance ?? null
  const memory = result.memory_available ? result.memory_analysis : null
  const concurrency = workload.concurrency
  const kvUsed = finite(memory?.kv_cache_used_gb)
  const tokensPerSecond = finite(performance?.throughput_tokens_per_sec)
  const perGpu = tokensPerSecond != null && gpus > 0 ? tokensPerSecond / gpus : null

  const figures: ComparisonFigures = {
    gpus,
    ttftMs: finite(performance?.ttft_ms),
    tpotMs: finite(performance?.tpot_ms),
    requestLatencyMs: finite(performance?.request_latency_ms),
    tokensPerSecond,
    tokensPerSecondPerGpu: perGpu,
    tokensPerSecondPerUser: finite(performance?.throughput_tokens_per_sec_per_user),
    weightMemoryGb: finite(memory?.weight_gb),
    kvCachePerRequestMb: kvUsed != null && concurrency > 0 ? (kvUsed / concurrency) * 1000 : null,
    kvCacheUsedGb: kvUsed,
    totalVramGb: finite(memory?.total_vram_gb),
    ...comparisonCost({ gpus, cloudRatePerHour, hardwareCostUsd }),
  }

  const modelName = model.split('/').filter(Boolean).pop() ?? model

  return {
    id,
    savedAt: now.toISOString(),
    name: `${modelName} · ${systemLabel}`,
    link,
    model,
    system,
    systemLabel,
    simulator: result.simulator,
    engineVersion: result.engine_version ?? null,
    backend,
    backendVersion,
    mode: result.mode === 'disagg' ? 'disagg' : 'agg',
    workload,
    topology: topologyFor(result),
    figures,
    deployment: result.engine
      ? { engine: result.engine, modelConfigData: deploymentModelConfigData ?? null }
      : null,
  }
}

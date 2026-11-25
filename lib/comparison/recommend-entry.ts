import type { RecommendResult } from '../api/recommend'
import { comparisonCost } from './cost'
import { finite, newComparisonId } from './entry-common'
import type {
  ComparisonEntry,
  ComparisonFigures,
  ComparisonSizingGoal,
  ComparisonTopology,
} from './types'

/**
 * The request fields a sizing snapshot keeps about its goal. Structural rather
 * than importing the page's request type: a saved run only needs the target it
 * was asked to meet, not the rest of the request.
 */
export interface SizingGoalSource {
  sla_mode: 'enforce' | 'none'
  objective?: 'min_gpus' | 'throughput_demand' | 'pareto' | 'concurrency_capacity'
  ttft?: number
  tpot?: number
  request_latency?: number
  target_request_rate?: number
  target_concurrency?: number
}

/**
 * Project the submitted request's target into the snapshot's goal.
 *
 * The objective is the one the API answered, so an omitted objective follows
 * the same sla_mode mapping the request builder uses. An end-to-end bound
 * replaces the decomposed TTFT/TPOT pair; the two are never reported together.
 */
export function sizingGoalFrom(source: SizingGoalSource): ComparisonSizingGoal {
  const objective = source.objective
    ?? (source.sla_mode === 'enforce' ? 'min_gpus' : 'throughput_demand')

  if (objective === 'concurrency_capacity') {
    return {
      kind: 'concurrency',
      targetRequestLatencyMs: null,
      targetTtftMs: null,
      targetTpotMs: null,
      targetRequestRate: null,
      targetConcurrency: finite(source.target_concurrency),
    }
  }
  if (objective === 'throughput_demand') {
    return {
      kind: 'throughput',
      targetRequestLatencyMs: null,
      targetTtftMs: null,
      targetTpotMs: null,
      targetRequestRate: finite(source.target_request_rate),
      targetConcurrency: null,
    }
  }

  const requestLatency = finite(source.request_latency)
  return {
    kind: 'latency',
    targetRequestLatencyMs: requestLatency,
    targetTtftMs: requestLatency == null ? finite(source.ttft) : null,
    targetTpotMs: requestLatency == null ? finite(source.tpot) : null,
    targetRequestRate: null,
    targetConcurrency: null,
  }
}

/**
 * The result page's label for an API quantization mode.
 *
 * Modes are named by the engine, so the same precision arrives under several
 * spellings (`half`, `float16`, `bf16`, `fp8_e4m3`). Unrecognized text is kept
 * verbatim rather than guessed at; an absent mode means the run did not resolve
 * an explicit one, which the row calls `auto`.
 */
function quantizationLabel(mode: string | null): string {
  if (mode == null) return 'auto'
  const token = mode.toLowerCase().replace(/[\s_-]/g, '')
  if (token.includes('nvfp4')) return 'NVFP4'
  if (token.includes('mxfp4')) return 'MXFP4'
  if (token.includes('fp8') || token.includes('float8')) return 'FP8'
  if (token.includes('int8')) return 'INT8'
  if (token.includes('int4')) return 'INT4'
  if (token.includes('fp4')) return 'NVFP4'
  if (token.includes('fp16') || token.includes('bf16') || token.includes('float16') || token === 'half') {
    return 'FP16'
  }
  return mode
}

function topologyFor(result: RecommendResult): ComparisonTopology {
  if (result.mode === 'disagg') {
    const { prefill, decode } = result.phases
    if (prefill == null || decode == null) {
      return { tp: null, pp: null, replicas: null, prefill: null, decode: null }
    }
    return {
      tp: null,
      pp: null,
      replicas: null,
      prefill: {
        tp: prefill.tensorParallelSize,
        pp: prefill.pipelineParallelSize,
        workers: prefill.workers,
      },
      decode: {
        tp: decode.tensorParallelSize,
        pp: decode.pipelineParallelSize,
        workers: decode.workers,
      },
    }
  }
  return {
    tp: result.recommendation.tensorParallelSize,
    pp: result.recommendation.pipelineParallelSize,
    replicas: result.recommendation.replicasNeeded,
    prefill: null,
    decode: null,
  }
}

export interface RecommendEntryInput {
  /** Shared-view link of the result being saved. */
  link: string
  model: string
  /** Catalog system identifier, as sent to the API. */
  system: string
  systemLabel: string
  backend: string | null
  /** Backend release the submitted request asked for. */
  backendVersion: string | null
  /** What the sizing run was asked to achieve. */
  goal: ComparisonSizingGoal
  cloudRatePerHour: number | null
  hardwareCostUsd: number | null
  result: RecommendResult
  /** The model's config.json when the run supplied one. */
  deploymentModelConfigData?: Record<string, unknown> | null
  /** Overrides for tests; the page uses the current time and a fresh id. */
  now?: Date
  id?: string
}

/**
 * Project a sizing recommendation into a comparison snapshot.
 *
 * Figures are the values the result panel displayed. Memory comes from the
 * breakdown the API reported, which is the field the predict row stores too;
 * it is not scaled by the GPU count. Total VRAM is not reported by a sizing
 * result, so it stays null instead of being inferred from the catalog.
 * Per-GPU throughput divides the reported total by the GPUs the page showed,
 * matching the predict row's definition.
 */
export function buildRecommendEntry({
  link,
  model,
  system,
  systemLabel,
  backend,
  backendVersion,
  goal,
  cloudRatePerHour,
  hardwareCostUsd,
  result,
  deploymentModelConfigData = null,
  now = new Date(),
  id = newComparisonId(),
}: RecommendEntryInput): ComparisonEntry {
  const gpus = result.recommendation.gpusNeeded
  const concurrency = result.performance.concurrency
  const memory = result.memoryAvailable ? result.memory.breakdown : null
  const kvUsed = finite(memory?.kvCacheGb)
  // A capacity run can size residency without a performance simulation. The
  // adapter fills the absent latency and throughput fields with zero for the
  // result panel to hide; a snapshot must not record those zeros as measured.
  const performanceReported = result.capacity?.performanceAvailable !== false
  const tokensPerSecond = performanceReported ? finite(result.throughput.tokensPerSecond) : null

  const figures: ComparisonFigures = {
    gpus,
    ttftMs: performanceReported ? finite(result.performance.ttftLatencyMs) : null,
    tpotMs: performanceReported ? finite(result.performance.tpotMs) : null,
    requestLatencyMs: performanceReported ? finite(result.performance.requestLatencyMs) : null,
    tokensPerSecond,
    tokensPerSecondPerGpu: tokensPerSecond != null && gpus > 0 ? tokensPerSecond / gpus : null,
    tokensPerSecondPerUser: performanceReported ? finite(result.throughput.tokensPerSecondPerUser) : null,
    weightMemoryGb: finite(memory?.weightsGb),
    kvCachePerRequestMb: kvUsed != null && concurrency > 0 ? (kvUsed / concurrency) * 1000 : null,
    kvCacheUsedGb: kvUsed,
    totalVramGb: null,
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
    engineVersion: result.engineVersion,
    backend,
    backendVersion,
    goal,
    mode: result.mode,
    workload: {
      isl: result.metadata.inputTokens,
      osl: result.metadata.outputTokens,
      concurrency,
      weightPrecision: quantizationLabel(result.precision.weights),
      kvPrecision: quantizationLabel(result.precision.kvCache),
    },
    topology: topologyFor(result),
    figures,
    deployment: result.engine
      ? { engine: result.engine, modelConfigData: deploymentModelConfigData ?? null }
      : null,
  }
}

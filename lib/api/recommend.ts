import type { RecommendRequest } from './schemas'
import { aicTimeoutSeconds } from './timeout'

// ─── Types ───────────────────────────────────────────────────────────────────

interface RecommendWarning {
  code: string
  message: string
}

export type ServingMode = 'agg' | 'disagg'

/**
 * One worker role in a disaggregated deployment (prefill / decode / encode).
 * `gpusPerWorker` = tp·pp·dp·cp (the aiconfigurator WORKER_GPU_DIMS; for MoE this
 * already equals etp·ep·pp — expert dims do not add GPUs). A replica's GPU count
 * is the sum of `workers × gpusPerWorker` across all phases.
 */
export interface PhaseConfig {
  workers: number
  gpusPerWorker: number
  tensorParallelSize: number
  pipelineParallelSize: number
  dataParallelSize: number
  contextParallelSize: number
  moeTensorParallelSize: number | null
  moeExpertParallelSize: number | null
  batchSize: number | null
  memoryGb: number | null
}

export interface RecommendResult {
  requestId: string
  status: 'completed'
  /** Serving mode the recommender chose. Disagg splits prefill/decode pools. */
  mode: ServingMode
  recommendation: {
    /** Aggregate total GPUs across all replicas and pools. */
    gpusNeeded: number
    /** GPUs in one replica (the smallest scalable unit). */
    gpusPerReplica: number
    /** GPUs per worker (agg). Equals gpusPerReplica for disagg. */
    totalGpus: number
    replicasNeeded: number
    tensorParallelSize: number
    pipelineParallelSize: number
    dataParallelSize: number
    contextParallelSize: number
    moeTensorParallelSize: number | null
    moeExpertParallelSize: number | null
    batchSize: number | null
  }
  /** Per-phase pools (disagg only; all null for agg). Encode is future-ready. */
  phases: {
    prefill: PhaseConfig | null
    decode: PhaseConfig | null
    encode: PhaseConfig | null
  }
  performance: {
    ttftLatencyMs: number
    tpotMs: number
    requestLatencyMs: number
    concurrency: number
  }
  throughput: {
    tokensPerSecond: number
    tokensPerSecondPerGpu: number
    tokensPerSecondPerUser: number
  }
  memory: {
    /** Worst-case peak memory usage per GPU (GB). Checked against one GPU's HBM. */
    value: number
    unit: 'GB'
  }
  metadata: {
    modelPath: string
    system: string
    inputTokens: number
    outputTokens: number
    targetTtftMs: number
    durationMs: number
  }
  warnings: RecommendWarning[]
}

export interface RecommendErrorResponse {
  requestId: string
  status: 'failed'
  error: {
    code: string
    message: string
  }
}

export type RecommendResponse = RecommendResult | RecommendErrorResponse

// ─── Request ID ──────────────────────────────────────────────────────────────

export function generateRequestId(): string {
  return 'size_' + crypto.randomUUID().replace(/-/g, '').slice(0, 12)
}

// ─── Phase (disagg worker) parsing ───────────────────────────────────────────

/** Shape of a WorkerConfig as returned by the AIConfigurator gateway. */
interface RawWorkerConfig {
  tp?: number | null
  pp?: number | null
  dp?: number | null
  cp?: number | null
  moe_tp?: number | null
  moe_ep?: number | null
  num_workers?: number | null
  batch_size?: number | null
  memory_gb?: number | null
}

/**
 * True when the model uses expert parallelism. Dense models report moe_tp/moe_ep
 * as null or 1; a value > 1 on either dimension means the experts are split.
 * Used only for labelling (EP/ETP) — NOT for the GPU-count math, where MoE
 * expert dims are laid out within the tp×dp GPUs (they do not add GPUs).
 */
export function isMoeConfig(moeTp: number | null, moeEp: number | null): boolean {
  return (moeTp != null && moeTp > 1) || (moeEp != null && moeEp > 1)
}

const dim = (v: number | null | undefined): number => (v != null && v > 0 ? v : 1)

/**
 * GPUs consumed by a single worker = tp·pp·dp·cp. This mirrors the
 * aiconfigurator SDK's WORKER_GPU_DIMS = (tp, pp, dp, cp) exactly and applies to
 * both dense and MoE models (for MoE, tp·pp·dp already equals etp·ep·pp; the
 * expert dims do not multiply the GPU count). cp (context parallel) is included
 * — omitting it undercounts prefill pools.
 */
function gpusPerWorker(tp: number, pp: number, dp: number, cp: number): number {
  return tp * pp * dp * cp
}

/** Parse a gateway WorkerConfig into a PhaseConfig, or null if absent. */
function parsePhase(raw: RawWorkerConfig | null | undefined): PhaseConfig | null {
  if (!raw || raw.tp == null) return null
  const tp = dim(raw.tp)
  const pp = dim(raw.pp)
  const dp = dim(raw.dp)
  const cp = dim(raw.cp)
  return {
    workers: raw.num_workers ?? 1,
    gpusPerWorker: gpusPerWorker(tp, pp, dp, cp),
    tensorParallelSize: tp,
    pipelineParallelSize: pp,
    dataParallelSize: dp,
    contextParallelSize: cp,
    moeTensorParallelSize: raw.moe_tp ?? null,
    moeExpertParallelSize: raw.moe_ep ?? null,
    batchSize: raw.batch_size ?? null,
    memoryGb: raw.memory_gb ?? null,
  }
}

// ─── Service ─────────────────────────────────────────────────────────────────

export async function callRecommend(
  request: RecommendRequest
): Promise<RecommendResponse> {
  const requestId = generateRequestId()
  const startTime = performance.now()

  const baseUrl = process.env.AICONFIGURATOR_GATEWAY_URL
  const timeoutSeconds = aicTimeoutSeconds()

  if (!baseUrl) {
    return makeError(requestId, 'AIC_NOT_CONFIGURED', 'AIConfigurator API URL is not configured')
  }

  const externalPayload: Record<string, unknown> = {
    model_path: request.model_path,
    system: request.system,
    backend: request.backend ?? 'vllm',
    isl: request.isl,
    osl: request.osl,
    ttft: request.ttft,
    tpot: request.tpot ?? 30,
    database_mode: request.database_mode ?? 'HYBRID',
    top_n: request.top_n ?? 5,
  }

  if (request.backend_version != null) externalPayload.backend_version = request.backend_version
  if (request.target_request_rate != null) externalPayload.target_request_rate = request.target_request_rate
  if (request.target_concurrency != null) externalPayload.target_concurrency = request.target_concurrency
  if (request.request_latency != null) externalPayload.request_latency = request.request_latency
  if (request.prefix != null) externalPayload.prefix = request.prefix
  if (request.model_config != null) externalPayload.model_config = request.model_config

  let response: Response
  try {
    response = await fetch(`${baseUrl}/recommend`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(externalPayload),
      signal: AbortSignal.timeout(timeoutSeconds * 1000),
    })
  } catch (err: unknown) {
    const durationMs = Math.round(performance.now() - startTime)
    if (err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      return makeError(requestId, 'AIC_TIMEOUT', `The AIConfigurator API did not respond within ${timeoutSeconds} seconds (waited ${durationMs}ms)`)
    }
    return makeError(requestId, 'AIC_UNAVAILABLE', 'AIConfigurator API is unreachable')
  }

  if (!response.ok) {
    let detail = `AIConfigurator API returned HTTP ${response.status}`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch { /* ignore parse errors */ }

    const code = response.status === 422 ? 'AIC_NO_CONFIGURATION' : 'AIC_UNAVAILABLE'
    return makeError(requestId, code, detail)
  }

  let rawData: Record<string, unknown>
  try {
    rawData = await response.json() as Record<string, unknown>
  } catch {
    return makeError(requestId, 'AIC_INVALID_RESPONSE', 'AIConfigurator API returned non-JSON response')
  }

  const configs = rawData.configs as Array<Record<string, unknown>> | undefined
  if (!configs || configs.length === 0) {
    return makeError(requestId, 'AIC_NO_CONFIGURATION', 'No valid GPU configuration found for this model and hardware combination.')
  }

  const best = configs[0]
  const chosenMode = typeof rawData.chosen_mode === 'string' ? rawData.chosen_mode : 'agg'
  const mode: ServingMode = chosenMode.startsWith('disagg') ? 'disagg' : 'agg'

  const totalGpusNeeded = (best.total_gpus_needed as number) ?? 0
  const replicasNeeded = (best.replicas_needed as number) ?? 1

  const prefill = parsePhase(best.prefill_config as RawWorkerConfig | undefined)
  const decode = parsePhase(best.decode_config as RawWorkerConfig | undefined)
  const encode = parsePhase(best.encode_config as RawWorkerConfig | undefined)

  const tp = dim(best.tp as number | undefined)
  const pp = dim(best.pp as number | undefined)
  const dp = dim(best.dp as number | undefined)
  const cp = dim(best.cp as number | undefined)
  const moeTp = (best.moe_tp as number | null) ?? null
  const moeEp = (best.moe_ep as number | null) ?? null
  const batchSize = (best.bs as number | null) ?? null

  const warnings: RecommendWarning[] = []
  let gpusPerReplica: number
  if (mode === 'disagg') {
    // A replica is one prefill/decode(/encode) set: sum of workers × gpus/worker.
    gpusPerReplica = [prefill, decode, encode].reduce(
      (sum, phase) => sum + (phase ? phase.workers * phase.gpusPerWorker : 0),
      0,
    )
  } else {
    // Agg: gpus/replica == gpus/worker == num_total_gpus (tp·pp·dp·cp).
    const numTotalGpus = (best.num_total_gpus as number) ?? gpusPerWorker(tp, pp, dp, cp)
    gpusPerReplica = numTotalGpus
    const parallelismProduct = gpusPerWorker(tp, pp, dp, cp)
    if (parallelismProduct !== numTotalGpus) {
      warnings.push({
        code: 'GPU_TOPOLOGY_MISMATCH',
        message: `Parallelism dimensions (TP=${tp} x PP=${pp} x DP=${dp} x CP=${cp} = ${parallelismProduct}) do not equal GPUs per worker (${numTotalGpus})`,
      })
    }
  }

  const durationMs = Math.round(performance.now() - startTime)

  // The gateway reports concurrency, request rate, and tokens/s per replica;
  // scale to cluster totals so the headline matches the whole deployment.
  // Per-GPU and per-user rates are already per-unit and stay as-is.
  const clusterConcurrency = Math.round(((best.concurrency as number) ?? 0) * replicasNeeded)
  const clusterTokensPerSecond = ((best.tokens_per_second as number) ?? 0) * replicasNeeded

  return {
    requestId,
    status: 'completed',
    mode,
    recommendation: {
      gpusNeeded: totalGpusNeeded,
      gpusPerReplica,
      totalGpus: gpusPerReplica,
      replicasNeeded,
      tensorParallelSize: tp,
      pipelineParallelSize: pp,
      dataParallelSize: dp,
      contextParallelSize: cp,
      moeTensorParallelSize: moeTp,
      moeExpertParallelSize: moeEp,
      batchSize,
    },
    phases: { prefill, decode, encode },
    performance: {
      ttftLatencyMs: (best.ttft as number) ?? 0,
      tpotMs: (best.tpot as number) ?? 0,
      requestLatencyMs: (best.request_latency as number) ?? 0,
      concurrency: clusterConcurrency,
    },
    throughput: {
      tokensPerSecond: clusterTokensPerSecond,
      tokensPerSecondPerGpu: (best.tokens_per_second_per_gpu as number) ?? 0,
      tokensPerSecondPerUser: (best.tokens_per_second_per_user as number) ?? 0,
    },
    memory: {
      value: (best.memory as number) ?? 0,
      unit: 'GB',
    },
    metadata: {
      modelPath: request.model_path,
      system: request.system,
      inputTokens: request.isl,
      outputTokens: request.osl,
      targetTtftMs: request.ttft,
      durationMs,
    },
    warnings,
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeError(requestId: string, code: string, message: string): RecommendErrorResponse {
  return { requestId, status: 'failed', error: { code, message } }
}

import type { RecommendRequest } from './schemas'
import { gatewayTimeoutSeconds } from './timeout'

// ─── Types ───────────────────────────────────────────────────────────────────

interface RecommendWarning {
  code: string
  message: string
}

export type ServingMode = 'agg' | 'disagg'

/**
 * One worker role in a disaggregated deployment (prefill / decode / encode).
 * `gpusPerWorker` = tp·pp·dp·cp (the aisimulate WORKER_GPU_DIMS; for MoE this
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
    /** Compatibility field for the smallest scalable unit; equals gpusPerReplica. */
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

export interface RecommendWindow {
  minGpus: number
  maxGpus: number
}

export type RecommendProgressEvent =
  | { type: 'search_started'; maxGpus: number }
  | { type: 'window_started'; window: RecommendWindow }
  | { type: 'window_completed'; window: RecommendWindow; candidateGpus: number | null }
  | { type: 'refining'; window: RecommendWindow; candidateGpus: number }
  | { type: 'completed'; response: RecommendResponse }

// ─── Request ID ──────────────────────────────────────────────────────────────

export function generateRequestId(): string {
  return 'size_' + crypto.randomUUID().replace(/-/g, '').slice(0, 12)
}

// ─── Phase (disagg worker) parsing ───────────────────────────────────────────

/** Shape of a WorkerConfig as returned by the AISimulators gateway. */
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
 * aisimulate SDK's WORKER_GPU_DIMS = (tp, pp, dp, cp) exactly and applies to
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
  request: RecommendRequest,
  options: { window?: RecommendWindow; timeoutSeconds?: number; signal?: AbortSignal } = {},
): Promise<RecommendResponse> {
  const requestId = generateRequestId()
  const startTime = performance.now()

  const baseUrl = process.env.AISIMULATORS_GATEWAY_URL
  const timeoutSeconds = gatewayTimeoutSeconds()
  const requestTimeoutSeconds = options.timeoutSeconds ?? timeoutSeconds

  if (!baseUrl) {
    return makeError(requestId, 'AISIM_NOT_CONFIGURED', 'AISimulators API URL is not configured')
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

  if (request.max_seq_len != null) externalPayload.max_seq_len = request.max_seq_len
  if (request.prefill_max_seq_len != null) externalPayload.prefill_max_seq_len = request.prefill_max_seq_len
  if (request.decode_max_seq_len != null) externalPayload.decode_max_seq_len = request.decode_max_seq_len

  if (request.backend_version != null) externalPayload.backend_version = request.backend_version
  if (request.target_request_rate != null) externalPayload.target_request_rate = request.target_request_rate
  if (request.target_concurrency != null) externalPayload.target_concurrency = request.target_concurrency
  if (request.request_latency != null) externalPayload.request_latency = request.request_latency
  if (request.prefix != null) externalPayload.prefix = request.prefix
  if (request.model_config != null) externalPayload.model_config = request.model_config
  if (options.window?.minGpus != null) externalPayload.min_candidate_gpus = options.window.minGpus
  if (options.window?.maxGpus != null) externalPayload.max_candidate_gpus = options.window.maxGpus

  let response: Response
  try {
    const timeoutSignal = AbortSignal.timeout(requestTimeoutSeconds * 1000)
    const signal = options.signal
      ? AbortSignal.any([timeoutSignal, options.signal])
      : timeoutSignal
    response = await fetch(`${baseUrl}/recommend`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(externalPayload),
      signal,
    })
  } catch (err: unknown) {
    if (options.signal?.aborted) throw err
    const durationMs = Math.round(performance.now() - startTime)
    if (err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      return makeError(requestId, 'AISIM_TIMEOUT', `The AISimulators API did not respond within ${requestTimeoutSeconds} seconds (waited ${durationMs}ms)`)
    }
    return makeError(requestId, 'AISIM_UNAVAILABLE', 'AISimulators API is unreachable')
  }

  if (!response.ok) {
    let detail = `AISimulators API returned HTTP ${response.status}`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch { /* ignore parse errors */ }

    // Current AISimulators deployments can surface an undersized bounded
    // search as a 500 NoViableParallelConfig. It is a normal signal for the
    // incremental search to try the next GPU window, not a service outage.
    // Keep the message check while older gateways are still in circulation;
    // the service now maps this condition to 422 as well.
    const noViableWindow = detail.includes('NoViableParallelConfig') ||
      detail.includes('no deployment_mode has a viable parallel config')
    const code = response.status === 422 || noViableWindow
      ? 'AISIM_NO_CONFIGURATION'
      : 'AISIM_UNAVAILABLE'
    return makeError(requestId, code, detail)
  }

  let rawData: Record<string, unknown>
  try {
    rawData = await response.json() as Record<string, unknown>
  } catch {
    return makeError(requestId, 'AISIM_INVALID_RESPONSE', 'AISimulators API returned non-JSON response')
  }

  const configs = rawData.configs as Array<Record<string, unknown>> | undefined
  if (!configs || configs.length === 0) {
    return makeError(requestId, 'AISIM_NO_CONFIGURATION', 'No valid GPU configuration found for this model and hardware combination.')
  }

  const best = configs[0]
  const chosenMode = typeof rawData.chosen_mode === 'string' ? rawData.chosen_mode : 'agg'
  const mode: ServingMode = chosenMode.startsWith('disagg') ? 'disagg' : 'agg'

  const rawTotalGpusNeeded = (best.total_gpus_needed as number) ?? 0
  const rawReplicasNeeded = (best.replicas_needed as number) ?? 0

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

  let gpusPerReplica: number
  let replicasNeeded: number
  if (mode === 'disagg') {
    // A replica is one prefill/decode(/encode) set: sum of workers × gpus/worker.
    gpusPerReplica = [prefill, decode, encode].reduce(
      (sum, phase) => sum + (phase ? phase.workers * phase.gpusPerWorker : 0),
      0,
    )
    replicasNeeded = Number.isFinite(rawReplicasNeeded) && rawReplicasNeeded > 0
      ? Math.ceil(rawReplicasNeeded)
      : 1
  } else {
    // Agg: GPUs per replica are the parallelism product. Older gateway builds
    // reported cluster-wide `used_gpus` as num_total_gpus, so deriving this
    // from the topology prevents a multi-replica recommendation from being
    // multiplied a second time by the cost calculator.
    const parallelismProduct = gpusPerWorker(tp, pp, dp, cp)
    replicasNeeded = Number.isFinite(rawReplicasNeeded) && rawReplicasNeeded > 0
      ? Math.ceil(rawReplicasNeeded)
      : rawTotalGpusNeeded > 0
        ? Math.max(Math.ceil(rawTotalGpusNeeded / parallelismProduct), 1)
        : 1
    const derivedPerReplica = rawTotalGpusNeeded > 0
      ? Math.max(Math.ceil(rawTotalGpusNeeded / replicasNeeded), 1)
      : parallelismProduct
    const hasTopology = best.tp != null || best.pp != null || best.dp != null || best.cp != null
    gpusPerReplica = hasTopology ? parallelismProduct : derivedPerReplica
  }

  const totalGpusNeeded = Number.isFinite(rawTotalGpusNeeded) && rawTotalGpusNeeded > 0
    ? Math.ceil(rawTotalGpusNeeded)
    : gpusPerReplica * replicasNeeded
  const topologyClusterGpus = gpusPerReplica * replicasNeeded
  if (totalGpusNeeded !== topologyClusterGpus) {
    return makeError(
      requestId,
      'AISIM_INVALID_RESPONSE',
      `AISimulators returned inconsistent GPU topology: cluster total ${totalGpusNeeded} does not equal ${gpusPerReplica} GPUs per replica x ${replicasNeeded} replicas.`,
    )
  }

  const durationMs = Math.round(performance.now() - startTime)

  // AISimulators candidate metrics describe the complete candidate cluster:
  // output_throughput_tok_s_per_gpu * used_gpus equals output_throughput_tok_s.
  // Do not multiply them by replicas again. The capacity calculator divides
  // this cluster value by replicas to obtain one scalable replica's capacity.
  const clusterConcurrency = Math.round((best.concurrency as number) ?? 0)
  const clusterTokensPerSecond = (best.tokens_per_second as number) ?? 0
  const ttftLatencyMs = (best.ttft as number) ?? 0
  const tpotMs = (best.tpot as number) ?? 0
  const requestLatencyMs = (best.request_latency as number) ?? 0

  const requiredValues = [
    totalGpusNeeded,
    gpusPerReplica,
    replicasNeeded,
    clusterTokensPerSecond,
    ttftLatencyMs,
    tpotMs,
  ]
  if (requiredValues.some(value => !Number.isFinite(value) || value <= 0)) {
    return makeError(
      requestId,
      'AISIM_INVALID_RESPONSE',
      'AISimulators returned an incomplete recommendation without positive GPU, throughput, TTFT, and TPOT values.',
    )
  }

  const missesLatencyTarget = request.request_latency != null
    ? !Number.isFinite(requestLatencyMs) || requestLatencyMs <= 0 || requestLatencyMs > request.request_latency
    : ttftLatencyMs > request.ttft || tpotMs > (request.tpot ?? 30)
  if (missesLatencyTarget) {
    return makeError(
      requestId,
      'AISIM_NO_CONFIGURATION',
      'AISimulators did not return a configuration that satisfies the requested latency targets.',
    )
  }

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
      ttftLatencyMs,
      tpotMs,
      requestLatencyMs,
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
    warnings: [],
  }
}

/**
 * Search progressively larger GPU windows, then refine the first feasible
 * window. A feasible result from a range is provisional until its lower
 * sub-range has been checked.
 */
export async function* incrementalRecommend(
  request: RecommendRequest,
  signal?: AbortSignal,
): AsyncGenerator<RecommendProgressEvent> {
  const maxGpus = configuredMaxCandidateGpus()
  const totalTimeoutSeconds = gatewayTimeoutSeconds()
  const deadline = performance.now() + Math.max(1, totalTimeoutSeconds - 1) * 1000
  let windowMin = 1
  let windowMax = 1
  let best: RecommendResult | null = null

  yield { type: 'search_started', maxGpus }

  while (windowMin <= maxGpus) {
    if (signal?.aborted) throw new DOMException('Recommendation search aborted', 'AbortError')
    const window = { minGpus: windowMin, maxGpus: windowMax }
    yield { type: 'window_started', window }
    const remainingSeconds = Math.ceil((deadline - performance.now()) / 1000)
    if (remainingSeconds <= 0) {
      yield {
        type: 'completed',
        response: best ?? makeError(generateRequestId(), 'AISIM_TIMEOUT', 'The incremental recommendation search exceeded its overall time budget.'),
      }
      return
    }
    const response = await callRecommend(request, {
      window,
      timeoutSeconds: remainingSeconds,
      signal,
    })

    if (response.status === 'failed') {
      if (response.error.code !== 'AISIM_NO_CONFIGURATION') {
        yield { type: 'completed', response: best ?? response }
        return
      }

      yield { type: 'window_completed', window, candidateGpus: null }
      if (best != null) {
        yield { type: 'completed', response: best }
        return
      }
      windowMin = windowMax + 1
      windowMax = Math.min(maxGpus, Math.max(windowMin, windowMax * 2))
      continue
    }

    const candidateGpus = response.recommendation.gpusNeeded
    if (best == null || candidateGpus < best.recommendation.gpusNeeded) best = response
    yield { type: 'window_completed', window, candidateGpus }

    if (candidateGpus <= windowMin) {
      yield { type: 'completed', response: best }
      return
    }

    const refinementMax = candidateGpus - 1
    if (refinementMax >= windowMin) {
      yield { type: 'refining', window: { minGpus: windowMin, maxGpus: refinementMax }, candidateGpus }
      windowMax = refinementMax
      continue
    }

    yield { type: 'completed', response: best }
    return
  }

  yield {
    type: 'completed',
    response: makeError(generateRequestId(), 'AISIM_NO_CONFIGURATION', 'No valid GPU configuration found for this workload.'),
  }
}

function configuredMaxCandidateGpus(): number {
  const value = Number(process.env.AISIMULATORS_MAX_CANDIDATE_GPUS)
  return Number.isInteger(value) && value > 0 ? value : 1024
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeError(requestId: string, code: string, message: string): RecommendErrorResponse {
  return { requestId, status: 'failed', error: { code, message } }
}

import type { RecommendRequest } from './schemas'
import { gatewayTimeoutSeconds } from './timeout'
import type { Candidate, CandidateViews, FrontierSweep } from '../types/pareto'
import { isRecord, parseFrontierBody } from '../pareto/parse'

// ─── Types ───────────────────────────────────────────────────────────────────

interface RecommendWarning {
  code: string
  message: string
}

export interface RecommendSearchSummary {
  strategy: string
  candidate_gpus: number[]
  evaluated_gpus: number[]
  max_candidates: number
  configured_max_gpus: number
  hardware_max_gpus: number
  effective_max_gpus: number
  time_budget_seconds: number
  budget_exhausted: boolean
  modes_searched?: string[]
  candidate_configurations?: number
  evaluated_configurations?: number
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

export interface RecommendDemandOutcome {
  targetRps: number
  achievedRps: number | null
  met: boolean
  observedConcurrency: number | null
  windowSeconds: number | null
  completedRequests: number | null
  evaluatedConfigurations: number | null
  bestObservedRps: number | null
  simulator: 'aisimulate' | 'blis'
}

export interface RecommendSweepMeta {
  runId: string | null
  algorithm: string | null
  seed: number | null
  maxTrials: number | null
  evaluated: number | null
  feasible: number | null
  timedOut: number | null
  budgetExhausted: boolean
  strictSla: boolean | null
  windowMinGpus: number | null
  windowMaxGpus: number | null
}

export interface RecommendResult {
  requestId: string
  status: 'completed'
  simulator: 'aisimulate' | 'blis'
  searchSummary?: RecommendSearchSummary
  /** Measured outcome of a throughput-demand run (demand objective only). */
  demand?: RecommendDemandOutcome
  /** Run-control facts for the sweep that produced the candidates. */
  sweep?: RecommendSweepMeta
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
    breakdown: {
      weightsGb: number
      kvCacheGb: number
      overheadGb: number
    } | null
  }
  memoryAvailable: boolean
  metadata: {
    modelPath: string
    system: string
    inputTokens: number
    outputTokens: number
    targetTtftMs: number | null
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

/**
 * Shared request mapping for the min_gpus and pareto paths. Both must send the
 * same workload shape; the objective only changes how the service searches it.
 */
function baseRequestPayload(request: RecommendRequest): Record<string, unknown> {
  const externalPayload: Record<string, unknown> = {
    model_path: request.model_path,
    system: request.system,
    backend: request.backend ?? 'vllm',
    isl: request.isl,
    osl: request.osl,
    database_mode: request.database_mode ?? 'HYBRID',
    top_n: request.top_n ?? 5,
    sla_mode: request.sla_mode ?? 'enforce',
  }

  const slaMode = request.sla_mode ?? 'enforce'
  if (slaMode === 'enforce' && request.request_latency == null) {
    // Preserve the long-standing default TPOT for callers that send only a
    // TTFT bound. A demand run (sla_mode=none) never sends either bound.
    if (request.ttft != null) externalPayload.ttft = request.ttft
    externalPayload.tpot = request.tpot ?? 30
  } else {
    if (request.ttft != null) externalPayload.ttft = request.ttft
    if (request.tpot != null) externalPayload.tpot = request.tpot
  }

  if (request.fabric !== undefined) externalPayload.fabric = request.fabric
  if (request.max_seq_len != null) externalPayload.max_seq_len = request.max_seq_len
  if (request.prefill_max_seq_len != null) externalPayload.prefill_max_seq_len = request.prefill_max_seq_len
  if (request.decode_max_seq_len != null) externalPayload.decode_max_seq_len = request.decode_max_seq_len

  if (request.backend_version != null) externalPayload.backend_version = request.backend_version
  if (request.target_request_rate != null) externalPayload.target_request_rate = request.target_request_rate
  if (request.target_concurrency != null) externalPayload.target_concurrency = request.target_concurrency
  if (request.request_latency != null) externalPayload.request_latency = request.request_latency
  if (request.prefix != null) externalPayload.prefix = request.prefix
  if (request.model_config != null) externalPayload.model_config = request.model_config
  if (request.simulator != null) externalPayload.simulator = request.simulator
  if (request.objective) externalPayload.objective = request.objective
  if (request.min_candidate_gpus != null) {
    externalPayload.min_candidate_gpus = request.min_candidate_gpus
  }
  if (request.max_candidate_gpus != null) {
    externalPayload.max_candidate_gpus = request.max_candidate_gpus
  }
  if (request.kv_transfer_bandwidth_gb_per_second != null) {
    externalPayload.kv_transfer_bandwidth_gb_per_second = request.kv_transfer_bandwidth_gb_per_second
  }
  return externalPayload
}

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

  const externalPayload = baseRequestPayload(request)

  if (request.simulator === 'blis') {
    const blisSearchBudgetSeconds = Math.min(60, requestTimeoutSeconds - 5)
    if (blisSearchBudgetSeconds < 1) {
      return makeError(
        requestId,
        'AISIM_TIMEOUT',
        'The configured gateway timeout is too short to run a bounded BLIS recommendation.',
      )
    }
    externalPayload.search_budget_seconds = blisSearchBudgetSeconds
  }
  if (options.window?.minGpus != null) externalPayload.min_candidate_gpus = options.window.minGpus
  if (options.window?.maxGpus != null) externalPayload.max_candidate_gpus = options.window.maxGpus

  let response: Response
  try {
    const timeoutSignal = AbortSignal.timeout(requestTimeoutSeconds * 1000)
    const signal = options.signal
      ? AbortSignal.any([timeoutSignal, options.signal])
      : timeoutSignal
    // Native AISimulators keeps the detailed memory estimate behind the
    // optional include flag; the sizing result renders that value directly.
    const recommendUrl = `${baseUrl}/recommend?include=memory`
    response = await fetch(recommendUrl, {
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
    let body: unknown = null
    try {
      body = await response.json()
    } catch { /* ignore parse errors */ }
    const parsedDetail = parseErrorDetail(body)
    const detail = parsedDetail?.message ?? `AISimulators API returned HTTP ${response.status}`
    const validationError = parsedDetail?.validation === true

    // Current AISimulators deployments can surface an undersized bounded
    // search as a 500 NoViableParallelConfig. It is a normal signal for the
    // incremental search to try the next GPU window, not a service outage.
    // Keep the message check while older gateways are still in circulation;
    // the service now maps this condition to 422 as well.
    const noViableWindow = detail.includes('NoViableParallelConfig') ||
      detail.includes('no deployment_mode has a viable parallel config')
    const code = response.status === 504
      ? 'AISIM_TIMEOUT'
      : noViableWindow
        ? 'AISIM_NO_CONFIGURATION'
      : response.status >= 500
        ? 'AISIM_UNAVAILABLE'
        : response.status === 422
          ? validationError ? 'INVALID_REQUEST' : 'AISIM_NO_CONFIGURATION'
          : 'AISIM_UNAVAILABLE'
    return makeError(requestId, code, detail)
  }

  let rawData: Record<string, unknown>
  try {
    rawData = await response.json() as Record<string, unknown>
  } catch {
    return makeError(requestId, 'AISIM_INVALID_RESPONSE', 'AISimulators API returned non-JSON response')
  }

  const responseSimulator = rawData.simulator === 'aisimulate' || rawData.simulator === 'blis'
    ? rawData.simulator
    : null
  if (rawData.simulator != null && responseSimulator == null) {
    return makeError(
      requestId,
      'AISIM_INVALID_RESPONSE',
      'The API returned an unknown simulator identifier.',
    )
  }
  const actualSimulator = responseSimulator ?? (request.simulator === 'aisimulate' ? 'aisimulate' : null)
  if (actualSimulator !== request.simulator) {
    return makeError(
      requestId,
      'SIMULATOR_MISMATCH',
      `Requested ${request.simulator}, but the API confirmed ${actualSimulator ?? 'no simulator'}. No fallback was accepted.`,
    )
  }

  const configs = rawData.configs as Array<Record<string, unknown>> | undefined
  if (!configs || configs.length === 0) {
    return makeError(requestId, 'AISIM_NO_CONFIGURATION', 'No valid GPU configuration found for this model and hardware combination.')
  }

  const best = configs[0]
  const chosenMode = typeof rawData.chosen_mode === 'string' ? rawData.chosen_mode : 'agg'
  const mode: ServingMode = chosenMode.startsWith('disagg') ? 'disagg' : 'agg'
  const simulator = actualSimulator
  const searchSummaryRaw = rawData.search_summary && typeof rawData.search_summary === 'object'
    ? rawData.search_summary as Record<string, unknown>
    : undefined
  const searchSummary: RecommendSearchSummary | undefined = searchSummaryRaw &&
    Array.isArray(searchSummaryRaw.candidate_gpus) && Array.isArray(searchSummaryRaw.evaluated_gpus)
    ? {
        strategy: typeof searchSummaryRaw.strategy === 'string' ? searchSummaryRaw.strategy : 'unknown',
        candidate_gpus: searchSummaryRaw.candidate_gpus.filter((gpu): gpu is number => typeof gpu === 'number'),
        evaluated_gpus: searchSummaryRaw.evaluated_gpus.filter((gpu): gpu is number => typeof gpu === 'number'),
        max_candidates: typeof searchSummaryRaw.max_candidates === 'number' ? searchSummaryRaw.max_candidates : 0,
        configured_max_gpus: typeof searchSummaryRaw.configured_max_gpus === 'number' ? searchSummaryRaw.configured_max_gpus : 0,
        hardware_max_gpus: typeof searchSummaryRaw.hardware_max_gpus === 'number' ? searchSummaryRaw.hardware_max_gpus : 0,
        effective_max_gpus: typeof searchSummaryRaw.effective_max_gpus === 'number' ? searchSummaryRaw.effective_max_gpus : 0,
        time_budget_seconds: typeof searchSummaryRaw.time_budget_seconds === 'number' ? searchSummaryRaw.time_budget_seconds : 0,
        budget_exhausted: searchSummaryRaw.budget_exhausted === true,
        modes_searched: Array.isArray(searchSummaryRaw.modes_searched)
          ? searchSummaryRaw.modes_searched.filter((mode): mode is string => typeof mode === 'string')
          : undefined,
        candidate_configurations: typeof searchSummaryRaw.candidate_configurations === 'number'
          ? searchSummaryRaw.candidate_configurations
          : undefined,
        evaluated_configurations: typeof searchSummaryRaw.evaluated_configurations === 'number'
          ? searchSummaryRaw.evaluated_configurations
          : undefined,
      }
    : undefined

  const demandRaw = rawData.demand && typeof rawData.demand === 'object'
    ? rawData.demand as Record<string, unknown>
    : null
  const demand: RecommendDemandOutcome | undefined = demandRaw
    ? {
        targetRps: typeof demandRaw.target_rps === 'number' ? demandRaw.target_rps : 0,
        achievedRps: typeof demandRaw.achieved_rps === 'number' ? demandRaw.achieved_rps : null,
        met: demandRaw.met === true,
        observedConcurrency: typeof demandRaw.observed_concurrency === 'number' ? demandRaw.observed_concurrency : null,
        windowSeconds: typeof demandRaw.window_seconds === 'number' ? demandRaw.window_seconds : null,
        completedRequests: typeof demandRaw.completed_requests === 'number' ? demandRaw.completed_requests : null,
        evaluatedConfigurations: typeof demandRaw.evaluated_configurations === 'number'
          ? demandRaw.evaluated_configurations
          : null,
        bestObservedRps: typeof demandRaw.best_observed_rps === 'number' ? demandRaw.best_observed_rps : null,
        simulator: demandRaw.simulator === 'blis' ? 'blis' : 'aisimulate',
      }
    : undefined
  const sweepRaw = rawData.sweep && typeof rawData.sweep === 'object'
    ? rawData.sweep as Record<string, unknown>
    : null
  const sweep: RecommendSweepMeta | undefined = sweepRaw
    ? {
        runId: typeof sweepRaw.run_id === 'string' ? sweepRaw.run_id : null,
        algorithm: typeof sweepRaw.algorithm === 'string' ? sweepRaw.algorithm : null,
        seed: typeof sweepRaw.seed === 'number' ? sweepRaw.seed : null,
        maxTrials: typeof sweepRaw.max_trials === 'number' ? sweepRaw.max_trials : null,
        evaluated: typeof sweepRaw.evaluated === 'number' ? sweepRaw.evaluated : null,
        feasible: typeof sweepRaw.feasible === 'number' ? sweepRaw.feasible : null,
        timedOut: typeof sweepRaw.timed_out === 'number' ? sweepRaw.timed_out : null,
        budgetExhausted: sweepRaw.budget_exhausted === true,
        strictSla: typeof sweepRaw.strict_sla === 'boolean' ? sweepRaw.strict_sla : null,
        windowMinGpus: typeof sweepRaw.window_min_gpus === 'number' ? sweepRaw.window_min_gpus : null,
        windowMaxGpus: typeof sweepRaw.window_max_gpus === 'number' ? sweepRaw.window_max_gpus : null,
      }
    : undefined

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
  const memoryBreakdown = best.memory_breakdown as Record<string, unknown> | undefined
  const bytes = (field: string): number => {
    const value = memoryBreakdown?.[field]
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : 0
  }
  const memoryBreakdownGb = memoryBreakdown
    ? {
        weightsGb: bytes('weights_bytes') / (1024 ** 3),
        kvCacheGb: bytes('kv_cache_bytes') / (1024 ** 3),
        overheadGb: (bytes('activations_bytes') + bytes('runtime_overhead_bytes') + bytes('comm_overhead_bytes')) / (1024 ** 3),
      }
    : null
  const memoryFromBreakdown = memoryBreakdown
    ? (memoryBreakdownGb!.weightsGb + memoryBreakdownGb!.kvCacheGb + memoryBreakdownGb!.overheadGb)
    : 0
  const memoryGb = typeof best.memory === 'number' && Number.isFinite(best.memory)
    ? best.memory
    : memoryFromBreakdown
  const memoryAvailable = typeof best.memory === 'number' || memoryBreakdown != null

  const requiredValues = [
    totalGpusNeeded,
    gpusPerReplica,
    replicasNeeded,
    clusterTokensPerSecond,
  ]
  // A demand run has no latency SLA, so latency columns are informational and
  // must not make an otherwise usable sizing result fail validation.
  if ((request.sla_mode ?? 'enforce') === 'enforce') {
    requiredValues.push(ttftLatencyMs, tpotMs)
  }
  if (requiredValues.some(value => !Number.isFinite(value) || value <= 0)) {
    return makeError(
      requestId,
      'AISIM_INVALID_RESPONSE',
      'AISimulators returned an incomplete recommendation without positive GPU, throughput, TTFT, and TPOT values.',
    )
  }

  const missesLatencyTarget = (request.sla_mode ?? 'enforce') === 'none'
    ? false
    : request.request_latency != null
      ? !Number.isFinite(requestLatencyMs) || requestLatencyMs <= 0 || requestLatencyMs > request.request_latency
      : ttftLatencyMs > (request.ttft ?? Number.POSITIVE_INFINITY) ||
        tpotMs > (request.tpot ?? 30)
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
    simulator,
    searchSummary,
    demand,
    sweep,
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
      value: memoryGb,
      unit: 'GB',
      breakdown: memoryBreakdownGb,
    },
    memoryAvailable,
    metadata: {
      modelPath: request.model_path,
      system: request.system,
      inputTokens: request.isl,
      outputTokens: request.osl,
      targetTtftMs: request.ttft ?? null,
      durationMs,
    },
    warnings: Array.isArray(best.warnings)
      ? best.warnings
          .filter((warning): warning is string => typeof warning === 'string')
          .map(message => ({ code: 'SIMULATOR_WARNING', message }))
      : [],
  }
}

// ─── Pareto sweep ────────────────────────────────────────────────────────────

export interface ParetoResult {
  status: 'completed'
  simulator: 'aisimulate'
  candidates: Candidate[]
  views: CandidateViews | null
  sweep: FrontierSweep | null
}

export interface ParetoErrorResponse {
  status: 'failed'
  error: { code: string; message: string }
}

export type ParetoResponse = ParetoResult | ParetoErrorResponse

/**
 * Run one bounded Pareto sweep and return the candidate ledger plus the
 * simulator-computed views. This is a companion to a normal sizing request —
 * it never replaces or substitutes for the min_gpus recommendation.
 */
export async function callPareto(
  request: RecommendRequest,
  options: { timeoutSeconds?: number; signal?: AbortSignal } = {},
): Promise<ParetoResponse> {
  if (request.simulator !== 'aisimulate') {
    return {
      status: 'failed',
      error: {
        code: 'AISIM_UNSUPPORTED',
        message: 'The Pareto frontier is available from AISimulate only; BLIS does not provide it.',
      },
    }
  }

  const baseUrl = process.env.AISIMULATORS_GATEWAY_URL
  const requestTimeoutSeconds = options.timeoutSeconds ?? gatewayTimeoutSeconds()
  if (!baseUrl) {
    return {
      status: 'failed',
      error: { code: 'AISIM_NOT_CONFIGURED', message: 'AISimulators API URL is not configured' },
    }
  }

  const basePayload = baseRequestPayload(request)
  // A Pareto sweep drives its own load. Strip the primary request's load target
  // and, in demand mode, its absent latency bounds; the sweep is strict-SLA
  // only when the primary request enforced one.
  delete basePayload.target_request_rate
  delete basePayload.target_concurrency
  if ((request.sla_mode ?? 'enforce') === 'none') {
    delete basePayload.ttft
    delete basePayload.tpot
    delete basePayload.request_latency
  }
  let response: Response
  try {
    response = await fetch(`${baseUrl}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ ...basePayload, objective: 'pareto' }),
      signal: options.signal
        ? AbortSignal.any([AbortSignal.timeout(requestTimeoutSeconds * 1000), options.signal])
        : AbortSignal.timeout(requestTimeoutSeconds * 1000),
    })
  } catch (err: unknown) {
    if (options.signal?.aborted) throw err
    if (err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      return {
        status: 'failed',
        error: { code: 'AISIM_TIMEOUT', message: `The Pareto sweep did not finish within ${requestTimeoutSeconds} seconds.` },
      }
    }
    return {
      status: 'failed',
      error: { code: 'AISIM_UNAVAILABLE', message: 'The AISimulators API is unreachable.' },
    }
  }

  let body: unknown = null
  try {
    body = await response.json()
  } catch {
    // Fall through to the status-based error below.
  }

  if (!response.ok) {
    const parsedDetail = parseErrorDetail(body)
    const detail = parsedDetail?.message
      ?? (isRecord(body) && isRecord(body.error) && typeof body.error.message === 'string'
        ? body.error.message
        : `Pareto sweep failed (${response.status})`)
    const validationError = parsedDetail?.validation === true
    const code = response.status === 429
      ? 'AISIM_BUSY'
      : response.status === 504
        ? 'AISIM_TIMEOUT'
        : response.status === 422
          ? validationError ? 'INVALID_REQUEST' : 'AISIM_NO_CONFIGURATION'
          : response.status >= 500
            ? 'AISIM_UNAVAILABLE'
            : 'INVALID_REQUEST'
    return { status: 'failed', error: { code, message: detail } }
  }

  const parsed = parseFrontierBody(body)
  if (parsed === null) {
    return {
      status: 'failed',
      error: {
        code: 'SIMULATOR_MISMATCH',
        message: 'The API did not confirm AISimulate as the Pareto sweep simulator.',
      },
    }
  }

  return {
    status: 'completed',
    simulator: 'aisimulate',
    candidates: parsed.candidates,
    views: parsed.views,
    sweep: parsed.sweep,
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
  if (request.simulator === 'blis') {
    if (signal?.aborted) throw new DOMException('Recommendation search aborted', 'AbortError')
    const totalTimeoutSeconds = gatewayTimeoutSeconds()
    // BLIS owns a bounded TP-only search in the service. It is one request,
    // not a sequence of AISimulate GPU windows; avoid publishing a guessed
    // max-GPU range that would misstate the search the service actually ran.
    const response = await callRecommend(request, {
      timeoutSeconds: Math.max(1, totalTimeoutSeconds - 1),
      signal,
    })
    yield { type: 'completed', response }
    return
  }

  if ((request.sla_mode ?? 'enforce') === 'none') {
    if (signal?.aborted) throw new DOMException('Recommendation search aborted', 'AbortError')
    const totalTimeoutSeconds = gatewayTimeoutSeconds()
    // Demand sizing is one bounded sweep over the full GPU window: the service
    // itself selects the fewest GPUs that sustain the rate, or the best
    // observed rate when none do. Replaying it as shrinking windows would
    // repeat the same sweep and could return an unmet window before a larger
    // one was ever tried.
    const response = await callRecommend(request, {
      timeoutSeconds: Math.max(1, totalTimeoutSeconds - 1),
      signal,
    })
    yield { type: 'completed', response }
    return
  }

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

export function configuredMaxCandidateGpus(): number {
  const value = Number(process.env.AISIMULATORS_MAX_CANDIDATE_GPUS)
  return Number.isInteger(value) && value > 0 ? value : 1024
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeError(requestId: string, code: string, message: string): RecommendErrorResponse {
  return { requestId, status: 'failed', error: { code, message } }
}

/**
 * Read a FastAPI error `detail`.
 *
 * A string is a domain message, such as "No configuration meets the specified
 * requirements." A list is a Pydantic request-validation failure, which means
 * the request we built was malformed rather than the workload infeasible — the
 * two must not collapse into the same "no configuration" state.
 */
function parseErrorDetail(body: unknown): { message: string; validation: boolean } | null {
  if (!isRecord(body)) return null
  const raw = body.detail
  if (typeof raw === 'string' && raw.length > 0) {
    return { message: raw, validation: false }
  }
  if (Array.isArray(raw)) {
    const parts = raw
      .map((item) => {
        if (!isRecord(item)) return null
        const location = Array.isArray(item.loc)
          ? item.loc.filter((part): part is string | number => typeof part === 'string' || typeof part === 'number').join('.')
          : ''
        const message = typeof item.msg === 'string' ? item.msg : null
        if (message == null) return null
        return location ? `${location}: ${message}` : message
      })
      .filter((part): part is string => part != null)
    return {
      message: parts.length > 0 ? parts.join('; ') : 'Request validation failed',
      validation: true,
    }
  }
  return null
}

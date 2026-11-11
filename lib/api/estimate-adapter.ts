/**
 * Adapter that calls the AISimulators /predict API and converts the response
 * into the InferenceConfigResult shape used by the Predict performance page.
 *
 * Unlike the recommend adapter, /predict takes explicit TP/PP/batch_size
 * and returns TTFT/TPOT for that specific configuration. The caller is
 * responsible for choosing the parallelism; this adapter does not search
 * for an optimal config.
 */

import type { EstimatePhase, InferenceConfigResult } from '@/lib/gpu-math/inference-config/types'

const GB = 1_073_741_824

export class EstimateError extends Error {
  constructor(public readonly code: string, message: string) {
    super(message)
    this.name = 'EstimateError'
  }
}

export interface EstimateAdapterInput {
  model_path: string
  system: string
  isl: number
  osl: number
  max_seq_len?: number
  prefill_max_seq_len?: number
  decode_max_seq_len?: number
  batch_size: number
  requests_per_load_unit?: number
  tp_size: number
  pp_size?: number
  /** Attention data-parallel width; omitted means the service default. */
  attention_dp_size?: number
  gpu_memory_utilization?: number
  max_num_seqs?: number
  enable_chunked_prefill?: boolean
  vram_gb?: number | null
  backend?: string
  backend_version?: string
  hf_model_config?: Record<string, unknown> | null
  moe_ep_size?: number
  moe_tp_size?: number
  moe_quant_mode?: string
  prefix?: number
  kvcache_quant_mode?: string | null
  gemm_quant_mode?: string | null
  /** Simulator identifier discovered from the catalog; open by design. */
  simulator?: string
  // Disagg: when mode='disagg', prefill/decode drive separate pools.
  mode?: 'agg' | 'disagg'
  fabric?: string
  prefill?: EstimatePhaseInput
  decode?: EstimatePhaseInput
}

export interface EstimatePhaseInput {
  tp: number
  pp: number
  workers: number
  batch: number
}

/**
 * Whether a model config describes a mixture-of-experts model.
 *
 * Expert counts live at the top level for most families and under
 * `text_config` for Gemma 4. Take the largest count rather than the first
 * non-nullish one: a top-level `1` must not mask a nested `128`.
 */
export function isMoeConfig(config: Record<string, unknown> | null | undefined): boolean {
  if (!config) return false
  const textConfig = config.text_config && typeof config.text_config === 'object'
    ? config.text_config as Record<string, unknown>
    : null

  if (config.enable_moe_block === true || textConfig?.enable_moe_block === true) return true

  const counts = [
    config.num_experts,
    config.num_local_experts,
    textConfig?.num_experts,
    textConfig?.num_local_experts,
  ]
  return counts.some(count => typeof count === 'number' && count > 1)
}

function deriveBottleneck(ttft: number, tpot: number) {
  if (ttft > 2000) {
    return {
      primary: 'TTFT' as const,
      risk: 'High TTFT — users will notice slow first-token response.',
      fix_suggestions: ['Increase TP to reduce prefill latency', 'Use chunked prefill'],
    }
  }
  if (tpot > 50) {
    return {
      primary: 'TPOT' as const,
      risk: 'High TPOT — generation will feel slow.',
      fix_suggestions: ['Reduce batch size', 'Use a smaller quantization (FP8)'],
    }
  }
  return {
    primary: 'THROUGHPUT' as const,
    risk: 'Balanced — no single bottleneck dominates.',
    fix_suggestions: [],
  }
}

export interface EstimateWithDebug {
  result: InferenceConfigResult
  debugRequest: Record<string, unknown>
  debugResponse: Record<string, unknown>
  debugStatus: number
  debugDuration: number
}

/**
 * Build the /api/predict body for an adapter input.
 *
 * Exported so replay surfaces (the debug panel and "Copy API request") use the
 * exact payload that is submitted instead of reconstructing it.
 */
export function buildPredictRequestBody(input: EstimateAdapterInput): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model_path: input.model_path,
    system: input.system,
    backend: input.backend ?? 'vllm',
    isl: input.isl,
    osl: input.osl,
    batch_size: input.batch_size,
    tp_size: input.tp_size,
  }

  if (input.max_seq_len != null) body.max_seq_len = input.max_seq_len
  if (input.requests_per_load_unit != null) body.requests_per_load_unit = input.requests_per_load_unit
  if (input.prefill_max_seq_len != null) body.prefill_max_seq_len = input.prefill_max_seq_len
  if (input.decode_max_seq_len != null) body.decode_max_seq_len = input.decode_max_seq_len

  if (input.simulator) body.simulator = input.simulator
  if (input.pp_size != null && input.pp_size > 1) body.pp_size = input.pp_size
  if (input.attention_dp_size != null && input.attention_dp_size > 1) body.attention_dp_size = input.attention_dp_size
  if (input.gpu_memory_utilization != null) body.gpu_memory_utilization = input.gpu_memory_utilization
  if (input.max_num_seqs != null) body.max_num_seqs = input.max_num_seqs
  if (input.enable_chunked_prefill != null) body.enable_chunked_prefill = input.enable_chunked_prefill
  if (input.backend_version) body.backend_version = input.backend_version
  if (input.prefix != null && input.prefix > 0) body.prefix = input.prefix
  if (input.kvcache_quant_mode) body.kvcache_quant_mode = input.kvcache_quant_mode
  if (input.gemm_quant_mode) body.gemm_quant_mode = input.gemm_quant_mode
  if (input.moe_quant_mode) body.moe_quant_mode = input.moe_quant_mode

  if (input.hf_model_config) {
    body.model_config = input.hf_model_config
  }

  // Disagg: send per-pool parallelism. The backend runs predict in
  // mode='disagg' and returns prefill_config / decode_config.
  if (input.mode === 'disagg' && input.prefill && input.decode) {
    body.mode = 'disagg'
    if (input.fabric) body.fabric = input.fabric
    body.prefill_tp_size = input.prefill.tp
    body.prefill_pp_size = input.prefill.pp
    body.prefill_num_workers = input.prefill.workers
    body.prefill_batch_size = input.prefill.batch
    body.decode_tp_size = input.decode.tp
    body.decode_pp_size = input.decode.pp
    body.decode_num_workers = input.decode.workers
    body.decode_batch_size = input.decode.batch
  }

  // Forward only widths the caller actually chose. When they are omitted the
  // API supplies the layout default (FFN tensor-sharded across the attention
  // width, no expert parallelism) and reports the resolved pair back.
  if (input.moe_ep_size != null) body.moe_ep_size = input.moe_ep_size
  if (input.moe_tp_size != null) body.moe_tp_size = input.moe_tp_size

  return body
}

export function fetchEstimateAsInferenceResult(
  input: EstimateAdapterInput,
  captureDebug: true
): Promise<EstimateWithDebug>
export function fetchEstimateAsInferenceResult(
  input: EstimateAdapterInput,
  captureDebug?: false
): Promise<InferenceConfigResult>
export async function fetchEstimateAsInferenceResult(
  input: EstimateAdapterInput,
  captureDebug: boolean = false
): Promise<EstimateWithDebug | InferenceConfigResult> {
  const body = buildPredictRequestBody(input)

  const startTime = performance.now()
  const res = await fetch('/api/predict?include=config,memory', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const duration = Math.round(performance.now() - startTime)

  const data = await res.json()

  if (!res.ok || data?.status === 'failed') {
    const code = (data?.error as Record<string, unknown>)?.code as string ?? 'AISIM_NO_CONFIGURATION'
    const message = (data?.error as Record<string, unknown>)?.message as string ?? data?.detail as string ?? 'AISimulators prediction failed'
    throw new EstimateError(code, message)
  }

  // Provenance is an open identifier. The response must name the same
  // simulator the caller requested; older builds that omit it are accepted as
  // the requested simulator. A response that names another simulator is never
  // accepted as a fallback.
  const requestedSimulator = typeof input.simulator === 'string' && input.simulator.length > 0
    ? input.simulator
    : null
  const responseSimulator = typeof data?.simulator === 'string' && data.simulator.trim().length > 0
    ? data.simulator
    : null
  if (data?.simulator != null && responseSimulator == null) {
    throw new EstimateError('AISIM_INVALID_RESPONSE', 'The API returned an unknown simulator identifier.')
  }
  const actualSimulator = responseSimulator ?? requestedSimulator
  if (actualSimulator == null) {
    throw new EstimateError(
      'AISIM_INVALID_RESPONSE',
      'The API did not identify the simulator that produced this estimate.',
    )
  }
  if (requestedSimulator != null && responseSimulator != null && actualSimulator !== requestedSimulator) {
    throw new EstimateError(
      'SIMULATOR_MISMATCH',
      `Requested ${requestedSimulator}, but the API confirmed ${actualSimulator}. No fallback was accepted.`,
    )
  }
  const tp = data.tp ?? input.tp_size
  const pp = data.pp ?? input.pp_size ?? 1

  // Validate PP is a positive integer before using it as a divisor
  if (!Number.isSafeInteger(pp) || pp < 1) {
    throw new EstimateError('INVALID_PP_SIZE', 'Pipeline parallel size must be a positive integer')
  }

  const sc = data.serving_config
  const mb = data.memory_breakdown
  const simulator: string = actualSimulator
  // The response names the engine release it actually used. AISimulate reports
  // `backend_version`; BLIS reports it under provenance. Null means the
  // response did not say, and callers must not infer one.
  const provenance = data?.provenance && typeof data.provenance === 'object'
    ? data.provenance as Record<string, unknown>
    : null
  const reportedEngineVersion = typeof data?.backend_version === 'string' && data.backend_version.trim()
    ? data.backend_version.trim()
    : typeof provenance?.engine_version === 'string' && provenance.engine_version.trim()
      ? provenance.engine_version.trim()
      : null
  const memoryAvailable = !!mb || typeof data.memory === 'number'
  const totalVramGb = input.vram_gb ?? null
  const gmu = sc?.gpu_memory_utilization ?? input.gpu_memory_utilization ?? 0.9
  const weightGb = memoryAvailable
    ? mb ? mb.weights_bytes / GB : (data.memory ?? 0) * 0.5
    : 0
  const kvCacheGb = memoryAvailable && mb && typeof mb.kv_cache_bytes === 'number'
    ? mb.kv_cache_bytes / GB
    : 0
  const usablePerGpu = totalVramGb != null ? totalVramGb * gmu : null
  const maxNumSeqs = sc?.max_num_seqs ?? input.batch_size
  // Null is a real answer: the simulator resolved the context window by policy
  // (for example BLIS's `max_model_len: auto`) and names that policy instead of
  // reporting a number. Do not invent a value the API did not give.
  const reportedMaxModelLen = sc?.max_model_len
  const maxModelLen = typeof reportedMaxModelLen === 'number'
    ? reportedMaxModelLen
    : reportedMaxModelLen === null
      ? null
      : input.max_seq_len ?? null
  const maxModelLenPolicy = typeof sc?.max_model_len_policy === 'string'
    ? sc.max_model_len_policy
    : null
  // Batch-token sizing still needs a concrete window; use the caller's chosen
  // length when the API reports the window by policy.
  const maxModelLenForBatch = maxModelLen ?? input.max_seq_len ?? input.isl + input.osl

  // With pipeline parallel, weights are sharded across TP×PP GPUs
  const shardCount = tp * pp

  // Disagg: fold the per-pool worker configs into EstimatePhase records.
  const buildPhase = (wc: Record<string, unknown> | null | undefined, fallback: EstimatePhaseInput): EstimatePhase => {
    const tpv = (wc?.tp as number) ?? fallback.tp
    const ppv = (wc?.pp as number) ?? fallback.pp
    return {
      workers: (wc?.num_workers as number) ?? fallback.workers,
      gpusPerWorker: tpv * ppv,
      tp_size: tpv,
      pp_size: ppv,
      batch_size: (wc?.batch_size as number) ?? fallback.batch,
      memory_gb: (wc?.memory_gb as number | null) ?? null,
      moe_tp_size: (wc?.moe_tp as number | null | undefined) ?? null,
      moe_ep_size: (wc?.moe_ep as number | null | undefined) ?? null,
    }
  }
  const isDisagg = data.mode === 'disagg' && !!input.prefill && !!input.decode
  const disagg = isDisagg
    ? { prefill: buildPhase(data.prefill_config, input.prefill!), decode: buildPhase(data.decode_config, input.decode!) }
    : undefined

  const ttft = data.ttft ?? 0
  const tpot = data.tpot ?? 0
  const hasTiming = Number.isFinite(ttft) && Number.isFinite(tpot)

  const warnings: string[] = memoryAvailable
    ? []
    : [typeof data.memory_unavailable_reason === 'string'
      ? data.memory_unavailable_reason
      : 'The selected simulator did not return a memory breakdown.']
  const rawWarnings: unknown = data.warnings
  if (Array.isArray(rawWarnings)) {
    warnings.push(...rawWarnings.filter((warning: unknown): warning is string => typeof warning === 'string'))
  }
  if (input.mode === 'disagg' && !isDisagg) {
    warnings.push('Disaggregated mode was requested but the backend returned an aggregated estimate.')
  }
  if (!hasTiming) {
    warnings.push('Timing metrics (TTFT, TPOT) missing or non-finite; performance prediction unavailable.')
  }
  const requestLatency = hasTiming
    ? typeof data.request_latency === 'number'
      ? data.request_latency
      : ttft + tpot * input.osl
    : 0
  const throughput = typeof data.tokens_per_second === 'number'
    ? data.tokens_per_second
    : hasTiming && input.osl > 0 && requestLatency > 0 ? (input.osl * 1000) / requestLatency : 0
  const concurrency = Math.max(0, Math.floor(maxNumSeqs))

  const result: InferenceConfigResult = {
    simulator,
    engine_version: reportedEngineVersion,
    memory_available: memoryAvailable,
    mode: isDisagg ? 'disagg' : 'agg',
    disagg,
    moe_tp_size: isDisagg ? null : (data.moe_tp as number | null | undefined) ?? null,
    moe_ep_size: isDisagg ? null : (data.moe_ep as number | null | undefined) ?? null,
    memory_analysis: {
      weight_gb: weightGb,
      weight_gb_per_gpu: weightGb / shardCount,
      total_vram_gb: totalVramGb,
      usable_hbm_per_gpu: usablePerGpu,
      tp_size: tp,
      replicas: 1,
      kv_cache_budget_gb: usablePerGpu != null ? usablePerGpu - weightGb / shardCount : null,
      kv_cache_used_gb: kvCacheGb,
      max_sequences_from_memory: maxNumSeqs,
      kv_category: memoryAvailable ? simulator : 'unavailable',
      kv_category_label: memoryAvailable ? `${simulator} prediction` : 'Memory unavailable',
    },
    vllm_config: {
      tensor_parallel_size: sc?.tensor_parallel_size ?? tp,
      max_model_len: maxModelLen,
      max_model_len_policy: maxModelLenPolicy,
      max_num_seqs: maxNumSeqs,
      gpu_memory_utilization: gmu,
      max_num_batched_tokens: sc?.max_num_batched_tokens ?? Math.min(maxModelLenForBatch * maxNumSeqs, 32768),
      enable_chunked_prefill: typeof sc?.enable_chunked_prefill === 'boolean'
        ? sc.enable_chunked_prefill
        : input.enable_chunked_prefill ?? null,
      // Null means the run did not state the setting. Do not present a default
      // as if the simulator chose it.
      enable_prefix_caching: typeof sc?.enable_prefix_caching === 'boolean'
        ? sc.enable_prefix_caching
        : null,
      quantization: sc?.quantization ?? 'auto',
    },
    parallelism_strategy: {
      strategy: pp > 1 ? 'PP_ACROSS_NODES' : 'TP_ONLY',
      pp_size: pp,
      topology_note: `TP=${tp}, PP=${pp}, batch=${input.batch_size}`,
    },
    performance: {
      ttft_ms: ttft,
      tpot_ms: tpot,
      request_latency_ms: requestLatency,
      throughput_tokens_per_sec: throughput,
      throughput_tokens_per_sec_per_user: typeof data.tokens_per_second_per_user === 'number'
        && Number.isFinite(data.tokens_per_second_per_user)
        ? data.tokens_per_second_per_user
        : null,
      concurrency: Math.max(0, concurrency),
    },
    bottleneck_analysis: deriveBottleneck(ttft, tpot),
    diagnostics: {
      nvidia_smi_watch: 'nvidia-smi dmon -s pucvmet -d 1',
      dcgm_metrics: ['DCGM_FI_PROF_GR_ENGINE_ACTIVE', 'DCGM_FI_DEV_FB_USED'],
      vllm_metrics: ['vllm:num_requests_running', 'vllm:gpu_cache_usage_perc'],
    },
    warnings,
  }

  if (captureDebug) {
    return {
      result,
      debugRequest: body,
      debugResponse: data,
      debugStatus: res.status,
      debugDuration: duration,
    }
  }

  return result
}

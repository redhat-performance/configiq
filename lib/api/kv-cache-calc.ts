import type { KvCacheCalcRequest } from './schemas'
import { gatewayTimeoutSeconds } from './timeout'
import { parseDebugRecord, type DebugRecord } from '../debug/types'

// ─── Types ───────────────────────────────────────────────────────────────────

export interface KvCacheCalcResult {
  requestId: string
  status: 'completed'
  kvCache: {
    totalBytes: number
    perTokenBytes: number
    totalTokens: number
  }
  memoryBreakdown: {
    weightsBytes: number
    activationsBytes: number
    runtimeOverheadBytes: number
    commOverheadBytes: number
  }
  gpuCapacity: {
    totalBytes: number
  }
  /** Commands and input files that reproduce this estimate, when the API sent them. */
  debug?: DebugRecord | null
  /**
   * Things the API asks the reader to know about the answer, such as an input
   * it could not apply. Empty when there is nothing to report.
   */
  warnings: string[]
  metadata: {
    /**
     * Simulator that produced the estimate, as the API reported it. The page
     * forwards the selected simulator and never assumes which one answered.
     */
    simulator: string
    modelPath: string
    backend: string
    backendVersion: string | null
    system: string
    maxNumTokens: number
    maxBatchSize: number
    tpSize: number
    ppSize: number
    moeTpSize: number | null
    moeEpSize: number | null
    memoryFractionKind: string
    /** The fraction the request stated; null when the simulator's own applied. */
    memoryFractionValue: number | null
    source: string
    durationMs: number
  }
}

export interface KvCacheCalcErrorResponse {
  requestId: string
  status: 'failed'
  error: {
    code: string
    message: string
  }
}

export type KvCacheCalcResponse = KvCacheCalcResult | KvCacheCalcErrorResponse

// ─── Request ID ──────────────────────────────────────────────────────────────

export function generateKvRequestId(): string {
  return 'kv_' + crypto.randomUUID().replace(/-/g, '').slice(0, 12)
}

/**
 * The `/memory` body for a KV cache request.
 *
 * Exported so the page's proxied request and the deep-link contract are audited
 * against the payload that is really sent, rather than against a copy of it.
 */
export function buildKvCachePayload(request: KvCacheCalcRequest): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    model_path: request.model_path,
    backend: request.backend,
    system: request.system,
    max_num_tokens: request.max_num_tokens,
    max_batch_size: request.max_batch_size,
    tp_size: request.tp_size,
    pp_size: request.pp_size,
    memory_fraction_kind: request.memory_fraction_kind,
  }
  // A fraction the caller did not state is left out, so the simulator resolves
  // its own default instead of receiving a number the view never showed.
  if (request.memory_fraction_value != null) {
    payload.memory_fraction_value = request.memory_fraction_value
  }
  if (request.simulator) payload.simulator = request.simulator
  if (request.backend_version) payload.backend_version = request.backend_version
  if (request.moe_tp_size != null) payload.moe_tp_size = request.moe_tp_size
  if (request.moe_ep_size != null) payload.moe_ep_size = request.moe_ep_size
  if (request.model_config != null) payload.model_config = request.model_config
  return payload
}

// ─── Service ─────────────────────────────────────────────────────────────────

export async function callKvCacheCalc(
  request: KvCacheCalcRequest
): Promise<KvCacheCalcResponse> {
  const requestId = generateKvRequestId()
  const startTime = performance.now()

  const baseUrl = process.env.AISIMULATORS_GATEWAY_URL
  const timeoutSeconds = gatewayTimeoutSeconds()

  if (!baseUrl) {
    return makeError(requestId, 'AISIM_NOT_CONFIGURED', 'AISimulators API URL is not configured')
  }

  const externalPayload = buildKvCachePayload(request)

  let response: Response
  try {
    response = await fetch(`${baseUrl}/memory?include=debug`, {
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
      return makeError(requestId, 'AISIM_TIMEOUT', `The AISimulators API did not respond within ${timeoutSeconds} seconds (waited ${durationMs}ms)`)
    }
    return makeError(requestId, 'AISIM_UNAVAILABLE', 'AISimulators API is unreachable')
  }

  if (!response.ok) {
    let detail = `AISimulators API returned HTTP ${response.status}`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch { /* ignore parse errors */ }

    const code = response.status === 422 ? 'AISIM_UNSUPPORTED' : 'AISIM_UNAVAILABLE'
    return makeError(requestId, code, detail)
  }

  let rawData: Record<string, unknown>
  try {
    const parsed = await response.json()
    if (parsed == null || typeof parsed !== 'object') {
      return makeError(requestId, 'AISIM_INVALID_RESPONSE', 'No valid memory data found for this model and hardware combination.')
    }
    rawData = parsed as Record<string, unknown>
  } catch {
    return makeError(requestId, 'AISIM_INVALID_RESPONSE', 'AISimulators API returned non-JSON response')
  }

  const breakdown = (rawData.memory_breakdown ?? {}) as Record<string, unknown>
  const durationMs = Math.round(performance.now() - startTime)

  // Provenance is an open identifier. The response must name the same simulator
  // the caller requested; older builds that omit it are accepted as the
  // requested simulator. A response that names another simulator is never
  // accepted as a fallback.
  const requestedSimulator = request.simulator && request.simulator.length > 0
    ? request.simulator
    : null
  const responseSimulator = typeof rawData.simulator === 'string' && rawData.simulator.trim().length > 0
    ? rawData.simulator.trim()
    : null
  if (requestedSimulator != null && responseSimulator != null && requestedSimulator !== responseSimulator) {
    return makeError(
      requestId,
      'SIMULATOR_MISMATCH',
      `Requested ${requestedSimulator}, but the API confirmed ${responseSimulator}. No fallback was accepted.`,
    )
  }

  return {
    requestId,
    status: 'completed',
    kvCache: {
      totalBytes: asNumber(rawData.total_kv_size_bytes, 0),
      perTokenBytes: asNumber(rawData.kv_size_per_token_bytes, 0),
      totalTokens: asNumber(rawData.total_kv_size_tokens, 0),
    },
    memoryBreakdown: {
      weightsBytes: asNumber(breakdown.weights_bytes, 0),
      activationsBytes: asNumber(breakdown.activations_bytes, 0),
      runtimeOverheadBytes: asNumber(breakdown.runtime_overhead_bytes, 0),
      commOverheadBytes: asNumber(breakdown.comm_overhead_bytes, 0),
    },
    gpuCapacity: {
      totalBytes: asNumber(rawData.total_gpu_capacity_bytes, 0),
    },
    debug: parseDebugRecord(rawData.debug),
    warnings: Array.isArray(rawData.warnings)
      ? rawData.warnings.filter((warning): warning is string => typeof warning === 'string')
      : [],
    metadata: {
      // A response that names no simulator is read as the one requested, or
      // as the documented service default for a legacy build.
      simulator: responseSimulator ?? requestedSimulator ?? 'aisim',
      modelPath: request.model_path,
      backend: request.backend,
      backendVersion: request.backend_version ?? null,
      system: request.system,
      maxNumTokens: request.max_num_tokens,
      maxBatchSize: request.max_batch_size,
      tpSize: request.tp_size,
      ppSize: request.pp_size,
      moeTpSize: request.moe_tp_size ?? null,
      moeEpSize: request.moe_ep_size ?? null,
      memoryFractionKind: request.memory_fraction_kind,
      memoryFractionValue: request.memory_fraction_value ?? null,
      source: typeof rawData.source === 'string' ? rawData.source : 'unknown',
      durationMs,
    },
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeError(requestId: string, code: string, message: string): KvCacheCalcErrorResponse {
  return { requestId, status: 'failed', error: { code, message } }
}

function asNumber(val: unknown, fallback: number): number {
  return typeof val === 'number' ? val : fallback
}

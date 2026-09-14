import type { KvCacheCalcRequest } from './schemas'
import { aicTimeoutSeconds } from './timeout'

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
  metadata: {
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
    memoryFractionValue: number
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

// ─── Service ─────────────────────────────────────────────────────────────────

export async function callKvCacheCalc(
  request: KvCacheCalcRequest
): Promise<KvCacheCalcResponse> {
  const requestId = generateKvRequestId()
  const startTime = performance.now()

  const baseUrl = process.env.AICONFIGURATOR_GATEWAY_URL
  const timeoutSeconds = aicTimeoutSeconds()

  if (!baseUrl) {
    return makeError(requestId, 'AIC_NOT_CONFIGURED', 'AIConfigurator API URL is not configured')
  }

  const externalPayload: Record<string, unknown> = {
    model_path: request.model_path,
    backend: request.backend,
    system: request.system,
    max_num_tokens: request.max_num_tokens,
    max_batch_size: request.max_batch_size,
    tp_size: request.tp_size,
    pp_size: request.pp_size,
    memory_fraction_kind: request.memory_fraction_kind,
    memory_fraction_value: request.memory_fraction_value,
  }
  if (request.backend_version) externalPayload.backend_version = request.backend_version
  if (request.moe_tp_size != null) externalPayload.moe_tp_size = request.moe_tp_size
  if (request.moe_ep_size != null) externalPayload.moe_ep_size = request.moe_ep_size
  if (request.model_config != null) externalPayload.model_config = request.model_config

  let response: Response
  try {
    response = await fetch(`${baseUrl}/memory`, {
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

    const code = response.status === 422 ? 'AIC_UNSUPPORTED' : 'AIC_UNAVAILABLE'
    return makeError(requestId, code, detail)
  }

  let rawData: Record<string, unknown>
  try {
    const parsed = await response.json()
    if (parsed == null || typeof parsed !== 'object') {
      return makeError(requestId, 'AIC_INVALID_RESPONSE', 'No valid memory data found for this model and hardware combination.')
    }
    rawData = parsed as Record<string, unknown>
  } catch {
    return makeError(requestId, 'AIC_INVALID_RESPONSE', 'AIConfigurator API returned non-JSON response')
  }

  const breakdown = (rawData.memory_breakdown ?? {}) as Record<string, unknown>
  const durationMs = Math.round(performance.now() - startTime)

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
    metadata: {
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
      memoryFractionValue: request.memory_fraction_value,
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

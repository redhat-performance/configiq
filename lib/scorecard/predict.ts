import type { ScorecardPrediction } from './types';

/**
 * One direct call to the AISimulators `/predict` endpoint.
 *
 * This is the build job's client, not the browser's: it talks to the gateway
 * over the public URL and never goes through a Next.js proxy. It deliberately
 * does not reuse `fetchEstimateAsInferenceResult`, which fetches the relative
 * `/api/predict` path and reshapes the response for the Predict page; the
 * scorecard needs the raw simulator provenance and the failure classification.
 */

export type PredictStatus = 'supported' | 'unsupported' | 'inconclusive';

export interface PredictOutcome {
  status: PredictStatus;
  /** The service's own message, or our diagnosis of a transport failure. */
  reason: string | null;
  prediction: ScorecardPrediction | null;
  engineVersion: string | null;
  /** Simulator the response named, when it named one. */
  simulator: string | null;
  httpStatus: number | null;
  durationMs: number;
}

function detailOf(data: unknown): string | null {
  if (!data || typeof data !== 'object') return null;
  const record = data as Record<string, unknown>;
  const error = record.error && typeof record.error === 'object' ? record.error as Record<string, unknown> : null;
  const candidates = [error?.message, record.detail, record.message];
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim().length > 0) return candidate.trim();
  }
  return null;
}

function reportedSimulator(data: unknown): string | null {
  if (!data || typeof data !== 'object') return null;
  const value = (data as Record<string, unknown>).simulator;
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;
}

function reportedEngineVersion(data: unknown): string | null {
  if (!data || typeof data !== 'object') return null;
  const record = data as Record<string, unknown>;
  if (typeof record.backend_version === 'string' && record.backend_version.trim().length > 0) {
    return record.backend_version.trim();
  }
  const provenance = record.provenance && typeof record.provenance === 'object'
    ? record.provenance as Record<string, unknown>
    : null;
  if (typeof provenance?.engine_version === 'string' && provenance.engine_version.trim().length > 0) {
    return provenance.engine_version.trim();
  }
  return null;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * Whether a 422 is asking for the model configuration rather than reporting a
 * capability gap. Those two are retried differently: the first is our omission,
 * the second is the simulator's answer.
 */
export function isModelConfigRejection(detail: string | null): boolean {
  if (!detail) return false;
  return /model_config|missing key|config\.json|incomplete model/i.test(detail);
}

export interface PredictOptions {
  timeoutSeconds: number;
  expectedSimulator: string;
}

export async function predictPoint(
  gatewayUrl: string,
  body: Record<string, unknown>,
  options: PredictOptions,
): Promise<PredictOutcome> {
  const started = Date.now();
  const url = `${gatewayUrl.replace(/\/+$/, '')}/predict?include=config,memory`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body),
      cache: 'no-store',
      signal: AbortSignal.timeout(options.timeoutSeconds * 1000),
    });
  } catch (error: unknown) {
    const timedOut = error instanceof Error
      && (error.name === 'TimeoutError' || error.name === 'AbortError');
    return {
      status: 'inconclusive',
      reason: timedOut
        ? `No answer within ${options.timeoutSeconds}s.`
        : `Request failed: ${error instanceof Error ? error.message : String(error)}`,
      prediction: null,
      engineVersion: null,
      simulator: null,
      httpStatus: null,
      durationMs: Date.now() - started,
    };
  }

  let data: unknown = null;
  try {
    data = await response.json();
  } catch {
    // Fall through: a non-JSON 5xx is a transport-level failure, not a verdict
    // about whether the simulator can model the run.
  }

  const durationMs = Date.now() - started;
  const detail = detailOf(data);

  if (!response.ok || (data && typeof data === 'object'
    && (data as Record<string, unknown>).status === 'failed')) {
    // 422 is the service saying "this simulator has no data or cannot model
    // this run". Everything else is us failing to get an answer, and must not
    // be reported as a capability gap.
    const status: PredictStatus = response.status === 422 ? 'unsupported' : 'inconclusive';
    return {
      status,
      reason: detail ?? `HTTP ${response.status}`,
      prediction: null,
      engineVersion: null,
      simulator: reportedSimulator(data),
      httpStatus: response.status,
      durationMs,
    };
  }

  if (!data || typeof data !== 'object') {
    return {
      status: 'inconclusive',
      reason: 'The response was not a JSON object.',
      prediction: null,
      engineVersion: null,
      simulator: null,
      httpStatus: response.status,
      durationMs,
    };
  }

  const simulator = reportedSimulator(data);
  if (simulator !== null && simulator !== options.expectedSimulator) {
    return {
      status: 'inconclusive',
      reason: `Requested ${options.expectedSimulator}, but the service answered as ${simulator}.`,
      prediction: null,
      engineVersion: null,
      simulator,
      httpStatus: response.status,
      durationMs,
    };
  }

  const record = data as Record<string, unknown>;
  const ttftMs = finiteNumber(record.ttft);
  const tpotMs = finiteNumber(record.tpot);
  if (ttftMs === null || tpotMs === null) {
    return {
      status: 'inconclusive',
      reason: 'The response did not include finite TTFT and TPOT values.',
      prediction: null,
      engineVersion: reportedEngineVersion(data),
      simulator,
      httpStatus: response.status,
      durationMs,
    };
  }

  return {
    status: 'supported',
    reason: null,
    prediction: {
      ttftMs,
      tpotMs,
      requestLatencyMs: finiteNumber(record.request_latency),
      outputTokensPerSecond: finiteNumber(record.tokens_per_second),
    },
    engineVersion: reportedEngineVersion(data),
    simulator,
    httpStatus: response.status,
    durationMs,
  };
}

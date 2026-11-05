'use client';

import * as React from 'react';
import {
  configuredMaxCandidateGpus,
  type RecommendProgressEvent,
  type RecommendResult,
} from '@/lib/api/recommend';
import { readRecommendStream } from '@/lib/api/recommend-stream';
import { fetchFrontier } from '@/lib/api/pareto';
import type { FrontierState } from '@/lib/types/pareto';

/**
 * Companion Pareto sweeps are sized from the primary result rather than a fixed
 * GPU cap: the frontier should cover the recommended point with room on both
 * sides, up to the simulator's own enumerated ceiling.
 */
const FRONTIER_MIN_WINDOW = 32;
const FRONTIER_WINDOW_MULTIPLIER = 2;

export type SizingGoal = 'latency' | 'demand';

interface RecommendParams {
  model_path: string;
  system: string;
  isl: number;
  osl: number;
  max_seq_len?: number;
  prefill_max_seq_len?: number;
  decode_max_seq_len?: number;
  /** `enforce` applies the latency SLA; `none` is a throughput-demand run. */
  sla_mode: 'enforce' | 'none';
  ttft?: number;
  tpot?: number;
  backend?: string;
  backend_version?: string;
  target_concurrency?: number;
  target_request_rate?: number;
  request_latency?: number;
  prefix?: number;
  model_config?: Record<string, unknown> | null;
  simulator?: 'aisimulate' | 'blis';
  fabric?: string | null;
}

interface RecommendState {
  params: RecommendParams | null;
  isLoading: boolean;
  result: RecommendResult | null;
  error: string | null;
  errorCode: string | null;
  elapsed: number;
  debugRequest: Record<string, unknown> | null;
  debugResponse: Record<string, unknown> | null;
  debugStatus: number | null;
  debugDuration: number | null;
  progress: RecommendProgressEvent | null;
  progressHistory: RecommendProgressEvent[];
  /**
   * Companion Pareto sweep for the same workload. It starts after the primary
   * recommendation completes so the primary numbers are visible while the
   * sweep runs, and it never replaces the primary result.
   */
  frontier: FrontierState;
  startSizing: (params: RecommendParams) => void;
  reset: () => void;
}

const RecommendContext = React.createContext<RecommendState>({
  params: null,
  isLoading: false,
  result: null,
  error: null,
  errorCode: null,
  elapsed: 0,
  debugRequest: null,
  debugResponse: null,
  debugStatus: null,
  debugDuration: null,
  progress: null,
  progressHistory: [],
  frontier: { status: 'idle' },
  startSizing: () => {},
  reset: () => {},
});

function requestBodyFor(p: RecommendParams): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model_path: p.model_path,
    system: p.system,
    isl: p.isl,
    osl: p.osl,
    sla_mode: p.sla_mode,
  };

  if (p.sla_mode === 'enforce') {
    body.objective = 'min_gpus';
    if (p.request_latency != null) body.request_latency = p.request_latency;
    else {
      if (p.ttft != null) body.ttft = p.ttft;
      if (p.tpot != null) body.tpot = p.tpot;
    }
    if (p.target_request_rate != null) body.target_request_rate = p.target_request_rate;
    else body.target_concurrency = p.target_concurrency ?? 32;
  } else {
    // A demand run measures an arrival rate. It carries no latency target and
    // no concurrency target, so the service cannot fabricate an SLA for it.
    body.objective = 'throughput_demand';
    body.target_request_rate = p.target_request_rate;
  }

  if (p.max_seq_len != null) body.max_seq_len = p.max_seq_len;
  if (p.prefill_max_seq_len != null) body.prefill_max_seq_len = p.prefill_max_seq_len;
  if (p.decode_max_seq_len != null) body.decode_max_seq_len = p.decode_max_seq_len;
  if (p.backend) body.backend = p.backend;
  if (p.backend_version) body.backend_version = p.backend_version;
  if (p.prefix != null && p.prefix > 0) body.prefix = p.prefix;
  if (p.model_config != null) body.model_config = p.model_config;
  if (p.simulator) body.simulator = p.simulator;
  if (p.fabric !== undefined) body.fabric = p.fabric;
  return body;
}

/** Window that surrounds the primary recommendation without a fixed GPU cap. */
function frontierWindow(primaryGpus: number): { minGpus: number; maxGpus: number } {
  const ceiling = configuredMaxCandidateGpus();
  const primary = Number.isFinite(primaryGpus) && primaryGpus > 0 ? primaryGpus : 1;
  const maxGpus = Math.min(
    ceiling,
    Math.max(FRONTIER_MIN_WINDOW, Math.ceil(primary * FRONTIER_WINDOW_MULTIPLIER)),
  );
  const minGpus = Math.max(1, Math.min(maxGpus, Math.floor(primary / 2)));
  return { minGpus, maxGpus };
}

export function RecommendProvider({ children }: { children: React.ReactNode }) {
  const [params, setParams] = React.useState<RecommendParams | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);
  const [result, setResult] = React.useState<RecommendResult | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [errorCode, setErrorCode] = React.useState<string | null>(null);
  const [elapsed, setElapsed] = React.useState(0);
  const [debugRequest, setDebugRequest] = React.useState<Record<string, unknown> | null>(null);
  const [debugResponse, setDebugResponse] = React.useState<Record<string, unknown> | null>(null);
  const [debugStatus, setDebugStatus] = React.useState<number | null>(null);
  const [debugDuration, setDebugDuration] = React.useState<number | null>(null);
  const [progress, setProgress] = React.useState<RecommendProgressEvent | null>(null);
  const [progressHistory, setProgressHistory] = React.useState<RecommendProgressEvent[]>([]);
  const [frontier, setFrontier] = React.useState<FrontierState>({ status: 'idle' });

  const abortRef = React.useRef<AbortController | null>(null);
  const frontierAbortRef = React.useRef<AbortController | null>(null);
  const timerRef = React.useRef<ReturnType<typeof setInterval> | null>(null);

  const clearTimer = React.useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  React.useEffect(() => {
    if (!isLoading) { clearTimer(); return; }
    setElapsed(0);
    timerRef.current = setInterval(() => setElapsed(e => e + 1), 1000);
    return clearTimer;
  }, [isLoading, clearTimer]);

  const launchFrontier = React.useCallback(
    (body: Record<string, unknown>, primaryGpus: number, simulator: string | undefined) => {
      if (simulator === 'blis') {
        setFrontier({
          status: 'unavailable',
          reason: 'Pareto frontiers are computed by AISimulate only. Select AISimulate in Settings to see one.',
        });
        return;
      }
      if (frontierAbortRef.current) frontierAbortRef.current.abort();
      const controller = new AbortController();
      frontierAbortRef.current = controller;
      setFrontier({ status: 'loading' });
      const window = frontierWindow(primaryGpus);
      void fetchFrontier(
        {
          ...body,
          simulator: 'aisimulate',
          min_candidate_gpus: window.minGpus,
          max_candidate_gpus: window.maxGpus,
        },
        controller.signal,
      )
        .then(({ candidates, views, sweep }) => {
          if (controller.signal.aborted) return;
          setFrontier({ status: 'ready', candidates, views, sweep });
        })
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          setFrontier({
            status: 'error',
            message: err instanceof Error ? err.message : 'Pareto sweep failed',
          });
        });
    },
    [],
  );

  const startSizing = React.useCallback((p: RecommendParams) => {
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    if (frontierAbortRef.current) frontierAbortRef.current.abort();
    frontierAbortRef.current = null;

    const requestBody = requestBodyFor(p);

    setParams(p);
    setIsLoading(true);
    setResult(null);
    setError(null);
    setErrorCode(null);
    setDebugRequest(requestBody);
    setDebugResponse(null);
    setDebugStatus(null);
    setDebugDuration(null);
    setProgress(null);
    setProgressHistory([]);
    setFrontier({ status: 'idle' });

    const t0 = performance.now();

    const run = async () => {
      try {
        const res = await fetch('/api/recommend', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });
        if (!controller.signal.aborted) setDebugStatus(res.status);
        if (!res.ok) {
          const data = await res.json().catch(() => null) as Record<string, unknown> | null;
          const error = data?.error as Record<string, unknown> | undefined;
          if (!controller.signal.aborted) {
            setDebugResponse(data ?? { status: res.status });
            setError(typeof error?.message === 'string' ? error.message : `Recommendation request failed (${res.status})`);
            setErrorCode(typeof error?.code === 'string' ? error.code : 'NETWORK_ERROR');
            setDebugDuration(Math.round(performance.now() - t0));
          }
          return;
        }
        const response = await readRecommendStream(res, event => {
          setProgress(event);
          setProgressHistory(history => [...history, event]);
        });
        setDebugResponse(response as unknown as Record<string, unknown>);
        setDebugDuration(Math.round(performance.now() - t0));
        if (response.status === 'failed') {
          setError(response.error.message);
          setErrorCode(response.error.code);
          return;
        }
        // Publish the primary result first, then start the companion sweep, so
        // the sizing numbers are readable while the frontier is still running.
        setResult(response);
        launchFrontier(requestBody, response.recommendation.gpusNeeded, p.simulator);
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        setDebugDuration(Math.round(performance.now() - t0));
        setError(err instanceof Error ? err.message : 'Network error');
        setErrorCode('NETWORK_ERROR');
      } finally {
        if (!controller.signal.aborted) setIsLoading(false);
      }
    };
    void run();
  }, [launchFrontier]);

  const reset = React.useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    if (frontierAbortRef.current) frontierAbortRef.current.abort();
    frontierAbortRef.current = null;
    setIsLoading(false);
    setResult(null);
    setError(null);
    setErrorCode(null);
    setElapsed(0);
    setParams(null);
    setDebugRequest(null);
    setDebugResponse(null);
    setDebugStatus(null);
    setDebugDuration(null);
    setProgress(null);
    setProgressHistory([]);
    setFrontier({ status: 'idle' });
  }, []);

  const value = React.useMemo<RecommendState>(
    () => ({ params, isLoading, result, error, errorCode, elapsed, debugRequest, debugResponse, debugStatus, debugDuration, progress, progressHistory, frontier, startSizing, reset }),
    [params, isLoading, result, error, errorCode, elapsed, debugRequest, debugResponse, debugStatus, debugDuration, progress, progressHistory, frontier, startSizing, reset]
  );

  return (
    <RecommendContext.Provider value={value}>
      {children}
    </RecommendContext.Provider>
  );
}

export function useRecommend(): RecommendState {
  return React.useContext(RecommendContext);
}

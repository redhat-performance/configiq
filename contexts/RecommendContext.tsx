'use client';

import * as React from 'react';
import type { RecommendProgressEvent, RecommendResult } from '@/lib/api/recommend';
import { readRecommendStream } from '@/lib/api/recommend-stream';

interface RecommendParams {
  model_path: string;
  system: string;
  isl: number;
  osl: number;
  max_seq_len?: number;
  prefill_max_seq_len?: number;
  decode_max_seq_len?: number;
  ttft: number;
  tpot?: number;
  backend?: string;
  target_concurrency?: number;
  target_request_rate?: number;
  request_latency?: number;
  prefix?: number;
  model_config?: Record<string, unknown> | null;
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
  startSizing: () => {},
  reset: () => {},
});

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

  const abortRef = React.useRef<AbortController | null>(null);
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

  const startSizing = React.useCallback((p: RecommendParams) => {
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const requestBody: Record<string, unknown> = {
      model_path: p.model_path,
      system: p.system,
      isl: p.isl,
      osl: p.osl,
      ttft: p.ttft,
      tpot: p.tpot ?? 30,
      target_concurrency: p.target_concurrency ?? 32,
    };
    if (p.max_seq_len != null) requestBody.max_seq_len = p.max_seq_len;
    if (p.prefill_max_seq_len != null) requestBody.prefill_max_seq_len = p.prefill_max_seq_len;
    if (p.decode_max_seq_len != null) requestBody.decode_max_seq_len = p.decode_max_seq_len;
    if (p.backend) requestBody.backend = p.backend;
    if (p.target_request_rate != null) {
      delete requestBody.target_concurrency;
      requestBody.target_request_rate = p.target_request_rate;
    }
    if (p.request_latency != null) requestBody.request_latency = p.request_latency;
    if (p.prefix != null && p.prefix > 0) requestBody.prefix = p.prefix;
    if (p.model_config != null) requestBody.model_config = p.model_config;

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
        } else {
          setResult(response);
        }
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
  }, []);

  const reset = React.useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
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
  }, []);

  const value = React.useMemo<RecommendState>(
    () => ({ params, isLoading, result, error, errorCode, elapsed, debugRequest, debugResponse, debugStatus, debugDuration, progress, progressHistory, startSizing, reset }),
    [params, isLoading, result, error, errorCode, elapsed, debugRequest, debugResponse, debugStatus, debugDuration, progress, progressHistory, startSizing, reset]
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

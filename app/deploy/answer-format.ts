import type { RecommendResult } from '@/lib/api/recommend'

export type LatencyVerdict = 'meets' | 'over'

/**
 * Display-only comparison of a predicted mean against the user's target. The
 * run itself is unbounded (`sla_mode: 'none'`), so this is a verdict on the
 * answer, never an enforced constraint.
 */
export function latencyVerdict(predicted: number | null, target: number): LatencyVerdict | null {
  if (predicted == null || !Number.isFinite(predicted)) return null
  return predicted <= target ? 'meets' : 'over'
}

/**
 * The load the response actually served, verbatim — never the user's input
 * echoed back. Capacity runs report concurrency; demand runs report the
 * achieved request rate, which can fall short of the target.
 */
export function servedLoadLabel(
  result: RecommendResult,
  loadUnit: 'users' | 'rps',
): { value: number; unit: string } | null {
  if (loadUnit === 'users') {
    const concurrency = result.performance.concurrency
    return Number.isFinite(concurrency) ? { value: concurrency, unit: 'users' } : null
  }
  const achieved = result.demand?.achievedRps
  return achieved != null && Number.isFinite(achieved) ? { value: achieved, unit: 'req/s' } : null
}

/** One-line deployment shape for the details disclosure. */
export function topologySummary(result: RecommendResult): string {
  const { replicasNeeded, tensorParallelSize } = result.recommendation
  if (result.mode === 'disagg') {
    return `disaggregated prefill/decode · ${replicasNeeded} × TP ${tensorParallelSize}`
  }
  const replicas = replicasNeeded === 1 ? '1 replica' : `${replicasNeeded} replicas`
  return `${replicas} × TP ${tensorParallelSize}`
}

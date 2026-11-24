import type { Simulator } from '@/contexts/SettingsContext'

export interface HybridSizingIntent {
  model: string
  backend: string
  simulator: Simulator
  /** Search shape the selected simulator reports; a proxy hint only. */
  searchStrategy?: 'windowed' | 'single'
  backendVersion: string
  averageInputTokens: number
  averageOutputTokens: number
  targetTtftMs: number
  targetTpotMs: number
  targetConcurrency: number
  prefixTokens: number
  modelConfig?: Record<string, unknown>
}

/** Convert the validated workload to the actual `/api/recommend` payload. */
export function buildHybridSizingRequest(input: HybridSizingIntent, systemId: string) {
  return {
    model_path: input.model,
    system: systemId,
    backend: input.backend,
    simulator: input.simulator,
    search_strategy: input.searchStrategy ?? 'windowed',
    ...(input.backendVersion ? { backend_version: input.backendVersion } : {}),
    isl: input.averageInputTokens,
    osl: input.averageOutputTokens,
    ttft: input.targetTtftMs,
    tpot: input.targetTpotMs,
    // Reference load measures capacity; the cost engine scales monthly demand.
    target_concurrency: input.targetConcurrency,
    prefix: input.prefixTokens,
    top_n: 5,
    ...(input.modelConfig ? { model_config: input.modelConfig } : {}),
  }
}

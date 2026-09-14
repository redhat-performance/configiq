'use client'

import { useState, useEffect } from 'react'
import { DEFAULT_AIC_TIMEOUT_SECONDS } from '@/lib/api/timeout'

export interface GpuOption {
  systemId: string
  label: string
  vendor: string | null
  architecture: string | null
  vramGb: number | null
  bandwidthTbps: number | null
  tflopsBf16: number | null
  tdpWatts: number | null
  gpusPerNode: number | null
  /** Fraction of GPU memory available for the model + KV cache.
   *  TODO: replace with per-backend value from AIC /memory response. */
  gpuMemoryUtilization: number
}

/**
 * Per-model metadata returned by GET /models?include=specs.
 *
 * MoE detection:  num_experts != null && num_experts > 1
 * Vision model:   architecture?.endsWith('ForConditionalGeneration')
 */
export interface ModelSpec {
  id: string
  /** Total number of experts (null for dense models). */
  num_experts: number | null
  /** Experts activated per token — top-k (null for dense models). */
  num_experts_per_tok: number | null
  /** Maximum context window in tokens. */
  context_length: number | null
  /** Number of query attention heads. */
  num_attn_heads: number | null
  /** Number of KV heads (< num_attn_heads when GQA is used). */
  num_kv_heads: number | null
  /** Raw HuggingFace architecture class name. */
  architecture: string | null
}

export interface AicCatalog {
  gpuOptions: GpuOption[]
  modelOptions: string[]
  modelSpecs: Map<string, ModelSpec>
  /** Effective AIC request timeout (seconds) reported by the server; falls back
   *  to DEFAULT_AIC_TIMEOUT_SECONDS until the catalog resolves. */
  timeoutSeconds: number
  isLoading: boolean
  error: string | null
}

const GB = 1_073_741_824

function mapSystem(s: Record<string, unknown>): GpuOption {
  const memBytes = typeof s.memory_bytes === 'number' ? s.memory_bytes : 0
  const bwBytes = typeof s.memory_bandwidth_bytes === 'number' ? s.memory_bandwidth_bytes : 0
  return {
    systemId: typeof s.id === 'string' ? s.id : '',
    label: typeof s.name === 'string' ? s.name : (typeof s.id === 'string' ? s.id : ''),
    vendor: typeof s.vendor === 'string' ? s.vendor : null,
    architecture: typeof s.architecture === 'string' ? s.architecture : null,
    vramGb: memBytes > 0 ? Math.round(memBytes / GB) : null,
    bandwidthTbps: bwBytes > 0 ? bwBytes / 1e12 : null,
    tflopsBf16: typeof s.bf16_tflops === 'number' && s.bf16_tflops > 0 ? s.bf16_tflops : null,
    tdpWatts: typeof s.tdp_watts === 'number' ? s.tdp_watts : null,
    gpusPerNode: typeof s.gpus_per_node === 'number' ? s.gpus_per_node : null,
    gpuMemoryUtilization: typeof s.gpu_memory_utilization === 'number' ? s.gpu_memory_utilization : 0.9,
  }
}

// Module-level cache — shared across all components, survives re-renders
// TTL of 10 minutes so new models appear without a hard refresh
const CACHE_TTL_MS = 10 * 60 * 1000
let cachedGpus: GpuOption[] | null = null
let cachedModels: string[] | null = null
let cachedModelSpecs: Map<string, ModelSpec> | null = null
let cachedTimeoutSeconds: number = DEFAULT_AIC_TIMEOUT_SECONDS
let cacheTimestamp: number | null = null
let fetchPromise: Promise<void> | null = null

function isCacheValid(): boolean {
  return cachedGpus !== null && cachedModels !== null && cachedModelSpecs !== null &&
    cacheTimestamp !== null && Date.now() - cacheTimestamp < CACHE_TTL_MS
}

export function useAicCatalog(): AicCatalog {
  const [gpuOptions, setGpuOptions] = useState<GpuOption[]>(isCacheValid() ? cachedGpus! : [])
  const [modelOptions, setModelOptions] = useState<string[]>(isCacheValid() ? cachedModels! : [])
  const [modelSpecs, setModelSpecs] = useState<Map<string, ModelSpec>>(isCacheValid() ? cachedModelSpecs! : new Map())
  const [timeoutSeconds, setTimeoutSeconds] = useState<number>(cachedTimeoutSeconds)
  const [isLoading, setIsLoading] = useState(!isCacheValid())
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (isCacheValid()) {
      setGpuOptions(cachedGpus!)
      setModelOptions(cachedModels!)
      setModelSpecs(cachedModelSpecs!)
      setTimeoutSeconds(cachedTimeoutSeconds)
      setIsLoading(false)
      return
    }

    fetchPromise = null  // reset so stale cache triggers a fresh fetch

    let cancelled = false

    async function fetchCatalog() {
      if (!fetchPromise) {
        fetchPromise = (async () => {
          // Same-origin proxy (app/api/catalog) resolves the per-host gateway
          // server-side, so no build-time NEXT_PUBLIC_AICONFIGURATOR_API_URL.
          const res = await fetch('/api/catalog')
          if (!res.ok) throw new Error(`Catalog fetch failed (${res.status})`)

          const data = await res.json()

          const systems = (data.systems ?? []) as Record<string, unknown>[]
          const gpus = systems.map(mapSystem)

          const rawModels = (data.models ?? []) as unknown[]
          const modelList: string[] = []
          const specsMap = new Map<string, ModelSpec>()
          for (const m of rawModels) {
            if (typeof m === 'string') {
              modelList.push(m)
            } else if (m && typeof m === 'object' && typeof (m as Record<string, unknown>).id === 'string') {
              const spec = m as ModelSpec
              modelList.push(spec.id)
              specsMap.set(spec.id, spec)
            }
          }

          if (gpus.length === 0 || modelList.length === 0) {
            throw new Error('AIC returned empty catalog')
          }

          cachedGpus = gpus
          cachedModels = modelList
          cachedModelSpecs = specsMap
          if (typeof data.timeoutSeconds === 'number' && data.timeoutSeconds > 0) {
            cachedTimeoutSeconds = data.timeoutSeconds
          }
          cacheTimestamp = Date.now()
        })()
      }

      try {
        await fetchPromise
        if (!cancelled) {
          setGpuOptions(cachedGpus!)
          setModelOptions(cachedModels!)
          setModelSpecs(cachedModelSpecs!)
          setTimeoutSeconds(cachedTimeoutSeconds)
        }
      } catch (err) {
        fetchPromise = null
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to fetch catalog')
        }
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    fetchCatalog()
    return () => { cancelled = true }
  }, [])

  return { gpuOptions, modelOptions, modelSpecs, timeoutSeconds, isLoading, error }
}

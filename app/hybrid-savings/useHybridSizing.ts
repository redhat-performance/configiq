'use client'

import * as React from 'react'
import type { RecommendResult } from '@/lib/api/recommend'
import { readRecommendStream } from '@/lib/api/recommend-stream'
import type { GpuOption } from '@/lib/hooks/useCatalog'
import type { InfrastructureCandidate } from '@/lib/hybrid-savings/calc'
import { resolveRentedCloudOffers, type RentedCloudOffer } from '@/lib/hybrid-savings/cloud-offer-catalog'
import { resolveCompatibleServerPurchaseConfigurations, type ServerPurchaseConfiguration } from '@/lib/hybrid-savings/purchase-catalog'

function candidateFromResult(
  result: RecommendResult,
  gpu: GpuOption,
  cloudOffer: RentedCloudOffer | null,
  purchaseConfiguration: ServerPurchaseConfiguration | null,
): InfrastructureCandidate {
  return {
    configurationId: cloudOffer?.id ?? (purchaseConfiguration ? `${gpu.systemId}:server:${purchaseConfiguration.gpuCount}` : undefined),
    systemId: gpu.systemId,
    label: gpu.label,
    gpusPerReplica: result.recommendation.gpusPerReplica,
    replicasNeeded: result.recommendation.replicasNeeded,
    clusterOutputTokensPerSecond: result.throughput.tokensPerSecond,
    cloudHourlyCostPerInstance: cloudOffer?.hourlyCost ?? null,
    cloudDirectInfrastructureMonthly: cloudOffer?.directInfrastructureMonthly ?? null,
    cloudRatePerGpuHour: cloudOffer
      ? cloudOffer.hourlyCost / cloudOffer.gpuCount
      : null,
    cloudGpusPerInstance: cloudOffer?.gpuCount ?? null,
    cloudInstanceName: cloudOffer?.instanceName ?? null,
    cloudProvider: cloudOffer?.provider ?? null,
    cloudProviderRegion: cloudOffer?.providerRegion ?? null,
    cloudRateKind: cloudOffer?.rateKind ?? null,
    cloudMaxInstancesPerReplica: cloudOffer?.maxInstancesPerReplica ?? null,
    cloudPriceSource: cloudOffer?.sourceLabel ?? null,
    cloudPriceSourceUrl: cloudOffer?.sourceUrl ?? null,
    cloudPriceSourceDate: cloudOffer?.sourceDate ?? null,
    purchaseGpusPerServer: purchaseConfiguration?.gpuCount ?? null,
    purchasePricePerReplica: purchaseConfiguration?.purchasePrice ?? null,
    purchaseInstallationPerReplica: purchaseConfiguration?.installationCost ?? null,
    purchasePriceIndicative: purchaseConfiguration?.indicative ?? true,
    purchasePriceSource: purchaseConfiguration?.sourceLabel ?? null,
    purchasePriceSourceUrl: purchaseConfiguration?.sourceUrl ?? null,
    purchasePriceSourceDate: purchaseConfiguration?.sourceDate ?? null,
    tdpWattsPerGpu: gpu.tdpWatts,
  }
}

interface SizingInput {
  signature: string
  model: string
  backend: string
  averageInputTokens: number
  averageOutputTokens: number
  targetTtftMs: number
  targetTpotMs: number
  targetConcurrency: number
  prefixTokens: number
  gpuOptions: GpuOption[]
  preferredCloudProvider: string | null
  modelConfig?: Record<string, unknown>
}

/** Four concurrent live searches; only the current, uncancelled run may publish results. */
export function useHybridSizing() {
  const [candidates, setCandidates] = React.useState<InfrastructureCandidate[]>([])
  const [isSizing, setIsSizing] = React.useState(false)
  const [progress, setProgress] = React.useState({ completed: 0, total: 0 })
  const [sizingError, setSizingError] = React.useState<string | null>(null)
  const [failedSystems, setFailedSystems] = React.useState(0)
  const [lastSizingSignature, setLastSizingSignature] = React.useState<string | null>(null)
  const abortRef = React.useRef<AbortController | null>(null)
  const runIdRef = React.useRef(0)

  React.useEffect(() => () => {
    runIdRef.current += 1
    abortRef.current?.abort()
  }, [])

  const cancel = React.useCallback(() => {
    runIdRef.current += 1
    abortRef.current?.abort()
    setIsSizing(false)
  }, [])

  const runSizing = async (input: SizingInput) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    const runId = ++runIdRef.current
    const current = () => !controller.signal.aborted && runIdRef.current === runId
    const results: InfrastructureCandidate[] = []
    let failures = 0
    let nextIndex = 0

    setIsSizing(true)
    setSizingError(null)
    setCandidates([])
    setLastSizingSignature(null)
    setFailedSystems(0)
    setProgress({ completed: 0, total: input.gpuOptions.length })

    const worker = async () => {
      while (nextIndex < input.gpuOptions.length && current()) {
        const gpu = input.gpuOptions[nextIndex++]
        try {
          const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
            signal: controller.signal,
            body: JSON.stringify({
              model_path: input.model,
              system: gpu.systemId,
              backend: input.backend,
              isl: Math.max(Math.round(input.averageInputTokens), 1),
              osl: Math.max(Math.round(input.averageOutputTokens), 1),
              ttft: Math.max(input.targetTtftMs, 1),
              tpot: Math.max(input.targetTpotMs, 1),
              // Preset reference load measures capacity; monthly demand is scaled by the cost engine.
              target_concurrency: input.targetConcurrency,
              prefix: input.prefixTokens,
              top_n: 5,
              ...(input.modelConfig ? { model_config: input.modelConfig } : {}),
            }),
          })
          const data = await readRecommendStream(response)
          if (!current()) return
          if (data.status !== 'completed') {
            failures += 1
            continue
          }
          const cloudOffers = resolveRentedCloudOffers(gpu.systemId, input.preferredCloudProvider)
          const purchases = resolveCompatibleServerPurchaseConfigurations(gpu.systemId, data.recommendation.gpusPerReplica)
          results.push(...cloudOffers.map(offer => candidateFromResult(data, gpu, offer, null)))
          results.push(...purchases.map(server => candidateFromResult(data, gpu, null, server)))
          // Retain unpriced successes for honest sizing/coverage diagnostics.
          if (cloudOffers.length === 0 && purchases.length === 0) {
            results.push(candidateFromResult(data, gpu, null, null))
          }
        } catch {
          if (!current()) return
          failures += 1
        } finally {
          if (current()) setProgress(previous => ({ ...previous, completed: previous.completed + 1 }))
        }
      }
    }

    await Promise.all(Array.from({ length: Math.min(4, input.gpuOptions.length) }, worker))
    if (!current()) return
    setCandidates(results)
    setFailedSystems(failures)
    setLastSizingSignature(input.signature)
    setIsSizing(false)
    if (results.length === 0) {
      setSizingError('AISimulators could not find a configuration that meets this model and workload target.')
    }
  }

  return { candidates, isSizing, progress, sizingError, failedSystems, lastSizingSignature, runSizing, cancel }
}

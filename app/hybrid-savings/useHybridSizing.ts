'use client'

import * as React from 'react'
import type { RecommendResult } from '@/lib/api/recommend'
import { readRecommendStream } from '@/lib/api/recommend-stream'
import type { GpuOption } from '@/lib/hooks/useCatalog'
import type { InfrastructureCandidate } from '@/lib/hybrid-savings/calc'
import { resolveRentedCloudOffers, type RentedCloudOffer } from '@/lib/hybrid-savings/cloud-offer-catalog'
import { resolveCompatibleServerPurchaseConfigurations, type ServerPurchaseConfiguration } from '@/lib/hybrid-savings/purchase-catalog'
import type { InfrastructureOffers } from '@/lib/api/infrastructure-offers'
import type { Simulator } from '@/contexts/SettingsContext'

function candidateFromResult(
  result: RecommendResult,
  gpu: GpuOption,
  cloudOffer: RentedCloudOffer | null,
  purchaseConfiguration: ServerPurchaseConfiguration | null,
): InfrastructureCandidate {
  return {
    configurationId: cloudOffer?.id ?? purchaseConfiguration?.id,
    systemId: gpu.systemId,
    label: gpu.label,
    gpusPerReplica: result.recommendation.gpusPerReplica,
    replicasNeeded: result.recommendation.replicasNeeded,
    clusterOutputTokensPerSecond: result.throughput.tokensPerSecond,
    cloudHourlyCostPerInstance: cloudOffer?.hourlyCost ?? null,
    cloudRatePerGpuHour: cloudOffer
      ? cloudOffer.hourlyCost / cloudOffer.gpuCount
      : null,
    cloudGpusPerInstance: cloudOffer?.gpuCount ?? null,
    cloudInstanceName: cloudOffer?.instanceName ?? null,
    cloudProvider: cloudOffer?.provider ?? null,
    cloudProviderRegion: cloudOffer?.providerRegion ?? null,
    cloudRateKind: cloudOffer?.rateKind ?? null,
    cloudMaxInstancesPerReplica: cloudOffer?.maxInstancesPerReplica ?? null,
    cloudInterconnect: cloudOffer?.interconnect ?? null,
    cloudPriceSource: cloudOffer?.sourceLabel ?? null,
    cloudPriceSourceUrl: cloudOffer?.sourceUrl ?? null,
    cloudPriceSourceDate: cloudOffer?.sourceDate ?? null,
    cloudPriceCollectionKind: cloudOffer?.collectionKind,
    cloudPriceRetrievedAt: cloudOffer?.retrievedAt,
    cloudPriceStale: cloudOffer?.stale,
    purchaseGpusPerServer: purchaseConfiguration?.gpuCount ?? null,
    purchasePricePerReplica: purchaseConfiguration?.purchasePrice ?? null,
    purchaseInstallationPerReplica: purchaseConfiguration?.installationCost ?? null,
    purchasePriceIndicative: purchaseConfiguration?.indicative ?? true,
    purchasePriceSource: purchaseConfiguration?.sourceLabel ?? null,
    purchasePriceSourceUrl: purchaseConfiguration?.sourceUrl ?? null,
    purchasePriceSourceDate: purchaseConfiguration?.sourceDate ?? null,
    purchasePriceCollectionKind: purchaseConfiguration?.collectionKind,
    purchasePriceRetrievedAt: purchaseConfiguration?.retrievedAt,
    purchasePriceStale: purchaseConfiguration?.stale,
    tdpWattsPerGpu: gpu.tdpWatts,
  }
}

interface SizingInput {
  signature: string
  model: string
  backend: string
  simulator: Simulator
  backendVersion: string
  averageInputTokens: number
  averageOutputTokens: number
  targetTtftMs: number
  targetTpotMs: number
  targetConcurrency: number
  prefixTokens: number
  gpuOptions: GpuOption[]
  modelConfig?: Record<string, unknown>
}

/** Four concurrent live searches; only the current, uncancelled run may publish results. */
export function useHybridSizing(offers: InfrastructureOffers, preferredCloudProvider: string | null) {
  const [sized, setSized] = React.useState<{ result: RecommendResult; gpu: GpuOption }[]>([])
  // Pricing can refresh without re-running simulation or mixing model results.
  const candidates = React.useMemo(() => sized.flatMap(({ result, gpu }) => {
    const cloud = resolveRentedCloudOffers(offers.rented, gpu.systemId, preferredCloudProvider)
    const purchases = resolveCompatibleServerPurchaseConfigurations(offers.purchased, gpu.systemId, result.recommendation.gpusPerReplica)
    const priced = [...cloud.map(offer => candidateFromResult(result, gpu, offer, null)),
      ...purchases.map(server => candidateFromResult(result, gpu, null, server))]
    return priced.length ? priced : [candidateFromResult(result, gpu, null, null)]
  }), [sized, offers.rented, offers.purchased, preferredCloudProvider])
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
    const results: { result: RecommendResult; gpu: GpuOption }[] = []
    let failures = 0
    let nextIndex = 0

    setIsSizing(true)
    setSizingError(null)
    setSized([])
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
              simulator: input.simulator,
              ...(input.backendVersion ? { backend_version: input.backendVersion } : {}),
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
          if (data.status !== 'completed' || data.simulator !== input.simulator) {
            failures += 1
            continue
          }
          results.push({ result: data, gpu })
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
    setSized(results)
    setFailedSystems(failures)
    setLastSizingSignature(input.signature)
    setIsSizing(false)
    if (results.length === 0) {
      setSizingError('AISimulators could not find a configuration that meets this model and workload target.')
    }
  }

  const resultSimulator = sized[0]?.result.simulator ?? null
  return { candidates, resultSimulator, isSizing, progress, sizingError, failedSystems, lastSizingSignature, runSizing, cancel }
}

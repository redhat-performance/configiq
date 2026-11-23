'use client'

import * as React from 'react'
import type { RecommendResult } from '@/lib/api/recommend'
import { readRecommendStream } from '@/lib/api/recommend-stream'
import type { GpuOption } from '@/lib/hooks/useCatalog'
import type { InfrastructureCandidate } from '@/lib/hybrid-savings/calc'
import { resolveRentedCloudOffers, type RentedCloudOffer } from '@/lib/hybrid-savings/cloud-offer-catalog'
import { resolveCompatibleServerPurchaseConfigurations, type ServerPurchaseConfiguration } from '@/lib/hybrid-savings/purchase-catalog'
import type { InfrastructureOffers } from '@/lib/api/infrastructure-offers'
import { buildHybridSizingRequest, type HybridSizingIntent } from '@/lib/hybrid-savings/sizing-request'

function candidateFromResult(
  result: RecommendResult,
  gpu: GpuOption,
  cloudOffer: RentedCloudOffer | null,
  purchaseConfiguration: ServerPurchaseConfiguration | null,
  referenceTargetConcurrency: number,
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
    purchaseIncludes: purchaseConfiguration?.includes,
    tdpWattsPerGpu: gpu.tdpWatts,
    // Reference-performance evidence retained from the sizing result.
    referenceMode: result.mode,
    referenceTtftMs: result.performance.ttftLatencyMs,
    referenceTpotMs: result.performance.tpotMs,
    referenceTargetConcurrency,
    referenceReportedConcurrency: result.performance.concurrency > 0 ? result.performance.concurrency : null,
  }
}

interface SizingInput extends HybridSizingIntent {
  signature: string
  gpuOptions: GpuOption[]
}

interface SizedSystem {
  result: RecommendResult
  gpu: GpuOption
  targetConcurrency: number
}

interface SizingIssue {
  message: string
  systems: string[]
}

interface SizingFailure {
  summary: string
  issues: SizingIssue[]
  timedOutSystems: number
  canReviewTargets: boolean
}

const DEFAULT_PARALLEL_SEARCHES = 4

function sizingParallelism(): number {
  const configured = Number(process.env.NEXT_PUBLIC_HYBRID_SIZING_PARALLELISM)
  return Number.isInteger(configured) && configured >= 1 && configured <= DEFAULT_PARALLEL_SEARCHES
    ? configured
    : DEFAULT_PARALLEL_SEARCHES
}

function hasValidRecommendation(result: RecommendResult, input: SizingInput): boolean {
  return result.simulator === input.simulator &&
    [result.recommendation.gpusPerReplica, result.recommendation.replicasNeeded].every(value => Number.isSafeInteger(value) && value > 0) &&
    [result.throughput.tokensPerSecond, result.performance.ttftLatencyMs, result.performance.tpotMs].every(value => Number.isFinite(value) && value > 0)
}

function meetsTargets(result: RecommendResult, input: SizingInput): boolean {
  return result.performance.ttftLatencyMs <= input.targetTtftMs && result.performance.tpotMs <= input.targetTpotMs
}

/** Bound parallel searches without changing the request sent to each GPU system. */
export function useHybridSizing(offers: InfrastructureOffers, preferredCloudProvider: string | null) {
  const [sized, setSized] = React.useState<SizedSystem[]>([])
  // Pricing can refresh without re-running simulation or mixing model results.
  const candidates = React.useMemo(() => sized.flatMap(({ result, gpu, targetConcurrency }) => {
    const cloud = resolveRentedCloudOffers(offers.rented, gpu.systemId, preferredCloudProvider)
    const purchases = resolveCompatibleServerPurchaseConfigurations(offers.purchased, gpu.systemId, result.recommendation.gpusPerReplica)
    const fallback = result.aggregatedFallback
    const fallbackPurchases = fallback
      ? resolveCompatibleServerPurchaseConfigurations(offers.purchased, gpu.systemId, fallback.recommendation.gpusPerReplica)
      : []
    const priced = [...cloud.map(offer => candidateFromResult(result, gpu, offer, null, targetConcurrency)),
      ...purchases.map(server => candidateFromResult(result, gpu, null, server, targetConcurrency)),
      ...(fallback ? fallbackPurchases.filter(server => !purchases.some(primary => primary.id === server.id))
        .map(server => candidateFromResult(fallback, gpu, null, server, targetConcurrency)) : [])]
    return priced.length ? priced : [candidateFromResult(result, gpu, null, null, targetConcurrency)]
  }), [sized, offers.rented, offers.purchased, preferredCloudProvider])
  const [isSizing, setIsSizing] = React.useState(false)
  const [progress, setProgress] = React.useState({ completed: 0, total: 0 })
  const [sizingError, setSizingError] = React.useState<SizingFailure | null>(null)
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
    setLastSizingSignature(null)
    setSized([])
    setSizingError(null)
    setFailedSystems(0)
    setIsSizing(false)
  }, [])

  const runSizing = async (input: SizingInput) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    const runId = ++runIdRef.current
    const current = () => !controller.signal.aborted && runIdRef.current === runId
    const results: SizedSystem[] = []
    const reasons = new Map<string, string[]>()
    const fail = (gpu: GpuOption, reason: string, code?: string) => {
      failures += 1
      // Some gateways still report bounded search exhaustion as a 422 with
      // a message rather than a distinct code. It is not a demonstrated SLA miss.
      if (code === 'AISIM_TIMEOUT' || /\b(search reached its time budget|search budget exhausted)\b/i.test(reason)) {
        timeLimited += 1
      } else if (code === 'AISIM_NO_CONFIGURATION' || code === 'SLA_MISS') {
        noMatch += 1
      }
      const systems = reasons.get(reason) ?? []
      systems.push(gpu.label)
      reasons.set(reason, systems)
    }
    let failures = 0
    let timeLimited = 0
    let noMatch = 0
    let nextIndex = 0

    setIsSizing(true)
    setSizingError(null)
    setSized([])
    setLastSizingSignature(input.signature)
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
            body: JSON.stringify(buildHybridSizingRequest(input, gpu.systemId)),
          })
          const data = await readRecommendStream(response)
          if (!current()) return
          if (data.status !== 'completed') {
            fail(gpu, data.error.message, data.error.code)
            continue
          }
          if (!hasValidRecommendation(data, input)) {
            fail(gpu, 'The recommendation returned invalid performance data or a different simulator.')
            continue
          }
          if (!meetsTargets(data, input)) {
            fail(gpu, 'The returned configuration misses the selected latency targets.', 'SLA_MISS')
            continue
          }
          const fallback = data.aggregatedFallback
          const aggregatedFallback = data.mode === 'disagg' && fallback?.mode === 'agg' &&
            hasValidRecommendation(fallback, input) && meetsTargets(fallback, input) ? fallback : undefined
          results.push({ result: { ...data, aggregatedFallback }, gpu, targetConcurrency: input.targetConcurrency })
          // Publish verified options as searches finish. A slow system must
          // not hold back priced results from systems that already passed.
          setSized([...results])
        } catch (error) {
          if (!current()) return
          fail(gpu, error instanceof Error ? error.message : 'Simulator request failed.')
        } finally {
          if (current()) setProgress(previous => ({ ...previous, completed: previous.completed + 1 }))
        }
      }
    }

    await Promise.all(Array.from({ length: Math.min(sizingParallelism(), input.gpuOptions.length) }, worker))
    if (!current()) return
    setSized(results)
    setFailedSystems(failures)
    setLastSizingSignature(input.signature)
    setIsSizing(false)
    if (results.length === 0) {
      const issues = [...reasons.entries()].sort((a, b) => b[1].length - a[1].length)
        .map(([message, systems]) => ({ message, systems: systems.sort() }))
      const summary = timeLimited > 0
        ? `The search timed out on ${timeLimited} of ${input.gpuOptions.length} GPU systems. No configuration was verified for the selected targets.`
        : failures > 0 && noMatch === failures
          ? 'The search did not find a configuration that met the selected targets.'
          : 'The sizing service did not return a usable configuration. See details or retry.'
      setSizingError({ summary, issues, timedOutSystems: timeLimited, canReviewTargets: timeLimited + noMatch > 0 })
    }
  }

  const resultSimulator = sized[0]?.result.simulator ?? null
  return { candidates, resultSimulator, isSizing, progress, sizingError, failedSystems, lastSizingSignature, runSizing, cancel }
}

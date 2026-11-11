'use client'
import * as React from 'react'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { useCostAssumptions } from '@/contexts/CostAssumptionsContext'
import {
  Alert,
  AlertActionCloseButton,
  Button,
  Card,
  CardBody,
  ExpandableSection,
  ExpandableSectionToggle,
  Form,
  FormGroup,
  FormSelect,
  FormSelectOption,
  Progress,
  Spinner,
  TextInput,
  ToggleGroup,
  ToggleGroupItem,
} from '@patternfly/react-core'
import { ComboBox } from '@/components/ModelComboBox/ModelComboBox'
import { InfoStrip, InfoStripAction } from '@/components/ui/InfoStrip'
import { SimulatorBadge } from '@/components/ui/SimulatorBadge'
import { buildModelItems, needsHfConfig } from '@/lib/model-options'
import { fetchModelConfig } from '@/lib/huggingface/fetch-config'
import { HybridCostCard } from './HybridCostCard'
import { AppliedCostAssumptions } from './AppliedCostAssumptions'
import { useHybridSizing } from './useHybridSizing'
import { useSettings } from '@/contexts/SettingsContext'
import { useRememberDefaultModel } from '@/lib/hooks/useRememberedDefaults'
import { useCatalog } from '@/lib/hooks/useCatalog'
import { useInfrastructureOffers } from '@/lib/hooks/useInfrastructureOffers'
import { useTestedModels } from '@/lib/hooks/useTestedModels'
import { getAppConfig } from '@/lib/app-config'
import { DEFAULT_WORKLOAD, type WorkloadPreset } from '@/lib/workload-presets'
import { matchConcurrencyPresetKey, type ConcurrencyPreset } from '@/lib/concurrency-presets'
import {
  useCostings,
  type FrontierModel,
} from '@/lib/hooks/useCostings'
import {
  calculateHybridComparison,
  infrastructureCandidateKey,
  infrastructureOptionsAtVolume,
  HYBRID_PLANNING_HORIZON_TOKENS,
  workloadFacts,
  workloadOutputShare,
  type CloudBillingMode,
  type CostAssumptions,
  type CostLens,
  type CostOption,
  type HybridWorkload,
  type InfrastructurePath,
} from '@/lib/hybrid-savings/calc'
import {
  hasServerPurchaseConfiguration,
} from '@/lib/hybrid-savings/purchase-catalog'
import {
  hasRentedCloudOffer,
} from '@/lib/hybrid-savings/cloud-offer-catalog'
import { hostedPriceForWorkload, hostedTokenUsageCost, resolveHostedPricing } from '@/lib/hybrid-savings/hosted-pricing'
import { buildApiModelItems, hostedOfferLabel, hostedPricingSource, isStandardTokenOffer, SAME_API_MODEL } from '@/lib/hybrid-savings/api-models'
import { DEFAULT_HYBRID_WORKLOAD, analyzeRequestMix, matchRequestShapePreset, proposeRequestMixAlignment } from '@/lib/hybrid-savings/workload-basis'
import { captureFixedDeployment, fixedDeploymentSamples, type FixedDeploymentCapture } from '@/lib/hybrid-savings/fixed-deployment'
import { formatCostPerMillionOutputTokens } from '@/lib/hybrid-savings/token-economics'
import { normalizeModelId } from '@/lib/model-metadata'
import { formatLatencyBound } from '@/lib/utils/format'
import styles from './hybrid-savings.module.css'

const CostComparisonChart = dynamic(() => import('./CostComparisonChart'), {
  ssr: false,
  loading: () => (
    <div className={styles.chartLoading} role="status">
      <Spinner size="md" />
      <span>Preparing comparison chart…</span>
    </div>
  ),
})

const FixedDeploymentChart = dynamic(() => import('./FixedDeploymentChart'), {
  ssr: false,
  loading: () => (
    <div className={styles.chartLoading} role="status">
      <Spinner size="md" />
      <span>Preparing fixed-deployment chart…</span>
    </div>
  ),
})

const formatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 })
const tokenUsageFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 2,
})
const collectionDateFormatter = new Intl.DateTimeFormat('en-GB', {
  day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
  timeZone: 'UTC', timeZoneName: 'short',
})
function collectionDate(value: string | null | undefined, fallback = 'Not supplied'): string {
  if (!value) return fallback
  const date = new Date(value)
  return Number.isFinite(date.getTime()) ? collectionDateFormatter.format(date) : value
}
function collectionLabel(source: string): string {
  const labels: Record<string, string> = {
    'models.litellm': 'LiteLLM', 'models.openrouter': 'OpenRouter',
  }
  return labels[source] ?? source
}
function compactNumber(value: number): string {
  return new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 2,
  }).format(value)
}

function preciseRate(value: number): string {
  if (value >= 1) return formatter.format(value)
  if (value >= 0.01) return value.toFixed(3)
  return value.toPrecision(3)
}

function hostedOfferKey(model: FrontierModel): string {
  return `${model.provider.trim().toLowerCase()}::${model.id.trim().toLowerCase()}::${model.source ?? ''}`
}

function numberValue(raw: string, fallback = 0): number {
  const parsed = Number(raw)
  return Number.isFinite(parsed) ? parsed : fallback
}

function billingModeLabel(mode: CloudBillingMode): string {
  if (mode === 'scale-to-zero') return 'Scale to zero'
  if (mode === 'active-window') return 'Active window'
  return 'Always on'
}

function instanceLabel(provider: string | null | undefined, sku: string | null | undefined): string {
  return provider === 'aws' && sku && !sku.startsWith('EC2 ') ? `EC2 ${sku}` : sku ?? ''
}

function requestCountLabel(value: number | null): string {
  return value === null || !Number.isFinite(value) ? 'Not available' : `${formatter.format(value)}/month`
}

/** Compact per-path sizing evidence and whole-deployment layout for Calculation details. */
function deploymentRows(option: CostOption | undefined, path: InfrastructurePath): React.ReactNode {
  const candidate = option?.candidate
  const deployment = option?.deployment ?? null
  if (!candidate) {
    return <div><dt>Status</dt><dd>Available after sizing</dd></div>
  }
  const unit = path === 'rented' ? 'instance' : 'server'
  return <>
    <div><dt>Configuration</dt><dd>{candidate.label}</dd></div>
    <div><dt>Serving mode</dt><dd>{candidate.referenceMode === 'disagg' ? 'Disaggregated prefill / decode' : 'Aggregate'}</dd></div>
    <div><dt>Reference throughput</dt><dd>{formatter.format(candidate.clusterOutputTokensPerSecond)} output tokens/s</dd></div>
    <div><dt>Reference concurrency</dt><dd>{candidate.referenceConcurrency != null ? `${formatter.format(candidate.referenceConcurrency)} concurrent requests` : 'Not supplied'}</dd></div>
    <div><dt>Reference latency</dt><dd>{candidate.referenceTtftMs != null && candidate.referenceTpotMs != null ? `${formatter.format(candidate.referenceTtftMs)} ms TTFT · ${formatter.format(candidate.referenceTpotMs)} ms TPOT` : 'Not supplied'}</dd></div>
    {deployment ? <>
      <div><dt>Required replicas</dt><dd>{formatter.format(deployment.requiredReplicas)} · {formatter.format(deployment.requiredGpus)} GPUs</dd></div>
      <div><dt>Whole {unit}s</dt><dd>{formatter.format(deployment.wholeUnits)} · {formatter.format(deployment.availableReplicaSlots)} replica slots</dd></div>
      <div><dt>Planning capacity</dt><dd>{compactNumber(deployment.planningTokenCapacity)} tokens/month · {deployment.planningCapacityUsedPct === null ? 'usage not available' : `${deployment.planningCapacityUsedPct.toFixed(0)}% used`}</dd></div>
      <div><dt>Output capacity</dt><dd>{compactNumber(deployment.planningOutputCapacity)} output tokens/month</dd></div>
      {path === 'rented' && <>
        <div><dt>Effective billing mode</dt><dd>{deployment.effectiveBillingMode ? billingModeLabel(deployment.effectiveBillingMode) : 'Not supplied'}</dd></div>
        <div><dt>Billable instance-hours</dt><dd>{deployment.billableInstanceHours === null ? 'Not supplied' : `${formatter.format(deployment.billableInstanceHours)} / month`}</dd></div>
      </>}
    </> : <div><dt>Deployment</dt><dd>Capacity not available for this configuration</dd></div>}
  </>
}

function NumberField({
  id,
  label,
  value,
  onChange,
  min = 0,
  max,
  step,
  formatWithCommas = false,
}: {
  id: string
  label: string
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  formatWithCommas?: boolean
}) {
  const displayValue = formatWithCommas
    ? Math.round(value).toLocaleString('en-US')
    : value
  const [draft, setDraft] = React.useState<string | null>(null)

  const commit = () => {
    if (draft === null) return
    const normalized = formatWithCommas ? draft.replace(/,/g, '') : draft
    const parsed = Math.max(numberValue(normalized, min), min)
    onChange(max === undefined ? parsed : Math.min(parsed, max))
    setDraft(null)
  }

  return (
    <FormGroup label={label} fieldId={id}>
      <TextInput
        id={id}
        type={formatWithCommas ? 'text' : 'number'}
        inputMode={formatWithCommas ? 'numeric' : undefined}
        min={min}
        max={max}
        step={step}
        value={draft ?? displayValue}
        onFocus={() => setDraft(String(displayValue))}
        onChange={(_event, raw) => {
          if (formatWithCommas && !/^[\d,]*$/.test(raw)) return
          setDraft(raw)
        }}
        onBlur={commit}
        onKeyDown={event => {
          if (event.key === 'Enter') {
            event.preventDefault()
            event.currentTarget.blur()
          }
        }}
      />
    </FormGroup>
  )
}

export default function HybridSavings() {
  const {
    hydrated,
    defaultModel,
    inferenceBackend,
    simulator,
    backendVersion,
    preferredCloudProvider,
    pricingSource,
    hfToken,
  } = useSettings()
  const { validatedModelIds, isAvailable: testedModelsAvailable } = useTestedModels()
  const {
    gpuOptions: allGpuOptions,
    modelOptions: allModelOptions,
    simulatorSupport,
    backendOptions,
    modelSpecs,
    isLoading: catalogLoading,
    error: catalogError,
  } = useCatalog()
  const support = simulatorSupport[simulator]
  const modelOptions = support?.models ?? allModelOptions
  const selectedBackend = (support?.backends ?? backendOptions).find(backend => backend.id === inferenceBackend)
  const gpuOptions = React.useMemo(() => allGpuOptions.filter(gpu =>
    (!support || support.systems.includes(gpu.systemId)) &&
    (!backendVersion || (backendVersion === 'current' && selectedBackend?.defaultVersion === 'current'
      ? !!selectedBackend.systems[gpu.systemId]?.length
      : selectedBackend?.systems[gpu.systemId]?.includes(backendVersion)))),
  [allGpuOptions, support, selectedBackend, backendVersion])
  const backendUnavailable = !!support && !selectedBackend
  const costings = useCostings(hydrated, pricingSource, 'models')
  const infrastructure = useInfrastructureOffers(hydrated)
  const hostedSources = React.useMemo(
    () => Object.entries(costings.health?.sources ?? {})
      .filter(([source]) => pricingSource === 'merged' ? source.startsWith('models.') : source === `models.${pricingSource}`),
    [costings.health, pricingSource],
  )
  const staleCostingSources = hostedSources.filter(([, status]) => status.stale).map(([source]) => source)
  const infrastructureStates = Object.values(infrastructure.sources)
  const costingInputsStale = costings.modelsStale || staleCostingSources.length > 0 || infrastructure.stale ||
    infrastructureStates.some(source => source.state === 'stale') ||
    infrastructure.rented.some(offer => offer.stale) || infrastructure.purchased.some(offer => offer.stale)
  const freshnessSummary = costingInputsStale ? 'Stale · check Sources'
    : infrastructure.error || infrastructureStates.some(source => source.state === 'unavailable') ? 'Some sources unavailable · check Sources'
      : infrastructureStates.some(source => source.state === 'unconfigured') ? 'Some collectors unconfigured · curated estimates may apply'
        : infrastructureStates.length === 0 ? 'Infrastructure source status unavailable' : 'No stale feed reported'

  const [model, setModel] = React.useState('')
  // Only an explicit model pick should replace the saved default; the automatic
  // starting point must never overwrite it.
  const modelTouched = React.useRef(false)
  // A simulator that reports a complete model catalog cannot price a model
  // outside it. Show that explicitly rather than replacing the selection.
  const modelUnsupported = model !== '' && (support?.modelsExhaustive ?? false) && !modelOptions.includes(model)
  const [monthlyInputTokens, setMonthlyInputTokens] = React.useState(DEFAULT_HYBRID_WORKLOAD.monthlyInputTokens)
  const [monthlyOutputTokens, setMonthlyOutputTokens] = React.useState(DEFAULT_HYBRID_WORKLOAD.monthlyOutputTokens)
  const [averageInputTokens, setAverageInputTokens] = React.useState(DEFAULT_HYBRID_WORKLOAD.averageInputTokens)
  const [averageOutputTokens, setAverageOutputTokens] = React.useState(DEFAULT_HYBRID_WORKLOAD.averageOutputTokens)
  const [targetTtftMs, setTargetTtftMs] = React.useState(1000)
  const [targetTpotMs, setTargetTpotMs] = React.useState(30)
  const [targetConcurrency, setTargetConcurrency] = React.useState(1)
  // Selection follows the committed value, so typing a matching concurrency
  // selects the preset and changing it deselects without extra state.
  const activeConcurrencyPreset = matchConcurrencyPresetKey(getAppConfig().concurrencyPresets, targetConcurrency)
  const [prefixTokens, setPrefixTokens] = React.useState(0)
  const [activeHoursPerMonth, setActiveHoursPerMonth] = React.useState(730)
  const [peakToAverage, setPeakToAverage] = React.useState(1)
  const { assumptions: planningAssumptions, hydrated: assumptionsHydrated } = useCostAssumptions()
  const [costLens, setCostLens] = React.useState<CostLens>('fully-loaded')
  const [cloudBillingMode, setCloudBillingMode] = React.useState<CloudBillingMode>('scale-to-zero')
  const assumptions = React.useMemo<CostAssumptions>(() => ({
    ...planningAssumptions, costLens, cloudBillingMode,
  }), [planningAssumptions, costLens, cloudBillingMode])
  const [rentedSelection, setRentedSelection] = React.useState<string | null>(null)
  const [ownedSelection, setOwnedSelection] = React.useState<string | null>(null)
  const [selectionNotice, setSelectionNotice] = React.useState<string | null>(null)
  const [selectedHostedOfferKey, setSelectedHostedOfferKey] = React.useState<string | null>(null)
  const [apiModel, setApiModel] = React.useState(SAME_API_MODEL)
  const [workloadAdvancedOpen, setWorkloadAdvancedOpen] = React.useState(false)
  const [chartOpen, setChartOpen] = React.useState(false)
  const [chartView, setChartView] = React.useState<'workload' | 'fixed'>('workload')
  const [fixedCapture, setFixedCapture] = React.useState<FixedDeploymentCapture | null>(null)
  const [fixedNotice, setFixedNotice] = React.useState<string | null>(null)
  const [calculationDetailsOpen, setCalculationDetailsOpen] = React.useState(false)
  const { candidates, resultSimulator, isSizing, progress, sizingError, failedSystems, lastSizingSignature, runSizing: sizeConfigurations, cancel } = useHybridSizing(infrastructure, preferredCloudProvider)
  const [modelConfig, setModelConfig] = React.useState<{ model: string; config: Record<string, unknown> } | null>(null)
  const [configLoading, setConfigLoading] = React.useState(false)
  const [configError, setConfigError] = React.useState<string | null>(null)
  React.useEffect(() => {
    if (!hydrated || model || modelOptions.length === 0) return
    const configured = modelOptions.find(id => id === defaultModel)
    const qwenStartingPoint = modelOptions.find(id => normalizeModelId(id) === 'qwen/qwen3-8b')
    const firstPriced = modelOptions.find(id =>
      resolveHostedPricing(id, costings.models, 1, 1).selected !== null,
    )
    setModel(configured ?? qwenStartingPoint ?? firstPriced ?? modelOptions[0])
  }, [hydrated, model, modelOptions, defaultModel, costings.models])

  const modelItems = React.useMemo(() => hydrated ? buildModelItems(modelOptions, modelSpecs, validatedModelIds) : [], [modelOptions, modelSpecs, validatedModelIds, hydrated])
  React.useEffect(() => {
    setModelConfig(null)
    setConfigError(null)
    setConfigLoading(false)
    if (catalogLoading || !needsHfConfig(model, modelOptions)) return
    let cancelled = false
    const timer = setTimeout(async () => {
      setConfigLoading(true)
      try {
        const data = await fetchModelConfig(model, hfToken)
        if (cancelled) return
        if (data.success && data.config) setModelConfig({ model, config: data.config })
        else setConfigError(data.error ?? 'Could not load this model configuration.')
      } catch {
        if (!cancelled) setConfigError('Could not load the model configuration. Check your connection or Hugging Face credentials in Settings.')
      } finally {
        if (!cancelled) setConfigLoading(false)
      }
    }, 500)
    return () => { clearTimeout(timer); cancelled = true }
  }, [model, modelOptions, catalogLoading, hfToken])

  useRememberDefaultModel(
    model,
    modelTouched.current && !modelUnsupported &&
      (modelOptions.includes(model) || modelConfig?.model === model),
  )

  const apiModelItems = React.useMemo(() => buildApiModelItems(costings.models, model), [costings.models, model])
  const standardHostedOffers = React.useMemo(() => costings.models.filter(isStandardTokenOffer), [costings.models])
  const hostedComparisonModel = apiModel === SAME_API_MODEL ? model : apiModel
  const requestShape = React.useMemo(() => ({ averageInputTokens, averageOutputTokens }), [averageInputTokens, averageOutputTokens])
  const hostedPricing = React.useMemo(
    () => resolveHostedPricing(
      hostedComparisonModel,
      standardHostedOffers,
      monthlyInputTokens,
      monthlyOutputTokens,
      requestShape,
    ),
    [hostedComparisonModel, standardHostedOffers, monthlyInputTokens, monthlyOutputTokens, requestShape],
  )
  const hostedProviderOffers = hostedPricing.providerMatches
  const hostedModel = selectedHostedOfferKey === null ? hostedProviderOffers[0] ?? null
    : hostedProviderOffers.find(offer => hostedOfferKey(offer) === selectedHostedOfferKey) ?? null
  const hostedPrice = React.useMemo(
    () => {
      const price = hostedModel ? hostedPriceForWorkload(hostedModel, requestShape) : null
      return price && hostedModel ? { ...price, label: `${hostedOfferLabel(hostedModel)} hosted API` } : null
    },
    [hostedModel, requestShape],
  )
  React.useEffect(() => {
    setSelectedHostedOfferKey(null)
  }, [hostedComparisonModel, monthlyInputTokens, monthlyOutputTokens])
  const apiModelSelector = <ComboBox id="hybrid-api-model" label="API model"
    value={apiModel} items={apiModelItems} portalMenu
    onChange={value => { setApiModel(value || SAME_API_MODEL); setSelectedHostedOfferKey(null) }} />
  const workloadPresets = hydrated ? [DEFAULT_WORKLOAD, ...getAppConfig().workloadPresets] : [DEFAULT_WORKLOAD]
  const activePreset = matchRequestShapePreset({ averageInputTokens, averageOutputTokens }, workloadPresets)
  const workload = React.useMemo<HybridWorkload>(() => ({
    monthlyInputTokens,
    monthlyOutputTokens,
    averageInputTokens,
    averageOutputTokens,
    activeHoursPerMonth,
    peakToAverage,
  }), [
    monthlyInputTokens,
    monthlyOutputTokens,
    averageInputTokens,
    averageOutputTokens,
    activeHoursPerMonth,
    peakToAverage,
  ])
  const facts = React.useMemo(() => workloadFacts(workload), [workload])
  const requestMix = analyzeRequestMix(workload)
  const mixAlignment = proposeRequestMixAlignment(workload, prefixTokens)
  const outputShare = workloadOutputShare(workload)
  const hasOutput = facts.monthlyOutputTokens > 0
  const invalidWorkload =
    facts.monthlyTokens <= 0 ||
    facts.monthlyTokens > HYBRID_PLANNING_HORIZON_TOKENS ||
    ![monthlyInputTokens, monthlyOutputTokens, activeHoursPerMonth, peakToAverage, targetTtftMs, targetTpotMs].every(Number.isFinite) ||
    monthlyInputTokens < 0 || monthlyOutputTokens < 0 ||
    !Number.isSafeInteger(averageInputTokens) || averageInputTokens <= 0 ||
    !Number.isSafeInteger(averageOutputTokens) || averageOutputTokens <= 0 ||
    activeHoursPerMonth <= 0 ||
    activeHoursPerMonth > assumptions.hoursPerMonth ||
    peakToAverage < 1 ||
    targetTtftMs <= 0 ||
    targetTpotMs <= 0 ||
    targetConcurrency < 1 ||
    !Number.isSafeInteger(targetConcurrency) ||
    prefixTokens < 0 ||
    !Number.isSafeInteger(prefixTokens) ||
    prefixTokens > averageInputTokens

  // Only inputs that change simulator requests belong in this signature.
  // Demand-only changes (monthly volumes, processing hours, peak) rescale
  // existing validated results, and prices never invalidate them.
  const sizingSignature = React.useMemo(() => JSON.stringify({
    model,
    modelConfig: modelConfig?.model === model ? modelConfig.config : null,
    averageInputTokens,
    averageOutputTokens,
    targetTtftMs,
    targetTpotMs,
    targetConcurrency,
    prefixTokens,
    inferenceBackend,
    simulator,
    backendVersion,
    gpuSystems: gpuOptions.map(gpu => gpu.systemId),
  }), [
    model,
    modelConfig,
    averageInputTokens,
    averageOutputTokens,
    targetTtftMs,
    targetTpotMs,
    targetConcurrency,
    prefixTokens,
    inferenceBackend,
    simulator,
    backendVersion,
    gpuOptions,
  ])
  const isStale = lastSizingSignature !== null && lastSizingSignature !== sizingSignature
  const comparison = React.useMemo(() => {
    if (!hostedPrice || candidates.length === 0 || lastSizingSignature === null || isStale || invalidWorkload) return null
    return calculateHybridComparison(workload, hostedPrice, candidates, assumptions, { rented: rentedSelection, owned: ownedSelection })
  }, [workload, hostedPrice, candidates, assumptions, lastSizingSignature, isStale, invalidWorkload, rentedSelection, ownedSelection])
  const hostedOption = comparison?.options.find(option => option.key === 'hosted')
  const rentedOption = comparison?.options.find(option => option.key === 'rented')
  const ownedOption = comparison?.options.find(option => option.key === 'owned')

  const rentedOptions = React.useMemo(() => infrastructureOptionsAtVolume('rented', workload, candidates, assumptions, facts.monthlyTokens), [workload, candidates, assumptions, facts.monthlyTokens])
  const ownedOptions = React.useMemo(() => infrastructureOptionsAtVolume('owned', workload, candidates, assumptions, facts.monthlyTokens), [workload, candidates, assumptions, facts.monthlyTokens])
  React.useEffect(() => {
    if (lastSizingSignature !== null && !sizingError) setSelectionNotice(null)
  }, [lastSizingSignature, sizingError])
  React.useEffect(() => {
    // Do not erase a valid manual choice while a new sizing run is in progress.
    if (isSizing || isStale || infrastructure.isLoading || lastSizingSignature === null) return
    const reset: string[] = []
    if (rentedSelection && !rentedOptions.some(option => option.candidate && infrastructureCandidateKey(option.candidate, 'rented') === rentedSelection)) {
      setRentedSelection(null)
      reset.push('rented offer')
    }
    if (ownedSelection && !ownedOptions.some(option => option.candidate && infrastructureCandidateKey(option.candidate, 'owned') === ownedSelection)) {
      setOwnedSelection(null)
      reset.push('server configuration')
    }
    if (reset.length) setSelectionNotice(`The selected ${reset.join(' and ')} ${reset.length === 1 ? 'is' : 'are'} no longer eligible. Returned to automatic selection.`)
  }, [isSizing, isStale, infrastructure.isLoading, lastSizingSignature, rentedSelection, ownedSelection, rentedOptions, ownedOptions])

  // Fixed-deployment exploration. The basis signature covers every input the
  // captured physical deployment depends on; prices, cost view and financial
  // assumptions are deliberately excluded so they reprice the same hardware.
  const fixedBasisSignature = React.useMemo(() => JSON.stringify({
    monthlyInputTokens,
    monthlyOutputTokens,
    averageInputTokens,
    averageOutputTokens,
    targetTtftMs,
    targetTpotMs,
    targetConcurrency,
    prefixTokens,
    activeHoursPerMonth,
    peakToAverage,
    planningCapacityUsePct: planningAssumptions.planningCapacityUsePct,
    model,
    modelConfig: modelConfig?.model === model ? modelConfig.config : null,
    simulator,
    inferenceBackend,
    backendVersion,
    rentedSelection,
    ownedSelection,
  }), [
    monthlyInputTokens, monthlyOutputTokens, averageInputTokens, averageOutputTokens,
    targetTtftMs, targetTpotMs, targetConcurrency, prefixTokens, activeHoursPerMonth,
    peakToAverage, planningAssumptions.planningCapacityUsePct, model, modelConfig,
    simulator, inferenceBackend, backendVersion, rentedSelection, ownedSelection,
  ])
  const enterFixedView = (path?: InfrastructurePath) => {
    const chosen = path ?? (rentedOption?.deployment ? 'rented' : 'owned')
    const option = chosen === 'rented' ? rentedOption : ownedOption
    // Automatic selection is resolved to the configuration actually chosen.
    const capture = option ? captureFixedDeployment(chosen, option, fixedBasisSignature) : null
    setFixedCapture(capture)
    if (capture) {
      setChartView('fixed')
      setFixedNotice(null)
    } else {
      // Unreachable through the controls (the Fixed deployment option is
      // disabled without a freezable deployment); kept as a defensive guard
      // with an accurate cause.
      setChartView('workload')
      setFixedNotice('Fixed deployment needs an eligible hardware configuration with modeled capacity.')
    }
  }
  const fixedCandidate = React.useMemo(() => {
    if (!fixedCapture) return null
    return candidates.find(entry => infrastructureCandidateKey(entry, fixedCapture.path) === fixedCapture.candidateKey) ?? null
  }, [fixedCapture, candidates])
  const fixedSamples = React.useMemo(() => {
    if (chartView !== 'fixed' || !fixedCapture || !fixedCandidate || !hostedPrice) return []
    return fixedDeploymentSamples(workload, hostedPrice, assumptions, fixedCandidate, fixedCapture)
  }, [chartView, fixedCapture, fixedCandidate, hostedPrice, workload, assumptions])
  // Current usage of the FROZEN deployment, not of whatever the automatic
  // selection currently ranks cheapest: a reprice-only change can re-rank the
  // automatic choice while the capture stays valid.
  const fixedCurrentCapacityUsedPct = fixedCapture && fixedCapture.planningTokenCapacity > 0
    ? Math.min(facts.monthlyTokens / fixedCapture.planningTokenCapacity * 100, 100)
    : null
  React.useEffect(() => {
    if (chartView !== 'fixed') return
    // Stay while the captured configuration still resolves among the sized
    // candidates and the basis signature is unchanged. Reprice-only changes
    // (prices, cost view, financial assumptions) may re-rank the automatic
    // selection without invalidating the capture; a refreshed offer that
    // changes the physical layout or availability changes the candidate key,
    // so the capture stops resolving and the view exits.
    if (fixedCapture && fixedCapture.basisSignature === fixedBasisSignature && fixedCandidate) return
    setChartView('workload')
    setFixedCapture(null)
    setFixedNotice('The fixed configuration is no longer available. Showing the workload forecast.')
  }, [chartView, fixedCapture, fixedBasisSignature, fixedCandidate])

  const rentedReadyGpus = React.useMemo(() => gpuOptions.filter(gpu => (
    hasRentedCloudOffer(infrastructure.rented, gpu.systemId)
  )), [gpuOptions, infrastructure.rented])
  const purchaseReadyGpus = React.useMemo(() => gpuOptions.filter(gpu => (
    hasServerPurchaseConfiguration(infrastructure.purchased, gpu.systemId)
  )), [gpuOptions, infrastructure.purchased])
  const priceReadyGpus = React.useMemo(() => gpuOptions.filter(gpu => (
    rentedReadyGpus.some(candidate => candidate.systemId === gpu.systemId) ||
    purchaseReadyGpus.some(candidate => candidate.systemId === gpu.systemId)
  )), [gpuOptions, rentedReadyGpus, purchaseReadyGpus])
  const sizedSystemCount = React.useMemo(
    () => new Set(candidates.map(candidate => candidate.systemId)).size,
    [candidates],
  )

  const applyProfile = (profile: WorkloadPreset) => {
    setAverageInputTokens(profile.isl)
    setAverageOutputTokens(profile.osl)
    setWorkloadAdvancedOpen(true)
  }

  const applyConcurrencyPreset = (preset: ConcurrencyPreset) => {
    setTargetConcurrency(preset.concurrency)
  }

  const runSizing = () => {
    // Performance inputs are unchanged: reuse the validated results instead of
    // repeating simulator requests (demand-only edits already rescaled them).
    if (lastSizingSignature === sizingSignature && !sizingError && candidates.length > 0) {
      setSelectionNotice(null)
      return
    }
    sizeConfigurations({
      signature: sizingSignature, model, backend: inferenceBackend, simulator, backendVersion,
      searchStrategy: support?.recommendSearch ?? 'windowed',
      averageInputTokens, averageOutputTokens, targetTtftMs, targetTpotMs,
      targetConcurrency, prefixTokens,
      gpuOptions,
      ...(modelConfig?.model === model ? { modelConfig: modelConfig.config } : {}),
    })
  }

  const canCalculate = assumptionsHydrated && !backendUnavailable && !modelUnsupported && !invalidWorkload && !!hostedPrice && !!model && gpuOptions.length > 0 &&
    !catalogLoading && !costings.isLoading && !isSizing && !configLoading &&
    (!needsHfConfig(model, modelOptions) || modelConfig?.model === model)

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.pageTitle}>Hybrid savings</h1>
          <p className={styles.subtitle}>
            Compare hosted API, rented GPU and purchased hardware costs per million output tokens.
          </p>
        </div>
      </header>

      {(catalogError || costings.error) && (
        <Alert title="Some shared data could not be loaded" variant="warning" isInline>
          {catalogError ?? costings.error}. Refresh the page or check the Sources page before using the result.
        </Alert>
      )}
      {(infrastructure.error || infrastructure.rejectedRecords > 0) && <Alert title="Infrastructure pricing needs attention" variant="warning" isInline>
        {infrastructure.error ?? `${infrastructure.rejectedRecords} incomplete or duplicate pricing records were excluded.`}
        {infrastructure.stale && ' Showing the last successfully loaded snapshot; prices may be stale.'}
      </Alert>}
      {selectionNotice && <Alert title="Hardware selection updated" variant="info" isInline
        actionClose={<AlertActionCloseButton onClose={() => setSelectionNotice(null)} />}>{selectionNotice}</Alert>}
      {backendUnavailable && <Alert title="Selected backend unavailable" variant="warning" isInline>
        This simulator does not support the backend selected in Settings. Choose a supported backend there to calculate.
      </Alert>}
      {modelUnsupported && <Alert title="Model not supported by the selected simulator" variant="danger" isInline>
        {model} is not in the {support?.label ?? 'selected simulator'} model catalog, which lists every model it can price.
        Choose a catalog model or switch simulators in Settings.
      </Alert>}

      <Form onSubmit={event => { event.preventDefault(); if (canCalculate) void runSizing() }}>
        <Card className={styles.inputCard}>
          <CardBody>
            <h2>Model and workload</h2>
            <div className={styles.inputGrid}>
              <div className={styles.modelField}>
                <ComboBox id="hybrid-model" value={model} onChange={value => { modelTouched.current = true; setModel(value) }} items={modelItems}
                  allowCustom supportedModels={testedModelsAvailable ? validatedModelIds : undefined} hfToken={hfToken}
                  helperText="Choose a model checkpoint to compare." />
              </div>
              <NumberField id="hybrid-input-tokens" label="Monthly input tokens" value={monthlyInputTokens} max={HYBRID_PLANNING_HORIZON_TOKENS} step={1_000_000} formatWithCommas onChange={setMonthlyInputTokens} />
              <NumberField id="hybrid-output-tokens" label="Monthly output tokens" value={monthlyOutputTokens} max={HYBRID_PLANNING_HORIZON_TOKENS} step={1_000_000} formatWithCommas onChange={setMonthlyOutputTokens} />
            </div>
            <div className={styles.inputActions}>
              <div className={styles.workloadPresets} role="group" aria-label="Workload type" id="hybrid-profile">
                <span className={styles.workloadLabel}>Workload:</span>
                {workloadPresets.map(profile => (
                  <Button key={profile.key} type="button" className="gc-preset-button" variant={activePreset?.key === profile.key ? 'secondary' : 'tertiary'} size="sm" aria-pressed={activePreset?.key === profile.key} onClick={() => applyProfile(profile)}>
                    {profile.label}
                  </Button>
                ))}
              </div>
              {hydrated && getAppConfig().concurrencyPresets.length > 0 && (
                <div className={styles.concurrencyPresets} role="group" aria-label="Concurrency">
                  <span className={styles.workloadLabel}>Concurrency:</span>
                  {getAppConfig().concurrencyPresets.map(preset => (
                    <Button key={preset.key} type="button" className="gc-preset-button" variant={activeConcurrencyPreset === preset.key ? 'secondary' : 'tertiary'} size="sm" aria-pressed={activeConcurrencyPreset === preset.key} onClick={() => applyConcurrencyPreset(preset)}>
                      {preset.label}
                    </Button>
                  ))}
                </div>
              )}
              <div className={styles.calculateAction}>
                <Button type="submit" variant="primary" isDisabled={!canCalculate}>Calculate</Button>
              </div>
            </div>
            {isSizing && (
              <div className={styles.sizingProgress} role="status">
                <div className={styles.progressHeading}><span><Spinner size="sm" /> Calculating configurations…</span><Button variant="link" onClick={cancel}>Cancel</Button></div>
                <Progress value={progress.total > 0 ? progress.completed / progress.total * 100 : 0} size="sm" aria-label="GPU sizing progress" />
                <span>{progress.completed} / {progress.total} GPU systems checked</span>
              </div>
            )}
            {configLoading && <p role="status">Loading model configuration…</p>}
            {configError && <Alert title="Model configuration could not be loaded" variant="warning" isInline>{configError}</Alert>}
            <button
              type="button"
              id="hybrid-workload-toggle"
              className={styles.advancedToggle}
              aria-expanded={workloadAdvancedOpen}
              aria-controls="hybrid-advanced-workload"
              onClick={() => setWorkloadAdvancedOpen(open => !open)}
            >
              <span className={styles.advancedToggleIcon} aria-hidden="true">{workloadAdvancedOpen ? '▾' : '▸'}</span>
              <span>Advanced workload</span>
              {!workloadAdvancedOpen && (
                <span className={styles.advancedSummary}>
                  {formatLatencyBound(targetTtftMs)}s TTFT · {formatter.format(targetTpotMs)} ms TPOT · {formatter.format(targetConcurrency)} concurrent {targetConcurrency === 1 ? 'user' : 'users'} · {formatter.format(prefixTokens)} prefix tokens · {formatter.format(activeHoursPerMonth)} hours/month · {formatter.format(peakToAverage)}× peak · {billingModeLabel(assumptions.cloudBillingMode)}
                </span>
              )}
            </button>
            <div id="hybrid-advanced-workload" className={styles.advancedBody} role="region" aria-labelledby="hybrid-workload-toggle" hidden={!workloadAdvancedOpen}>
              <p className={styles.advancedLead}>Change these values only when you have workload measurements or a clear response target.</p>
              <h3 className={styles.advancedSectionLabel}>Request shape and targets</h3>
              <div className={styles.workloadGrid}>
                <NumberField id="hybrid-average-input" label="Average input tokens per request" value={averageInputTokens} min={1} onChange={setAverageInputTokens} />
                <NumberField id="hybrid-average-output" label="Average output tokens per request" value={averageOutputTokens} min={1} onChange={setAverageOutputTokens} />
                <NumberField id="hybrid-ttft" label="Maximum TTFT (ms)" value={targetTtftMs} min={1} onChange={setTargetTtftMs} />
                <NumberField id="hybrid-tpot" label="Maximum TPOT (ms/token)" value={targetTpotMs} min={1} onChange={setTargetTpotMs} />
                <NumberField id="hybrid-concurrency" label="Sizing concurrency (requests)" value={targetConcurrency} min={1} step={1} onChange={setTargetConcurrency} />
                <NumberField id="hybrid-prefix" label="Cached prefix tokens per request" value={prefixTokens} min={0} max={averageInputTokens} step={1} onChange={setPrefixTokens} />
              </div>
              {requestMix.twoSided && !requestMix.aligned && (
                <div className={styles.alignRequestMix}>
                  <Button type="button" variant="secondary" size="sm"
                    isDisabled={mixAlignment.proposedInputTokens === null}
                    onClick={() => {
                      if (mixAlignment.proposedInputTokens !== null) setAverageInputTokens(mixAlignment.proposedInputTokens)
                    }}>
                    Align request mix
                  </Button>
                  <p className={styles.alignRequestMixNote}>
                    {mixAlignment.proposedInputTokens !== null
                      ? `Monthly totals imply ${formatter.format(requestMix.inputImpliedRequests!)} input-side and ${formatter.format(requestMix.outputImpliedRequests!)} output-side requests. Hardware sizing uses the larger request count implied by the input or output totals. Aligning sets the average input length to ${formatter.format(mixAlignment.proposedInputTokens)} tokens so both sides imply one request count; monthly totals and every other input stay unchanged.`
                      : mixAlignment.reason}
                  </p>
                </div>
              )}
              <h3 className={styles.advancedSectionLabel}>Processing and billing</h3>
              <div className={styles.workloadGrid}>
                <NumberField id="hybrid-active-hours" label="Processing hours per month" value={activeHoursPerMonth} min={1} max={assumptions.hoursPerMonth} onChange={setActiveHoursPerMonth} />
                <NumberField id="hybrid-peak" label="Peak demand multiplier" value={peakToAverage} min={1} step={0.1} onChange={setPeakToAverage} />
                <FormGroup label="Rented billing policy" fieldId="hybrid-billing-mode" className={styles.workloadBilling}>
                  <FormSelect id="hybrid-billing-mode" value={assumptions.cloudBillingMode} onChange={(_event, value) => setCloudBillingMode(value as CloudBillingMode)}>
                    <FormSelectOption value="scale-to-zero" label="Scale to zero — pay for serving time" />
                    <FormSelectOption value="active-window" label="Active window — keep required GPUs warm" />
                    <FormSelectOption value="always-on" label="Always on — full month" />
                  </FormSelect>
                </FormGroup>
              </div>
            </div>
          </CardBody>
        </Card>
      </Form>
      <InfoStrip>
        <span className={styles.workloadSummary}>
          <span><strong>Based on {activePreset?.label ?? 'Custom'}</strong> · {formatter.format(averageInputTokens)} input / {formatter.format(averageOutputTokens)} output tokens per request · {formatter.format(activeHoursPerMonth)} hours/month · {billingModeLabel(assumptions.cloudBillingMode)} · {compactNumber(facts.monthlyTokens)} total tokens/month</span>
          <InfoStripAction onClick={() => setWorkloadAdvancedOpen(true)}>Adjust</InfoStripAction>
        </span>
      </InfoStrip>
            {invalidWorkload && <Alert title="Review the workload inputs" variant="warning" isInline>The combined monthly token total must be positive and no more than 1T, performance targets must be positive, processing hours must be between 1 and {assumptions.hoursPerMonth}, and peak demand must be at least 1× average. Concurrency must be a positive whole number; cached prefix tokens must be a whole number between zero and the average input length.</Alert>}
            {!costings.isLoading && hostedModel === null && model && <Alert title="Hosted comparison unavailable" variant="warning" isInline>
              No compatible standard text price was found for this API model and request length. Choose another API model to compare; rented and purchased sizing still uses {model}.
              <div className={styles.apiModelFallback}>{apiModelSelector}</div>
            </Alert>}
            {!costings.isLoading && costingInputsStale && (
              <Alert title="Some pricing inputs are stale" variant="warning" isInline>
                The comparison may use the last successfully collected prices
                {staleCostingSources.length > 0 ? ` for ${staleCostingSources.join(', ')}` : ''}. Validate current rates before a purchasing decision.
              </Alert>
            )}
            {isStale && !isSizing && <Alert title="Inputs changed" variant="info" isInline>Refresh the forecast so AISimulators can size the updated workload.</Alert>}
            {sizingError && <Alert title="Comparison unavailable" variant="danger" isInline>{sizingError}</Alert>}



      {comparison && (
        <section className={styles.resultsSection} aria-label="Cost comparison">
          <div className={styles.sectionHeading}>
            <h2>Cost comparison</h2>
            {resultSimulator && <SimulatorBadge simulator={resultSimulator} />}
            <FormGroup label="Cost view" fieldId="hybrid-cost-view" className={styles.lensControl}>
              <FormSelect id="hybrid-cost-view" value={assumptions.costLens} onChange={(_event, value) => setCostLens(value as CostLens)}>
                <FormSelectOption value="fully-loaded" label="Full TCO" />
                <FormSelectOption value="marginal" label="Direct running costs" />
              </FormSelect>
            </FormGroup>
          </div>
          <div className={styles.resultGrid}>
            <HybridCostCard title="Hosted API" option={hostedOption} cheapest={comparison.cheapest?.key === 'hosted'} lens={assumptions.costLens}
              requestedOutputTokens={facts.monthlyOutputTokens}
              selector={<div className={styles.hostedSelectors}>
                {apiModelSelector}
                {hostedProviderOffers.length > 1 ? <ComboBox id="hybrid-hosted-provider" label="Provider" portalMenu preserveOrder
                  value={hostedModel ? hostedOfferKey(hostedModel) : ''} onChange={value => setSelectedHostedOfferKey(value || null)}
                  items={hostedProviderOffers.map((offer, index) => {
                    const usageCost = hostedTokenUsageCost(offer, monthlyInputTokens, monthlyOutputTokens, requestShape)
                    const price = usageCost > 0 && usageCost < 0.01 ? '<$0.01' : tokenUsageFormatter.format(usageCost)
                    return { value: hostedOfferKey(offer), label: hostedOfferLabel(offer), group: '',
                      description: `${price}/month API usage${index === 0 ? ' · cheapest' : ''}` }
                  })} />
                  : <p className={styles.apiSourceLine}>{hostedModel && hostedOfferLabel(hostedModel)}</p>}
              </div>} />
            <HybridCostCard title="Rented GPU" option={rentedOption} cheapest={comparison.cheapest?.key === 'rented'} lens={assumptions.costLens}
              requestedOutputTokens={facts.monthlyOutputTokens}
              unavailable={infrastructure.isLoading ? 'Loading complete instance prices…' : 'No sized configuration has an eligible complete cloud offer.'}
              selector={<ComboBox id="hybrid-rented-configuration" label="Hardware configuration" portalMenu preserveOrder
                value={rentedSelection ?? ''} onChange={value => { setRentedSelection(value || null); setSelectionNotice(null) }}
                items={[
                  { value: '', label: `Automatic · ${rentedOptions[0]?.candidate?.cloudGpusPerInstance ?? 1} × ${rentedOptions[0]?.candidate?.label ?? ''}`, group: '', description: 'Cheapest eligible configuration' },
                  ...rentedOptions.flatMap(option => option.candidate ? [{ value: infrastructureCandidateKey(option.candidate, 'rented'),
                    label: `${option.candidate.cloudGpusPerInstance ?? 1} × ${option.candidate.label}`, group: '',
                    description: `${option.candidate.cloudProvider?.toUpperCase()} · ${instanceLabel(option.candidate.cloudProvider, option.candidate.cloudInstanceName)} · ${option.candidate.cloudProviderRegion} · ${formatCostPerMillionOutputTokens(option.costPerMillionOutputTokens)} / 1M output tokens` }] : []),
                ]} />}>
              <p className={styles.costSelectionNote}>{rentedOption?.candidate?.cloudProvider?.toUpperCase()} · {instanceLabel(rentedOption?.candidate?.cloudProvider, rentedOption?.candidate?.cloudInstanceName)}</p>
            </HybridCostCard>
            <HybridCostCard title="Purchased hardware" option={ownedOption} cheapest={comparison.cheapest?.key === 'owned'} lens={assumptions.costLens}
              requestedOutputTokens={facts.monthlyOutputTokens}
              unavailable={infrastructure.isLoading ? 'Loading complete-server prices…' : 'No sized configuration has an eligible complete-server purchase price.'}
              selector={<ComboBox id="hybrid-owned-configuration" label="Hardware configuration" portalMenu preserveOrder
                value={ownedSelection ?? ''} onChange={value => { setOwnedSelection(value || null); setSelectionNotice(null) }}
                items={[
                  { value: '', label: `Automatic · ${ownedOptions[0]?.candidate?.purchaseGpusPerServer ?? ownedOptions[0]?.candidate?.gpusPerReplica ?? 1} × ${ownedOptions[0]?.candidate?.label ?? ''} server`, group: '', description: 'Cheapest eligible configuration' },
                  ...ownedOptions.flatMap(option => option.candidate ? [{ value: infrastructureCandidateKey(option.candidate, 'owned'),
                    label: `${option.candidate.purchaseGpusPerServer ?? option.candidate.gpusPerReplica} × ${option.candidate.label} server`, group: '',
                    description: `${formatCostPerMillionOutputTokens(option.costPerMillionOutputTokens)} / 1M output tokens` }] : []),
                ]} />}>
              {ownedOption?.candidate?.purchasePriceIndicative && <p className={styles.costSelectionNote}>Indicative hardware price</p>}
            </HybridCostCard>
          </div>
          <Card className={styles.disclosureCard}><CardBody>
            <ExpandableSection isExpanded={chartOpen} onToggle={(_event, expanded) => setChartOpen(expanded)} toggleText="Cost vs workload (break-even analysis)">
              {hasOutput ? <>
                <div className={styles.chartControls}>
                  <ToggleGroup id="hybrid-chart-view" className={styles.chartViewSwitch} isFill aria-label="Chart view">
                    <ToggleGroupItem text="Growing demand" buttonId="hybrid-chart-demand" isSelected={chartView === 'workload'}
                      onChange={(_event, selected) => { if (selected) { setChartView('workload'); setFixedNotice(null) } }} />
                    <ToggleGroupItem text="Hardware usage" buttonId="hybrid-chart-hardware" isSelected={chartView === 'fixed'}
                      isDisabled={!rentedOption?.deployment && !ownedOption?.deployment}
                      onChange={(_event, selected) => { if (selected) enterFixedView() }} />
                  </ToggleGroup>
                  {chartView === 'fixed' && fixedCapture && (
                    <div className={styles.fixedPathChoice}>
                      <span>Deployment</span>
                      <ToggleGroup isCompact aria-label="Fixed deployment path">
                        <ToggleGroupItem text="Rented" isSelected={fixedCapture.path === 'rented'} isDisabled={!rentedOption?.deployment}
                          onChange={(_event, selected) => { if (selected) enterFixedView('rented') }} />
                        <ToggleGroupItem text="Purchased" isSelected={fixedCapture.path === 'owned'} isDisabled={!ownedOption?.deployment}
                          onChange={(_event, selected) => { if (selected) enterFixedView('owned') }} />
                      </ToggleGroup>
                    </div>
                  )}
                </div>
                {fixedNotice && <p className={styles.chartControlNote} role="status">{fixedNotice}</p>}
                {chartView === 'fixed' && fixedCapture ? (
                  <FixedDeploymentChart
                    samples={fixedSamples}
                    currentCapacityUsedPct={fixedCurrentCapacityUsedPct}
                    path={fixedCapture.path}
                    hardwareLabel={fixedCandidate?.label}
                    hostedModelLabel={apiModel !== SAME_API_MODEL ? hostedComparisonModel : undefined}
                    infrastructureModelLabel={apiModel !== SAME_API_MODEL ? model : undefined}
                  />
                ) : (
                  <CostComparisonChart
                    points={comparison.chartPoints}
                    currentTokens={comparison.monthlyTokens}
                    outputShare={outputShare}
                    rentedBreakEvenTokens={comparison.rentedBreakEvenTokens}
                    ownedBreakEvenTokens={comparison.ownedBreakEvenTokens}
                    rentedLowestCostTokens={comparison.rentedLowestCostTokens}
                    ownedLowestCostTokens={comparison.ownedLowestCostTokens}
                    transitionsVerified={comparison.transitionsVerified}
                    hostedModelLabel={apiModel !== SAME_API_MODEL ? hostedComparisonModel : undefined}
                    infrastructureModelLabel={apiModel !== SAME_API_MODEL ? model : undefined}
                  />
                )}
              </> : <p className={styles.chartEmpty} role="status">Enter monthly output tokens above zero to see this graph. Monthly costs remain available on the cards.</p>}
            </ExpandableSection>
          </CardBody></Card>
        </section>
      )}

      <Card className={styles.disclosureCard}><CardBody>
        <div className={styles.calculationDetailsHeader}>
          <ExpandableSectionToggle
            isExpanded={calculationDetailsOpen} onToggle={setCalculationDetailsOpen}
            toggleId="hybrid-calculation-details-toggle" contentId="hybrid-calculation-details"
          >Data and assumptions</ExpandableSectionToggle>
          <Link className={styles.assumptionsLink} href="/sources#cost-assumptions">Edit cost assumptions in Sources</Link>
        </div>
        <ExpandableSection isDetached isExpanded={calculationDetailsOpen}
          toggleId="hybrid-calculation-details-toggle" contentId="hybrid-calculation-details">
          <section className={styles.dataSourcesSection} aria-labelledby="hybrid-workload-basis-heading">
            <h3 id="hybrid-workload-basis-heading">Workload basis</h3>
            <dl className={styles.calculationList}>
              <div><dt>Monthly input</dt><dd>{compactNumber(monthlyInputTokens)} tokens</dd></div>
              <div><dt>Monthly output</dt><dd>{compactNumber(monthlyOutputTokens)} tokens</dd></div>
              <div><dt>Request shape</dt><dd>{formatter.format(averageInputTokens)} input / {formatter.format(averageOutputTokens)} output tokens per request</dd></div>
              <div><dt>Input-implied requests</dt><dd>{requestCountLabel(requestMix.inputImpliedRequests)}</dd></div>
              <div><dt>Output-implied requests</dt><dd>{requestCountLabel(requestMix.outputImpliedRequests)}</dd></div>
              <div><dt>Requests used for sizing</dt><dd>{requestCountLabel(requestMix.sizingRequests)}</dd></div>
              <div><dt>Simulated output work</dt><dd>{requestMix.sizingRequests !== null ? `${compactNumber(requestMix.sizingRequests * averageOutputTokens)} output tokens` : 'Not available'}</dd></div>
            </dl>
            {!requestMix.twoSided
              ? <p className={styles.calculationNote}>Only one monthly side is positive, so there is no two-sided request mix. Sizing follows the positive side&apos;s implied request count; the hosted invoice uses the entered token totals.</p>
              : !requestMix.aligned && <p className={styles.calculationNote}>Hardware sizing uses the larger request count implied by the input or output totals. The hosted invoice uses the entered token totals, not the larger simulated work.</p>}
          </section>
          <section className={styles.dataSourcesSection} aria-labelledby="hybrid-sizing-deployment-heading">
            <h3 id="hybrid-sizing-deployment-heading">Sizing and deployment</h3>
            <div className={styles.assumptionColumns}>
              <section className={styles.calculationGroup} aria-labelledby="hybrid-reference-performance-heading">
                <h4 id="hybrid-reference-performance-heading">Reference performance</h4>
                <dl className={styles.calculationList}>
                  <div><dt>Backend</dt><dd>{inferenceBackend}{backendVersion ? ` · ${backendVersion} requested` : ''}</dd></div>
                  <div><dt>Simulator</dt><dd>{resultSimulator && !isStale ? <SimulatorBadge simulator={resultSimulator} /> : 'Not yet calculated for current settings'}</dd></div>
                  <div><dt>Capacity reference</dt><dd>{formatter.format(targetConcurrency)} concurrent {targetConcurrency === 1 ? 'request' : 'requests'}</dd></div>
                  <div><dt>Response targets</dt><dd>{formatLatencyBound(targetTtftMs)}s TTFT · {formatter.format(targetTpotMs)} ms TPOT</dd></div>
                  <div><dt>Peak demand</dt><dd>{preciseRate(facts.peakRequestsPerSecond)} requests/s · {formatter.format(peakToAverage)}× average</dd></div>
                  <div><dt>Failover reserve</dt><dd>None</dd></div>
                </dl>
              </section>
              <section className={styles.calculationGroup} aria-labelledby="hybrid-rented-deployment-heading">
                <h4 id="hybrid-rented-deployment-heading">Rented deployment</h4>
                <dl className={styles.calculationList}>{deploymentRows(rentedOption, 'rented')}</dl>
              </section>
              <section className={styles.calculationGroup} aria-labelledby="hybrid-owned-deployment-heading">
                <h4 id="hybrid-owned-deployment-heading">Purchased deployment</h4>
                <dl className={styles.calculationList}>{deploymentRows(ownedOption, 'owned')}</dl>
              </section>
            </div>
            <p className={styles.calculationNote}>Capacity is modeled from the selected request shape and reference concurrency. It is not a measured maximum-throughput benchmark.</p>
          </section>
          <section className={styles.dataSourcesSection} aria-labelledby="hybrid-cost-calculation-heading">
            <h3 id="hybrid-cost-calculation-heading">Cost calculation</h3>
            <dl className={styles.calculationList}>
              <div><dt>Rented billing policy</dt><dd>{billingModeLabel(assumptions.cloudBillingMode)}</dd></div>
              <div><dt>Processing window</dt><dd>{formatter.format(activeHoursPerMonth)} hours/month</dd></div>
              <div><dt>Always-on month</dt><dd>{formatter.format(assumptions.hoursPerMonth)} hours</dd></div>
              <div><dt>Requested output denominator</dt><dd>{compactNumber(facts.monthlyOutputTokens)} output tokens/month</dd></div>
              <div><dt>Cost per 1M output tokens</dt><dd>Selected monthly cost × 1,000,000 ÷ requested monthly output tokens</dd></div>
            </dl>
            {apiModel !== SAME_API_MODEL && (
              <p className={styles.calculationNote}>Hosted API: {hostedComparisonModel} · Rented / purchased: {model}. Costs use the same requested token volumes. Different models are not assumed to provide equivalent answer quality or capability.</p>
            )}
          </section>
          <AppliedCostAssumptions assumptions={assumptions} workload={workload} rented={rentedOption} owned={ownedOption} />
          <section className={styles.dataSourcesSection} aria-labelledby="hybrid-data-sources-heading">
            <h3 id="hybrid-data-sources-heading">Data sources</h3>
            <div className={styles.assumptionColumns}>
            <section className={styles.calculationGroup} aria-labelledby="hybrid-catalogue-details-heading">
              <h4 id="hybrid-catalogue-details-heading">Catalogue coverage</h4>
              <dl className={styles.calculationList}>
                <div><dt>Live catalogue</dt><dd>{modelOptions.length} models · {gpuOptions.length} GPU systems</dd></div>
                <div><dt>Sizing results</dt><dd>{sizedSystemCount} successful · {failedSystems} failed systems</dd></div>
                <div><dt>Price-ready systems</dt><dd>{priceReadyGpus.length} in the catalogue</dd></div>
                <div><dt>Eligible configurations</dt><dd>{rentedOptions.length} rented · {ownedOptions.length} purchased</dd></div>
                <div><dt>Price coverage</dt><dd>{rentedReadyGpus.length} rented-ready · {purchaseReadyGpus.length} purchase-ready systems</dd></div>
                <div><dt>Cloud filter</dt><dd>{preferredCloudProvider || 'All providers'}</dd></div>
              </dl>
            </section>
            <section className={styles.calculationGroup} aria-labelledby="hybrid-pricing-details-heading">
              <h4 id="hybrid-pricing-details-heading">Prices and freshness</h4>
              <dl className={styles.calculationList}>
                <div><dt>Pricing freshness</dt><dd>{freshnessSummary}{!costings.health && ' · source health unavailable'}</dd></div>
                <div><dt>Hosted prices</dt><dd>Live aicostings feed · {pricingSource}</dd></div>
              {hostedModel && <>
                  <div><dt>Selected API model</dt><dd>{hostedModel.id}</dd></div>
                  <div><dt>Token rates / 1M</dt><dd>${formatter.format(hostedPrice?.inputPerMillion ?? 0)} input · ${formatter.format(hostedPrice?.outputPerMillion ?? 0)} output · standard uncached text{hostedPrice?.requestPerRequest ? ` + $${hostedPrice.requestPerRequest}/request` : ''}</dd></div>
                  <div><dt>Selected pricing source</dt><dd>{hostedPricingSource(hostedModel)}</dd></div>
                  <div><dt>Offer updated</dt><dd>{collectionDate(hostedModel.updated_at)}</dd></div>
                </>}
                <div><dt>Hosted feed updated</dt><dd>{collectionDate(costings.modelsUpdatedAt)}</dd></div>
                <div><dt>Infrastructure prices</dt><dd>Complete-offers API · {infrastructure.rented.length} rental offers · {infrastructure.purchased.length} server configurations</dd></div>
                <div><dt>Excluded rental rates</dt><dd>{infrastructure.rented.filter(offer => offer.eligibility !== 'standard' || offer.rateKind !== 'on_demand').length} Spot, Capacity Block or inference-platform-only offers excluded from general rental selection</dd></div>
                {Object.entries(infrastructure.sources).map(([source, status]) => <div key={source}>
                  <dt>{source} collection</dt><dd>{status.state}{status.message && ` · ${status.message}`}</dd>
                </div>)}
                {costings.health && <>
                  <div><dt>Tracked hosted sources</dt><dd>{hostedSources.length}</dd></div>
                  <div><dt>Stale sources</dt><dd>{staleCostingSources.map(collectionLabel).join(', ') || 'None reported'}</dd></div>
                  <div><dt>Refresh failures</dt><dd>{hostedSources.filter(([, status]) => status.last_error).map(([source]) => collectionLabel(source)).join(', ') || 'None reported'}</dd></div>
                </>}
              </dl>
              <Link className={styles.sourceDetailsLink} href="/sources">View source collection times</Link>
            </section>
            </div>
            <p className={styles.calculationNote}>Planning estimates, not production guarantees. Validate the selected workload and commercial terms. Freshness describes collection, not a price’s effective date; each flip card identifies collected prices or curated estimates.</p>
          </section>
        </ExpandableSection>
      </CardBody></Card>

    </div>
  )
}

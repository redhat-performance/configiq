'use client'
import * as React from 'react'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { useCostAssumptions } from '@/contexts/CostAssumptionsContext'
import {
  Alert,
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
} from '@patternfly/react-core'
import { ComboBox } from '@/components/ModelComboBox/ModelComboBox'
import { InfoStrip, InfoStripAction } from '@/components/ui/InfoStrip'
import { buildModelItems, needsHfConfig } from '@/lib/model-options'
import { fetchModelConfig } from '@/lib/huggingface/fetch-config'
import { HybridCostCard } from './HybridCostCard'
import { AppliedCostAssumptions } from './AppliedCostAssumptions'
import { useHybridSizing } from './useHybridSizing'
import { useSettings } from '@/contexts/SettingsContext'
import { useCatalog } from '@/lib/hooks/useCatalog'
import { getAppConfig } from '@/lib/app-config'
import { DEFAULT_WORKLOAD, type WorkloadPreset } from '@/lib/workload-presets'
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
  type CloudBillingMode,
  type CostAssumptions,
  type CostLens,
  type HostedPrice,
  type HybridWorkload,
} from '@/lib/hybrid-savings/calc'
import {
  hasServerPurchaseConfiguration,
} from '@/lib/hybrid-savings/purchase-catalog'
import {
  hasRentedCloudOffer,
} from '@/lib/hybrid-savings/cloud-offer-catalog'
import { hostedTokenUsageCost, resolveHostedPricing } from '@/lib/hybrid-savings/hosted-pricing'
import { buildApiModelItems, hostedOfferLabel, hostedPricingSource, isStandardTokenOffer, SAME_API_MODEL } from '@/lib/hybrid-savings/api-models'
import { normalizeModelId } from '@/lib/model-metadata'
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

const formatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 })
const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
})
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
    'cloud.aws': 'AWS', 'cloud.azure': 'Azure', 'cloud.vastai': 'Vast.ai',
    hardware_costs: 'Hardware prices', 'models.litellm': 'LiteLLM', 'models.openrouter': 'OpenRouter',
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

function toHostedPrice(model: FrontierModel): HostedPrice {
  return {
    modelId: model.id,
    label: `${hostedOfferLabel(model)} hosted API`,
    inputPerMillion: model.price_per_m_input,
    outputPerMillion: model.price_per_m_output,
  }
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
    testedModels,
    inferenceBackend,
    costingsEnabled,
    preferredCloudProvider,
    pricingSource,
    hfToken,
  } = useSettings()
  const {
    gpuOptions,
    modelOptions,
    modelSpecs,
    isLoading: catalogLoading,
    error: catalogError,
  } = useCatalog()
  const costings = useCostings(costingsEnabled, pricingSource)
  const staleCostingSources = React.useMemo(
    () => Object.entries(costings.health?.sources ?? {})
      .filter(([, status]) => status.stale)
      .map(([source]) => source),
    [costings.health],
  )
  const costingInputsStale = costings.modelsStale || staleCostingSources.length > 0

  const [model, setModel] = React.useState('')
  const [profileId, setProfileId] = React.useState(DEFAULT_WORKLOAD.key)
  const [monthlyInputTokens, setMonthlyInputTokens] = React.useState(2_000_000_000)
  const [monthlyOutputTokens, setMonthlyOutputTokens] = React.useState(500_000_000)
  const [averageInputTokens, setAverageInputTokens] = React.useState(DEFAULT_WORKLOAD.isl)
  const [averageOutputTokens, setAverageOutputTokens] = React.useState(DEFAULT_WORKLOAD.osl)
  const [targetTtftMs, setTargetTtftMs] = React.useState(DEFAULT_WORKLOAD.ttft)
  const [targetTpotMs, setTargetTpotMs] = React.useState(DEFAULT_WORKLOAD.tpot)
  const [targetConcurrency, setTargetConcurrency] = React.useState(DEFAULT_WORKLOAD.concurrency)
  const [prefixTokens, setPrefixTokens] = React.useState(DEFAULT_WORKLOAD.prefix)
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
  const [selectedHostedOfferKey, setSelectedHostedOfferKey] = React.useState<string | null>(null)
  const [apiModel, setApiModel] = React.useState(SAME_API_MODEL)
  const [workloadAdvancedOpen, setWorkloadAdvancedOpen] = React.useState(false)
  const [chartOpen, setChartOpen] = React.useState(false)
  const [calculationDetailsOpen, setCalculationDetailsOpen] = React.useState(false)
  const { candidates, isSizing, progress, sizingError, failedSystems, lastSizingSignature, runSizing: sizeConfigurations, cancel } = useHybridSizing()
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

  const modelItems = React.useMemo(() => hydrated ? buildModelItems(modelOptions, modelSpecs) : [], [modelOptions, modelSpecs, hydrated])
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

  const apiModelItems = React.useMemo(() => buildApiModelItems(costings.models, model), [costings.models, model])
  const standardHostedOffers = React.useMemo(() => costings.models.filter(isStandardTokenOffer), [costings.models])
  const hostedComparisonModel = apiModel === SAME_API_MODEL ? model : apiModel
  const hostedPricing = React.useMemo(
    () => resolveHostedPricing(
      hostedComparisonModel,
      standardHostedOffers,
      monthlyInputTokens,
      monthlyOutputTokens,
    ),
    [hostedComparisonModel, standardHostedOffers, monthlyInputTokens, monthlyOutputTokens],
  )
  const hostedProviderOffers = hostedPricing.providerMatches
  const hostedModel = selectedHostedOfferKey === null ? hostedProviderOffers[0] ?? null
    : hostedProviderOffers.find(offer => hostedOfferKey(offer) === selectedHostedOfferKey) ?? null
  const hostedPrice = React.useMemo(
    () => hostedModel ? toHostedPrice(hostedModel) : null,
    [hostedModel],
  )
  React.useEffect(() => {
    setSelectedHostedOfferKey(null)
  }, [hostedComparisonModel, monthlyInputTokens, monthlyOutputTokens])
  const apiModelSelector = <ComboBox id="hybrid-api-model" label="API model"
    value={apiModel} items={apiModelItems} portalMenu
    onChange={value => { setApiModel(value || SAME_API_MODEL); setSelectedHostedOfferKey(null) }} />
  const workloadPresets = hydrated ? [DEFAULT_WORKLOAD, ...getAppConfig().workloadPresets] : [DEFAULT_WORKLOAD]
  const selectedProfile = workloadPresets.find(profile => profile.key === profileId) ?? DEFAULT_WORKLOAD
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
  const sizingSignature = React.useMemo(() => JSON.stringify({
    model,
    monthlyInputTokens,
    monthlyOutputTokens,
    averageInputTokens,
    averageOutputTokens,
    activeHoursPerMonth,
    peakToAverage,
    targetTtftMs,
    targetTpotMs,
    targetConcurrency,
    prefixTokens,
    inferenceBackend,
  }), [
    model,
    monthlyInputTokens,
    monthlyOutputTokens,
    averageInputTokens,
    averageOutputTokens,
    activeHoursPerMonth,
    peakToAverage,
    targetTtftMs,
    targetTpotMs,
    targetConcurrency,
    prefixTokens,
    inferenceBackend,
  ])
  const isStale = lastSizingSignature !== null && lastSizingSignature !== sizingSignature
  const comparison = React.useMemo(() => {
    if (!hostedPrice || candidates.length === 0 || lastSizingSignature === null || isStale) return null
    return calculateHybridComparison(workload, hostedPrice, candidates, assumptions, { rented: rentedSelection, owned: ownedSelection })
  }, [workload, hostedPrice, candidates, assumptions, lastSizingSignature, isStale, rentedSelection, ownedSelection])

  const rentedOptions = React.useMemo(() => infrastructureOptionsAtVolume('rented', workload, candidates, assumptions, facts.monthlyTokens), [workload, candidates, assumptions, facts.monthlyTokens])
  const ownedOptions = React.useMemo(() => infrastructureOptionsAtVolume('owned', workload, candidates, assumptions, facts.monthlyTokens), [workload, candidates, assumptions, facts.monthlyTokens])
  React.useEffect(() => {
    // Do not erase a valid manual choice while a new sizing run is in progress.
    if (isSizing || isStale || lastSizingSignature === null) return
    if (rentedSelection && !rentedOptions.some(option => option.candidate && infrastructureCandidateKey(option.candidate, 'rented') === rentedSelection)) setRentedSelection(null)
    if (ownedSelection && !ownedOptions.some(option => option.candidate && infrastructureCandidateKey(option.candidate, 'owned') === ownedSelection)) setOwnedSelection(null)
  }, [isSizing, isStale, lastSizingSignature, rentedSelection, ownedSelection, rentedOptions, ownedOptions])

  const rentedReadyGpus = React.useMemo(() => gpuOptions.filter(gpu => (
    hasRentedCloudOffer(gpu.systemId)
  )), [gpuOptions])
  const purchaseReadyGpus = React.useMemo(() => gpuOptions.filter(gpu => (
    hasServerPurchaseConfiguration(gpu.systemId)
  )), [gpuOptions])
  const priceReadyGpus = React.useMemo(() => gpuOptions.filter(gpu => (
    rentedReadyGpus.some(candidate => candidate.systemId === gpu.systemId) ||
    purchaseReadyGpus.some(candidate => candidate.systemId === gpu.systemId)
  )), [gpuOptions, rentedReadyGpus, purchaseReadyGpus])
  const sizedSystemCount = React.useMemo(
    () => new Set(candidates.map(candidate => candidate.systemId)).size,
    [candidates],
  )

  const applyProfile = (profile: WorkloadPreset) => {
    setProfileId(profile.key)
    setAverageInputTokens(profile.isl)
    setAverageOutputTokens(profile.osl)
    setTargetTtftMs(profile.ttft)
    setTargetTpotMs(profile.tpot)
    setTargetConcurrency(profile.concurrency)
    setPrefixTokens(profile.prefix)
    setWorkloadAdvancedOpen(true)
  }

  const runSizing = () => sizeConfigurations({
    signature: sizingSignature, model, backend: inferenceBackend,
    averageInputTokens, averageOutputTokens, targetTtftMs, targetTpotMs,
    targetConcurrency, prefixTokens,
    gpuOptions, preferredCloudProvider,
    ...(modelConfig?.model === model ? { modelConfig: modelConfig.config } : {}),
  })

  const invalidWorkload =
    facts.monthlyTokens <= 0 ||
    facts.monthlyTokens > HYBRID_PLANNING_HORIZON_TOKENS ||
    averageInputTokens <= 0 ||
    averageOutputTokens <= 0 ||
    activeHoursPerMonth <= 0 ||
    activeHoursPerMonth > assumptions.hoursPerMonth ||
    peakToAverage < 1 ||
    targetTtftMs <= 0 ||
    targetTpotMs <= 0 ||
    targetConcurrency < 1 ||
    !Number.isInteger(targetConcurrency) ||
    prefixTokens < 0 ||
    !Number.isInteger(prefixTokens) ||
    prefixTokens > averageInputTokens

  const canCalculate = assumptionsHydrated && !invalidWorkload && !!hostedPrice && !!model && gpuOptions.length > 0 &&
    !catalogLoading && !costings.isLoading && !isSizing && !configLoading &&
    (!needsHfConfig(model, modelOptions) || modelConfig?.model === model)
  const hostedOption = comparison?.options.find(option => option.key === 'hosted')
  const rentedOption = comparison?.options.find(option => option.key === 'rented')
  const ownedOption = comparison?.options.find(option => option.key === 'owned')

  if (hydrated && !costingsEnabled) {
    return (
      <div className={styles.page}>
        <header className={styles.header}>
          <div>
            <h1 className={styles.pageTitle}>Hybrid savings</h1>
            <p className={styles.subtitle}>Compare hosted API, rented GPU and purchased hardware costs for one workload.</p>
          </div>
        </header>
        <Alert title="Enable experimental costings" variant="info" isInline>
          Hybrid Savings uses the shared costing catalogue. Enable experimental costings in Settings, then return here.
          <div className={styles.alertAction}><Button component="a" href="/settings" variant="link" isInline>Open Settings</Button></div>
        </Alert>
      </div>
    )
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.pageTitle}>Hybrid savings</h1>
          <p className={styles.subtitle}>
            Compare the monthly cost of hosted API, rented GPU and purchased hardware for your workload.
          </p>
        </div>
      </header>

      {(catalogError || costings.error) && (
        <Alert title="Some shared data could not be loaded" variant="warning" isInline>
          {catalogError ?? costings.error}. Refresh the page or check the Sources page before using the result.
        </Alert>
      )}

      <Form onSubmit={event => { event.preventDefault(); if (canCalculate) void runSizing() }}>
        <Card className={styles.inputCard}>
          <CardBody>
            <h2>Model and workload</h2>
            <div className={styles.inputGrid}>
              <div className={styles.modelField}>
                <ComboBox id="hybrid-model" value={model} onChange={setModel} items={modelItems}
                  allowCustom supportedModels={testedModels} hfToken={hfToken}
                  helperText="Choose a model checkpoint to compare." />
              </div>
              <NumberField id="hybrid-input-tokens" label="Monthly input tokens" value={monthlyInputTokens} max={HYBRID_PLANNING_HORIZON_TOKENS} step={1_000_000} formatWithCommas onChange={setMonthlyInputTokens} />
              <NumberField id="hybrid-output-tokens" label="Monthly output tokens" value={monthlyOutputTokens} max={HYBRID_PLANNING_HORIZON_TOKENS} step={1_000_000} formatWithCommas onChange={setMonthlyOutputTokens} />
            </div>
            <div className={styles.inputActions}>
              <div className={styles.workloadPresets} role="group" aria-label="Workload type" id="hybrid-profile">
                <span className={styles.workloadLabel}>Workload:</span>
                {workloadPresets.map(profile => (
                  <Button key={profile.key} type="button" variant={profileId === profile.key ? 'secondary' : 'tertiary'} size="sm" aria-pressed={profileId === profile.key} onClick={() => applyProfile(profile)}>
                    {profile.label}
                  </Button>
                ))}
              </div>
              <Button type="submit" variant="primary" isDisabled={!canCalculate}>Calculate</Button>
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
                  {formatter.format(targetTtftMs)} ms TTFT · {formatter.format(targetTpotMs)} ms TPOT · {formatter.format(targetConcurrency)} concurrent · {formatter.format(prefixTokens)} prefix tokens · {formatter.format(activeHoursPerMonth)} hours/month · {formatter.format(peakToAverage)}× peak · {billingModeLabel(assumptions.cloudBillingMode)}
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
          <span><strong>Based on {selectedProfile.label}</strong> · {formatter.format(averageInputTokens)} input / {formatter.format(averageOutputTokens)} output tokens per request · {formatter.format(activeHoursPerMonth)} hours/month · {billingModeLabel(assumptions.cloudBillingMode)} · {compactNumber(facts.monthlyTokens)} total tokens/month</span>
          <InfoStripAction onClick={() => setWorkloadAdvancedOpen(true)}>Adjust</InfoStripAction>
        </span>
      </InfoStrip>
            {invalidWorkload && <Alert title="Review the workload inputs" variant="warning" isInline>The combined monthly token total must be positive and no more than 1T, performance targets must be positive, processing hours must be between 1 and {assumptions.hoursPerMonth}, and peak demand must be at least 1× average. Concurrency must be a positive whole number; cached prefix tokens must be a whole number between zero and the average input length.</Alert>}
            {!costings.isLoading && hostedModel === null && model && <Alert title="Hosted comparison unavailable" variant="warning" isInline>
              No valid standard token price was found for the selected API model or offer. Choose another API model to compare; rented and purchased sizing still uses {model}.
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
            <FormGroup label="Cost view" fieldId="hybrid-cost-view" className={styles.lensControl}>
              <FormSelect id="hybrid-cost-view" value={assumptions.costLens} onChange={(_event, value) => setCostLens(value as CostLens)}>
                <FormSelectOption value="fully-loaded" label="Full TCO" />
                <FormSelectOption value="marginal" label="Marginal" />
              </FormSelect>
            </FormGroup>
          </div>
          <div className={styles.resultGrid}>
            <HybridCostCard title="Hosted API" option={hostedOption} cheapest={comparison.cheapest?.key === 'hosted'} lens={assumptions.costLens}
              hostedOfferId={hostedModel?.id}
              selector={<div className={styles.hostedSelectors}>
                {apiModelSelector}
                {hostedProviderOffers.length > 1 ? <ComboBox id="hybrid-hosted-provider" label="Provider" portalMenu preserveOrder
                  value={hostedModel ? hostedOfferKey(hostedModel) : ''} onChange={value => setSelectedHostedOfferKey(value || null)}
                  items={hostedProviderOffers.map((offer, index) => {
                    const usageCost = hostedTokenUsageCost(offer, monthlyInputTokens, monthlyOutputTokens)
                    const price = usageCost > 0 && usageCost < 0.01 ? '<$0.01' : tokenUsageFormatter.format(usageCost)
                    return { value: hostedOfferKey(offer), label: hostedOfferLabel(offer), group: '',
                      description: `${price}/month token usage${index === 0 ? ' · cheapest' : ''}` }
                  })} />
                  : <p className={styles.apiSourceLine}>{hostedModel && hostedOfferLabel(hostedModel)}</p>}
              </div>}>
              <dl className={styles.metricList}>
                <div><dt>Pricing source</dt><dd>{hostedModel && hostedPricingSource(hostedModel)}</dd></div>
              </dl>
            </HybridCostCard>
            <HybridCostCard title="Rented GPU" option={rentedOption} cheapest={comparison.cheapest?.key === 'rented'} lens={assumptions.costLens}
              unavailable="No sized configuration has an eligible complete cloud offer."
              selector={<ComboBox id="hybrid-rented-configuration" label="Hardware configuration" portalMenu preserveOrder
                value={rentedSelection ?? ''} onChange={value => setRentedSelection(value || null)}
                items={[
                  { value: '', label: `Automatic · ${rentedOptions[0]?.candidate?.cloudGpusPerInstance ?? 1} × ${rentedOptions[0]?.candidate?.label ?? ''}`, group: '', description: 'Cheapest eligible configuration' },
                  ...rentedOptions.flatMap(option => option.candidate ? [{ value: infrastructureCandidateKey(option.candidate, 'rented'),
                    label: `${option.candidate.cloudGpusPerInstance ?? 1} × ${option.candidate.label}`, group: '',
                    description: `${option.candidate.cloudProvider?.toUpperCase()} · ${option.candidate.cloudInstanceName} · ${option.candidate.cloudProviderRegion} · ${currencyFormatter.format(option.monthlyCost)}/mo` }] : []),
                ]} />}>
              <dl className={styles.metricList}>
                <div><dt>Provider / instance</dt><dd>{rentedOption?.candidate?.cloudProvider?.toUpperCase()} · {rentedOption?.candidate?.cloudInstanceName}</dd></div>
                <div><dt>Availability</dt><dd>Rented ready</dd></div>
              </dl>
            </HybridCostCard>
            <HybridCostCard title="Purchased hardware" option={ownedOption} cheapest={comparison.cheapest?.key === 'owned'} lens={assumptions.costLens}
              unavailable="No sized configuration has an eligible complete-server purchase price."
              selector={<ComboBox id="hybrid-owned-configuration" label="Hardware configuration" portalMenu preserveOrder
                value={ownedSelection ?? ''} onChange={value => setOwnedSelection(value || null)}
                items={[
                  { value: '', label: `Automatic · ${ownedOptions[0]?.candidate?.purchaseGpusPerServer ?? ownedOptions[0]?.candidate?.gpusPerReplica ?? 1} × ${ownedOptions[0]?.candidate?.label ?? ''} server`, group: '', description: 'Cheapest eligible configuration' },
                  ...ownedOptions.flatMap(option => option.candidate ? [{ value: infrastructureCandidateKey(option.candidate, 'owned'),
                    label: `${option.candidate.purchaseGpusPerServer ?? option.candidate.gpusPerReplica} × ${option.candidate.label} server`, group: '',
                    description: `${currencyFormatter.format(option.monthlyCost)}/mo` }] : []),
                ]} />}>
              <dl className={styles.metricList}>
                <div><dt>System</dt><dd>{ownedOption?.candidate?.label}</dd></div>
                <div><dt>Availability</dt><dd>Purchase ready · indicative price</dd></div>
              </dl>
            </HybridCostCard>
          </div>
          <Card className={styles.disclosureCard}><CardBody>
            <ExpandableSection isExpanded={chartOpen} onToggle={(_event, expanded) => setChartOpen(expanded)} toggleText="Cost vs workload (break-even analysis)">
                <CostComparisonChart
                  points={comparison.chartPoints}
                  currentTokens={comparison.monthlyTokens}
                  costLens={assumptions.costLens}
                  rentedBreakEvenTokens={comparison.rentedBreakEvenTokens}
                  ownedBreakEvenTokens={comparison.ownedBreakEvenTokens}
                  rentedLowestCostTokens={comparison.rentedLowestCostTokens}
                  ownedLowestCostTokens={comparison.ownedLowestCostTokens}
                  transitionsVerified={comparison.transitionsVerified}
                  hostedModelLabel={apiModel !== SAME_API_MODEL ? hostedComparisonModel : undefined}
                  infrastructureModelLabel={apiModel !== SAME_API_MODEL ? model : undefined}
                />

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
          <AppliedCostAssumptions assumptions={assumptions} workload={workload} rented={rentedOption} owned={ownedOption} />
          <section className={styles.dataSourcesSection} aria-labelledby="hybrid-data-sources-heading">
            <h3 id="hybrid-data-sources-heading">Data sources</h3>
            <div className={styles.assumptionColumns}>
            <section className={styles.calculationGroup} aria-labelledby="hybrid-sizing-details-heading">
              <h4 id="hybrid-sizing-details-heading">Sizing and capacity</h4>
              <dl className={styles.calculationList}>
                <div><dt>Backend</dt><dd>{inferenceBackend}</dd></div>
                <div><dt>Capacity reference</dt><dd>{formatter.format(targetConcurrency)} concurrent {targetConcurrency === 1 ? 'request' : 'requests'}</dd></div>
                <div><dt>Response targets</dt><dd>{formatter.format(targetTtftMs)} ms TTFT · {formatter.format(targetTpotMs)} ms TPOT</dd></div>
                <div><dt>Peak demand</dt><dd>{preciseRate(facts.peakRequestsPerSecond)} requests/s · {formatter.format(peakToAverage)}× average</dd></div>
                <div><dt>Failover reserve</dt><dd>None</dd></div>
              </dl>
            </section>
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
                <div><dt>Pricing freshness</dt><dd>{costingInputsStale ? 'Stale · check Sources' : 'No stale feed reported'}{!costings.health && ' · source health unavailable'}</dd></div>
                <div><dt>Hosted prices</dt><dd>Live aicostings feed · {pricingSource}</dd></div>
              {hostedModel && <>
                  <div><dt>Selected API model</dt><dd>{hostedModel.id}</dd></div>
                  <div><dt>Token rates / 1M</dt><dd>${formatter.format(hostedModel.price_per_m_input)} input · ${formatter.format(hostedModel.price_per_m_output)} output</dd></div>
                  <div><dt>Selected pricing source</dt><dd>{hostedPricingSource(hostedModel)}</dd></div>
                  <div><dt>Offer updated</dt><dd>{collectionDate(hostedModel.updated_at)}</dd></div>
                </>}
                <div><dt>Hosted feed updated</dt><dd>{collectionDate(costings.modelsUpdatedAt)}</dd></div>
                {costings.health && <>
                  <div><dt>Tracked sources</dt><dd>{Object.keys(costings.health.sources).length}</dd></div>
                  <div><dt>Stale sources</dt><dd>{staleCostingSources.map(collectionLabel).join(', ') || 'None reported'}</dd></div>
                  <div><dt>Refresh failures</dt><dd>{Object.entries(costings.health.sources).filter(([, status]) => status.last_error).map(([source]) => collectionLabel(source)).join(', ') || 'None reported'}</dd></div>
                </>}
              </dl>
              <Link className={styles.sourceDetailsLink} href="/sources">View source collection times</Link>
            </section>
            </div>
            <p className={styles.calculationNote}>Freshness describes collection, not a price’s effective date. Cloud and server prices are dated catalogue records, not live quotes; each flip card includes its price source and date.</p>
          </section>
        </ExpandableSection>
      </CardBody></Card>

    </div>
  )
}

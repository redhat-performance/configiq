'use client'

import * as React from 'react'
import dynamic from 'next/dynamic'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardBody,
  ExpandableSection,
  Form,
  FormGroup,
  FormHelperText,
  FormSelect,
  FormSelectOption,
  HelperText,
  HelperTextItem,
  Label,
  Progress,
  Spinner,
  TextInput,
  ToggleGroup,
  ToggleGroupItem,
} from '@patternfly/react-core'
import { useSettings } from '@/contexts/SettingsContext'
import type { RecommendResult } from '@/lib/api/recommend'
import { readRecommendStream } from '@/lib/api/recommend-stream'
import { useCatalog, type GpuOption, type ModelSpec } from '@/lib/hooks/useCatalog'
import {
  useCostings,
  type FrontierModel,
} from '@/lib/hooks/useCostings'
import {
  calculateHybridComparison,
  hostedCostAtVolume,
  HYBRID_PLANNING_HORIZON_TOKENS,
  workloadFacts,
  type CloudBillingMode,
  type CostAssumptions,
  type CostLens,
  type CostOption,
  type HostedPrice,
  type HybridWorkload,
  type InfrastructureCandidate,
} from '@/lib/hybrid-savings/calc'
import {
  hasServerPurchaseConfiguration,
  resolveCompatibleServerPurchaseConfigurations,
  type ServerPurchaseConfiguration,
} from '@/lib/hybrid-savings/purchase-catalog'
import {
  hasRentedCloudOffer,
  resolveRentedCloudOffers,
  type RentedCloudOffer,
} from '@/lib/hybrid-savings/cloud-offer-catalog'
import { resolveHostedPricing } from '@/lib/hybrid-savings/hosted-pricing'
import {
  isModelListedAsTested,
  modelParameterBillions,
  modelSizeLabel,
  modelTierLabel,
} from '@/lib/hybrid-savings/model-catalogue'
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

type WorkloadProfileId = 'assistant' | 'rag' | 'agentic' | 'batch'

interface WorkloadProfile {
  id: WorkloadProfileId
  label: string
  description: string
  averageInputTokens: number
  averageOutputTokens: number
  targetTtftMs: number
  targetTpotMs: number
  activeHoursPerMonth: number
  peakToAverage: number
}

const WORKLOAD_PROFILES: WorkloadProfile[] = [
  {
    id: 'assistant',
    label: 'General assistant',
    description: 'Concise enterprise questions and answers.',
    averageInputTokens: 2_048,
    averageOutputTokens: 512,
    targetTtftMs: 5_000,
    targetTpotMs: 60,
    activeHoursPerMonth: 730,
    peakToAverage: 1,
  },
  {
    id: 'rag',
    label: 'RAG / search',
    description: 'Retrieval, grounding and a sourced answer.',
    averageInputTokens: 4_000,
    averageOutputTokens: 800,
    targetTtftMs: 4_000,
    targetTpotMs: 70,
    activeHoursPerMonth: 730,
    peakToAverage: 1,
  },
  {
    id: 'agentic',
    label: 'Agentic / deep research',
    description: 'Long context, tools and reasoning-oriented responses.',
    averageInputTokens: 8_000,
    averageOutputTokens: 1_200,
    targetTtftMs: 5_000,
    targetTpotMs: 90,
    activeHoursPerMonth: 730,
    peakToAverage: 1,
  },
  {
    id: 'batch',
    label: 'Batch / offline',
    description: 'Asynchronous processing where throughput matters most.',
    averageInputTokens: 2_000,
    averageOutputTokens: 512,
    targetTtftMs: 10_000,
    targetTpotMs: 100,
    activeHoursPerMonth: 160,
    peakToAverage: 1,
  },
]

const DEFAULT_ASSUMPTIONS: CostAssumptions = {
  costLens: 'fully-loaded',
  cloudBillingMode: 'scale-to-zero',
  cloudRuntimeBufferPct: 10,
  planningCapacityUsePct: 90,
  hoursPerMonth: 730,
  analysisMonths: 36,
  loadedMonthlyCostPerFte: 18_000,
  hostedOperationsFte: 0.05,
  hostedImplementation: 15_000,
  rentedDirectInfrastructureMonthly: 1_800,
  rentedOperationsFte: 0.2,
  rentedImplementation: 40_000,
  hardwareLifeYears: 4,
  hardwareResidualPct: 20,
  annualCostOfCapitalPct: 8,
  annualMaintenancePct: 5,
  electricityPerKwh: 0.12,
  pue: 1.4,
  ownedBaseSystemPowerWattsPerServer: 450,
  ownedInstallationPerServer: 0,
  ownedFacilityMonthlyPerServer: 200,
  ownedDirectInfrastructureMonthly: 1_500,
  ownedOperationsFte: 0.25,
  ownedImplementation: 50_000,
  hostedFixedMonthly: 0,
  rentedFixedMonthly: 0,
  ownedFixedMonthly: 0,
}

// Capacity comparisons need a repeatable multi-request load. The original
// Hybrid Savings request sent the workload's (often tiny) peak request rate,
// so the returned throughput could describe offered demand rather than the
// sustainable capacity of a serving replica. Concurrency 32 is the
// AISimulators reference load used for recommendation
// examples and is high enough to exercise continuous batching while the SLA
// constraints still reject overloaded configurations.
const AISIMULATORS_CAPACITY_CONCURRENCY = 32

const formatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 })
const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
})
const preciseCurrencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

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

function modelTypeLabel(spec: ModelSpec | undefined): string {
  if (spec?.num_experts && spec.num_experts > 1) return 'Mixture of experts'
  if (spec?.architecture?.toLowerCase().includes('conditionalgeneration')) return 'Multimodal'
  return 'Dense model'
}

function modelSummary(modelId: string, spec: ModelSpec | undefined): string {
  const organization = modelId.includes('/') ? modelId.split('/')[0] : 'the catalogue provider'
  if (spec?.num_experts && spec.num_experts > 1) {
    const active = spec.num_experts_per_tok ? ` with ${spec.num_experts_per_tok} active per token` : ''
    return `Mixture-of-experts checkpoint from ${organization}: ${spec.num_experts} experts${active}.`
  }
  if (spec?.architecture?.toLowerCase().includes('conditionalgeneration')) {
    return `Multimodal checkpoint from ${organization} for text and visual inputs.`
  }
  return `Open-weight checkpoint from ${organization}, ready for live sizing after selection.`
}

function formatMonthlyUsage(value: number): string {
  if (value >= 1_000) return `$${compactNumber(value)}/mo`
  if (value >= 1) return `${currencyFormatter.format(value)}/mo`
  return `${preciseCurrencyFormatter.format(value)}/mo`
}

function normalizeModelId(value: string): string {
  return value.trim().toLowerCase()
}

function hostedProviderLabel(value: string): string {
  const labels: Record<string, string> = {
    openrouter: 'OpenRouter',
    ovhcloud: 'OVHcloud',
    deepinfra: 'DeepInfra',
    togetherai: 'Together AI',
  }
  return labels[value.trim().toLowerCase()] ?? value
}

function toHostedPrice(model: NonNullable<ReturnType<typeof resolveHostedPricing>['selected']>): HostedPrice {
  return {
    modelId: model.id,
    label: `${hostedProviderLabel(model.provider)} hosted API`,
    inputPerMillion: model.price_per_m_input,
    outputPerMillion: model.price_per_m_output,
  }
}

function hostedOfferKey(model: FrontierModel): string {
  return `${model.provider.trim().toLowerCase()}::${model.id.trim().toLowerCase()}::${model.source ?? ''}`
}

interface HostedOfferChoice {
  key: string
  provider: string
  monthlyCost: number
  cheapest: boolean
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

function candidateFromResult(
  result: RecommendResult,
  gpu: GpuOption,
  cloudOffer: RentedCloudOffer | null,
  purchaseConfiguration: ServerPurchaseConfiguration | null,
): InfrastructureCandidate {
  return {
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
    cloudInterconnect: cloudOffer?.interconnect ?? null,
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
    ttftMs: result.performance.ttftLatencyMs,
    tpotMs: result.performance.tpotMs,
    source: 'AISimulators estimate',
  }
}

function OptionCard({
  option,
  cheapest,
  lens,
  hostedOfferChoices,
  selectedHostedOfferKey,
  onSelectHostedOffer,
}: {
  option: CostOption
  cheapest: boolean
  lens: CostLens
  hostedOfferChoices?: HostedOfferChoice[]
  selectedHostedOfferKey?: string | null
  onSelectHostedOffer?: (key: string) => void
}) {
  const candidate = option.candidate
  const label = option.key === 'hosted'
    ? 'Hosted API'
    : option.key === 'rented'
      ? 'Rented infrastructure'
      : 'Purchased hardware'
  const hostedUsage = option.key === 'hosted'
    ? option.breakdown
      .filter(item => item.label === 'Input token usage' || item.label === 'Output token usage')
      .reduce((sum, item) => sum + item.monthlyCost, 0)
    : 0
  const hostedOther = Math.max(option.monthlyCost - hostedUsage, 0)

  return (
    <Card className={`${styles.resultCard} ${cheapest ? styles.resultCardWinner : ''}`}>
      <CardBody>
        <div className={styles.resultHeading}>
          <div>
            <span className={styles.eyebrow}>{label}</span>
            <h3>{option.label}</h3>
          </div>
          {cheapest && <Label color="green">Lowest modeled cost</Label>}
        </div>
        {option.key === 'hosted' && hostedOfferChoices && hostedOfferChoices.length > 0 && (
          <div className={styles.hostedProviderControl}>
            <span className={styles.hostedProviderPrompt}>Choose hosted API provider</span>
            <div className={styles.hostedProviderSelector} role="radiogroup" aria-label="Choose hosted API provider">
              {hostedOfferChoices.map(choice => {
                const selected = choice.key === selectedHostedOfferKey
                return (
                  <button
                    key={choice.key}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    className={selected ? styles.hostedProviderSelected : ''}
                    onClick={() => onSelectHostedOffer?.(choice.key)}
                  >
                    <strong>{choice.provider}</strong>
                    <small className={choice.cheapest ? styles.hostedProviderCheapest : undefined}>
                      {choice.cheapest ? 'Cheapest' : `${currencyFormatter.format(choice.monthlyCost)}/mo`}
                    </small>
                  </button>
                )
              })}
            </div>
          </div>
        )}
        <div className={styles.costValue}>{currencyFormatter.format(option.monthlyCost)}</div>
        <div className={styles.costPeriod}>
          {lens === 'fully-loaded' ? 'monthly full TCO' : 'monthly marginal cost'}
        </div>
        {option.key === 'hosted' && (
          <div className={styles.hostedCostSummary} aria-label="Hosted API monthly cost split">
            <span>
              <strong>{currencyFormatter.format(hostedUsage)}</strong>
              <small>token usage</small>
            </span>
            <i aria-hidden="true">+</i>
            <span>
              <strong>{currencyFormatter.format(hostedOther)}</strong>
              <small>{lens === 'fully-loaded' ? 'fees, people and implementation' : 'other recurring charges'}</small>
            </span>
          </div>
        )}
        <dl className={styles.metricList}>
          <div>
            <dt>Effective cost</dt>
            <dd>{preciseCurrencyFormatter.format(option.costPerMillionTokens)} / 1M tokens</dd>
          </div>
          <div>
            <dt>Average request</dt>
            <dd>${preciseRate(option.costPerRequest)}</dd>
          </div>
          {candidate && (
            <>
              <div>
                <dt>Required deployment</dt>
                <dd>{`${option.gpuCount} GPU${option.gpuCount === 1 ? '' : 's'} · ${option.replicas} replica${option.replicas === 1 ? '' : 's'}`}</dd>
              </div>
              {option.key === 'rented' && candidate.cloudGpusPerInstance != null && candidate.cloudGpusPerInstance > 1 && (
                <div>
                  <dt>Cloud billing unit</dt>
                  <dd>{candidate.cloudGpusPerInstance}-GPU instance minimum</dd>
                </div>
              )}
              <div>
                <dt>Capacity used</dt>
                <dd>{option.utilizationPct == null ? '—' : `${option.utilizationPct.toFixed(option.utilizationPct < 0.1 ? 3 : 1)}%`}</dd>
              </div>
              <div>
                <dt>Estimated performance</dt>
                <dd>{Math.round(candidate.ttftMs)} ms TTFT · {candidate.tpotMs.toFixed(1)} ms TPOT</dd>
              </div>
            </>
          )}
        </dl>
        {candidate && option.key === 'rented' && (
          <p className={styles.sourceLine}>
            {candidate.cloudProvider?.toUpperCase()} · {candidate.cloudInstanceName ?? 'cloud GPU instance'} ·{' '}
            {candidate.cloudRateKind === 'spot'
              ? 'spot/marketplace'
              : candidate.cloudRateKind === 'capacity_block'
                ? 'capacity block · always on'
                : 'on-demand'} ·{' '}
            {preciseCurrencyFormatter.format(candidate.cloudHourlyCostPerInstance ?? 0)}/whole instance-hour
            {candidate.cloudPriceSourceDate ? ` · source ${candidate.cloudPriceSourceDate}` : ''}
          </p>
        )}
        {candidate && option.key === 'owned' && (
          <p className={styles.sourceLine}>
            {preciseCurrencyFormatter.format(candidate.purchasePricePerReplica ?? 0)} / complete {candidate.purchaseGpusPerServer ?? candidate.gpusPerReplica}-GPU server
            {candidate.purchasePriceIndicative ? ' · indicative, quote required' : ''}
            {candidate.purchasePriceSourceDate ? ` · source ${candidate.purchasePriceSourceDate}` : ''}
          </p>
        )}
        <ExpandableSection toggleText="Monthly cost breakdown" className={styles.breakdownToggle}>
          <ul className={styles.breakdownList}>
            {option.breakdown.map(item => (
              <li key={item.label} className={lens === 'marginal' && !item.includedInMarginal ? styles.excludedCost : ''}>
                <span>{item.label}{lens === 'marginal' && !item.includedInMarginal ? ' · excluded' : ''}</span>
                <strong>{currencyFormatter.format(item.monthlyCost)}</strong>
              </li>
            ))}
          </ul>
        </ExpandableSection>
      </CardBody>
    </Card>
  )
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
          if (event.key === 'Enter') event.currentTarget.blur()
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
  const [modelSearch, setModelSearch] = React.useState('')
  const [profileId, setProfileId] = React.useState<WorkloadProfileId>('assistant')
  const [monthlyInputTokens, setMonthlyInputTokens] = React.useState(2_000_000_000)
  const [monthlyOutputTokens, setMonthlyOutputTokens] = React.useState(500_000_000)
  const [averageInputTokens, setAverageInputTokens] = React.useState(2_048)
  const [averageOutputTokens, setAverageOutputTokens] = React.useState(512)
  const [targetTtftMs, setTargetTtftMs] = React.useState(5_000)
  const [targetTpotMs, setTargetTpotMs] = React.useState(60)
  const [activeHoursPerMonth, setActiveHoursPerMonth] = React.useState(730)
  const [peakToAverage, setPeakToAverage] = React.useState(WORKLOAD_PROFILES[0].peakToAverage)
  const [assumptions, setAssumptions] = React.useState<CostAssumptions>(DEFAULT_ASSUMPTIONS)
  const [selectedHostedOfferKey, setSelectedHostedOfferKey] = React.useState<string | null>(null)
  const [workloadAdvancedOpen, setWorkloadAdvancedOpen] = React.useState(false)
  const [assumptionsOpen, setAssumptionsOpen] = React.useState(false)
  const [chartOpen, setChartOpen] = React.useState(false)
  const [candidates, setCandidates] = React.useState<InfrastructureCandidate[]>([])
  const [isSizing, setIsSizing] = React.useState(false)
  const [progress, setProgress] = React.useState({ completed: 0, total: 0 })
  const [sizingError, setSizingError] = React.useState<string | null>(null)
  const [failedSystems, setFailedSystems] = React.useState(0)
  const [lastSizingSignature, setLastSizingSignature] = React.useState<string | null>(null)
  const abortRef = React.useRef<AbortController | null>(null)
  const runIdRef = React.useRef(0)

  React.useEffect(() => {
    if (!hydrated || model || modelOptions.length === 0) return
    const configured = modelOptions.find(id => id === defaultModel)
    const qwenStartingPoint = modelOptions.find(id => normalizeModelId(id) === 'qwen/qwen3-8b')
    const firstPriced = modelOptions.find(id =>
      resolveHostedPricing(id, costings.models, 1, 1).selected !== null,
    )
    setModel(configured ?? qwenStartingPoint ?? firstPriced ?? modelOptions[0])
  }, [hydrated, model, modelOptions, defaultModel, costings.models])

  React.useEffect(() => () => abortRef.current?.abort(), [])

  const hostedPricing = React.useMemo(
    () => resolveHostedPricing(
      model,
      costings.models,
      monthlyInputTokens,
      monthlyOutputTokens,
    ),
    [model, costings.models, monthlyInputTokens, monthlyOutputTokens],
  )
  const hostedProviderOffers = React.useMemo(
    () => hostedPricing.providerMatches.slice(0, 3),
    [hostedPricing.providerMatches],
  )
  const hostedModel = hostedProviderOffers.find(
    offer => hostedOfferKey(offer) === selectedHostedOfferKey,
  ) ?? hostedProviderOffers[0] ?? hostedPricing.selected
  const hostedPrice = React.useMemo(
    () => hostedModel ? toHostedPrice(hostedModel) : null,
    [hostedModel],
  )
  React.useEffect(() => {
    setSelectedHostedOfferKey(null)
  }, [model, monthlyInputTokens, monthlyOutputTokens])
  const selectedModelSpec = modelSpecs.get(model)
  const selectedProfile = WORKLOAD_PROFILES.find(profile => profile.id === profileId) ?? WORKLOAD_PROFILES[0]
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
  const requestMixMismatch = React.useMemo(() => {
    const monthlyTokens = monthlyInputTokens + monthlyOutputTokens
    const averageTokens = averageInputTokens + averageOutputTokens
    if (monthlyTokens <= 0 || averageTokens <= 0) return false
    return Math.abs(
      monthlyInputTokens / monthlyTokens - averageInputTokens / averageTokens,
    ) > 0.01
  }, [monthlyInputTokens, monthlyOutputTokens, averageInputTokens, averageOutputTokens])
  const hostedOfferChoices = React.useMemo<HostedOfferChoice[]>(
    () => hostedProviderOffers.map((offer, index) => ({
      key: hostedOfferKey(offer),
      provider: hostedProviderLabel(offer.provider),
      monthlyCost: hostedCostAtVolume(
        workload,
        toHostedPrice(offer),
        assumptions,
        facts.monthlyTokens,
      ).monthlyCost,
      cheapest: index === 0,
    })),
    [hostedProviderOffers, workload, assumptions, facts.monthlyTokens],
  )
  const featuredModels = React.useMemo(() => {
    const preferredIds = [
      'qwen/qwen3-8b',
      'qwen/qwen3-32b',
      'qwen/qwen3-235b-a22b',
    ]
    const targets = [8, 32, 235]
    const picked = preferredIds.map((preferredId, index) => {
      const preferred = modelOptions.find(id => normalizeModelId(id) === preferredId)
      if (preferred) return preferred
      return [...modelOptions]
        .filter(id => modelParameterBillions(id) !== null)
        .sort((left, right) => (
          Math.abs((modelParameterBillions(left) ?? targets[index]) - targets[index])
          - Math.abs((modelParameterBillions(right) ?? targets[index]) - targets[index])
        ))[0]
    }).filter((id): id is string => Boolean(id))

    if (model && !picked.includes(model)) {
      const tierIndex = modelTierLabel(model) === 'Small model'
        ? 0
        : modelTierLabel(model) === 'Medium model'
          ? 1
          : 2
      picked[tierIndex] = model
    }
    return picked.filter((id, index) => picked.indexOf(id) === index)
  }, [model, modelOptions])
  const modelMatches = React.useMemo(() => {
    const query = normalizeModelId(modelSearch)
    if (!query) return featuredModels
    return modelOptions.filter(id => {
      const spec = modelSpecs.get(id)
      return [id, id.split('/').pop(), modelSizeLabel(id), modelTypeLabel(spec), spec?.architecture]
        .filter((value): value is string => Boolean(value))
        .some(value => normalizeModelId(value).includes(query))
    })
  }, [featuredModels, modelOptions, modelSearch, modelSpecs])
  const visibleModels = React.useMemo(
    () => modelMatches.slice(0, modelSearch.trim() ? 24 : 3),
    [modelMatches, modelSearch],
  )
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
    inferenceBackend,
  ])
  const isStale = lastSizingSignature !== null && lastSizingSignature !== sizingSignature
  const comparison = React.useMemo(() => {
    if (!hostedPrice || candidates.length === 0 || lastSizingSignature === null || isStale) return null
    return calculateHybridComparison(workload, hostedPrice, candidates, assumptions)
  }, [workload, hostedPrice, candidates, assumptions, lastSizingSignature, isStale])

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

  const applyProfile = (nextId: WorkloadProfileId) => {
    const profile = WORKLOAD_PROFILES.find(item => item.id === nextId) ?? WORKLOAD_PROFILES[0]
    setProfileId(nextId)
    setAverageInputTokens(profile.averageInputTokens)
    setAverageOutputTokens(profile.averageOutputTokens)
    setTargetTtftMs(profile.targetTtftMs)
    setTargetTpotMs(profile.targetTpotMs)
    setActiveHoursPerMonth(profile.activeHoursPerMonth)
    setPeakToAverage(profile.peakToAverage)
  }

  const updateAssumption = <K extends keyof CostAssumptions>(key: K, value: CostAssumptions[K]) => {
    setAssumptions(previous => ({ ...previous, [key]: value }))
  }

  const runSizing = async () => {
    if (!hostedPrice || facts.monthlyTokens <= 0 || gpuOptions.length === 0) return
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    const runId = runIdRef.current + 1
    runIdRef.current = runId
    const signature = sizingSignature
    const results: InfrastructureCandidate[] = []
    let failures = 0
    let nextIndex = 0

    setIsSizing(true)
    setSizingError(null)
    setCandidates([])
    setFailedSystems(0)
    setProgress({ completed: 0, total: gpuOptions.length })

    const worker = async () => {
      while (nextIndex < gpuOptions.length && !controller.signal.aborted) {
        const gpu = gpuOptions[nextIndex]
        nextIndex += 1
        try {
          const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'text/event-stream',
            },
            signal: controller.signal,
            body: JSON.stringify({
              model_path: model,
              system: gpu.systemId,
              backend: inferenceBackend,
              isl: Math.max(Math.round(averageInputTokens), 1),
              osl: Math.max(Math.round(averageOutputTokens), 1),
              ttft: Math.max(targetTtftMs, 1),
              tpot: Math.max(targetTpotMs, 1),
              // Use a stable, SLA-constrained capacity load rather than the
              // workload's current request rate. The latter can measure only
              // sparse offered demand, not serving capacity. The response is
              // normalized to one replica before the cost engine scales it.
              target_concurrency: AISIMULATORS_CAPACITY_CONCURRENCY,
              top_n: 5,
            }),
          })
          const data = await readRecommendStream(response)
          if (data.status === 'completed') {
            const cloudOffers = resolveRentedCloudOffers(
              gpu.systemId,
              preferredCloudProvider,
            )
            const purchaseConfigurations = resolveCompatibleServerPurchaseConfigurations(
              gpu.systemId,
              data.recommendation.gpusPerReplica,
            )
            if (cloudOffers.length > 0) {
              results.push(...cloudOffers.map(cloudOffer => (
                candidateFromResult(data, gpu, cloudOffer, null)
              )))
            }
            if (purchaseConfigurations.length > 0) {
              results.push(...purchaseConfigurations.map(configuration => (
                candidateFromResult(data, gpu, null, configuration)
              )))
            }
            if (cloudOffers.length === 0 && purchaseConfigurations.length === 0) {
              // Retain successfully sized systems even when pricing is absent.
              // Cost formulas ignore null prices, while the UI can accurately
              // distinguish sizing support from cost-ready coverage.
              results.push(candidateFromResult(data, gpu, null, null))
            }
          } else {
            failures += 1
          }
        } catch (error: unknown) {
          if (error instanceof DOMException && error.name === 'AbortError') return
          failures += 1
        } finally {
          if (!controller.signal.aborted && runIdRef.current === runId) {
            setProgress(previous => ({ ...previous, completed: previous.completed + 1 }))
          }
        }
      }
    }

    await Promise.all(Array.from({ length: Math.min(4, gpuOptions.length) }, worker))
    if (controller.signal.aborted || runIdRef.current !== runId) return

    setCandidates(results)
    setFailedSystems(failures)
    setLastSizingSignature(signature)
    setIsSizing(false)
    if (results.length === 0) {
      setSizingError('AISimulators could not find a priced configuration that meets this model and workload target.')
    }
  }

  const invalidWorkload =
    facts.monthlyTokens <= 0 ||
    facts.monthlyTokens > HYBRID_PLANNING_HORIZON_TOKENS ||
    averageInputTokens <= 0 ||
    averageOutputTokens <= 0 ||
    activeHoursPerMonth <= 0 ||
    activeHoursPerMonth > assumptions.hoursPerMonth ||
    peakToAverage < 1 ||
    targetTtftMs <= 0 ||
    targetTpotMs <= 0

  if (hydrated && !costingsEnabled) {
    return (
      <div className={styles.page}>
        <header className={styles.header}>
          <h1 className={styles.pageTitle}>Hybrid savings</h1>
          <p className={styles.subtitle}>Compare hosted API, rented GPU and purchased hardware costs for one workload.</p>
        </header>
        <Alert title="Enable experimental costings" variant="info" isInline>
          Hybrid Savings uses the shared costing catalogue. Enable experimental costings in Settings, then return here.
          <div className={styles.alertAction}><Button component="a" href="/settings" variant="link" isInline>Open Settings</Button></div>
        </Alert>
      </div>
    )
  }

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div>
          <span className={styles.pageEyebrow}>Hybrid cost estimator</span>
          <h1 className={styles.pageTitle}>Which deployment option costs less?</h1>
          <p className={styles.subtitle}>
            Compare one open-weight model across a hosted API, rented GPUs and purchased hardware.
          </p>
        </div>
        <Label color="blue" isCompact>Planning estimate</Label>
      </header>

      {(catalogError || costings.error) && (
        <Alert title="Some shared data could not be loaded" variant="warning" isInline>
          {catalogError ?? costings.error}. Refresh the page or check the Sources page before using the result.
        </Alert>
      )}

      <Form className={styles.workflow}>
        <Card className={styles.stepCard}>
          <CardBody>
            <div className={styles.stepIntro}>
              <span className={styles.stepNumber}>01</span>
              <div>
                <h2>Enter the monthly workload</h2>
                <p>Start with the token usage you already know. ConfigIQ derives the request demand used for GPU sizing.</p>
              </div>
            </div>

            <div className={styles.workloadGrid}>
              <NumberField id="hybrid-input-tokens" label="Input tokens per month" value={monthlyInputTokens} min={0} max={HYBRID_PLANNING_HORIZON_TOKENS} step={1_000_000} formatWithCommas onChange={setMonthlyInputTokens} />
              <NumberField id="hybrid-output-tokens" label="Output tokens per month" value={monthlyOutputTokens} min={0} max={HYBRID_PLANNING_HORIZON_TOKENS} step={1_000_000} formatWithCommas onChange={setMonthlyOutputTokens} />
              <div className={styles.totalUsage}>
                <span>Total monthly usage</span>
                <strong>{compactNumber(facts.monthlyTokens)}</strong>
                <small>billed tokens</small>
              </div>
            </div>

            <p className={styles.workloadExplanation}>
              The {selectedProfile.label.toLowerCase()} profile implies approximately <strong>{compactNumber(facts.monthlyRequests)} requests/month</strong>. Your entered token totals remain the hosted billing basis.
            </p>

            {requestMixMismatch && (
              <Alert title="Token mix differs from the average request" variant="info" isInline>
                GPU sizing uses the higher request count implied by the input or output volume so no work is omitted. Align the monthly and average-request ratios for a tighter estimate.
              </Alert>
            )}

            <div className={styles.assumptionStrip}>
              <div><span>Workload shape</span><strong>{selectedProfile.label}</strong></div>
              <div><span>Processing window</span><strong>{formatter.format(activeHoursPerMonth)} hours/month</strong></div>
              <div><span>Peak demand</span><strong>{peakToAverage.toFixed(1)}× average</strong></div>
              <div><span>Rented compute</span><strong>{billingModeLabel(assumptions.cloudBillingMode)}</strong></div>
            </div>

            <ExpandableSection
              isExpanded={workloadAdvancedOpen}
              onToggle={(_event, expanded) => setWorkloadAdvancedOpen(expanded)}
              toggleText={workloadAdvancedOpen ? 'Hide advanced workload' : 'Advanced workload'}
              className={styles.inlineExpandable}
            >
              <div className={styles.advancedLead}>Change these values only when you have workload measurements or a clear response target.</div>
              <div className={styles.advancedGrid}>
                <FormGroup label="Workload shape" fieldId="hybrid-profile">
                  <FormSelect id="hybrid-profile" value={profileId} onChange={(_event, value) => applyProfile(value as WorkloadProfileId)}>
                    {WORKLOAD_PROFILES.map(profile => <FormSelectOption key={profile.id} value={profile.id} label={profile.label} />)}
                  </FormSelect>
                  <FormHelperText><HelperText><HelperTextItem>{selectedProfile.description}</HelperTextItem></HelperText></FormHelperText>
                </FormGroup>
                <NumberField id="hybrid-average-input" label="Average input tokens per request" value={averageInputTokens} min={1} onChange={setAverageInputTokens} />
                <NumberField id="hybrid-average-output" label="Average output tokens per request" value={averageOutputTokens} min={1} onChange={setAverageOutputTokens} />
                <NumberField id="hybrid-ttft" label="Maximum TTFT (ms)" value={targetTtftMs} min={1} onChange={setTargetTtftMs} />
                <NumberField id="hybrid-tpot" label="Maximum TPOT (ms/token)" value={targetTpotMs} min={1} onChange={setTargetTpotMs} />
                <NumberField id="hybrid-active-hours" label="Processing hours per month" value={activeHoursPerMonth} min={1} max={assumptions.hoursPerMonth} onChange={setActiveHoursPerMonth} />
                <NumberField id="hybrid-peak" label="Peak demand multiplier" value={peakToAverage} min={1} step={0.1} onChange={setPeakToAverage} />
                <FormGroup label="Rented billing policy" fieldId="hybrid-billing-mode">
                  <FormSelect id="hybrid-billing-mode" value={assumptions.cloudBillingMode} onChange={(_event, value) => updateAssumption('cloudBillingMode', value as CloudBillingMode)}>
                    <FormSelectOption value="scale-to-zero" label="Scale to zero — pay for serving time" />
                    <FormSelectOption value="active-window" label="Active window — keep required GPUs warm" />
                    <FormSelectOption value="always-on" label="Always on — full month" />
                  </FormSelect>
                </FormGroup>
              </div>
            </ExpandableSection>
          </CardBody>
        </Card>

        <Card className={styles.stepCard}>
          <CardBody>
            <div className={styles.stepIntro}>
              <span className={styles.stepNumber}>02</span>
              <div>
                <h2>Select a model from the live catalogue</h2>
                <p>Choose the model you want to evaluate. We will use its supported configuration to size the hardware in the next step.</p>
              </div>
            </div>
            <section className={styles.modelCatalogue} aria-label="Live model catalogue">
              <div className={styles.modelCatalogueControls}>
                <FormGroup label="Search supported models" fieldId="hybrid-model-search">
                  <TextInput
                    id="hybrid-model-search"
                    type="search"
                    value={modelSearch}
                    onChange={(_event, value) => setModelSearch(value)}
                    placeholder="Search Qwen, Llama, DeepSeek or another model"
                    aria-label="Search supported models"
                  />
                </FormGroup>
                <div className={styles.liveCatalogueStatus} aria-live="polite">
                  <span>{catalogLoading ? 'Loading live catalogue' : 'Live AISimulators catalogue'}</span>
                  <strong>
                    {catalogLoading
                      ? 'Checking supported models and GPUs…'
                      : `${modelOptions.length} models · ${gpuOptions.length} GPU systems`}
                  </strong>
                  {!catalogLoading && <small>Search any supported checkpoint; select a card to compare it.</small>}
                </div>
              </div>
              <div className={styles.modelGrid}>
                {visibleModels.map(candidateId => {
                  const spec = modelSpecs.get(candidateId)
                  const pricing = resolveHostedPricing(
                    candidateId,
                    costings.models,
                    monthlyInputTokens,
                    monthlyOutputTokens,
                  )
                  const selectedPrice = pricing.selected
                  const selected = candidateId === model
                  const isTestedModel = isModelListedAsTested(candidateId, testedModels)
                  return (
                    <Card
                      component="button"
                      type="button"
                      isClickable
                      isSelectable
                      isSelected={selected}
                      className={styles.modelCard}
                      key={candidateId}
                      onClick={() => setModel(candidateId)}
                      aria-pressed={selected}
                      aria-label={`Select ${candidateId.split('/').pop() ?? candidateId}`}
                    >
                      <CardBody>
                        <div className={styles.modelCardTop}>
                          <span className={styles.modelSize}>{modelSizeLabel(candidateId)}</span>
                          <Label
                            color={isTestedModel ? 'blue' : 'green'}
                            isCompact
                            title={isTestedModel
                              ? "Listed as tested in ConfigIQ's app config; the live catalogue doesn't report system-specific test status."
                              : 'Available in the live AISimulators catalogue'}
                          >
                            {isTestedModel ? 'Tested' : 'In catalog'}
                          </Label>
                        </div>
                        {!modelSearch.trim() && <span className={styles.modelTier}>Typical {modelTierLabel(candidateId).toLowerCase()}</span>}
                        <h3>{candidateId.split('/').pop() ?? candidateId}</h3>
                        <p>{modelSummary(candidateId, spec)}</p>
                        <code>{candidateId}</code>
                        <div className={styles.modelCardMeta}>
                          {modelTypeLabel(spec)} · {spec?.context_length ? `${compactNumber(spec.context_length)} context` : 'context not listed'} · sizing checked after selection
                        </div>
                        <div className={styles.modelCardCost}>
                          <div>
                            <span>Hosted API</span>
                            <strong>
                              {costings.isLoading
                                ? 'Checking…'
                                : selectedPrice && pricing.selectedMonthlyUsageCost !== null
                                  ? formatMonthlyUsage(pricing.selectedMonthlyUsageCost)
                                  : 'Price unavailable'}
                            </strong>
                          </div>
                          <small>
                            {selectedPrice
                              ? `${preciseCurrencyFormatter.format(selectedPrice.price_per_m_input)} input · ${preciseCurrencyFormatter.format(selectedPrice.price_per_m_output)} output / 1M · ${selectedPrice.provider}${pricing.matches.length > 1 ? ` · ${pricing.matches.length} offers` : ''}`
                              : 'No exact hosted offer for this checkpoint'}
                          </small>
                          <small>{compactNumber(facts.monthlyTokens)} billed tokens/month</small>
                        </div>
                      </CardBody>
                    </Card>
                  )
                })}
              </div>
              {!catalogLoading && visibleModels.length === 0 && (
                <p className={styles.emptyCatalogue}>No model in the current AISimulators catalogue matches this search.</p>
              )}
              {!catalogLoading && modelMatches.length > visibleModels.length && (
                <p className={styles.catalogueCount}>
                  Showing {visibleModels.length} of {modelMatches.length} matching models. Refine the search to find a specific checkpoint.
                </p>
              )}
              {catalogError && <Alert title="Live catalogue unavailable" variant="warning" isInline>{catalogError}</Alert>}
              {!catalogLoading && model && (
                <p className={styles.selectedModelNote}>
                  <strong>Selected:</strong> {model.split('/').pop()} · {selectedModelSpec?.context_length ? `${compactNumber(selectedModelSpec.context_length)} token context` : 'live catalogue checkpoint'}
                  {hostedPricing.selected ? ` · ${hostedProviderLabel(hostedPricing.selected.provider)} is the lowest-priced of ${hostedPricing.providerMatches.length} compatible hosted provider${hostedPricing.providerMatches.length === 1 ? '' : 's'} for this workload.` : ' · hosted price unavailable.'}
                  {hostedModel && hostedPricing.selected && hostedOfferKey(hostedModel) !== hostedOfferKey(hostedPricing.selected) ? ` Comparing ${hostedProviderLabel(hostedModel.provider)}.` : ''}
                </p>
              )}
            </section>
          </CardBody>
        </Card>

        <Card className={styles.stepCard}>
          <CardBody>
            <div className={styles.stepIntro}>
              <span className={styles.stepNumber}>03</span>
              <div><h2>Let ConfigIQ size the infrastructure</h2><p>AISimulators checks model fit and performance. ConfigIQ selects the lowest-cost eligible rented and purchased setup independently.</p></div>
            </div>
            <div className={styles.sizingChoice}>
              <div className={styles.sizingIcon} aria-hidden="true">GPU</div>
              <div>
                <span className={styles.eyebrow}>Recommended</span>
                <strong>Choose the best supported setup</strong>
                <p>
                  {costings.isLoading || catalogLoading
                    ? 'Loading the live model, GPU and pricing catalogues…'
                    : `Check all ${gpuOptions.length} supported GPU systems. ${rentedReadyGpus.length} have stable whole-instance rented prices and ${purchaseReadyGpus.length} have complete-server purchase prices; ${Math.max(gpuOptions.length - priceReadyGpus.length, 0)} are sizing-only. Spot-only offers are excluded from the default comparison.`}
                </p>
              </div>
              <Button variant="primary" onClick={runSizing} isDisabled={isSizing || invalidWorkload || hostedPrice === null || gpuOptions.length === 0 || costings.isLoading || catalogLoading} isLoading={isSizing}>
                {isSizing ? 'Calculating forecast' : candidates.length > 0 ? 'Refresh forecast' : 'Calculate forecast'}
              </Button>
            </div>
            {isSizing && (
              <div className={styles.loadingPanel} aria-live="polite">
                <div className={styles.loadingHeading}><Spinner size="md" /><div><strong>Checking supported GPU systems</strong><span>{progress.completed} of {progress.total} complete</span></div></div>
                <Progress value={progress.total > 0 ? progress.completed / progress.total * 100 : 0} size="sm" aria-label="GPU sizing progress" />
              </div>
            )}
            {invalidWorkload && <Alert title="Review the workload inputs" variant="warning" isInline>The combined monthly token total must be positive and no more than 1T, performance targets must be positive, processing hours must be between 1 and {assumptions.hoursPerMonth}, and peak demand must be at least 1× average.</Alert>}
            {!costings.isLoading && hostedModel === null && model && <Alert title="Hosted comparison unavailable" variant="warning" isInline>No exact hosted API price was found for this checkpoint in the selected pricing feed. The app does not silently substitute a base or differently quantized model.</Alert>}
            {!costings.isLoading && costingInputsStale && (
              <Alert title="Some pricing inputs are stale" variant="warning" isInline>
                The comparison may use the last successfully collected prices
                {staleCostingSources.length > 0 ? ` for ${staleCostingSources.join(', ')}` : ''}. Validate current rates before a purchasing decision.
              </Alert>
            )}
            {isStale && !isSizing && <Alert title="Inputs changed" variant="info" isInline>Refresh the forecast so AISimulators can size the updated workload.</Alert>}
            {sizingError && <Alert title="Comparison unavailable" variant="danger" isInline>{sizingError}</Alert>}
          </CardBody>
        </Card>
      </Form>

      {comparison && (
        <section className={styles.resultsSection} aria-live="polite">
          <Card className={styles.resultShell}>
            <CardBody>
              <div className={styles.resultSummaryHead}>
                <div className={styles.stepIntro}>
                  <span className={styles.stepNumber}>04</span>
                  <div><span className={styles.eyebrow}>Lowest planning cost</span><h2>{comparison.cheapest?.key === 'hosted' ? 'Hosted API' : comparison.cheapest?.key === 'rented' ? 'Rented infrastructure' : 'Purchased hardware'}</h2><p>{compactNumber(comparison.monthlyTokens)} billed tokens per month · {model.split('/').pop()}</p></div>
                </div>
                <div className={styles.lensControl}>
                  <span>Compare using</span>
                  <ToggleGroup aria-label="Cost comparison view">
                    <ToggleGroupItem text="Full TCO" isSelected={assumptions.costLens === 'fully-loaded'} onChange={() => updateAssumption('costLens', 'fully-loaded')} />
                    <ToggleGroupItem text="Marginal" isSelected={assumptions.costLens === 'marginal'} onChange={() => updateAssumption('costLens', 'marginal')} />
                  </ToggleGroup>
                  <small>{assumptions.costLens === 'fully-loaded' ? 'People, implementation and recurring costs included' : 'Operating costs only; acquisition and initial implementation are treated as already committed'}</small>
                </div>
              </div>
              <p className={styles.comparisonFrame}>
                <strong>Comparison basis:</strong> {hostedModel ? hostedProviderLabel(hostedModel.provider) : 'Hosted provider'} · {assumptions.costLens === 'fully-loaded' ? 'Full TCO' : 'Marginal cost'} · rented: {billingModeLabel(assumptions.cloudBillingMode)} · purchased power: full TDP for {formatter.format(assumptions.hoursPerMonth)} h/month · {formatter.format(peakToAverage)}× peak · {formatter.format(assumptions.planningCapacityUsePct)}% planning capacity · no failover reserve
              </p>
              <div className={styles.resultGrid}>
                {comparison.options.map(option => (
                  <OptionCard
                    key={option.key}
                    option={option}
                    cheapest={comparison.cheapest?.key === option.key}
                    lens={assumptions.costLens}
                    hostedOfferChoices={option.key === 'hosted' ? hostedOfferChoices : undefined}
                    selectedHostedOfferKey={option.key === 'hosted' && hostedModel ? hostedOfferKey(hostedModel) : null}
                    onSelectHostedOffer={option.key === 'hosted' ? setSelectedHostedOfferKey : undefined}
                  />
                ))}
                {!comparison.options.some(option => option.key === 'rented') && <Card className={styles.unavailableCard}><CardBody><strong>Rented infrastructure</strong><p>No sized GPU option has a usable cloud rate.</p></CardBody></Card>}
                {!comparison.options.some(option => option.key === 'owned') && <Card className={styles.unavailableCard}><CardBody><strong>Purchased hardware</strong><p>No sized GPU option has a usable purchase price.</p></CardBody></Card>}
              </div>
              <p className={styles.effectiveCostNote}>Effective cost divides the selected monthly cost view by billed tokens. For self-managed options it includes unused capacity in the required whole-GPU deployment.</p>
            </CardBody>
          </Card>

          <Card className={styles.comparisonShell}>
            <CardBody>
              <div className={styles.currentWorkloadHeading}>
                <div><span className={styles.eyebrow}>Current workload</span><h2>{model.split('/').pop()}</h2></div>
                <div className={styles.evidenceSummary}>
                  <Badge>{sizedSystemCount} systems sized</Badge>
                  <span>{rentedReadyGpus.length} rented-ready</span>
                  <span>{purchaseReadyGpus.length} purchase-ready</span>
                  {failedSystems > 0 && <span>{failedSystems} could not be sized</span>}
                </div>
              </div>
              <div className={styles.workloadFlow} aria-label="Current workload calculation">
                <div className={`${styles.workloadStep} ${styles.workloadStepModel}`}><span>Selected model</span><strong>{model.split('/').pop()}</strong></div><b aria-hidden="true">→</b>
                <div className={styles.workloadStep}><span>Input usage</span><strong>{compactNumber(facts.monthlyInputTokens)}</strong><small>tokens / month</small></div><b aria-hidden="true">→</b>
                <div className={styles.workloadStep}><span>Output usage</span><strong>{compactNumber(facts.monthlyOutputTokens)}</strong><small>tokens / month</small></div><b aria-hidden="true">→</b>
                <div className={`${styles.workloadStep} ${styles.workloadStepTotal}`}><span>Billed usage</span><strong>{compactNumber(facts.monthlyTokens)}</strong><small>tokens / month</small></div>
              </div>
              <p className={styles.workloadProfileNote}><strong>{selectedProfile.label} sizing:</strong> {compactNumber(facts.averageTokensPerRequest)} average tokens per request, {formatter.format(targetTpotMs)} ms maximum TPOT and {preciseRate(facts.peakRequestsPerSecond)} peak requests/second. AISimulators measures serving capacity at {AISIMULATORS_CAPACITY_CONCURRENCY} concurrent requests while enforcing the latency targets.</p>
              <div className={styles.crossoverGrid}>
                <div><span>Rented transition</span><strong>{!comparison.transitionsVerified ? 'Transition not verified for this workload' : comparison.rentedLowestCostTokens === null ? 'Not the lowest-cost option by 1T tokens/month' : `First becomes the lowest-cost option at ≈ ${compactNumber(comparison.rentedLowestCostTokens)} tokens/month`}</strong></div>
                <div><span>Purchased transition</span><strong>{!comparison.transitionsVerified ? 'Transition not verified for this workload' : comparison.ownedLowestCostTokens === null ? 'Not the lowest-cost option by 1T tokens/month' : `First becomes the lowest-cost option at ≈ ${compactNumber(comparison.ownedLowestCostTokens)} tokens/month`}</strong></div>
              </div>
              <ExpandableSection isExpanded={chartOpen} onToggle={(_event, expanded) => setChartOpen(expanded)} toggleText={chartOpen ? 'Hide comparison chart' : 'Show comparison chart'} className={styles.chartToggle}>
                <CostComparisonChart
                  points={comparison.chartPoints}
                  currentTokens={comparison.monthlyTokens}
                  costLens={assumptions.costLens}
                  rentedBreakEvenTokens={comparison.rentedBreakEvenTokens}
                  ownedBreakEvenTokens={comparison.ownedBreakEvenTokens}
                  rentedLowestCostTokens={comparison.rentedLowestCostTokens}
                  ownedLowestCostTokens={comparison.ownedLowestCostTokens}
                  transitionsVerified={comparison.transitionsVerified}
                />
              </ExpandableSection>
            </CardBody>
          </Card>
        </section>
      )}

      <Card className={styles.assumptionsCard}>
        <CardBody>
          <ExpandableSection isExpanded={assumptionsOpen} onToggle={(_event, expanded) => setAssumptionsOpen(expanded)} toggleText={assumptionsOpen ? 'Hide advanced assumptions' : 'Advanced assumptions'}>
            <div className={styles.assumptionsSummary}><span>Current planning basis</span><strong>{billingModeLabel(assumptions.cloudBillingMode)} · {assumptions.planningCapacityUsePct}% planning capacity · {assumptions.hardwareLifeYears}-year hardware life · {preciseCurrencyFormatter.format(assumptions.loadedMonthlyCostPerFte)}/FTE-month</strong></div>
            <div className={styles.assumptionGroups}>
              <section><h3>Hosted API</h3><p>Vendor usage plus internal adoption and management costs.</p><div className={styles.advancedGrid}>
                <NumberField id="hosted-fixed" label="Fixed fees per month" value={assumptions.hostedFixedMonthly} onChange={value => updateAssumption('hostedFixedMonthly', value)} />
                <NumberField id="hosted-ops" label="Operations FTE" value={assumptions.hostedOperationsFte} step={0.01} onChange={value => updateAssumption('hostedOperationsFte', value)} />
                <NumberField id="hosted-implementation" label="One-time implementation" value={assumptions.hostedImplementation} onChange={value => updateAssumption('hostedImplementation', value)} />
              </div></section>
              <section><h3>Rented infrastructure</h3><p>Cloud compute, supporting infrastructure and model operations.</p><div className={styles.advancedGrid}>
                <NumberField id="runtime-buffer" label="Scale-to-zero runtime buffer (%)" value={assumptions.cloudRuntimeBufferPct} onChange={value => updateAssumption('cloudRuntimeBufferPct', value)} />
                <NumberField id="rented-infra" label="Fallback infrastructure per month" value={assumptions.rentedDirectInfrastructureMonthly} onChange={value => updateAssumption('rentedDirectInfrastructureMonthly', value)} />
                <NumberField id="rented-ops" label="Operations FTE" value={assumptions.rentedOperationsFte} step={0.01} onChange={value => updateAssumption('rentedOperationsFte', value)} />
                <NumberField id="rented-implementation" label="One-time implementation" value={assumptions.rentedImplementation} onChange={value => updateAssumption('rentedImplementation', value)} />
                <NumberField id="rented-other" label="Other monthly costs" value={assumptions.rentedFixedMonthly} onChange={value => updateAssumption('rentedFixedMonthly', value)} />
              </div></section>
              <section><h3>Purchased hardware</h3><p>Capital, power, facilities, implementation and platform operations.</p><div className={styles.advancedGrid}>
                <NumberField id="hardware-life" label="Hardware life (years)" value={assumptions.hardwareLifeYears} min={1} onChange={value => updateAssumption('hardwareLifeYears', value)} />
                <NumberField id="hardware-residual" label="Residual value (%)" value={assumptions.hardwareResidualPct} max={100} onChange={value => updateAssumption('hardwareResidualPct', Math.min(value, 100))} />
                <NumberField id="capital-cost" label="Annual cost of capital (%)" value={assumptions.annualCostOfCapitalPct} onChange={value => updateAssumption('annualCostOfCapitalPct', value)} />
                <NumberField id="maintenance" label="Annual maintenance (%)" value={assumptions.annualMaintenancePct} onChange={value => updateAssumption('annualMaintenancePct', value)} />
                <NumberField id="electricity" label="Electricity ($ / kWh)" value={assumptions.electricityPerKwh} step={0.01} onChange={value => updateAssumption('electricityPerKwh', value)} />
                <NumberField id="pue" label="Power usage effectiveness" value={assumptions.pue} min={1} step={0.1} onChange={value => updateAssumption('pue', value)} />
                <NumberField id="base-server-power" label="Base server power per server (W)" value={assumptions.ownedBaseSystemPowerWattsPerServer} onChange={value => updateAssumption('ownedBaseSystemPowerWattsPerServer', value)} />
                <NumberField id="installation" label="Additional installation per server" value={assumptions.ownedInstallationPerServer} onChange={value => updateAssumption('ownedInstallationPerServer', value)} />
                <NumberField id="facility" label="Facilities per server / month" value={assumptions.ownedFacilityMonthlyPerServer} onChange={value => updateAssumption('ownedFacilityMonthlyPerServer', value)} />
                <NumberField id="owned-infra" label="Shared infrastructure and platform / month" value={assumptions.ownedDirectInfrastructureMonthly} onChange={value => updateAssumption('ownedDirectInfrastructureMonthly', value)} />
                <NumberField id="owned-ops" label="Operations FTE" value={assumptions.ownedOperationsFte} step={0.01} onChange={value => updateAssumption('ownedOperationsFte', value)} />
                <NumberField id="owned-implementation" label="One-time implementation" value={assumptions.ownedImplementation} onChange={value => updateAssumption('ownedImplementation', value)} />
                <NumberField id="owned-other" label="Other monthly costs" value={assumptions.ownedFixedMonthly} onChange={value => updateAssumption('ownedFixedMonthly', value)} />
              </div></section>
              <section><h3>Shared planning inputs</h3><p>Applied consistently across the three deployment paths.</p><div className={styles.advancedGrid}>
                <NumberField id="planning-capacity" label="Planning capacity use (%)" value={assumptions.planningCapacityUsePct} min={1} max={100} onChange={value => updateAssumption('planningCapacityUsePct', Math.min(value, 100))} />
                <NumberField id="analysis-months" label="Analysis period (months)" value={assumptions.analysisMonths} min={1} onChange={value => updateAssumption('analysisMonths', value)} />
                <NumberField id="loaded-fte" label="Loaded cost per FTE-month" value={assumptions.loadedMonthlyCostPerFte} onChange={value => updateAssumption('loadedMonthlyCostPerFte', value)} />
              </div></section>
            </div>
          </ExpandableSection>
          {!assumptionsOpen && <div className={styles.closedAssumptionSummary}><span>Prices, runtime, labour, power and hardware lifecycle</span><strong>{billingModeLabel(assumptions.cloudBillingMode)} · {assumptions.planningCapacityUsePct}% capacity · {assumptions.hardwareLifeYears}-year life · {assumptions.analysisMonths}-month analysis</strong></div>}
        </CardBody>
      </Card>

      <aside className={styles.planningNote} aria-label="Planning guidance">
        <strong>Planning estimate</strong>
        <p>
          AISimulators sizing is estimated. “Tested” listings have real-hardware benchmark evidence, but may not match your workload. Catalogue prices can differ from negotiated rates—validate performance and pricing before production.
        </p>
      </aside>
    </main>
  )
}

'use client'

import * as React from 'react'
import Link from 'next/link'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionToggle,
  Alert,
  Button,
  FormSelect,
  FormSelectOption,
  PageSection,
  Spinner,
  Switch,
  TextInput,
} from '@patternfly/react-core'
import { ComboBox, type ComboBoxItem } from '@/components/ModelComboBox/ModelComboBox'
import { GpuSystemInput } from '@/components/ui/GpuSystemInput'
import { useSettings } from '@/contexts/SettingsContext'
import {
  engineVersionSupport,
  useCatalog,
  type BackendOption,
  type GpuOption,
  type SimulatorSupport,
} from '@/lib/hooks/useCatalog'
import { fetchEstimateAsInferenceResult } from '@/lib/api/estimate-adapter'
import type { InferenceConfigResult } from '@/lib/gpu-math/inference-config/types'
import {
  comparisonRequestsPerLoadUnit,
  measuredPoolTopology,
  perPoolBatch,
  predictionFabric,
  type ServingMode,
} from '@/lib/validation/benchmark-predictions'
import {
  acceleratorNamesMatch,
  type BenchmarkAdjustment,
  type BenchmarkUnavailable,
} from '@/lib/validation/benchmark-resolve'
import { backendVersionIdentity, backendVersionsComparable, backendVersionsMatch } from '@/lib/validation/benchmark-versions'
import {
  BENCHMARK_FILTER_KEYS,
  type BenchmarkFilterKey,
  type BenchmarkFilters,
  type BenchmarkPoint,
  type ConfigurationOption,
  type DatasetConfiguration,
  type DatasetSource,
} from '@/lib/validation/benchmark-types'
import ComparisonPreview, {
  predictionColor,
  type EstimatePoint,
  type PredictionSeries,
} from './ComparisonPreview'
import styles from './validation-compare.module.css'

type Notice = { variant: 'info' | 'warning' | 'success' | 'danger'; title: string; message: string }
type InferenceXSource = { url: string; retrievedAt: string; rowCount: number }
type InferenceXRun = { url: string; date: string | null }
type BenchmarkState = 'idle' | 'loaded' | 'empty' | 'unavailable'

type SelectionKey = 'accelerator' | 'model' | 'isl' | 'osl' | 'tp'
type Selection = Pick<BenchmarkFilters, SelectionKey>

const SELECTION_KEYS: SelectionKey[] = ['accelerator', 'model', 'isl', 'osl', 'tp']
const EMPTY_SELECTION: Selection = { accelerator: '', model: '', isl: '', osl: '', tp: '' }
const EMPTY_OPTIONS: Record<BenchmarkFilterKey, string[]> = {
  accelerator: [], model: [], backend: [], backendVersion: [], isl: [], osl: [], tp: [],
}
const MAX_PREDICTION_LEVELS = 24

const BACKEND_LABELS: Record<string, string> = {
  vllm: 'vLLM',
  'tensorrt-llm': 'TensorRT-LLM',
  sglang: 'SGLang',
}

interface ResolutionResponse {
  status: 'ok' | 'failed'
  error?: string
  source?: DatasetSource | null
  inferencexSource?: InferenceXSource | null
  benchmarkError?: string | null
  inferencexError?: string | null
  filters?: BenchmarkFilters
  adjustments?: BenchmarkAdjustment[]
  unavailable?: BenchmarkUnavailable | null
  versionFallback?: { value: string; message: string } | null
  options?: Record<BenchmarkFilterKey, string[]>
  configurations?: ConfigurationOption[]
}

interface DataResponse {
  status: 'ok' | 'failed'
  error?: string
  source?: DatasetSource | null
  inferencexSource?: InferenceXSource | null
  benchmarkError?: string | null
  inferencexError?: string | null
  matchedRows?: number
  benchmarkPoints?: BenchmarkPoint[]
  inferencexPoints?: BenchmarkPoint[]
  benchmarkVersion?: string | null
  inferencexVersion?: string | null
  inferencexRunUrl?: string | null
  inferencexDate?: string | null
  inferencexRuns?: InferenceXRun[]
}

interface PredictionAssumptions {
  servingMode: ServingMode
  weightPrecision: string
  kvCachePrecision: string
  ppSize: string
  dpSize: string
  moeTpSize: string
  moeEpSize: string
  fabric: string
}

interface SimulatorPlan {
  simulator: string
  label: string
  backend: BackendOption | null
  /** Literal release to send, or null to let the simulator resolve its default. */
  version: string | null
  requiresFabric: boolean
  skipReason: string | null
}

interface ComparisonResults {
  benchmark: BenchmarkPoint[]
  inferencex: BenchmarkPoint[]
  /** Release the Red Hat rows report, as resolved by the API. */
  benchmarkVersion: string | null
  /** Release the InferenceX rows report, as resolved by the API. */
  inferencexVersion: string | null
  runs: InferenceXRun[]
  series: PredictionSeries[]
  skipped: string[]
}

const EMPTY_RESULTS: ComparisonResults = {
  benchmark: [], inferencex: [], benchmarkVersion: null, inferencexVersion: null, runs: [], series: [], skipped: [],
}

function optionLabel(key: BenchmarkFilterKey, value: string): string {
  if (key === 'isl' || key === 'osl') return `${Number(value).toLocaleString('en-US')} tokens`
  if (key === 'tp') return `TP ${value}`
  if (key === 'backend') return BACKEND_LABELS[value] ?? value
  return value
}

function isHttpsUrl(value: string | null | undefined): value is string {
  if (!value) return false
  try {
    return new URL(value).protocol === 'https:'
  } catch {
    return false
  }
}

function findCatalogGpu(accelerator: string, gpus: GpuOption[]): GpuOption | null {
  const normalized = accelerator.trim().toLocaleLowerCase()
  if (!normalized) return null
  return gpus.find(gpu => gpu.label.toLocaleLowerCase() === normalized)
    ?? gpus.find(gpu => acceleratorNamesMatch(gpu.label, accelerator))
    ?? null
}

function matchAccelerator(options: string[], requested: string): string {
  const normalized = requested.trim().toLocaleLowerCase()
  if (!normalized) return options[0] ?? ''
  return options.find(option => option.toLocaleLowerCase() === normalized)
    ?? options.find(option => acceleratorNamesMatch(option, requested))
    ?? options[0]
    ?? ''
}

function backendVersionsFor(backend: BackendOption, systemId: string | null): string[] {
  if (!systemId) return backend.versions
  const listed = backend.systems[systemId]
  // Some simulators scope releases to a system, others report one shared list.
  // An empty system entry means "not scoped", not "unsupported".
  return listed && listed.length > 0 ? listed : backend.versions
}

function choosePredictionLevels(points: BenchmarkPoint[], limit = MAX_PREDICTION_LEVELS): BenchmarkPoint[] {
  if (points.length <= limit) return points
  const indexes = new Set<number>()
  for (let index = 0; index < limit; index += 1) {
    indexes.add(Math.round((index * (points.length - 1)) / (limit - 1)))
  }
  return [...indexes].sort((left, right) => left - right).map(index => points[index])
}

async function mapWithLimit<T, R>(
  values: T[],
  limit: number,
  mapper: (value: T) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(values.length)
  let nextIndex = 0
  const workers = Array.from({ length: Math.min(limit, values.length) }, async () => {
    while (nextIndex < values.length) {
      const index = nextIndex
      nextIndex += 1
      results[index] = await mapper(values[index])
    }
  })
  await Promise.all(workers)
  return results
}

function quantLabel(value: string | null | undefined): string {
  if (!value) return ''
  const token = value.toLocaleLowerCase().replace(/[\s_-]/g, '')
  if (token.includes('nvfp4')) return 'NVFP4'
  if (token.includes('mxfp4')) return 'MXFP4'
  if (token.includes('fp8')) return 'FP8'
  if (token.includes('int8')) return 'INT8'
  if (token.includes('int4')) return 'INT4'
  if (token.includes('fp4')) return 'NVFP4'
  if (token.includes('fp16') || token.includes('bf16')) return 'FP16'
  return ''
}

function gemmQuantMode(label: string): string | undefined {
  switch (label) {
    case 'FP8': return 'fp8'
    case 'INT8': return 'int8_wo'
    case 'INT4': return 'int4_wo'
    case 'MXFP4': return 'mxfp4'
    case 'NVFP4': return 'nvfp4'
    default: return undefined
  }
}

function kvcacheQuantMode(label: string): string | undefined {
  switch (label) {
    case 'FP8': return 'fp8'
    case 'NVFP4': return 'nvfp4'
    default: return undefined
  }
}

function measuredServingMode(configuration: ConfigurationOption | undefined): ServingMode {
  const topology = configuration?.values.inferencex_topology?.toLocaleLowerCase() ?? ''
  return topology.startsWith('disaggregated') ? 'disagg' : 'agg'
}

function derivedAssumptions(
  configuration: ConfigurationOption | undefined,
  defaultFabric: string,
  disaggAvailable: boolean,
): PredictionAssumptions {
  const mode = measuredServingMode(configuration)
  return {
    servingMode: mode === 'disagg' && disaggAvailable ? 'disagg' : 'agg',
    weightPrecision: quantLabel(configuration?.values.weight_quantization ?? configuration?.values.precision),
    kvCachePrecision: quantLabel(configuration?.values.kv_quantization),
    // Benchmark rows do not record PP, DP, or MoE widths. Default to a single
    // stage and the simulator's own width resolution, and say so in the UI.
    ppSize: '1',
    dpSize: '1',
    moeTpSize: '',
    moeEpSize: '',
    fabric: defaultFabric,
  }
}

function assumptionsEqual(left: PredictionAssumptions, right: PredictionAssumptions): boolean {
  return left.servingMode === right.servingMode
    && left.weightPrecision === right.weightPrecision
    && left.kvCachePrecision === right.kvCachePrecision
    && left.ppSize === right.ppSize
    && left.dpSize === right.dpSize
    && left.moeTpSize === right.moeTpSize
    && left.moeEpSize === right.moeEpSize
    && left.fabric === right.fabric
}

function positiveInteger(value: string): number | null {
  const parsed = Number(value)
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null
}

/**
 * How different two measured runtime configurations are. Keeping the same
 * source matters most, then the recorded precision and runtime settings, so a
 * changed workload dimension repairs to the run that still describes the same
 * deployment.
 */
function configurationDistance(left: DatasetConfiguration, right: DatasetConfiguration): number {
  let score = left.measurement_source === right.measurement_source ? 0 : 100
  for (const field of Object.keys(left)) {
    if (field === 'measurement_source') continue
    const leftValue = left[field] ?? ''
    const rightValue = right[field] ?? ''
    if (leftValue !== rightValue) score += leftValue && rightValue ? 4 : 1
  }
  return score
}

function nearestConfiguration(
  options: ConfigurationOption[],
  previous: DatasetConfiguration | null,
): ConfigurationOption | null {
  if (options.length === 0) return null
  if (!previous) return options[0]
  return [...options].sort((left, right) =>
    configurationDistance(left.values, previous) - configurationDistance(right.values, previous),
  )[0]
}

function simulatorPlans({
  simulators,
  simulatorSupport,
  selectedSystem,
  model,
  backendId,
  measuredVersion,
  assumptions,
}: {
  simulators: Array<{ id: string; label: string }>
  simulatorSupport: Record<string, SimulatorSupport>
  selectedSystem: GpuOption | null
  model: string
  backendId: string
  measuredVersion: string | null
  assumptions: PredictionAssumptions
}): SimulatorPlan[] {
  return simulators.map(simulator => {
    const support = simulatorSupport[simulator.id]
    const plan = (skipReason: string, backend: BackendOption | null = null): SimulatorPlan => ({
      simulator: simulator.id,
      label: support?.label ?? simulator.label,
      backend,
      version: null,
      requiresFabric: false,
      skipReason,
    })

    if (!support) return plan('The catalog does not list this simulator.')
    if (!support.available) return plan('Its catalog capabilities could not be read.')
    if (!selectedSystem) return plan('The benchmark GPU is not in the catalog, so it cannot run a prediction.')
    if (!support.systems.includes(selectedSystem.systemId)) {
      return plan(`The catalog does not list ${selectedSystem.label} for this simulator.`)
    }
    if (support.modelsExhaustive && !support.models.includes(model)) {
      return plan('The catalog does not list this model for this simulator.')
    }
    const backend = support.backends.find(option => option.id === backendId) ?? null
    if (!backend) return plan(`The catalog does not list ${BACKEND_LABELS[backendId] ?? backendId} for this simulator.`)

    const listedVersions = backendVersionsFor(backend, selectedSystem.systemId)
    const matchedVersion = measuredVersion
      ? listedVersions.find(version => backendVersionsMatch(version, measuredVersion)) ?? null
      : null
    const effectiveVersion = matchedVersion ?? backend.defaultVersion
    const versionSupport = engineVersionSupport(backend, effectiveVersion)
    if (assumptions.servingMode === 'disagg') {
      if (versionSupport && !versionSupport.disaggregated) {
        return plan(`Release ${effectiveVersion ?? 'default'} does not model prefill/decode disaggregation.`, backend)
      }
      if (backend.disaggregationRequiresFabric && (!assumptions.fabric || !support.networks.includes(assumptions.fabric))) {
        return plan(
          assumptions.fabric
            ? `The ${assumptions.fabric} fabric is not available in this simulator's catalog.`
            : 'Choose an inter-node fabric for disaggregated serving.',
          backend,
        )
      }
    }

    return {
      simulator: simulator.id,
      label: support.label,
      backend,
      version: matchedVersion,
      requiresFabric: backend.disaggregationRequiresFabric,
      skipReason: null,
    }
  })
}

export default function BenchmarkComparisonPage() {
  const {
    hydrated,
    defaultModel,
    defaultSystem,
    inferenceBackend,
    backendVersion,
  } = useSettings()
  const {
    gpuOptions,
    modelOptions,
    modelSpecs,
    simulatorSupport,
    simulators,
    isLoading: catalogLoading,
    error: catalogError,
  } = useCatalog()

  const [selection, setSelection] = React.useState<Selection>(EMPTY_SELECTION)
  const [resolution, setResolution] = React.useState<ResolutionResponse | null>(null)
  const [rootOptions, setRootOptions] = React.useState<Record<BenchmarkFilterKey, string[]> | null>(null)
  const [seeded, setSeeded] = React.useState(false)
  const [configurationKey, setConfigurationKey] = React.useState('')
  const configurationRef = React.useRef<ConfigurationOption | null>(null)
  const [source, setSource] = React.useState<DatasetSource | null>(null)
  const [inferencexSource, setInferenceXSource] = React.useState<InferenceXSource | null>(null)
  const [results, setResults] = React.useState<ComparisonResults>(EMPTY_RESULTS)
  const [benchmarkState, setBenchmarkState] = React.useState<BenchmarkState>('idle')
  const [isResolving, setIsResolving] = React.useState(true)
  const [isComparing, setIsComparing] = React.useState(false)
  const [notice, setNotice] = React.useState<Notice | null>(null)
  const [overrideAssumptions, setOverrideAssumptions] = React.useState(false)
  const [assumptionInputs, setAssumptionInputs] = React.useState<PredictionAssumptions | null>(null)
  const [assumptionsOpen, setAssumptionsOpen] = React.useState(false)
  const requestId = React.useRef(0)
  // Incremented whenever the visible result set is invalidated, so a slower
  // comparison cannot overwrite results for a newer selection.
  const compareRequestId = React.useRef(0)
  const settingsSnapshot = React.useRef(`${inferenceBackend}|${backendVersion}`)

  const resolvedFilters = resolution?.filters ?? null
  const unavailable = resolution?.unavailable ?? null
  const options = resolution?.options ?? EMPTY_OPTIONS
  const configurations = resolution?.configurations ?? []
  const selectedConfiguration = configurations.find(option => option.key === configurationKey)
  const currentGpu = React.useMemo(
    () => resolvedFilters ? findCatalogGpu(resolvedFilters.accelerator, gpuOptions) : null,
    [resolvedFilters, gpuOptions],
  )

  const clearResults = React.useCallback(() => {
    compareRequestId.current += 1
    setResults(EMPTY_RESULTS)
    setBenchmarkState('idle')
    setIsComparing(false)
  }, [])

  const resolve = React.useCallback(async (next: Selection): Promise<ResolutionResponse | null> => {
    const currentRequest = requestId.current + 1
    requestId.current = currentRequest
    setIsResolving(true)
    try {
      const params = new URLSearchParams({ mode: 'resolve' })
      for (const key of SELECTION_KEYS) {
        if (next[key]) params.set(key, next[key])
      }
      if (inferenceBackend) params.set('backend', inferenceBackend)
      if (backendVersion) params.set('backendVersion', backendVersion)
      const response = await fetch(`/api/validation/benchmark?${params.toString()}`, { cache: 'no-store' })
      const data = await response.json() as ResolutionResponse
      if (currentRequest !== requestId.current) return null
      if (!response.ok || data.status !== 'ok' || !data.filters || !data.options) {
        throw new Error(data.error ?? `Benchmark request failed (${response.status})`)
      }

      setResolution(data)
      setSource(data.source ?? null)
      setInferenceXSource(data.inferencexSource ?? null)
      setSelection({
        accelerator: data.filters.accelerator,
        model: data.filters.model,
        isl: data.filters.isl,
        osl: data.filters.osl,
        tp: data.filters.tp,
      })
      if (!data.filters.model) setRootOptions(data.options)
      const configurations = data.configurations ?? []
      const previous = configurationRef.current
      const nextConfiguration = configurations.find(option => option.key === previous?.key)
        ?? nearestConfiguration(configurations, previous?.values ?? null)
      configurationRef.current = nextConfiguration
      setConfigurationKey(nextConfiguration?.key ?? '')

      if (data.unavailable) {
        setNotice({
          variant: 'warning',
          title: 'No benchmark data for this selection',
          message: data.unavailable.message,
        })
      } else if (data.benchmarkError) {
        setNotice({
          variant: 'warning',
          title: 'Red Hat benchmark data unavailable',
          message: `InferenceX measurements remain available. ${data.benchmarkError}`,
        })
      } else if (data.inferencexError) {
        setNotice({
          variant: 'warning',
          title: 'InferenceX measurements unavailable',
          message: `Red Hat benchmark data remains available. ${data.inferencexError}`,
        })
      } else if ((data.adjustments?.length ?? 0) > 0 || data.versionFallback) {
        const reasons = [
          ...(data.adjustments ?? []).map(adjustment => adjustment.reason),
          ...(data.versionFallback ? [data.versionFallback.message] : []),
        ]
        setNotice({
          variant: 'info',
          title: 'Selection adjusted to measured values',
          message: reasons.join(' '),
        })
      } else {
        setNotice(null)
      }
      return data
    } catch (error) {
      if (currentRequest === requestId.current) {
        setBenchmarkState(previous => previous === 'loaded' ? previous : 'unavailable')
        setNotice({
          variant: 'danger',
          title: 'Benchmark data unavailable',
          message: error instanceof Error ? error.message : 'Could not load the benchmark dataset.',
        })
      }
      return null
    } finally {
      if (currentRequest === requestId.current) setIsResolving(false)
    }
  }, [inferenceBackend, backendVersion])

  // Load the root option lists once the catalogue and settings are ready. The
  // root response chooses no model, so it lists every benchmarked model.
  React.useEffect(() => {
    if (!hydrated || catalogLoading || seeded) return
    void resolve(EMPTY_SELECTION).then(data => {
      if (data?.options) setRootOptions(data.options)
    })
  }, [catalogLoading, hydrated, resolve, seeded])

  // Seed model and GPU from the cross-tool defaults when the benchmark covers
  // them, otherwise from the first measured option. This is first load only;
  // later changes are repaired by the resolver, never silently replaced here.
  React.useEffect(() => {
    if (!rootOptions || seeded) return
    const root = rootOptions
    const model = root.model.includes(defaultModel) ? defaultModel : root.model[0] ?? ''
    const defaultGpuLabel = gpuOptions.find(gpu => gpu.systemId === defaultSystem)?.label ?? defaultSystem
    const accelerator = matchAccelerator(root.accelerator, defaultGpuLabel)
    const isl = root.isl.includes('8192') ? '8192' : root.isl[0] ?? ''
    const osl = root.osl.includes('1024') ? '1024' : root.osl[0] ?? ''
    const tp = root.tp.includes('1') ? '1' : root.tp[0] ?? ''
    const next = { model, accelerator, isl, osl, tp }
    setSeeded(true)
    settingsSnapshot.current = `${inferenceBackend}|${backendVersion}`
    if (!model) return
    setSelection(next)
    void resolve(next)
  }, [backendVersion, defaultModel, defaultSystem, gpuOptions, inferenceBackend, resolve, rootOptions, seeded])

  // Settings owns the serving backend and version. A change there re-resolves
  // the same benchmark selection instead of resetting the page.
  React.useEffect(() => {
    const snapshot = `${inferenceBackend}|${backendVersion}`
    if (!seeded) {
      settingsSnapshot.current = snapshot
      return
    }
    if (settingsSnapshot.current === snapshot) return
    settingsSnapshot.current = snapshot
    clearResults()
    void resolve(selection)
  }, [backendVersion, clearResults, inferenceBackend, resolve, seeded, selection])

  function handleSelectionChange(key: SelectionKey, value: string) {
    if (isComparing) return
    const next = { ...selection, [key]: value }
    setSelection(next)
    clearResults()
    setNotice(null)
    void resolve(next)
  }

  const modelItems = React.useMemo<ComboBoxItem[]>(() => options.model.map(model => ({
    value: model,
    label: model,
    group: model.includes('/') ? model.slice(0, model.indexOf('/')) : 'Models',
  })), [options.model])

  const gpuInputOptions = React.useMemo<GpuOption[]>(() => {
    const extras = options.accelerator
      .filter(accelerator => !findCatalogGpu(accelerator, gpuOptions))
      .map<GpuOption>(accelerator => ({
        systemId: accelerator,
        label: accelerator,
        vendor: null,
        architecture: null,
        vramGb: null,
        bandwidthTbps: null,
        tflopsBf16: null,
        tdpWatts: null,
        gpusPerNode: null,
      }))
    return [...gpuOptions, ...extras]
  }, [gpuOptions, options.accelerator])

  const testedSystemIds = React.useMemo(() => options.accelerator.flatMap(accelerator => {
    const gpu = findCatalogGpu(accelerator, gpuOptions)
    return gpu ? [gpu.systemId] : []
  }), [gpuOptions, options.accelerator])

  const gpuValue = findCatalogGpu(resolvedFilters?.accelerator ?? '', gpuOptions)?.systemId
    ?? resolvedFilters?.accelerator
    ?? selection.accelerator

  const modelName = resolvedFilters?.model ?? ''
  const modelSpec = React.useMemo(() => {
    if (!modelName) return undefined
    for (const simulator of simulators) {
      const spec = simulatorSupport[simulator.id]?.modelSpecs.get(modelName)
      if (spec) return spec
    }
    return modelSpecs.get(modelName)
  }, [modelName, modelSpecs, simulatorSupport, simulators])
  const isMoe = (modelSpec?.num_experts ?? 0) > 1

  const capabilityPlans = React.useMemo(() => simulatorPlans({
    simulators,
    simulatorSupport,
    selectedSystem: currentGpu,
    model: resolvedFilters?.model ?? '',
    backendId: inferenceBackend,
    measuredVersion: resolvedFilters?.backendVersion ?? null,
    assumptions: {
      servingMode: 'agg',
      weightPrecision: '', kvCachePrecision: '', ppSize: '1', dpSize: '1',
      moeTpSize: '', moeEpSize: '', fabric: '',
    },
  }), [currentGpu, inferenceBackend, resolvedFilters?.backendVersion, resolvedFilters?.model, simulatorSupport, simulators])

  const disaggAvailable = capabilityPlans.some(plan => {
    if (!plan.backend) return false
    const version = plan.version ?? plan.backend.defaultVersion
    return engineVersionSupport(plan.backend, version)?.disaggregated ?? true
  })
  const fabricRequired = capabilityPlans.some(plan => plan.backend?.disaggregationRequiresFabric ?? false)
  const fabricOptions = React.useMemo(() => [...new Set(
    simulators.flatMap(simulator => simulatorSupport[simulator.id]?.networks ?? []),
  )], [simulatorSupport, simulators])
  const defaultFabric = React.useMemo(
    () => simulators.map(simulator => simulatorSupport[simulator.id]?.defaultNetwork)
      .find((network): network is string => Boolean(network)) ?? fabricOptions[0] ?? '',
    [fabricOptions, simulatorSupport, simulators],
  )

  const defaults = React.useMemo(
    () => derivedAssumptions(selectedConfiguration, defaultFabric, disaggAvailable),
    [defaultFabric, disaggAvailable, selectedConfiguration],
  )
  const assumptions = overrideAssumptions
    ? assumptionInputs ?? defaults
    : defaults
  const assumptionsOverridden = overrideAssumptions && !assumptionsEqual(assumptions, defaults)

  function setAssumption<K extends keyof PredictionAssumptions>(key: K, value: PredictionAssumptions[K]) {
    if (isComparing) return
    setAssumptionInputs(current => ({ ...(current ?? defaults), [key]: value }))
    clearResults()
  }

  const measuredMode = measuredServingMode(selectedConfiguration)
  const layoutComparable = measuredMode === assumptions.servingMode

  const modelConfigurationRequired = Boolean(
    resolvedFilters?.model
    && !modelOptions.includes(resolvedFilters.model)
    && !modelSpecs.has(resolvedFilters.model),
  )

  async function loadModelConfig(): Promise<Record<string, unknown> | null> {
    if (!resolvedFilters?.model || !modelConfigurationRequired) return null
    const response = await fetch(`/api/hf-config?model=${encodeURIComponent(resolvedFilters.model)}`)
    const data = await response.json() as { config?: Record<string, unknown>; message?: string }
    if (!response.ok || !data.config) throw new Error(data.message ?? 'Model configuration unavailable')
    return data.config
  }

  async function runComparison() {
    if (!resolvedFilters || unavailable || !selectedConfiguration) {
      setNotice({
        variant: 'warning',
        title: 'Choose a complete benchmark run',
        message: 'Select a model, GPU, benchmark runtime configuration, and a serving backend that the benchmark measured.',
      })
      return
    }

    const runId = compareRequestId.current + 1
    compareRequestId.current = runId
    const stale = () => runId !== compareRequestId.current
    setIsComparing(true)
    setBenchmarkState('idle')
    setResults(EMPTY_RESULTS)
    setNotice({ variant: 'info', title: 'Loading comparison', message: 'Fetching the selected benchmark rows.' })
    let measuredLoaded = false

    try {
      const params = new URLSearchParams({ mode: 'data', configuration: selectedConfiguration.key })
      for (const key of BENCHMARK_FILTER_KEYS) params.set(key, resolvedFilters[key])
      const response = await fetch(`/api/validation/benchmark?${params.toString()}`, { cache: 'no-store' })
      const data = await response.json() as DataResponse
      if (stale()) return
      if (!response.ok || data.status !== 'ok' || !data.benchmarkPoints || !data.inferencexPoints) {
        throw new Error(data.error ?? `Comparison request failed (${response.status})`)
      }

      const benchmarkPoints = data.benchmarkPoints
      const inferencexPoints = data.inferencexPoints
      const measuredLevels = [...new Map([...benchmarkPoints, ...inferencexPoints]
        .map(point => [point.concurrency, point])).values()]
        .sort((left, right) => left.concurrency - right.concurrency)
      setSource(data.source ?? null)
      setInferenceXSource(data.inferencexSource ?? null)

      if (measuredLevels.length === 0) {
        setBenchmarkState('empty')
        setResults({
          benchmark: benchmarkPoints,
          inferencex: inferencexPoints,
          benchmarkVersion: data.benchmarkVersion ?? null,
          inferencexVersion: data.inferencexVersion ?? null,
          runs: data.inferencexRuns ?? [],
          series: [],
          skipped: [],
        })
        setNotice({
          variant: 'warning',
          title: 'No matching benchmark rows',
          message: 'Adjust the benchmark inputs and select Compare again.',
        })
        return
      }

      setBenchmarkState('loaded')
      measuredLoaded = true
      setResults({
        benchmark: benchmarkPoints,
        inferencex: inferencexPoints,
        benchmarkVersion: data.benchmarkVersion ?? null,
        inferencexVersion: data.inferencexVersion ?? null,
        runs: data.inferencexRuns ?? [],
        series: [],
        skipped: [],
      })

      const plans = simulatorPlans({
        simulators,
        simulatorSupport,
        selectedSystem: currentGpu,
        model: resolvedFilters.model,
        backendId: inferenceBackend,
        measuredVersion: resolvedFilters.backendVersion,
        assumptions,
      })
      const skipped = plans.flatMap(plan => plan.skipReason ? [`${plan.label}: ${plan.skipReason}`] : [])
      const runnable = plans.filter(plan => plan.skipReason === null)
      if (runnable.length === 0) {
        setResults(previous => ({ ...previous, skipped }))
        setNotice({
          variant: 'warning',
          title: 'Measurements loaded; no simulator can predict this run',
          message: skipped.join(' '),
        })
        return
      }

      const modelConfig = await loadModelConfig()
      if (stale()) return
      const levels = choosePredictionLevels(measuredLevels)
      const ppSize = positiveInteger(assumptions.ppSize) ?? 1
      const dpSize = positiveInteger(assumptions.dpSize) ?? 1
      const moeTpSize = positiveInteger(assumptions.moeTpSize)
      const moeEpSize = positiveInteger(assumptions.moeEpSize)
      const tpSize = Number(resolvedFilters.tp)
      const measuredTopology = measuredPoolTopology(selectedConfiguration?.values, tpSize)
      // Deltas are only meaningful when the predicted P/D pools match the
      // measured run. Aggregated runs have no per-pool shape to compare.
      const topologyComparable = assumptions.servingMode === 'agg' || measuredTopology.known
      const series: PredictionSeries[] = []

      for (const plan of runnable) {
        const attempts = await mapWithLimit(levels, 3, async measured => {
          try {
            const input = {
              model_path: resolvedFilters.model,
              system: currentGpu?.systemId ?? '',
              isl: Number(resolvedFilters.isl),
              osl: Number(resolvedFilters.osl),
              batch_size: measured.concurrency,
              requests_per_load_unit: comparisonRequestsPerLoadUnit(measured.concurrency),
              tp_size: tpSize,
              pp_size: ppSize,
              attention_dp_size: dpSize,
              backend: inferenceBackend,
              backend_version: plan.version ?? undefined,
              gemm_quant_mode: gemmQuantMode(assumptions.weightPrecision),
              kvcache_quant_mode: kvcacheQuantMode(assumptions.kvCachePrecision),
              moe_tp_size: moeTpSize ?? undefined,
              moe_ep_size: moeEpSize ?? undefined,
              simulator: plan.simulator,
              vram_gb: currentGpu?.vramGb ?? null,
              hf_model_config: modelConfig,
              mode: assumptions.servingMode,
              // A fabric is a BLIS P/D input. Forward it only to a simulator
              // whose catalog says it consumes one; AISimulate rejects it.
              fabric: predictionFabric(assumptions.servingMode, plan.requiresFabric, assumptions.fabric),
              prefill: assumptions.servingMode === 'disagg'
                ? {
                  tp: measuredTopology.prefillTp,
                  pp: ppSize,
                  workers: measuredTopology.prefillWorkers,
                  batch: perPoolBatch(measured.concurrency, measuredTopology.prefillWorkers),
                }
                : undefined,
              decode: assumptions.servingMode === 'disagg'
                ? {
                  tp: measuredTopology.decodeTp,
                  pp: ppSize,
                  workers: measuredTopology.decodeWorkers,
                  batch: perPoolBatch(measured.concurrency, measuredTopology.decodeWorkers),
                }
                : undefined,
            }
            const result: InferenceConfigResult = await fetchEstimateAsInferenceResult(input)
            const performance = result.performance
            if (!performance) throw new Error(`${plan.label} did not return performance metrics.`)
            const outputTokensPerSecond = performance.throughput_tokens_per_sec
            const outputTokensPerSecondPerUser = performance.throughput_tokens_per_sec_per_user
              ?? (measured.concurrency > 0 ? outputTokensPerSecond / measured.concurrency : null)
            const point: EstimatePoint = {
              concurrency: measured.concurrency,
              ttftSeconds: performance.ttft_ms / 1000,
              tpotSeconds: performance.tpot_ms / 1000,
              outputTokensPerSecond,
              outputTokensPerSecondPerUser,
              requestLatencySeconds: performance.request_latency_ms / 1000,
            }
            return { point, version: result.engine_version ?? null, error: null as string | null }
          } catch (error) {
            return {
              point: null,
              version: null,
              error: error instanceof Error ? error.message : `${plan.label} prediction failed`,
            }
          }
        })

        const failures = [...new Set(attempts.flatMap(attempt => attempt.error ? [attempt.error] : []))]
        const points = attempts.flatMap(attempt => attempt.point ? [attempt.point] : [])
        if (points.length === 0) {
          const reason = failures[0] ?? 'No prediction points were returned.'
          skipped.push(`${plan.label}: ${reason}`)
          continue
        }
        if (failures.length > 0) {
          // A level the simulator refuses (for example because its concurrency
          // exceeds a request limit) must be visible, not silently dropped from
          // an otherwise plotted series.
          skipped.push(
            `${plan.label}: ${levels.length - points.length} of ${levels.length} concurrency levels skipped. ${failures[0]}`,
          )
        }
        const reportedVersion = attempts.find(attempt => attempt.version)?.version ?? null
        const versionComparable = backendVersionsComparable(resolvedFilters.backendVersion, reportedVersion)
        series.push({
          simulator: plan.simulator,
          label: plan.label,
          version: reportedVersion,
          comparable: versionComparable && layoutComparable && topologyComparable && !assumptionsOverridden,
          overridden: assumptionsOverridden,
          points,
        })
      }

      if (stale()) return
      setResults(previous => ({ ...previous, series, skipped }))
      const failed = attemptsFailedCount(series, levels.length, runnable.length, skipped)
      setNotice({
        variant: failed > 0 || skipped.length > 0 ? 'warning' : 'success',
        title: skipped.length > 0 ? 'Comparison complete with skipped predictions' : 'Comparison complete',
        message: [
          `${data.matchedRows?.toLocaleString('en-US') ?? 0} measured rows across ${measuredLevels.length} concurrency levels.`,
          series.length > 0 ? `${series.length} simulator series plotted.` : '',
          skipped.length > 0 ? skipped.join(' ') : '',
        ].filter(Boolean).join(' '),
      })
    } catch (error) {
      if (stale()) return
      if (!measuredLoaded) setBenchmarkState('unavailable')
      setNotice({
        variant: 'danger',
        title: 'Comparison failed',
        message: error instanceof Error ? error.message : 'Could not compare this benchmark run.',
      })
    } finally {
      if (!stale()) setIsComparing(false)
    }
  }

  const measuredCount = new Set([...results.benchmark, ...results.inferencex].map(point => point.concurrency)).size
  const safeInferenceXRuns = results.runs.filter(run => isHttpsUrl(run.url))
  const datasetDate = source?.lastModified ? new Date(source.lastModified).toLocaleDateString() : null
  const selectedVersionLabel = backendVersion || 'Simulator default'
  const measuredVersionLabel = resolvedFilters?.backendVersion ?? null
  // Core-equivalent labels are comparable, but a measured vendor image build is
  // still worth showing when it is more specific than the Settings pin.
  const measuredVersionDiffers = Boolean(measuredVersionLabel)
    && backendVersionIdentity(measuredVersionLabel).label !== backendVersionIdentity(backendVersion).label
  const canCompare = Boolean(
    resolvedFilters
    && SELECTION_KEYS.every(key => resolvedFilters[key])
    && measuredVersionLabel
    && selectedConfiguration
    && !unavailable
    && !isResolving
    && !isComparing,
  )

  return (
    <PageSection className={styles.page}>
      <div className={styles.heading}>
        <div>
          <p className={styles.eyebrow}>Validation</p>
          <h1>Benchmark comparison</h1>
          <p className={styles.intro}>
            Compare measured benchmark data with every simulator in the catalog that supports this model, GPU, and serving backend.
          </p>
        </div>
        {(source || inferencexSource) && (
          <div className={styles.sourceBadges}>
            {source && <div className={styles.sourceBadge}>
              <a href={source.url} target="_blank" rel="noreferrer">Benchmark data from Red Hat · Hugging Face</a>
              <span>{source.rowCount.toLocaleString('en-US')} rows · CC BY 4.0{datasetDate ? ` · updated ${datasetDate}` : ''}</span>
            </div>}
            {inferencexSource && <div className={styles.sourceBadge}>
              {isHttpsUrl(inferencexSource.url)
                ? <a href={inferencexSource.url} target="_blank" rel="noreferrer">Benchmark data from SemiAnalysis InferenceX</a>
                : <span>Benchmark data from SemiAnalysis InferenceX</span>}
              <span>{inferencexSource.rowCount.toLocaleString('en-US')} 8k/1k rows · snapshot {new Date(inferencexSource.retrievedAt).toLocaleDateString()}</span>
            </div>}
          </div>
        )}
      </div>

      {isResolving && !source && !inferencexSource && (
        <Alert variant="info" isInline title="Loading benchmark data" className={styles.notice}>
          <span className={styles.loadingCopy}><Spinner size="sm" /> Loading measured benchmark rows.</span>
        </Alert>
      )}
      {notice && !isResolving && (
        <Alert variant={notice.variant} isInline title={notice.title} className={styles.notice}>
          <p>{notice.message}</p>
          {unavailable && (unavailable.field === 'backend' || unavailable.field === 'backendVersion') && (
            <p><Link href="/settings">Change the serving backend or version in Settings</Link></p>
          )}
          {notice.variant === 'danger' && !source && !inferencexSource && (
            <Button variant="link" onClick={() => void resolve(selection)}>Retry dataset load</Button>
          )}
        </Alert>
      )}

      <section className={styles.filterCard} aria-labelledby="benchmark-inputs-title" aria-busy={isResolving}>
        <div className={styles.cardHead}>
          <div>
            <h2 id="benchmark-inputs-title">Benchmark inputs</h2>
            <p>Start with the model and GPU system. The remaining options come from the measurements that match them.</p>
          </div>
        </div>

        <fieldset disabled={isComparing} className={styles.filterFieldset}>
          <legend className={styles.visuallyHidden}>Benchmark comparison inputs</legend>
          <div className={styles.primaryGrid}>
            <ComboBox
              id="benchmark-model"
              label="Model"
              value={resolvedFilters?.model ?? selection.model}
              onChange={value => handleSelectionChange('model', value)}
              items={modelItems}
              placeholder="Search measured models..."
              helperText="Models with benchmark measurements in this dataset."
            />
            <GpuSystemInput
              id="benchmark-gpu"
              value={gpuValue}
              onChange={value => {
                const option = gpuInputOptions.find(entry => entry.systemId === value)
                handleSelectionChange('accelerator', option?.label ?? value)
              }}
              gpuOptions={gpuInputOptions}
              testedSystemIds={testedSystemIds}
            />
          </div>

          <div className={styles.secondaryGrid}>
            {(['isl', 'osl', 'tp'] as const).map(key => (
              <div className={styles.field} key={key}>
                <label htmlFor={`benchmark-${key}`}>
                  {key === 'isl' ? 'Input tokens' : key === 'osl' ? 'Output tokens' : 'Tensor parallel size'}
                </label>
                <div id={`benchmark-${key}`}>
                  <FormSelect
                    value={resolvedFilters?.[key] ?? selection[key]}
                    onChange={(_event, value) => handleSelectionChange(key, String(value))}
                    aria-label={key === 'isl' ? 'Input tokens' : key === 'osl' ? 'Output tokens' : 'Tensor parallel size'}
                    isDisabled={isResolving || options[key].length === 0}
                  >
                    {options[key].map(value => (
                      <FormSelectOption key={value} value={value} label={optionLabel(key, value)} />
                    ))}
                  </FormSelect>
                </div>
              </div>
            ))}
            <div className={`${styles.field} ${styles.configurationField}`}>
              <label htmlFor="benchmark-configuration">Measured runtime configuration</label>
              <div id="benchmark-configuration">
                <FormSelect
                  value={configurationKey}
                  onChange={(_event, value) => {
                    const key = String(value)
                    configurationRef.current = configurations.find(option => option.key === key) ?? null
                    setConfigurationKey(key)
                    clearResults()
                  }}
                  aria-label="Measured runtime configuration"
                  isDisabled={isResolving || configurations.length === 0}
                >
                  {configurations.length === 0 && <FormSelectOption value="" label="No matching runtime configuration" />}
                  {configurations.map(configuration => (
                    <FormSelectOption
                      key={configuration.key}
                      value={configuration.key}
                      label={`${configuration.label} · ${configuration.rows} rows`}
                    />
                  ))}
                </FormSelect>
              </div>
              <p className={styles.helperText}>
                Keeps measurement source, precision, quantization, and serving settings from different runs separate.
              </p>
            </div>
          </div>
        </fieldset>

        <div className={styles.settingsRow}>
          <div>
            <span className={styles.settingsLabel}>Serving backend from settings</span>
            <span className={styles.settingsValue}>
              {BACKEND_LABELS[inferenceBackend] ?? inferenceBackend}
              {' · '}
              {selectedVersionLabel}
              {measuredVersionDiffers
                ? ` · benchmark ${measuredVersionLabel}`
                : ''}
            </span>
          </div>
          <Link href="/settings" className={styles.settingsLink}>Change in settings</Link>
        </div>
        <p className={styles.helperText}>
          The serving backend and version are shared by every tool and set in Settings. Each simulator receives the
          measured release only when its catalog lists that release for this GPU; otherwise the service resolves its
          own default.
        </p>

        <Accordion asDefinitionList={false} className={styles.assumptionsCard}>
          <AccordionItem isExpanded={assumptionsOpen}>
            <AccordionToggle
              id="benchmark-assumptions-toggle"
              onClick={() => setAssumptionsOpen(open => !open)}
            >
              <span className={styles.accordionToggle}>
                <span className={styles.accordionTitle}>Prediction assumptions</span>
                {!assumptionsOpen && (
                  <span className={styles.accordionSummary}>
                    <span><span className={styles.accordionKey}>override</span> {overrideAssumptions ? 'on' : 'off'}</span>
                    <span><span className={styles.accordionKey}>mode</span> {assumptions.servingMode === 'agg' ? 'aggregated' : 'disaggregated'}</span>
                    <span><span className={styles.accordionKey}>weights</span> {assumptions.weightPrecision || 'auto'}</span>
                    <span><span className={styles.accordionKey}>KV</span> {assumptions.kvCachePrecision || 'auto'}</span>
                    <span><span className={styles.accordionKey}>PP</span> {assumptions.ppSize}</span>
                    <span><span className={styles.accordionKey}>DP</span> {assumptions.dpSize}</span>
                    {isMoe && <span><span className={styles.accordionKey}>MoE TP</span> {assumptions.moeTpSize || 'auto'}</span>}
                    {isMoe && <span><span className={styles.accordionKey}>EP</span> {assumptions.moeEpSize || 'auto'}</span>}
                    {assumptions.servingMode === 'disagg' && fabricRequired && (
                      <span><span className={styles.accordionKey}>fabric</span> {assumptions.fabric || 'none'}</span>
                    )}
                  </span>
                )}
              </span>
            </AccordionToggle>
            <AccordionContent>
              <div className={styles.assumptionsHead}>
                <div>
                  <p>
                    Defaults follow the selected benchmark run. Turn on the override to change topology, precision, or
                    parallelism for every prediction; percentage differences are hidden when assumptions differ.
                  </p>
                </div>
                <Switch
                  id="benchmark-override-assumptions"
                  label="Override prediction assumptions"
                  isChecked={overrideAssumptions}
                  isDisabled={isComparing}
                  onChange={(_event, checked) => {
                    setOverrideAssumptions(checked)
                    setAssumptionInputs(checked ? { ...defaults } : null)
                    clearResults()
                  }}
                />
              </div>

              <div className={styles.assumptionGrid}>
                <div className={styles.assumptionField}>
                  <span className={styles.assumptionLabel}>Serving mode</span>
                  <div className={styles.servingModeRow}>
                    <Button
                      variant={assumptions.servingMode === 'agg' ? 'secondary' : 'tertiary'}
                      size="sm"
                      aria-pressed={assumptions.servingMode === 'agg'}
                      isDisabled={!overrideAssumptions || isComparing}
                      onClick={() => setAssumption('servingMode', 'agg')}
                    >
                      Aggregated
                    </Button>
                    <Button
                      variant={assumptions.servingMode === 'disagg' ? 'secondary' : 'tertiary'}
                      size="sm"
                      aria-pressed={assumptions.servingMode === 'disagg'}
                      isDisabled={!overrideAssumptions || !disaggAvailable || isComparing}
                      onClick={() => setAssumption('servingMode', 'disagg')}
                    >
                      Disaggregated
                    </Button>
                  </div>
                  <p className={styles.assumptionNote}>
                    {measuredMode === 'disagg'
                      ? 'The benchmark run is disaggregated.'
                      : 'The benchmark run is aggregated.'}
                    {!disaggAvailable ? ' No catalog simulator models disaggregation for this release.' : ''}
                  </p>
                </div>

                <div className={styles.assumptionField}>
                  <label className={styles.assumptionLabel} htmlFor="benchmark-weight-precision">Weight precision</label>
                  <FormSelect
                    id="benchmark-weight-precision"
                    value={assumptions.weightPrecision}
                    isDisabled={!overrideAssumptions || isComparing}
                    onChange={(_event, value) => setAssumption('weightPrecision', String(value))}
                  >
                    <FormSelectOption value="" label="Simulator default" />
                    {['FP16', 'FP8', 'INT8', 'INT4', 'MXFP4', 'NVFP4'].map(value => (
                      <FormSelectOption key={value} value={value} label={value} />
                    ))}
                  </FormSelect>
                </div>

                <div className={styles.assumptionField}>
                  <label className={styles.assumptionLabel} htmlFor="benchmark-kv-precision">KV cache precision</label>
                  <FormSelect
                    id="benchmark-kv-precision"
                    value={assumptions.kvCachePrecision}
                    isDisabled={!overrideAssumptions || isComparing}
                    onChange={(_event, value) => setAssumption('kvCachePrecision', String(value))}
                  >
                    <FormSelectOption value="" label="Simulator default" />
                    {['FP16', 'FP8', 'NVFP4'].map(value => (
                      <FormSelectOption key={value} value={value} label={value} />
                    ))}
                  </FormSelect>
                </div>

                {(['ppSize', 'dpSize'] as const).map(key => (
                  <div className={styles.assumptionField} key={key}>
                    <label className={styles.assumptionLabel} htmlFor={`benchmark-${key}`}>
                      {key === 'ppSize' ? 'Pipeline parallel (PP)' : 'Attention data parallel (DP)'}
                    </label>
                    <TextInput
                      id={`benchmark-${key}`}
                      type="number"
                      min={1}
                      value={assumptions[key]}
                      isDisabled={!overrideAssumptions || isComparing}
                      onChange={(_event, value) => setAssumption(key, value)}
                    />
                  </div>
                ))}

                {isMoe && (['moeTpSize', 'moeEpSize'] as const).map(key => (
                  <div className={styles.assumptionField} key={key}>
                    <label className={styles.assumptionLabel} htmlFor={`benchmark-${key}`}>
                      {key === 'moeTpSize' ? 'MoE tensor parallel (TP)' : 'Expert parallel (EP)'}
                    </label>
                    <TextInput
                      id={`benchmark-${key}`}
                      type="number"
                      min={1}
                      value={assumptions[key]}
                      isDisabled={!overrideAssumptions || isComparing}
                      placeholder="Simulator default"
                      onChange={(_event, value) => setAssumption(key, value)}
                    />
                  </div>
                ))}

                {assumptions.servingMode === 'disagg' && fabricRequired && (
                  <div className={styles.assumptionField}>
                    <label className={styles.assumptionLabel} htmlFor="benchmark-fabric">Inter-node fabric</label>
                    <FormSelect
                      id="benchmark-fabric"
                      value={assumptions.fabric}
                      isDisabled={!overrideAssumptions || fabricOptions.length === 0 || isComparing}
                      onChange={(_event, value) => setAssumption('fabric', String(value))}
                    >
                      {fabricOptions.length === 0 && <FormSelectOption value="" label="No fabrics in the catalog" />}
                      {fabricOptions.map(fabric => (
                        <FormSelectOption key={fabric} value={fabric} label={fabric} />
                      ))}
                    </FormSelect>
                    <p className={styles.assumptionNote}>
                      Used for prefill-to-decode KV transfer. The simulator catalog names the available fabrics.
                    </p>
                  </div>
                )}
              </div>

              <p className={`${styles.assumptionNote} ${styles.assumptionsFootnote}`}>
                Benchmark rows do not record pipeline, data, or expert parallelism. Predictions therefore start from
                PP 1, DP 1, and the simulator&apos;s own MoE width resolution. Changing them is an explicit override.
              </p>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        <div className={styles.filterFooter}>
          <Button variant="primary" isDisabled={!canCompare} onClick={() => void runComparison()}>
            {isComparing && <Spinner size="sm" className={styles.buttonSpinner} />}
            {isComparing ? 'Comparing…' : 'Compare'}
          </Button>
        </div>
      </section>

      <section aria-labelledby="comparison-charts-title" className={styles.results}>
        <div className={styles.resultsHead}>
          <div>
            <h2 id="comparison-charts-title">Performance by concurrency</h2>
            <p>{measuredCount > 0 ? `${measuredCount} measured concurrency levels` : 'Run a comparison to load measured and predicted performance.'}</p>
          </div>
          <div className={styles.legend} aria-label="Chart series">
            {results.benchmark.length > 0 && (
              <span>
                <i className={styles.benchmarkSwatch} />
                Red Hat benchmark · {results.benchmarkVersion ?? 'version not reported'}
              </span>
            )}
            {results.inferencex.length > 0 && (
              <span>
                <i className={styles.inferencexSwatch} />
                InferenceX benchmark · SemiAnalysis · {results.inferencexVersion ?? 'version not reported'}
              </span>
            )}
            {results.series.map((entry, index) => (
              <span key={entry.simulator}>
                <i className={styles.seriesSwatch} style={{ background: predictionColor(entry.simulator, index) }} />
                {entry.label} · {entry.version ?? 'version not reported'}
                {entry.overridden
                  ? ' · overridden assumptions'
                  : !entry.comparable ? ' · not comparable' : ''}
              </span>
            ))}
          </div>
        </div>

        {results.skipped.length > 0 && (
          <Alert variant="warning" isInline title="Skipped predictions" className={styles.notice}>
            <ul className={styles.adjustmentList}>
              {results.skipped.map(reason => <li key={reason}>{reason}</li>)}
            </ul>
          </Alert>
        )}

        {measuredCount > 0 ? (
          <ComparisonPreview
            benchmark={results.benchmark}
            inferencex={results.inferencex}
            series={results.series}
          />
        ) : benchmarkState === 'empty' ? (
          <div className={styles.emptyResults}>
            <h3>No matching benchmark rows</h3>
            <p>No measured rows match these inputs. Adjust the benchmark inputs and select Compare again.</p>
          </div>
        ) : benchmarkState === 'unavailable' ? (
          <div className={styles.emptyResults}>
            <h3>Benchmark data unavailable</h3>
            <p>The benchmark datasets could not be loaded.</p>
            <Button variant="link" onClick={() => void resolve(selection)}>Retry dataset load</Button>
          </div>
        ) : (
          <div className={styles.emptyResults}>
            <h3>Compare a benchmark run</h3>
            <p>Choose a measured runtime configuration and select Compare to plot Red Hat, InferenceX, and simulator predictions.</p>
          </div>
        )}

        {safeInferenceXRuns.length > 0 && (
          <p className={styles.methodNote}>
            {safeInferenceXRuns.length === 1 ? 'Selected run' : `${safeInferenceXRuns.length} averaged runs`}
            {safeInferenceXRuns.some(run => run.date) && ` (${safeInferenceXRuns.map(run => run.date).filter(Boolean).join(', ')})`}:{' '}
            {safeInferenceXRuns.map((run, index) => (
              <React.Fragment key={run.url}>
                {index > 0 && ', '}
                <a href={run.url} target="_blank" rel="noreferrer">Run {index + 1}</a>
              </React.Fragment>
            ))}
            .
          </p>
        )}

        {catalogError && (
          <p className={styles.methodNote}>Simulator catalog unavailable: {catalogError}</p>
        )}
      </section>
    </PageSection>
  )
}

function attemptsFailedCount(
  series: PredictionSeries[],
  levels: number,
  runnable: number,
  skipped: string[],
): number {
  const expected = levels * runnable
  const observed = series.reduce((total, entry) => total + entry.points.length, 0)
  return Math.max(0, expected - observed) + skipped.length
}

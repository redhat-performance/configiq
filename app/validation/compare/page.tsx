'use client'

import * as React from 'react'
import {
  Alert,
  Button,
  PageSection,
  Spinner,
} from '@patternfly/react-core'
import { ComboBox, type ComboBoxItem } from '@/components/ModelComboBox/ModelComboBox'
import { SimulatorBadge } from '@/components/ui/SimulatorBadge'
import { useSettings, type Simulator } from '@/contexts/SettingsContext'
import { CompareSelect } from './CompareSelect'
import { useCatalog, type BackendOption, type GpuOption } from '@/lib/hooks/useCatalog'
import { fetchEstimateAsInferenceResult } from '@/lib/api/estimate-adapter'
import {
  GROUND_TRUTH_FILTER_KEYS,
} from '@/lib/validation/ground-truth-types'
import type {
  ConfigurationOption,
  DatasetSource,
  GroundTruthFilterKey,
  GroundTruthFilters,
  GroundTruthPoint,
} from '@/lib/validation/ground-truth-types'
import ComparisonPreview, { type EstimatePoint } from './ComparisonPreview'
import styles from './validation-compare.module.css'

type Notice = { variant: 'info' | 'warning' | 'success' | 'danger'; title: string; message: string }
type InferenceXSource = { url: string; retrievedAt: string; rowCount: number }
type InferenceXRun = { url: string; date: string | null }

interface FilterResponse {
  status: 'ok' | 'failed'
  error?: string
  source?: DatasetSource | null
  inferencexSource?: InferenceXSource | null
  groundTruthError?: string | null
  inferencexError?: string | null
  options?: Record<GroundTruthFilterKey, string[]>
  configurations?: ConfigurationOption[]
}

interface GroundTruthResponse {
  status: 'ok' | 'failed'
  error?: string
  source?: DatasetSource | null
  inferencexSource?: InferenceXSource | null
  groundTruthError?: string | null
  inferencexError?: string | null
  matchedRows?: number
  groundTruthPoints?: GroundTruthPoint[]
  inferencexPoints?: GroundTruthPoint[]
  inferencexRunUrl?: string | null
  inferencexDate?: string | null
  inferencexRuns?: InferenceXRun[]
}

interface EstimateAttempt {
  point: EstimatePoint | null
  simulator: Simulator | null
  error: string | null
}

const FILTER_LABELS: Record<GroundTruthFilterKey, string> = {
  accelerator: 'GPU accelerator',
  model: 'Model',
  backend: 'Serving backend',
  backendVersion: 'Serving backend version',
  isl: 'Input tokens',
  osl: 'Output tokens',
  tp: 'Tensor parallel size',
}

const EMPTY_FILTERS: GroundTruthFilters = {
  accelerator: '',
  model: '',
  backend: '',
  backendVersion: '',
  isl: '',
  osl: '',
  tp: '',
}

const PREDICTION_REQUESTS_PER_LOAD_UNIT = 10

function optionLabel(key: GroundTruthFilterKey, value: string): string {
  if (key === 'isl' || key === 'osl') return `${Number(value).toLocaleString('en-US')} tokens`
  if (key === 'tp') return `TP ${value}`
  if (key === 'backend' && value === 'vllm') return 'vLLM'
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
  return gpus.find(gpu => gpu.label.toLocaleLowerCase().includes(normalized)) ?? null
}

function supportedBackendVersions(backend: BackendOption | undefined, systemId: string | null): string[] {
  if (!backend) return []
  if (!systemId) return backend.versions
  const systemVersions = backend.systems[systemId]
  return systemVersions?.length ? systemVersions : []
}

function matchingPredictionVersion(backend: string | null, measuredVersion: string | null, supportedVersions: string[]): string | null {
  if (!measuredVersion) return null
  const normalized = backend?.toLowerCase() === 'vllm' ? measuredVersion.replace(/^v(?=\d)/i, '') : measuredVersion
  return supportedVersions.find(version => version === normalized) ?? null
}

function choosePredictionLevels(points: GroundTruthPoint[], limit = 24): GroundTruthPoint[] {
  if (points.length <= limit) return points
  const indexes = new Set<number>()
  for (let point = 0; point < limit; point += 1) {
    indexes.add(Math.round((point * (points.length - 1)) / (limit - 1)))
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

export default function GroundTruthComparisonPage() {
  const { simulator } = useSettings()
  const { gpuOptions, modelOptions, modelSpecs, backendOptions, simulatorSupport, isLoading: catalogLoading, error: catalogError } = useCatalog()
  const support = simulatorSupport[simulator]
  const predictionBackendOptions = support?.backends ?? backendOptions
  const predictionModelIds = support?.models ?? modelOptions
  const predictionSystemIds = support?.systems ?? gpuOptions.map(gpu => gpu.systemId)
  const simulatorLabel = simulator === 'blis' ? 'BLIS' : 'AISimulate'
  const [filters, setFilters] = React.useState<GroundTruthFilters>(EMPTY_FILTERS)
  const [filterOptions, setFilterOptions] = React.useState<Record<GroundTruthFilterKey, string[]>>({
    accelerator: [], model: [], backend: [], backendVersion: [], isl: [], osl: [], tp: [],
  })
  const [configurations, setConfigurations] = React.useState<ConfigurationOption[]>([])
  const [configurationKey, setConfigurationKey] = React.useState('')
  const [source, setSource] = React.useState<DatasetSource | null>(null)
  const [inferencexSource, setInferenceXSource] = React.useState<InferenceXSource | null>(null)
  const [inferencexRunUrl, setInferenceXRunUrl] = React.useState<string | null>(null)
  const [inferencexDate, setInferenceXDate] = React.useState<string | null>(null)
  const [inferencexRuns, setInferenceXRuns] = React.useState<InferenceXRun[]>([])
  const [groundTruth, setGroundTruth] = React.useState<GroundTruthPoint[]>([])
  const [inferencex, setInferenceX] = React.useState<GroundTruthPoint[]>([])
  const [estimates, setEstimates] = React.useState<EstimatePoint[]>([])
  const [predictedBy, setPredictedBy] = React.useState<Simulator | null>(null)
  const [groundTruthState, setGroundTruthState] = React.useState<'idle' | 'loaded' | 'empty' | 'unavailable'>('idle')
  const [isLoadingDataset, setIsLoadingDataset] = React.useState(true)
  const [isComparing, setIsComparing] = React.useState(false)
  const [notice, setNotice] = React.useState<Notice | null>(null)
  const filterRequestId = React.useRef(0)
  const rootFilterOptions = React.useRef<Record<GroundTruthFilterKey, string[]> | null>(null)

  const [predictionBackend, setPredictionBackend] = React.useState('')
  const [predictionBackendVersion, setPredictionBackendVersion] = React.useState('')

  const currentGpu = React.useMemo(
    () => findCatalogGpu(filters.accelerator, gpuOptions),
    [filters.accelerator, gpuOptions],
  )
  const selectedConfiguration = configurations.find(option => option.key === configurationKey)
  const measuredBackend = selectedConfiguration?.values.backend ?? null
  const measuredVersion = selectedConfiguration?.values.backend_version ?? null
  const predictionGpuSupported = Boolean(currentGpu && predictionSystemIds.includes(currentGpu.systemId))
  const predictionModelSupported = predictionModelIds.includes(filters.model)
  const measuredPredictionBackend = predictionBackendOptions.find(backend => backend.id.toLowerCase() === measuredBackend?.toLowerCase())
  const matchingVersion = predictionGpuSupported && predictionModelSupported && currentGpu && measuredPredictionBackend
    ? matchingPredictionVersion(measuredBackend, measuredVersion, supportedBackendVersions(measuredPredictionBackend, currentGpu.systemId))
    : null
  const exactPrediction = measuredPredictionBackend && matchingVersion
    ? { backend: measuredPredictionBackend.id, version: matchingVersion }
    : null
  const predictionControlsLocked = exactPrediction !== null
  const activePredictionBackend = exactPrediction?.backend ?? predictionBackend
  const activePredictionVersion = exactPrediction?.version ?? predictionBackendVersion
  const currentBackend = predictionBackendOptions.find(backend => backend.id === activePredictionBackend)
  const backendVersions = currentBackend?.versions ?? []
  const availableBackendVersions = supportedBackendVersions(currentBackend, currentGpu?.systemId ?? null)
  const disaggregatedMeasurement = selectedConfiguration?.values.measurement_source === 'InferenceX'
    && Boolean(selectedConfiguration.values.inferencex_topology?.startsWith('disaggregated'))
  const predictionRuntimeMismatch = Boolean(selectedConfiguration && (!predictionControlsLocked || disaggregatedMeasurement))
  const modelItems = React.useMemo<ComboBoxItem[]>(() => filterOptions.model.map(model => ({
    value: model,
    label: model,
    group: model.includes('/') ? model.slice(0, model.indexOf('/')) : 'Models',
  })), [filterOptions.model])

  const loadFilters: (nextFilters: GroundTruthFilters, refresh?: boolean) => Promise<void> = React.useCallback(async (nextFilters: GroundTruthFilters, refresh = false) => {
    const requestId = filterRequestId.current + 1
    filterRequestId.current = requestId
    setIsLoadingDataset(true)
    try {
      const params = new URLSearchParams({ mode: 'filters' })
      for (const key of GROUND_TRUTH_FILTER_KEYS) {
        if (nextFilters[key]) params.set(key, nextFilters[key])
      }
      if (refresh) params.set('refresh', 'true')
      const response = await fetch(`/api/validation/ground-truth?${params.toString()}`, { cache: 'no-store' })
      const data = await response.json() as FilterResponse
      if (!response.ok || data.status !== 'ok' || !data.options || (!data.source && !data.inferencexSource)) {
        throw new Error(data.error ?? `Dataset request failed (${response.status})`)
      }
      if (requestId !== filterRequestId.current) return
      const options = data.options
      const invalidKey = [...GROUND_TRUTH_FILTER_KEYS].reverse().find(key =>
        nextFilters[key] && !options[key].includes(nextFilters[key]),
      )
      if (invalidKey) {
        const corrected = { ...nextFilters }
        for (const key of GROUND_TRUTH_FILTER_KEYS.slice(GROUND_TRUTH_FILTER_KEYS.indexOf(invalidKey))) {
          corrected[key] = ''
        }
        setFilters(corrected)
        setConfigurationKey('')
        void loadFilters(corrected)
        return
      }
      if (GROUND_TRUTH_FILTER_KEYS.every(key => !nextFilters[key])) {
        rootFilterOptions.current = options
      }
      setFilterOptions(options)
      setConfigurations(data.configurations ?? [])
      setSource(data.source ?? null)
      setInferenceXSource(data.inferencexSource ?? null)
      setGroundTruthState(previous => previous === 'unavailable' ? 'idle' : previous)
      setConfigurationKey(previous => {
        if ((data.configurations ?? []).some(option => option.key === previous)) return previous
        return data.configurations?.length === 1 ? data.configurations[0].key : ''
      })
      setNotice(data.groundTruthError ? {
        variant: 'warning', title: 'Internal GT unavailable',
        message: `InferenceX measurements remain available. ${data.groundTruthError}`,
      } : data.inferencexError ? {
        variant: 'warning', title: 'InferenceX unavailable',
        message: `Internal GT measurements remain available. ${data.inferencexError}`,
      } : refresh ? {
        variant: 'success',
        title: 'Dataset refreshed',
        message: `${(data.source?.rowCount ?? 0).toLocaleString('en-US')} internal GT rows and ${(data.inferencexSource?.rowCount ?? 0).toLocaleString('en-US')} InferenceX rows available.`,
      } : null)
    } catch (error) {
      if (requestId === filterRequestId.current) {
        setGroundTruthState(previous => previous === 'loaded' ? previous : 'unavailable')
        setNotice({
          variant: 'danger',
          title: 'Ground truth data unavailable',
          message: error instanceof Error ? error.message : 'Could not load the Hugging Face dataset.',
        })
      }
    } finally {
      if (requestId === filterRequestId.current) setIsLoadingDataset(false)
    }
  }, [])

  React.useEffect(() => {
    void loadFilters(EMPTY_FILTERS)
  }, [loadFilters])

  React.useEffect(() => {
    if (!predictionBackend && predictionBackendOptions.length > 0) {
      const preferred = predictionBackendOptions.find(option => option.id === 'vllm') ?? predictionBackendOptions[0]
      setPredictionBackend(preferred.id)
    }
  }, [predictionBackendOptions, predictionBackend])

  // A simulator change invalidates the backend/version choice and any estimates
  // produced by the previous simulator.
  React.useEffect(() => {
    setEstimates([])
    setPredictedBy(null)
    setPredictionBackendVersion('')
    setPredictionBackend(current =>
      predictionBackendOptions.some(option => option.id === current) ? current : '')
  }, [simulator, predictionBackendOptions])

  React.useEffect(() => {
    if (!selectedConfiguration || !measuredPredictionBackend || !currentGpu) return
    setPredictionBackend(measuredPredictionBackend.id)
    setPredictionBackendVersion(exactPrediction?.version ?? '')
  }, [selectedConfiguration, measuredPredictionBackend, currentGpu, exactPrediction?.version])

  React.useEffect(() => {
    if (!currentBackend) return
    if (predictionControlsLocked) {
      if (predictionBackendVersion !== exactPrediction?.version) setPredictionBackendVersion(exactPrediction?.version ?? '')
      return
    }
    if (!currentGpu) {
      if (predictionBackendVersion) setPredictionBackendVersion('')
      return
    }
    if (!availableBackendVersions.includes(predictionBackendVersion)) {
      setPredictionBackendVersion(
        [...currentBackend.versions].reverse().find(version => availableBackendVersions.includes(version)) ?? '',
      )
    }
  }, [availableBackendVersions, currentBackend, currentGpu, predictionBackendVersion, predictionControlsLocked, exactPrediction?.version])

  function handleFilterChange(key: GroundTruthFilterKey, value: string) {
    if (isComparing) return
    const changedIndex = GROUND_TRUTH_FILTER_KEYS.indexOf(key)
    const next = { ...filters, [key]: value }
    for (const downstreamKey of GROUND_TRUTH_FILTER_KEYS.slice(changedIndex + 1)) next[downstreamKey] = ''
    setFilters(next)
    setGroundTruth([])
    setGroundTruthState('idle')
    setInferenceX([])
    setInferenceXRunUrl(null)
    setInferenceXDate(null)
    setInferenceXRuns([])
    setEstimates([])
    setConfigurationKey('')
    setNotice(null)
    void loadFilters(next)
  }

  function resetInputs() {
    const options = rootFilterOptions.current
    if (!options || isComparing) return
    filterRequestId.current += 1
    setIsLoadingDataset(false)
    setFilters(EMPTY_FILTERS)
    setFilterOptions(options)
    setConfigurations([])
    setConfigurationKey('')
    setGroundTruth([])
    setInferenceX([])
    setInferenceXRunUrl(null)
    setInferenceXDate(null)
    setInferenceXRuns([])
    setEstimates([])
    setPredictedBy(null)
    setGroundTruthState('idle')
    setNotice(null)
    const defaultBackend = predictionBackendOptions.find(option => option.id === 'vllm') ?? predictionBackendOptions[0]
    setPredictionBackend(defaultBackend?.id ?? '')
    setPredictionBackendVersion('')
  }

  async function runComparison() {
    if (GROUND_TRUTH_FILTER_KEYS.some(key => !filters[key]) || !selectedConfiguration) {
      setNotice({
        variant: 'warning',
        title: 'Choose a complete benchmark configuration',
        message: 'Select the GPU, model, serving backend and version, token lengths, tensor parallel size, and measured runtime configuration.',
      })
      return
    }

    setIsComparing(true)
    setPredictedBy(null)
    setGroundTruthState('idle')
    setNotice({ variant: 'info', title: 'Loading comparison', message: 'Fetching the selected measured benchmark rows.' })
    setGroundTruth([])
    setInferenceX([])
    setInferenceXRunUrl(null)
    setInferenceXDate(null)
    setInferenceXRuns([])
    setEstimates([])

    try {
      const params = new URLSearchParams({ mode: 'data', configuration: selectedConfiguration.key })
      for (const key of GROUND_TRUTH_FILTER_KEYS) params.set(key, filters[key])
      const response = await fetch(`/api/validation/ground-truth?${params.toString()}`, { cache: 'no-store' })
      const data = await response.json() as GroundTruthResponse
      if (!response.ok || data.status !== 'ok' || !data.groundTruthPoints || !data.inferencexPoints) {
        throw new Error(data.error ?? `Comparison request failed (${response.status})`)
      }
      const internalPoints = data.groundTruthPoints
      const inferencexPoints = data.inferencexPoints
      const measuredLevels = [...new Map([...internalPoints, ...inferencexPoints]
        .map(point => [point.concurrency, point])).values()].sort((left, right) => left.concurrency - right.concurrency)
      if (measuredLevels.length === 0) {
        setGroundTruthState('empty')
        setNotice({ variant: 'warning', title: 'No matching benchmark rows', message: 'Try another benchmark configuration or adjust the filters.' })
        return
      }

      setSource(data.source ?? null)
      setInferenceXSource(data.inferencexSource ?? null)
      setInferenceXRunUrl(data.inferencexRunUrl ?? null)
      setInferenceXDate(data.inferencexDate ?? null)
      setInferenceXRuns(data.inferencexRuns ?? [])
      setGroundTruth(internalPoints)
      setInferenceX(inferencexPoints)
      setGroundTruthState('loaded')

      if (disaggregatedMeasurement) {
        setNotice({
          variant: 'warning',
          title: 'Measurements loaded; prediction is not comparable',
          message: 'This InferenceX configuration uses a disaggregated prefill/decode topology. AISimulators predicts an aggregated deployment for this selection, so no prediction is overlaid and no percentage difference is shown.',
        })
        return
      }

      if (!currentGpu || !currentBackend || !activePredictionVersion
        || !availableBackendVersions.includes(activePredictionVersion)
        || !predictionGpuSupported || !predictionModelSupported) {
        setNotice({
          variant: 'warning',
          title: `Measurements loaded; ${simulatorLabel} cannot predict this selection`,
          message: predictionBackendOptions.length === 0
            ? 'AISimulators serving backends are unavailable. The catalog will retry automatically; measured results remain available below.'
            : `The selected model, accelerator, or serving backend is not supported by ${simulatorLabel}. Measured results remain available below.`,
        })
        return
      }

      setNotice({
        variant: 'info',
        title: `Running ${simulatorLabel} predictions`,
        message: `Predicting ${Math.min(measuredLevels.length, 24)} measured concurrency levels. Measured data stays visible while predictions run.`,
      })

      const catalogModelSpec = modelSpecs.get(filters.model)
      let modelConfig: Record<string, unknown> | null = null
      if (!modelOptions.includes(filters.model) || !catalogModelSpec) {
        try {
          const configResponse = await fetch(`/api/hf-config?model=${encodeURIComponent(filters.model)}`)
          const configData = await configResponse.json() as { config?: Record<string, unknown>; message?: string }
          if (!configResponse.ok || !configData.config) throw new Error(configData.message ?? 'Model configuration unavailable')
          modelConfig = configData.config
        } catch (error) {
          setNotice({
            variant: 'warning',
            title: 'Measurements loaded; prediction is unavailable',
            message: error instanceof Error ? error.message : 'AISimulators model configuration is unavailable.',
          })
          return
        }
      }

      const predictionLevels = choosePredictionLevels(measuredLevels)
      const attempts = await mapWithLimit(predictionLevels, 3, async (measured): Promise<EstimateAttempt> => {
        try {
          const result = await fetchEstimateAsInferenceResult({
            model_path: filters.model,
            system: currentGpu.systemId,
            isl: Number(filters.isl),
            osl: Number(filters.osl),
            batch_size: measured.concurrency,
            requests_per_load_unit: PREDICTION_REQUESTS_PER_LOAD_UNIT,
            tp_size: Number(filters.tp),
            backend: activePredictionBackend,
            backend_version: activePredictionVersion,
            simulator,
            moe_ep_size: (catalogModelSpec?.num_experts ?? 0) > 1
              ? Number(filters.tp) : undefined,
            hf_model_config: modelConfig,
          })
          const performance = result.performance
          if (!performance) throw new Error(`${simulatorLabel} did not return performance metrics.`)
          const concurrency = measured.concurrency
          const outputTokensPerSecond = performance.throughput_tokens_per_sec
          const outputTokensPerSecondPerUser = performance.throughput_tokens_per_sec_per_user
            ?? (concurrency > 0 ? outputTokensPerSecond / concurrency : null)
          return {
            simulator: result.simulator,
            error: null,
            point: {
              concurrency,
              ttftSeconds: performance.ttft_ms / 1000,
              tpotSeconds: performance.tpot_ms / 1000,
              outputTokensPerSecond,
              outputTokensPerSecondPerUser,
              requestLatencySeconds: performance.request_latency_ms / 1000,
            },
          }
        } catch (error) {
          return {
            simulator: null,
            point: null,
            error: error instanceof Error ? error.message : `${simulatorLabel} prediction failed`,
          }
        }
      })
      const successfulEstimates = attempts.flatMap(attempt => attempt.point ? [attempt.point] : [])
      const failures = attempts.flatMap(attempt => attempt.error ? [attempt.error] : [])
      const actualSimulators = new Set(attempts.flatMap(attempt => attempt.simulator ? [attempt.simulator] : []))
      setPredictedBy(actualSimulators.size === 1 ? [...actualSimulators][0] : null)
      setEstimates(successfulEstimates)
      if (failures.length > 0) {
        const firstError = failures[0]
        setNotice({
          variant: successfulEstimates.length > 0 ? 'warning' : 'danger',
          title: successfulEstimates.length > 0 ? 'Some predictions could not be calculated' : `${simulatorLabel} could not predict this configuration`,
          message: `${successfulEstimates.length} of ${predictionLevels.length} predictions completed.${firstError ? ` ${firstError}` : ''}`,
        })
      } else {
        setNotice({
          variant: 'success',
          title: 'Comparison complete',
          message: `${data.matchedRows?.toLocaleString('en-US') ?? 0} measured rows across ${measuredLevels.length} concurrency levels.`,
        })
      }
    } catch (error) {
      setGroundTruthState(previous => previous === 'loaded' ? previous : 'unavailable')
      setNotice({
        variant: 'danger',
        title: 'Comparison failed',
        message: error instanceof Error ? error.message : 'Could not compare this benchmark configuration.',
      })
    } finally {
      setIsComparing(false)
    }
  }

  const modelSelected = GROUND_TRUTH_FILTER_KEYS.every(key => Boolean(filters[key]))
  const canCompare = modelSelected && Boolean(selectedConfiguration) && !isLoadingDataset && !isComparing
  const datasetDate = source?.lastModified ? new Date(source.lastModified).toLocaleDateString() : null
  const measuredCount = new Set([...groundTruth, ...inferencex].map(point => point.concurrency)).size
  const safeInferenceXRuns = inferencexRuns.filter(run => isHttpsUrl(run.url))
  const predictionLabel = predictedBy === 'blis' ? 'BLIS' : predictedBy === 'aisimulate' ? 'AISimulate' : simulatorLabel

  return (
    <PageSection className={styles.page}>
      <div className={styles.heading}>
        <div>
          <p className={styles.eyebrow}>Validation</p>
          <h1>Ground truth comparison</h1>
          <p className={styles.intro}>
            Compare measured benchmark data with AISimulators predictions across concurrency levels.
          </p>
        </div>
        {(source || inferencexSource) && (
          <div className={styles.sourceBadges}>
            {source && <div className={styles.sourceBadge}>
              <a href={source.url} target="_blank" rel="noreferrer">Internal GT · Hugging Face</a>
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

      {isLoadingDataset && !source && !inferencexSource && (
        <Alert variant="info" isInline title="Loading benchmark data" className={styles.notice}>
          <span className={styles.loadingCopy}><Spinner size="sm" /> Loading measured benchmark rows.</span>
        </Alert>
      )}
      {notice && !isLoadingDataset && (
        <Alert variant={notice.variant} isInline title={notice.title} className={styles.notice}>
          <p>{notice.message}</p>
          {notice.variant === 'danger' && !source && !inferencexSource && (
            <Button variant="link" onClick={() => void loadFilters(filters, true)}>Retry dataset load</Button>
          )}
        </Alert>
      )}

      <section className={styles.filterCard} aria-labelledby="comparison-filters-title" aria-busy={isLoadingDataset}>
        <div className={styles.cardHead}>
          <div>
            <h2 id="comparison-filters-title">Comparison inputs</h2>
            <p>Each option comes from benchmark rows that match your preceding selections.</p>
          </div>
          <Button
            variant="secondary"
            isDisabled={isComparing || !rootFilterOptions.current}
            onClick={resetInputs}
          >
            Reset inputs
          </Button>
        </div>

        <fieldset disabled={isComparing} className={styles.filterFieldset}>
          <legend className={styles.visuallyHidden}>Ground truth comparison inputs</legend>
          <div className={styles.filterGrid}>
          {GROUND_TRUTH_FILTER_KEYS.map(key => (
            <div className={styles.field} key={key}>
              {key === 'model' ? (
                <ComboBox
                  id="validation-model"
                  label={FILTER_LABELS[key]}
                  value={filters.model}
                  onChange={value => handleFilterChange('model', value)}
                  items={modelItems}
                  placeholder="Search measured models..."
                  helperText="Type to search or browse the measured models."
                />
              ) : <>
                <label htmlFor={`validation-${key}`}>{FILTER_LABELS[key]}</label>
                <CompareSelect
                  id={`validation-${key}`}
                  label={FILTER_LABELS[key]}
                  value={filters[key]}
                  isDisabled={filterOptions[key].length === 0 || (key === 'backendVersion' && !filters.backend)}
                  onChange={value => handleFilterChange(key, value)}
                  placeholder="Select an option"
                  placeholderIsSelectable
                  options={filterOptions[key].map(value => ({ value, label: optionLabel(key, value) }))}
                />
                {key === 'backendVersion' && <p className={styles.helperText}>Shared measurement version for internal GT and InferenceX. The AISimulators prediction version is selected separately below.</p>}
              </>}
            </div>
          ))}
          {modelSelected && (
            <div className={`${styles.field} ${styles.configurationField}`}>
              <label htmlFor="validation-configuration">Measured runtime configuration</label>
              <CompareSelect
                id="validation-configuration"
                label="Measured runtime configuration"
                value={configurationKey}
                isDisabled={configurations.length === 0 && !isLoadingDataset}
                onChange={value => {
                  setConfigurationKey(value)
                  setGroundTruth([])
                  setGroundTruthState('idle')
                  setInferenceX([])
                  setInferenceXRunUrl(null)
                  setInferenceXDate(null)
                  setInferenceXRuns([])
                  setEstimates([])
                  setNotice(null)
                }}
                placeholder={configurations.length > 1 ? 'Select a measured configuration' : 'No matching runtime configuration'}
                placeholderIsSelectable={configurations.length > 0}
                options={configurations.map(configuration => ({
                  value: configuration.key,
                  label: `${configuration.label} · ${configuration.rows} rows`,
                }))}
              />
              <p className={styles.helperText}>Shows the measurement source and keeps benchmark release, precision, quantization, and serving settings from different runs separate.</p>
            </div>
          )}
          </div>
        </fieldset>

        <div className={styles.predictSettings}>
          <div>
            <div className={styles.predictHeading}>
              <h3>{simulatorLabel} prediction</h3>
              <SimulatorBadge simulator={simulator} />
            </div>
            <p>Use the measured serving backend and version when {simulatorLabel} supports them. Otherwise, choose a supported prediction version.</p>
          </div>
          <div className={styles.predictGrid}>
            <div className={styles.field}>
              <label htmlFor="validation-prediction-backend">Prediction backend</label>
              <CompareSelect
                id="validation-prediction-backend"
                label="Prediction backend"
                value={activePredictionBackend}
                isDisabled={catalogLoading || predictionBackendOptions.length === 0 || isComparing || predictionControlsLocked}
                onChange={value => {
                  setPredictionBackend(value)
                  setPredictionBackendVersion('')
                  setEstimates([])
                }}
                placeholder="Select a backend"
                options={predictionBackendOptions.map(backend => ({ value: backend.id, label: backend.id }))}
              />
            </div>
            <div className={styles.field}>
              <label htmlFor="validation-prediction-version">{simulatorLabel} backend version</label>
              <CompareSelect
                id="validation-prediction-version"
                label={`${simulatorLabel} backend version`}
                value={activePredictionVersion}
                isDisabled={catalogLoading || !predictionGpuSupported || availableBackendVersions.length === 0 || isComparing || predictionControlsLocked}
                onChange={value => {
                  setPredictionBackendVersion(value)
                  setEstimates([])
                }}
                placeholder={!filters.accelerator
                    ? 'Select a GPU accelerator first'
                    : !currentGpu
                      ? 'GPU unavailable in the catalog'
                      : !predictionGpuSupported
                        ? `GPU unavailable in ${simulatorLabel}`
                        : availableBackendVersions.length === 0
                          ? 'No backend versions for this GPU'
                          : 'Select a backend version'}
                options={backendVersions.map(version => ({
                  value: version,
                  label: availableBackendVersions.includes(version) ? version : `${version} (no prediction data for this GPU)`,
                  isDisabled: !availableBackendVersions.includes(version),
                }))}
              />
              <p className={styles.helperText}>Version used for the {simulatorLabel} prediction. All reported versions are listed after choosing a GPU; versions without prediction data for it are disabled.</p>
            </div>
          </div>
          {selectedConfiguration && predictionControlsLocked && (
            <p className={styles.methodNote} role="status">
              {simulatorLabel} supports the measured {measuredBackend} {measuredVersion} on this GPU. The prediction backend and version are set to match.
            </p>
          )}
          {!catalogLoading && backendOptions.length === 0 && (
            <Alert variant="warning" isInline title="Prediction catalog unavailable">
              {catalogError ?? 'AISimulators returned no serving backends.'} The prediction backend and version will become available when the catalog recovers.
            </Alert>
          )}
          {selectedConfiguration && !predictionControlsLocked && !catalogLoading && backendOptions.length > 0 && currentGpu && (
            <Alert variant="warning" isInline title="Exact prediction version unavailable">
              The measured run uses {measuredBackend ?? 'an unreported backend'} {measuredVersion ?? 'an unreported version'}.
              Choose an available {simulatorLabel} backend and version. Its results are a different backend or version reference, so percentage differences are hidden.
            </Alert>
          )}
          <p className={styles.methodNote}>
            SemiAnalysis InferenceX is shown as a separate measured source. This snapshot includes 8k/1k single-turn runs for ConfigIQ approved models.
            {safeInferenceXRuns.length > 0 ? (
              <>
                {' '}
                {safeInferenceXRuns.length === 1 ? 'Selected run' : `${safeInferenceXRuns.length} averaged runs`}
                {safeInferenceXRuns.some(run => run.date) && ` (${safeInferenceXRuns.map(run => run.date).filter(Boolean).join(', ')})`}:{' '}
                {safeInferenceXRuns.map((run, index) => (
                  <React.Fragment key={run.url}>
                    {index > 0 && ', '}
                    <a href={run.url} target="_blank" rel="noreferrer">Run {index + 1}</a>
                  </React.Fragment>
                ))}
                .
              </>
            ) : inferencexDate ? <> Selected run: {inferencexDate}.</> : null}
            {safeInferenceXRuns.length === 0 && isHttpsUrl(inferencexRunUrl) && <> <a href={inferencexRunUrl} target="_blank" rel="noreferrer">View the InferenceX run</a>.</>}
          </p>
        </div>

        <div className={styles.filterFooter}>
          <p>Measurements are grouped by source and target concurrency. Repeated runs within each source are averaged.</p>
          <Button variant="primary" isDisabled={!canCompare} onClick={() => void runComparison()}>
            {isComparing && <Spinner size="sm" className={styles.buttonSpinner} />}
            {isComparing ? 'Comparing…' : 'Compare runs'}
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
            {groundTruth.length > 0 && <span><i className={styles.groundTruthSwatch} />Internal GT</span>}
            {inferencex.length > 0 && <span><i className={styles.inferencexSwatch} />InferenceX · SemiAnalysis</span>}
            {estimates.length > 0 && <span><i className={styles.aisimSwatch} />{predictionLabel} · {activePredictionBackend} {activePredictionVersion}{predictionRuntimeMismatch ? ' (different backend/version)' : ''}</span>}
          </div>
        </div>
        {measuredCount > 0 ? (
          <ComparisonPreview groundTruth={groundTruth} inferencex={inferencex} estimates={estimates} showDifference={!predictionRuntimeMismatch} predictionLabel={predictionLabel} />
        ) : groundTruthState === 'unavailable' ? (
          <div className={styles.emptyResults}>
            <h3>Measurements are unavailable</h3>
            <p>The benchmark datasets could not be loaded.</p>
            <Button variant="link" onClick={() => void loadFilters(filters, true)}>Retry dataset load</Button>
          </div>
        ) : groundTruthState === 'empty' ? (
          <div className={styles.emptyResults}>
            <h3>No matching benchmark rows</h3>
            <p>No measured rows match this configuration. Adjust the filters and try again.</p>
          </div>
        ) : (
          <div className={styles.emptyResults}>
            <h3>Choose benchmark inputs to begin</h3>
            <p>Select an internal GT or InferenceX runtime configuration to compare its measurements with AISimulators predictions.</p>
          </div>
        )}
      </section>
    </PageSection>
  )
}

'use client'

import * as React from 'react'
import {
  Alert,
  Button,
  FormSelect,
  FormSelectOption,
  PageSection,
  Spinner,
} from '@patternfly/react-core'
import { ComboBox, type ComboBoxItem } from '@/components/ModelComboBox/ModelComboBox'
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

interface FilterResponse {
  status: 'ok' | 'failed'
  error?: string
  source?: DatasetSource
  options?: Record<GroundTruthFilterKey, string[]>
  configurations?: ConfigurationOption[]
}

interface GroundTruthResponse {
  status: 'ok' | 'failed'
  error?: string
  source?: DatasetSource
  matchedRows?: number
  points?: GroundTruthPoint[]
}

interface EstimateAttempt {
  point: EstimatePoint | null
  error: string | null
}

const FILTER_LABELS: Record<GroundTruthFilterKey, string> = {
  accelerator: 'GPU accelerator',
  model: 'Model',
  version: 'Benchmark version',
  isl: 'Input tokens',
  osl: 'Output tokens',
  tp: 'Tensor parallel size',
}

const EMPTY_FILTERS: GroundTruthFilters = {
  accelerator: '',
  model: '',
  version: '',
  isl: '',
  osl: '',
  tp: '',
}

const PREDICTION_REQUESTS_PER_LOAD_UNIT = 10

function optionLabel(key: GroundTruthFilterKey, value: string): string {
  if (key === 'isl' || key === 'osl') return `${Number(value).toLocaleString('en-US')} tokens`
  if (key === 'tp') return `TP ${value}`
  return value
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
  const { gpuOptions, modelOptions, modelSpecs, backendOptions, isLoading: catalogLoading } = useCatalog()
  const [filters, setFilters] = React.useState<GroundTruthFilters>(EMPTY_FILTERS)
  const [filterOptions, setFilterOptions] = React.useState<Record<GroundTruthFilterKey, string[]>>({
    accelerator: [], model: [], version: [], isl: [], osl: [], tp: [],
  })
  const [configurations, setConfigurations] = React.useState<ConfigurationOption[]>([])
  const [configurationKey, setConfigurationKey] = React.useState('')
  const [source, setSource] = React.useState<DatasetSource | null>(null)
  const [groundTruth, setGroundTruth] = React.useState<GroundTruthPoint[]>([])
  const [estimates, setEstimates] = React.useState<EstimatePoint[]>([])
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
  const currentBackend = backendOptions.find(backend => backend.id === predictionBackend)
  const backendVersions = currentBackend?.versions ?? []
  const availableBackendVersions = supportedBackendVersions(currentBackend, currentGpu?.systemId ?? null)
  const selectedConfiguration = configurations.find(option => option.key === configurationKey)
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
      if (!response.ok || data.status !== 'ok' || !data.options || !data.source) {
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
      setSource(data.source)
      setConfigurationKey(previous => {
        if ((data.configurations ?? []).some(option => option.key === previous)) return previous
        return data.configurations?.length === 1 ? data.configurations[0].key : ''
      })
      setNotice(refresh ? {
        variant: 'success',
        title: 'Dataset refreshed',
        message: `${data.source.rowCount.toLocaleString('en-US')} rows loaded from Hugging Face.`,
      } : null)
    } catch (error) {
      if (requestId === filterRequestId.current) {
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
    if (!predictionBackend && backendOptions.length > 0) {
      const preferred = backendOptions.find(option => option.id === 'vllm') ?? backendOptions[0]
      setPredictionBackend(preferred.id)
    }
  }, [backendOptions, predictionBackend])

  React.useEffect(() => {
    if (!currentBackend) return
    if (!currentGpu) {
      if (predictionBackendVersion) setPredictionBackendVersion('')
      return
    }
    if (!availableBackendVersions.includes(predictionBackendVersion)) {
      setPredictionBackendVersion(
        [...currentBackend.versions].reverse().find(version => availableBackendVersions.includes(version)) ?? '',
      )
    }
  }, [availableBackendVersions, currentBackend, currentGpu, predictionBackendVersion])

  function handleFilterChange(key: GroundTruthFilterKey, value: string) {
    if (isComparing) return
    const changedIndex = GROUND_TRUTH_FILTER_KEYS.indexOf(key)
    const next = { ...filters, [key]: value }
    for (const downstreamKey of GROUND_TRUTH_FILTER_KEYS.slice(changedIndex + 1)) next[downstreamKey] = ''
    setFilters(next)
    setGroundTruth([])
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
    setEstimates([])
    setNotice(null)
    const defaultBackend = backendOptions.find(option => option.id === 'vllm') ?? backendOptions[0]
    setPredictionBackend(defaultBackend?.id ?? '')
    setPredictionBackendVersion('')
  }

  async function runComparison() {
    if (GROUND_TRUTH_FILTER_KEYS.some(key => !filters[key]) || !selectedConfiguration) {
      setNotice({
        variant: 'warning',
        title: 'Choose a complete benchmark configuration',
        message: 'Select the GPU, model, benchmark version, token lengths, tensor parallel size, and measured runtime configuration.',
      })
      return
    }

    setIsComparing(true)
    setNotice({ variant: 'info', title: 'Loading comparison', message: 'Fetching the selected measured benchmark rows.' })
    setGroundTruth([])
    setEstimates([])

    try {
      const params = new URLSearchParams({ mode: 'data', configuration: selectedConfiguration.key })
      for (const key of GROUND_TRUTH_FILTER_KEYS) params.set(key, filters[key])
      const response = await fetch(`/api/validation/ground-truth?${params.toString()}`, { cache: 'no-store' })
      const data = await response.json() as GroundTruthResponse
      if (!response.ok || data.status !== 'ok' || !data.source || !data.points) {
        throw new Error(data.error ?? `Comparison request failed (${response.status})`)
      }
      if (data.points.length === 0) {
        setNotice({ variant: 'warning', title: 'No matching benchmark rows', message: 'Try another benchmark configuration or adjust the filters.' })
        return
      }

      setSource(data.source)
      setGroundTruth(data.points)

      if (!currentGpu || !currentBackend || !predictionBackendVersion || !availableBackendVersions.includes(predictionBackendVersion)) {
        setNotice({
          variant: 'warning',
          title: 'Ground truth loaded; prediction is unavailable for this selection',
          message: 'The selected accelerator or serving backend is not supported by the AISimulators catalog. Measured results remain available below.',
        })
        return
      }

      setNotice({
        variant: 'info',
        title: 'Running AISimulators predictions',
        message: `Predicting ${Math.min(data.points.length, 24)} measured concurrency levels. Measured data stays visible while predictions run.`,
      })

      const catalogModelSpec = modelSpecs.get(filters.model)
      let modelConfig: Record<string, unknown> | null = null
      if (!modelOptions.includes(filters.model) || !catalogModelSpec) {
        const configResponse = await fetch(`/api/hf-config?model=${encodeURIComponent(filters.model)}`)
        const configData = await configResponse.json() as { config?: Record<string, unknown>; message?: string }
        if (!configResponse.ok || !configData.config) {
          throw new Error(configData.message ?? 'Could not load this model configuration from Hugging Face.')
        }
        modelConfig = configData.config
      }

      const predictionLevels = choosePredictionLevels(data.points)
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
            backend: predictionBackend,
            backend_version: predictionBackendVersion,
            moe_ep_size: (catalogModelSpec?.num_experts ?? 0) > 1
              ? Number(filters.tp) : undefined,
            hf_model_config: modelConfig,
          })
          const performance = result.performance
          if (!performance) throw new Error('AISimulators did not return performance metrics.')
          const concurrency = measured.concurrency
          const outputTokensPerSecondPerUser = performance.throughput_tokens_per_sec
          return {
            error: null,
            point: {
              concurrency,
              ttftSeconds: performance.ttft_ms / 1000,
              tpotSeconds: performance.tpot_ms / 1000,
              outputTokensPerSecond: outputTokensPerSecondPerUser * concurrency,
              outputTokensPerSecondPerUser,
              requestLatencySeconds: performance.request_latency_ms / 1000,
            },
          }
        } catch (error) {
          return {
            point: null,
            error: error instanceof Error ? error.message : 'AISimulators prediction failed',
          }
        }
      })
      const successfulEstimates = attempts.flatMap(attempt => attempt.point ? [attempt.point] : [])
      const failures = attempts.flatMap(attempt => attempt.error ? [attempt.error] : [])
      setEstimates(successfulEstimates)
      if (failures.length > 0) {
        const firstError = failures[0]
        setNotice({
          variant: successfulEstimates.length > 0 ? 'warning' : 'danger',
          title: successfulEstimates.length > 0 ? 'Some predictions could not be calculated' : 'AISimulators could not predict this configuration',
          message: `${successfulEstimates.length} of ${predictionLevels.length} predictions completed.${firstError ? ` ${firstError}` : ''}`,
        })
      } else {
        setNotice({
          variant: 'success',
          title: 'Comparison complete',
          message: `${data.matchedRows?.toLocaleString('en-US') ?? 0} matching ground truth rows across ${data.points.length} concurrency levels.`,
        })
      }
    } catch (error) {
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
        {source && (
          <div className={styles.sourceBadge}>
            <a href={source.url} target="_blank" rel="noreferrer">Hugging Face dataset</a>
            <span>{source.rowCount.toLocaleString('en-US')} rows · CC BY 4.0{datasetDate ? ` · updated ${datasetDate}` : ''}</span>
          </div>
        )}
      </div>

      {isLoadingDataset && !source && (
        <Alert variant="info" isInline title="Loading benchmark data" className={styles.notice}>
          <span className={styles.loadingCopy}><Spinner size="sm" /> Loading current rows from the Hugging Face dataset.</span>
        </Alert>
      )}
      {notice && !isLoadingDataset && (
        <Alert variant={notice.variant} isInline title={notice.title} className={styles.notice}>
          <p>{notice.message}</p>
          {notice.variant === 'danger' && !source && (
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
                <FormSelect
                  id={`validation-${key}`}
                  value={filters[key]}
                  isDisabled={filterOptions[key].length === 0}
                  onChange={(_event, value) => handleFilterChange(key, value)}
                  aria-label={FILTER_LABELS[key]}
                >
                  <FormSelectOption value="" label="Select an option" />
                  {filterOptions[key].map(value => (
                    <FormSelectOption key={value} value={value} label={optionLabel(key, value)} />
                  ))}
                </FormSelect>
              </>}
            </div>
          ))}
          {modelSelected && (
            <div className={`${styles.field} ${styles.configurationField}`}>
              <label htmlFor="validation-configuration">Measured runtime configuration</label>
              <FormSelect
                id="validation-configuration"
                value={configurationKey}
                isDisabled={configurations.length === 0 && !isLoadingDataset}
                onChange={(_event, value) => {
                  setConfigurationKey(value)
                  setGroundTruth([])
                  setEstimates([])
                  setNotice(null)
                }}
                aria-label="Measured runtime configuration"
              >
                <FormSelectOption value="" label={configurations.length > 1 ? 'Select a measured configuration' : 'No matching runtime configuration'} />
                {configurations.map(configuration => (
                  <FormSelectOption key={configuration.key} value={configuration.key} label={`${configuration.label} · ${configuration.rows} rows`} />
                ))}
              </FormSelect>
              <p className={styles.helperText}>Keeps backend, precision, quantization, and serving settings from different runs separate.</p>
            </div>
          )}
          </div>
        </fieldset>

        <div className={styles.predictSettings}>
          <div>
            <h3>AISimulators prediction</h3>
            <p>Choose the serving backend used to generate the comparison series.</p>
          </div>
          <div className={styles.predictGrid}>
            <div className={styles.field}>
              <label htmlFor="validation-prediction-backend">Prediction backend</label>
              <FormSelect
                id="validation-prediction-backend"
                value={predictionBackend}
                isDisabled={catalogLoading || backendOptions.length === 0 || isComparing}
                onChange={(_event, value) => {
                  setPredictionBackend(value)
                  setPredictionBackendVersion('')
                  setEstimates([])
                }}
              >
                {backendOptions.map(backend => <FormSelectOption key={backend.id} value={backend.id} label={backend.id} />)}
              </FormSelect>
            </div>
            <div className={styles.field}>
              <label htmlFor="validation-prediction-version">AISimulators backend version</label>
              <FormSelect
                id="validation-prediction-version"
                value={predictionBackendVersion}
                isDisabled={catalogLoading || !currentGpu || availableBackendVersions.length === 0 || isComparing}
                onChange={(_event, value) => {
                  setPredictionBackendVersion(value)
                  setEstimates([])
                }}
              >
                <FormSelectOption
                  value=""
                  label={!filters.accelerator
                    ? 'Select a GPU accelerator first'
                    : !currentGpu
                      ? 'GPU unavailable in AISimulators'
                      : availableBackendVersions.length === 0
                        ? 'No backend versions for this GPU'
                        : 'Select a backend version'}
                  isDisabled
                />
                {backendVersions.map(version => (
                  <FormSelectOption
                    key={version}
                    value={version}
                    label={availableBackendVersions.includes(version) ? version : `${version} (no prediction data for this GPU)`}
                    isDisabled={!availableBackendVersions.includes(version)}
                  />
                ))}
              </FormSelect>
              <p className={styles.helperText}>Version used for the AISimulators prediction. All reported versions are listed after choosing a GPU; versions without prediction data for it are disabled.</p>
            </div>
          </div>
          <p className={styles.methodNote}>
            Ground-truth TTFT and TPOT are converted from milliseconds; request latency is already in seconds. Predicted TPOT uses AISimulators&apos; reported TPOT, which excludes TTFT. Predictions use the selected model, accelerator, token lengths, TP, concurrency, and prediction backend, with 10 simulated requests per concurrency slot. Aggregate predicted throughput is the per-user rate multiplied by concurrency; other measured runtime settings are not inferred by AISimulators.
          </p>
        </div>

        <div className={styles.filterFooter}>
          <p>Ground truth is grouped by target concurrency. Repeated rows for the same setting are averaged.</p>
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
            <p>{groundTruth.length > 0 ? `${groundTruth.length} measured concurrency levels` : 'Run a comparison to load measured and predicted performance.'}</p>
          </div>
          <div className={styles.legend} aria-label="Chart series">
            <span><i className={styles.groundTruthSwatch} />Ground truth</span>
            <span><i className={styles.aisimSwatch} />AISimulators</span>
          </div>
        </div>
        {groundTruth.length > 0 ? (
          <ComparisonPreview groundTruth={groundTruth} estimates={estimates} />
        ) : (
          <div className={styles.emptyResults}>
            <h3>Choose benchmark inputs to begin</h3>
            <p>The charts and comparison table will use the matching measurements from the live Hugging Face dataset.</p>
          </div>
        )}
      </section>
    </PageSection>
  )
}

'use client'

import * as React from 'react'
import { Alert, Button, Label, Spinner } from '@patternfly/react-core'
import CheckCircleIcon from '@patternfly/react-icons/dist/esm/icons/check-circle-icon'
import { formatBytes } from '@/lib/utils/format'
import { useCountUp } from '@/app/predict/performanceHelpers'
import { hasDefaultMemoryFractionForVersion, systemMemoryFraction, useCatalog } from '@/lib/hooks/useCatalog'
import { useTestedModels } from '@/lib/hooks/useTestedModels'
import { useSettings, type InferenceBackend } from '@/contexts/SettingsContext'
import { useRememberDefaultModel } from '@/lib/hooks/useRememberedDefaults'
import { getAppConfig } from '@/lib/app-config'
import { DEFAULT_WORKLOAD, type WorkloadPreset } from '@/lib/workload-presets'
import type { ConcurrencyPreset } from '@/lib/concurrency-presets'
import { ModelInput, type ModelStatus } from '@/components/ui/ModelInput';
import { ComboBox, type ComboBoxItem } from '@/components/ModelComboBox/ModelComboBox';
import { buildModelItems, needsHfConfig } from '@/lib/model-options';
import { fetchModelConfig } from '@/lib/huggingface/fetch-config';
import { GpuSystemInput } from '@/components/ui/GpuSystemInput'
import { InfoStrip, InfoStripAction } from '@/components/ui/InfoStrip'
import { WarningPanel } from '@/components/ui/WarningPanel'
import { DebugPanel } from '@/components/DebugPanel/DebugPanel'
import type { KvCacheCalcResult } from '@/lib/api/kv-cache-calc'
import styles from './KvCacheCalc.module.css'


const BREAKDOWN_COLORS: Record<string, string> = {
  kv: '#0066cc',
  weights: '#5e40be',
  activations: '#f0ab00',
  runtime: '#3e8635',
  comm: '#009596',
}

/** Parallelism inputs for one disagg pool (string-backed, parsed at calc time). */
interface PhaseParallelInput {
  tp: string
  pp: string
  moeTp: string
  moeEp: string
}

/** A computed KV-cache result tagged with its pool label ('' for agg). */
interface PhaseResult {
  label: string
  result: KvCacheCalcResult
}

/** Parallelism fields sent to /api/memory for one pool. */
interface PhaseParallel {
  tp: number
  pp: number
  moeTp: number
  moeEp: number
}

function parsePhaseParallel(p: PhaseParallelInput): PhaseParallel {
  return {
    tp: Math.max(1, parseInt(p.tp, 10) || 1),
    pp: Math.max(1, parseInt(p.pp, 10) || 1),
    moeTp: parseInt(p.moeTp, 10) || 0,
    moeEp: parseInt(p.moeEp, 10) || 0,
  }
}

function invalidPhaseParallel(p: PhaseParallelInput): boolean {
  return p.tp === '' || parseInt(p.tp, 10) < 1 || p.pp === '' || parseInt(p.pp, 10) < 1
}

export default function KvCacheCalc() {
  const { hydrated, hfToken, defaultModel: settingsDefaultModel, defaultSystem: settingsDefaultSystem, inferenceBackend, backendVersion: settingsBackendVersion, setDefaultSystem } = useSettings()
  const { modelOptions: catalogModels, gpuOptions: catalogGpus, modelSpecs, backendOptions, isLoading: catalogLoading } = useCatalog()
  const { validatedModelIds, isAvailable: testedModelsAvailable } = useTestedModels()
  const MODEL_OPTIONS = catalogModels

  const [model, setModel] = React.useState('')
  const [system, setSystem] = React.useState('')
  const [gpuReady, setGpuReady] = React.useState(false)
  const [backend, setBackend] = React.useState(() => getAppConfig().defaultBackend)
  const [backendVersion, setBackendVersion] = React.useState('')

  // Initialise model, backend, and version from settings once context has loaded from localStorage
  const fromSettings = React.useRef(false)
  React.useEffect(() => {
    if (!hydrated || fromSettings.current) return
    fromSettings.current = true
    setModel(settingsDefaultModel)
    setBackend(inferenceBackend)
    setBackendVersion(settingsBackendVersion)
  }, [hydrated, settingsDefaultModel, inferenceBackend, settingsBackendVersion])

  // Start from the system remembered across tools. A saved system the current
  // catalog cannot offer falls back to the first available one without
  // replacing the stored preference.
  React.useEffect(() => {
    if (!hydrated || catalogLoading || gpuReady) return
    const preferred = settingsDefaultSystem && catalogGpus.some(gpu => gpu.systemId === settingsDefaultSystem)
      ? settingsDefaultSystem
      : catalogGpus[0]?.systemId ?? ''
    setSystem(preferred)
    setGpuReady(true)
  }, [hydrated, catalogLoading, gpuReady, settingsDefaultSystem, catalogGpus])

  React.useEffect(() => {
    if (!gpuReady || catalogGpus.length === 0) return
    if (!catalogGpus.some(gpu => gpu.systemId === system)) {
      setSystem(catalogGpus[0].systemId)
    }
  }, [gpuReady, system, catalogGpus])

  const defaultMaxNumTokens = DEFAULT_WORKLOAD.isl + DEFAULT_WORKLOAD.osl
  const [maxNumTokens, setMaxNumTokens] = React.useState(defaultMaxNumTokens)
  const [maxBatchSize, setMaxBatchSize] = React.useState(128)
  const [tpSize, setTpSize] = React.useState(1)
  const [ppSize, setPpSize] = React.useState(1)
  const [moeTpSize, setMoeTpSize] = React.useState('')
  const [moeEpSize, setMoeEpSize] = React.useState('')
  // Disagg: prefill and decode pools have independent parallelism.
  const [servingMode, setServingMode] = React.useState<'agg' | 'disagg'>('agg')
  const [prefillPar, setPrefillPar] = React.useState<PhaseParallelInput>({ tp: '1', pp: '1', moeTp: '', moeEp: '' })
  const [decodePar, setDecodePar] = React.useState<PhaseParallelInput>({ tp: '1', pp: '1', moeTp: '', moeEp: '' })
  const [memFractionKind, setMemFractionKind] = React.useState('of_total')
  const [memFractionValue, setMemFractionValue] = React.useState(1.0)
  const memFractionValueUserEdited = React.useRef(false)
  const [expanded, setExpanded] = React.useState<string[]>([])
  const [loading, setLoading] = React.useState(false)
  const [results, setResults] = React.useState<PhaseResult[]>([])
  const [error, setError] = React.useState<string | null>(null)
  const calcAbortRef = React.useRef<AbortController | null>(null)
  const [debugOpen, setDebugOpen] = React.useState(false)
  const [inputsDirty, setInputsDirty] = React.useState(false)
  const markInputsDirty = () => {
    if (results.length > 0 || loading) setInputsDirty(true)
  }
  const [debugRequest, setDebugRequest] = React.useState<Record<string, unknown> | Record<string, unknown>[] | null>(null)
  const [debugResponse, setDebugResponse] = React.useState<Record<string, unknown> | Record<string, unknown>[] | null>(null)
  const [debugStatus, setDebugStatus] = React.useState<number | null>(null)
  const [debugDuration, setDebugDuration] = React.useState<number | null>(null)

  const [maxNumTokensInput, setMaxNumTokensInput] = React.useState(String(defaultMaxNumTokens))
  const [maxBatchSizeInput, setMaxBatchSizeInput] = React.useState('128')
  const [tpSizeInput, setTpSizeInput] = React.useState('1')
  const [ppSizeInput, setPpSizeInput] = React.useState('1')
  const [memFractionValueInput, setMemFractionValueInput] = React.useState('1.0')

  const [activePreset, setActivePreset] = React.useState<string>('default')

  const applyPreset = (p: Pick<WorkloadPreset, 'key' | 'isl' | 'osl'>) => {
    if (calcAbortRef.current) {
      setLoading(false)
      calcAbortRef.current.abort()
    }
    const tokens = p.isl + p.osl
    setMaxNumTokens(tokens); setMaxNumTokensInput(String(tokens))
    setActivePreset(p.key)
    setResults([])
    setInputsDirty(false)
  }

  const resetToDefaults = () => applyPreset(DEFAULT_WORKLOAD)

  const [activeConcurrencyPreset, setActiveConcurrencyPreset] = React.useState<string>('')

  const applyConcurrencyPreset = (p: ConcurrencyPreset) => {
    if (activeConcurrencyPreset === p.key) {
      setActiveConcurrencyPreset('')
      return
    }
    setMaxBatchSize(p.concurrency); setMaxBatchSizeInput(String(p.concurrency))
    setActiveConcurrencyPreset(p.key)
    markInputsDirty()
  }

  // Use the selected backend's catalog defaults for the memory controls. The
  // fields remain editable after the backend default has been applied.
  // The fraction is version-keyed, so prefer the selected system's own default;
  // the backend-level figure only describes the fleet's majority version.
  React.useEffect(() => {
    if (!hydrated || catalogLoading) return
    const selectedBackend = backendOptions.find(option => option.id === backend)
    if (!selectedBackend) return

    const fraction = systemMemoryFraction(selectedBackend, system)
    if (fraction != null) {
      if (!memFractionValueUserEdited.current) {
        setMemFractionValue(fraction)
        setMemFractionValueInput(String(fraction))
      }
    } else if (!memFractionValueUserEdited.current) {
      setMemFractionValue(1.0)
      setMemFractionValueInput('1.0')
    }
    setMemFractionKind(selectedBackend.memoryFractionKind)
  }, [backend, backendOptions, catalogLoading, hydrated, system])

  const invalidMaxNumTokens = maxNumTokensInput === '' || parseInt(maxNumTokensInput, 10) < 1;
  const invalidMaxBatchSize = maxBatchSizeInput === '' || parseInt(maxBatchSizeInput, 10) < 1;
  const invalidTpSize = tpSizeInput === '' || parseInt(tpSizeInput, 10) < 1;
  const invalidPpSize = ppSizeInput === '' || parseInt(ppSizeInput, 10) < 1;
  const selectedBackend = backendOptions.find(option => option.id === backend)
  const needsExplicitMemFraction = Boolean(
    backendVersion.trim() &&
    !memFractionValueUserEdited.current &&
    !hasDefaultMemoryFractionForVersion(selectedBackend, system, backendVersion),
  )
  const invalidMemFractionValue = needsExplicitMemFraction || memFractionValueInput === '' || !Number.isFinite(Number(memFractionValueInput)) || Number(memFractionValueInput) < 0 || Number(memFractionValueInput) > 1;
  const invalidParallelism = servingMode === 'disagg'
    ? invalidPhaseParallel(prefillPar) || invalidPhaseParallel(decodePar)
    : invalidTpSize || invalidPpSize;

  const handleMaxNumTokensChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setMaxNumTokensInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setMaxNumTokens(n);
    setActivePreset('');
    markInputsDirty();
  };
  
  const handleMaxBatchSizeChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setMaxBatchSizeInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setMaxBatchSize(n);
    setActiveConcurrencyPreset('');
    markInputsDirty();
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps -- hydrated gates config.json readiness
  const modelItems: ComboBoxItem[] = React.useMemo(() => buildModelItems(catalogModels, modelSpecs, validatedModelIds), [catalogModels, modelSpecs, validatedModelIds, hydrated]);

  // HF config for models AISimulators can't resolve from its catalog (incl. tested models
  // outside the catalog). Fetched on model change, sent to /api/memory on calc.
  const [hfConfig, setHfConfig] = React.useState<Record<string, unknown> | null>(null);
  // Track which model the config belongs to so a stale fetch cannot validate a
  // newer, still-typing model ID.
  const [hfConfigModel, setHfConfigModel] = React.useState('');

  React.useEffect(() => {
    setHfConfig(null);
    setHfConfigModel('');
    if (catalogLoading || !needsHfConfig(model, catalogModels) || !model.includes('/')) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      fetchModelConfig(model, hfToken).then(r => {
        // Ignore a response for a model the user has since moved away from.
        if (!cancelled && r.success && r.config) {
          setHfConfig(r.config as Record<string, unknown>);
          setHfConfigModel(model);
        }
      });
    }, 500);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [model, hfToken, catalogModels, catalogLoading]);

  const handleTpSizeChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setTpSizeInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setTpSize(n);
    markInputsDirty();
  };

  const handlePpSizeChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setPpSizeInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setPpSize(n);
    markInputsDirty();
  };

  const handleMemFractionValueChange = (raw: string) => {
    memFractionValueUserEdited.current = true
    const cleaned = raw.replace(/[^0-9.]/g, '');
    setMemFractionValueInput(cleaned);
    const n = Number(cleaned);
    if (Number.isFinite(n) && n >= 0 && n <= 1) setMemFractionValue(n);
    markInputsDirty();
  };

  const catalogMatch = MODEL_OPTIONS.includes(model)
  const kvModelStatus: ModelStatus = validatedModelIds.includes(model)
    ? 'supported'
    : catalogMatch ? 'catalog'
    : catalogLoading ? 'fetching'
    : model ? 'idle' : 'idle'

  useRememberDefaultModel(model, validatedModelIds.includes(model) || catalogMatch || hfConfigModel === model)

  function buildRequestBody(par: PhaseParallel): Record<string, unknown> {
    const body: Record<string, unknown> = {
      model_path: model,
      system,
      backend,
      max_num_tokens: maxNumTokens,
      max_batch_size: maxBatchSize,
      tp_size: par.tp,
      pp_size: par.pp,
      memory_fraction_kind: memFractionKind,
      memory_fraction_value: memFractionValue,
    }
    if (backendVersion.trim()) body.backend_version = backendVersion.trim()
    if (needsHfConfig(model, catalogModels) && hfConfig) body.model_config = hfConfig
    if (par.moeTp > 0) body.moe_tp_size = par.moeTp
    if (par.moeEp > 0) body.moe_ep_size = par.moeEp
    return body
  }

  async function handleCalculate() {
    if (calcAbortRef.current) calcAbortRef.current.abort()
    const controller = new AbortController()
    calcAbortRef.current = controller

    setInputsDirty(false)
    setLoading(true)
    setError(null)
    setResults([])

    // One request per pool. Agg is a single unlabelled pool; disagg fans out to
    // prefill + decode, each with its own parallelism, against the same /memory
    // endpoint (which already takes tp/pp/moe dims).
    const phases: { label: string; par: PhaseParallel }[] = servingMode === 'disagg'
      ? [
          { label: 'Prefill', par: parsePhaseParallel(prefillPar) },
          { label: 'Decode', par: parsePhaseParallel(decodePar) },
        ]
      : [{ label: '', par: { tp: tpSize, pp: ppSize, moeTp: parseInt(moeTpSize, 10) || 0, moeEp: parseInt(moeEpSize, 10) || 0 } }]

    const requestBodies = phases.map(ph => ({ label: ph.label, body: buildRequestBody(ph.par) }))
    setDebugRequest(servingMode === 'disagg' ? requestBodies : requestBodies[0].body)
    setDebugResponse(null)
    setDebugStatus(null)
    setDebugDuration(null)

    const t0 = performance.now()

    try {
      const responses = await Promise.all(
        requestBodies.map(async ({ label, body }) => {
          const res = await fetch('/api/memory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: controller.signal,
          })
          const data = await res.json()
          return { label, status: res.status, data }
        }),
      )

      if (controller.signal.aborted) return

      const failed = responses.find(r => r.data?.status === 'failed')

      setDebugResponse(servingMode === 'disagg' ? responses.map(r => r.data) : responses[0].data)
      setDebugStatus(failed?.status ?? responses[responses.length - 1].status)
      setDebugDuration(Math.round(performance.now() - t0))

      if (failed) {
        setError(failed.data.error?.message ?? 'An unexpected error occurred')
        return
      }

      setResults(responses.map(r => ({ label: r.label, result: r.data as KvCacheCalcResult })))
    } catch {
      if (controller.signal.aborted) return
      setDebugDuration(Math.round(performance.now() - t0))
      setError('Failed to connect to the server. Please try again.')
    } finally {
      if (!controller.signal.aborted) setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      {/* Header */}
      <div className={styles.header}>
        <h1 className={styles.pageTitle}>KV cache calculator</h1>
        <p className={styles.subtitle}>
          Calculate KV cache memory requirements for any model on supported GPU systems
        </p>
        <div className={styles.helperText}>
          Memory estimates use AISimulate, independent of the global Simulator setting.
        </div>
      </div>

      {/* Input card */}
      <div className={`${styles.card} ${styles.inputCard}`}>
        <div className={styles.inputRow}>
          <div className={styles.field}>
            <ComboBox
              id="kv-model"
              value={model}
              onChange={(value) => { setModel(value); markInputsDirty(); }}
              items={modelItems}
              placeholder="Type model name or select from dropdown..."
              allowCustom
              supportedModels={testedModelsAvailable ? validatedModelIds : undefined}
            />
          </div>

          <GpuSystemInput id="kv-gpu" value={system} onChange={(value) => { setSystem(value); markInputsDirty(); if (value) setDefaultSystem(value); }} gpuOptions={catalogGpus} />
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 16 }}>
          <Button
            variant="primary"
            onClick={handleCalculate}
            isDisabled={loading || !model.trim() || invalidMaxNumTokens || invalidMaxBatchSize || invalidParallelism || invalidMemFractionValue}
          >
            {loading ? 'Calculating…' : 'Calculate'}
          </Button>
        </div>
      </div>
        {needsExplicitMemFraction && (
          <div className={styles.helperText} role="status" aria-live="polite">
            This version differs from the system default. Enter a memory fraction for the pinned version.
          </div>
        )}

      {inputsDirty && <WarningPanel warnings={['Inputs have changed since the last submitted calculation. Click Calculate to update results.']} />}

      {/* ---------- workload presets ---------- */}
      {hydrated && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
          <span style={{ fontSize: '12px', fontWeight: 600, fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.06em', color: '#3c3f42', whiteSpace: 'nowrap' }}>Workload:</span>
          <Button variant={activePreset === 'default' ? 'secondary' : 'tertiary'} size="sm" aria-pressed={activePreset === 'default'} onClick={resetToDefaults}>Default</Button>
          {getAppConfig().workloadPresets.map(p => (
            <Button key={p.key} variant={activePreset === p.key ? 'secondary' : 'tertiary'} size="sm" aria-pressed={activePreset === p.key} onClick={() => applyPreset(p)}>
              {p.label}
            </Button>
          ))}
        </div>
      )}

      {/* ---------- concurrency presets ---------- */}
      {hydrated && getAppConfig().concurrencyPresets.length > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
          <span style={{ fontSize: '12px', fontWeight: 600, fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.06em', color: '#3c3f42', whiteSpace: 'nowrap' }}>Concurrency:</span>
          {getAppConfig().concurrencyPresets.map(p => (
            <Button key={p.key} variant={activeConcurrencyPreset === p.key ? 'secondary' : 'tertiary'} size="sm" aria-pressed={activeConcurrencyPreset === p.key} onClick={() => applyConcurrencyPreset(p)}>
              {p.label}
            </Button>
          ))}
        </div>
      )}

      {/* ---------- info strip ---------- */}
      <InfoStrip>
        Based on your configuration — Maximum Number of Tokens (ISL + OSL): {maxNumTokens.toLocaleString()}, Batch Size: {maxBatchSize}. 
        {' '}<InfoStripAction onClick={() => setExpanded(expanded.includes('customize') ? expanded.filter(e => e !== 'customize') : [...expanded, 'customize'])}>
          Adjust? (edit fields below)
        </InfoStripAction>
      </InfoStrip>

      {/* ---------- customization section ---------- */}
      {expanded.includes('customize') && (
        <div className={styles.card} style={{ marginBottom: 16 }}>
          <div className={styles.advancedBody} style={{ padding: 20 }}>
            {/* Serving config */}
            <div className={styles.advancedSectionLabel}>Serving config</div>
            <div className={styles.advancedRow4}>
              <div className={styles.field}>
                <label htmlFor="kv-backend" className={styles.fieldLabel}>Backend</label>
                <select
                  id="kv-backend"
                  value={backend}
                  onChange={e => {
                    const nextBackend = e.target.value
                    if (systemMemoryFraction(backendOptions.find(option => option.id === nextBackend), system) != null) {
                      memFractionValueUserEdited.current = false
                    }
                    setBackend(nextBackend)
                    markInputsDirty()
                  }}
                  className={styles.gpuSelect}
                >
                  {backendOptions.map(option => (
                    <option key={option.id} value={option.id}>{option.id}</option>
                  ))}
                </select>
              </div>
              <div className={styles.field}>
                <label htmlFor="kv-backend-ver" className={styles.fieldLabel}>Backend version</label>
                <input
                  type="text"
                  id="kv-backend-ver"
                  value={backendVersion}
                  onChange={e => { setBackendVersion(e.target.value); markInputsDirty(); }}
                  placeholder={backendOptions.find(option => option.id === backend)?.defaultVersion ?? 'latest'}
                  className={styles.numberInput}
                />
              </div>
              <div className={styles.field}>
                <label htmlFor="kv-tokens" className={styles.fieldLabel}>Max num tokens</label>
                <input
                  type="number"
                  id="kv-tokens"
                  value={maxNumTokensInput}
                  onChange={e => handleMaxNumTokensChange(e.target.value)}
                  min={1}
                  className={invalidMaxNumTokens ? styles.paramInputInvalid : styles.numberInput}
                />
              </div>
              <div className={styles.field}>
                <label htmlFor="kv-batch" className={styles.fieldLabel}>Max batch size</label>
                <input
                  type="number"
                  id="kv-batch"
                  value={maxBatchSizeInput}
                  onChange={e => handleMaxBatchSizeChange(e.target.value)}
                  min={1}
                  className={invalidMaxBatchSize ? styles.paramInputInvalid : styles.numberInput}
                />
              </div>
            </div>

            {/* Parallelism */}
            <div className={styles.advancedSectionLabel}>Parallelism</div>

            {/* Serving mode (agg / disagg) */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginBottom: '8px', marginTop: '4px' }}>
              <span style={{ fontSize: '12px', fontWeight: 600, fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.06em', color: '#3c3f42', whiteSpace: 'nowrap' }}>Serving mode:</span>
              <Button variant={servingMode === 'agg' ? 'secondary' : 'tertiary'} size="sm" aria-pressed={servingMode === 'agg'} onClick={() => { setServingMode('agg'); setResults([]); setInputsDirty(false); }}>Aggregated</Button>
              <Button variant={servingMode === 'disagg' ? 'secondary' : 'tertiary'} size="sm" aria-pressed={servingMode === 'disagg'} onClick={() => { setServingMode('disagg'); setResults([]); setInputsDirty(false); }}>Disaggregated</Button>
              {servingMode === 'disagg' && (
                <span style={{ fontSize: '12px', color: '#54585c' }}>Prefill and decode run on separate GPU pools.</span>
              )}
            </div>

            {servingMode === 'agg' ? (
              <div className={styles.advancedRow4}>
                <div className={styles.field}>
                  <label htmlFor="kv-tp" className={styles.fieldLabel}>TP size</label>
                  <input
                    type="number"
                    id="kv-tp"
                    value={tpSizeInput}
                    onChange={e => handleTpSizeChange(e.target.value)}
                    min={1}
                    className={invalidTpSize ? styles.paramInputInvalid : styles.numberInput}
                  />
                </div>
                <div className={styles.field}>
                  <label htmlFor="kv-pp" className={styles.fieldLabel}>PP size</label>
                  <input
                    type="number"
                    id="kv-pp"
                    value={ppSizeInput}
                    onChange={e => handlePpSizeChange(e.target.value)}
                    min={1}
                    className={invalidPpSize ? styles.paramInputInvalid : styles.numberInput}
                  />
                </div>
                <div className={styles.field}>
                  <label htmlFor="kv-moe-tp" className={styles.fieldLabel}>MoE TP size</label>
                  <input
                    type="text"
                    id="kv-moe-tp"
                    value={moeTpSize}
                    onChange={e => { setMoeTpSize(e.target.value.replace(/[^0-9]/g, '')); markInputsDirty(); }}
                    placeholder="auto"
                    className={styles.numberInput}
                  />
                </div>
                <div className={styles.field}>
                  <label htmlFor="kv-moe-ep" className={styles.fieldLabel}>MoE EP size</label>
                  <input
                    type="text"
                    id="kv-moe-ep"
                    value={moeEpSize}
                    onChange={e => { setMoeEpSize(e.target.value.replace(/[^0-9]/g, '')); markInputsDirty(); }}
                    placeholder="auto"
                    className={styles.numberInput}
                  />
                </div>
              </div>
            ) : (
              <div className={styles.advancedRow4}>
                <div className={styles.field}>
                  <label htmlFor="kv-p-tp" className={styles.fieldLabel}>Prefill TP</label>
                  <input type="number" id="kv-p-tp" value={prefillPar.tp} min={1}
                    onChange={e => { setPrefillPar(p => ({ ...p, tp: e.target.value.replace(/[^0-9]/g, '') })); markInputsDirty(); }}
                    className={prefillPar.tp === '' || parseInt(prefillPar.tp, 10) < 1 ? styles.paramInputInvalid : styles.numberInput} />
                </div>
                <div className={styles.field}>
                  <label htmlFor="kv-p-pp" className={styles.fieldLabel}>Prefill PP</label>
                  <input type="number" id="kv-p-pp" value={prefillPar.pp} min={1}
                    onChange={e => { setPrefillPar(p => ({ ...p, pp: e.target.value.replace(/[^0-9]/g, '') })); markInputsDirty(); }}
                    className={prefillPar.pp === '' || parseInt(prefillPar.pp, 10) < 1 ? styles.paramInputInvalid : styles.numberInput} />
                </div>
                <div className={styles.field}>
                  <label htmlFor="kv-p-moe-tp" className={styles.fieldLabel}>Prefill MoE TP</label>
                  <input type="text" id="kv-p-moe-tp" value={prefillPar.moeTp} placeholder="auto"
                    onChange={e => { setPrefillPar(p => ({ ...p, moeTp: e.target.value.replace(/[^0-9]/g, '') })); markInputsDirty(); }}
                    className={styles.numberInput} />
                </div>
                <div className={styles.field}>
                  <label htmlFor="kv-p-moe-ep" className={styles.fieldLabel}>Prefill MoE EP</label>
                  <input type="text" id="kv-p-moe-ep" value={prefillPar.moeEp} placeholder="auto"
                    onChange={e => { setPrefillPar(p => ({ ...p, moeEp: e.target.value.replace(/[^0-9]/g, '') })); markInputsDirty(); }}
                    className={styles.numberInput} />
                </div>
                <div className={styles.field}>
                  <label htmlFor="kv-d-tp" className={styles.fieldLabel}>Decode TP</label>
                  <input type="number" id="kv-d-tp" value={decodePar.tp} min={1}
                    onChange={e => { setDecodePar(p => ({ ...p, tp: e.target.value.replace(/[^0-9]/g, '') })); markInputsDirty(); }}
                    className={decodePar.tp === '' || parseInt(decodePar.tp, 10) < 1 ? styles.paramInputInvalid : styles.numberInput} />
                </div>
                <div className={styles.field}>
                  <label htmlFor="kv-d-pp" className={styles.fieldLabel}>Decode PP</label>
                  <input type="number" id="kv-d-pp" value={decodePar.pp} min={1}
                    onChange={e => { setDecodePar(p => ({ ...p, pp: e.target.value.replace(/[^0-9]/g, '') })); markInputsDirty(); }}
                    className={decodePar.pp === '' || parseInt(decodePar.pp, 10) < 1 ? styles.paramInputInvalid : styles.numberInput} />
                </div>
                <div className={styles.field}>
                  <label htmlFor="kv-d-moe-tp" className={styles.fieldLabel}>Decode MoE TP</label>
                  <input type="text" id="kv-d-moe-tp" value={decodePar.moeTp} placeholder="auto"
                    onChange={e => { setDecodePar(p => ({ ...p, moeTp: e.target.value.replace(/[^0-9]/g, '') })); markInputsDirty(); }}
                    className={styles.numberInput} />
                </div>
                <div className={styles.field}>
                  <label htmlFor="kv-d-moe-ep" className={styles.fieldLabel}>Decode MoE EP</label>
                  <input type="text" id="kv-d-moe-ep" value={decodePar.moeEp} placeholder="auto"
                    onChange={e => { setDecodePar(p => ({ ...p, moeEp: e.target.value.replace(/[^0-9]/g, '') })); markInputsDirty(); }}
                    className={styles.numberInput} />
                </div>
              </div>
            )}

            {/* Memory */}
            <div className={styles.advancedSectionLabel}>Memory</div>
            <div className={styles.advancedRow2}>
              <div className={styles.field}>
                <label htmlFor="kv-mem-val" className={styles.fieldLabel}>Memory fraction</label>
                <input
                  type="number"
                  id="kv-mem-val"
                  value={memFractionValueInput}
                  onChange={e => handleMemFractionValueChange(e.target.value)}
                  min={0}
                  max={1}
                  step={0.05}
                  className={invalidMemFractionValue ? styles.paramInputInvalid : styles.numberInput}
                />
              </div>
              <div className={styles.field}>
                <label htmlFor="kv-mem-kind" className={styles.fieldLabel}>Fraction kind</label>
                <select
                  id="kv-mem-kind"
                  value={memFractionKind}
                  onChange={e => { setMemFractionKind(e.target.value); markInputsDirty(); }}
                  className={styles.gpuSelect}
                >
                  <option value="of_total">of_total</option>
                  <option value="of_free">of_free</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className={styles.errorAlert}>
          <Alert variant={error.toLowerCase().includes('unsupported') ? 'warning' : 'danger'} title="Calculation failed" isInline>
            <p>{error}</p>
            <p style={{ marginTop: 8 }}>
              You can also try using our{' '}
              <a href="/predict" className={styles.errorLink}>
                Predict performance
              </a>{' '}
              for an approximate KV cache calculation.
            </p>
          </Alert>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className={styles.loadingWrap}>
          <Spinner size="lg" />
          <span>Calculating KV cache requirements…</span>
        </div>
      )}

      {/* Placeholder */}
      {!loading && results.length === 0 && !error}

      {/* Results — one section per pool (agg = single unlabelled pool) */}
      {results.length > 0 && !loading && (
        <>
          {results.map(({ label, result }) => {
            const currentGpu = catalogGpus.find(g => g.systemId === result.metadata.system)
            return (
              <PhaseResults key={label || 'agg'} label={label} result={result} gpuLabel={currentGpu?.label} />
            )
          })}
        </>
      )}

      {/* Debug panel */}
      <DebugPanel
        request={debugRequest}
        response={debugResponse}
        status={debugStatus}
        duration={debugDuration}
        open={debugOpen}
        onToggle={setDebugOpen}
        endpoint="POST /api/memory"
      />
    </div>
  )
}

// ─── Sub-components ─────────────────────────────────────────────────────────


/** Full result rendering for one pool: tiles + memory breakdown + request detail. */
function PhaseResults({ label, result, gpuLabel }: PhaseResult & { gpuLabel?: string }) {
  const [flipped, setFlipped] = React.useState<Record<string, boolean>>({})
  const toggleFlip = (id: string) => setFlipped(prev => ({ ...prev, [id]: !prev[id] }))

  const animKv = useCountUp(result.kvCache.totalBytes, 800)
  const animPerToken = useCountUp(result.kvCache.perTokenBytes, 800)
  const animTokens = useCountUp(result.kvCache.totalTokens, 800)
  const animGpuCap = useCountUp(result.gpuCapacity.totalBytes, 800)

  return (
    <div className={styles.phaseBlock}>
      {label && <div className={styles.phaseTitle}>{label}</div>}

      <div className={styles.tilesGrid}>
        <TileCard
          id="total"
          dark
          label="Available for KV cache / GPU"
          value={formatBytes(animKv)}
          sub={`${result.kvCache.totalTokens.toLocaleString()} tokens capacity`}
          flipped={flipped.total ?? false}
          onFlip={() => toggleFlip('total')}
          backContent={
            <>
              <div className={styles.backTitle}>KV cache / GPU detail</div>
              <BackRow label="Raw bytes" value={result.kvCache.totalBytes.toLocaleString()} />
              <BackRow label="Total tokens" value={result.kvCache.totalTokens.toLocaleString()} />
              <BackRow label="Per token" value={`${result.kvCache.perTokenBytes.toLocaleString()} B`} />
              <BackRow label="Source" value={result.metadata.source} />
            </>
          }
        />

        <TileCard
          id="pertoken"
          label="Per token"
          value={formatBytes(animPerToken)}
          sub="KV cache memory per token"
          flipped={flipped.pertoken ?? false}
          onFlip={() => toggleFlip('pertoken')}
          backContent={
            <>
              <div className={styles.backTitle}>Per-token detail</div>
              <BackRow label="Bytes/token" value={result.kvCache.perTokenBytes.toLocaleString()} />
              <BackRow label="Total tokens" value={result.kvCache.totalTokens.toLocaleString()} />
            </>
          }
        />

        <TileCard
          id="tokens"
          label="Token capacity"
          value={animTokens.toLocaleString()}
          sub="Max tokens in KV cache"
          flipped={flipped.tokens ?? false}
          onFlip={() => toggleFlip('tokens')}
          backContent={
            <>
              <div className={styles.backTitle}>Capacity detail</div>
              <BackRow label="Total tokens" value={result.kvCache.totalTokens.toLocaleString()} />
              <BackRow label="KV size" value={formatBytes(result.kvCache.totalBytes)} />
              <BackRow label="Per token" value={`${result.kvCache.perTokenBytes.toLocaleString()} B`} />
            </>
          }
        />

        <TileCard
          id="gpu"
          label="GPU memory"
          value={formatBytes(animGpuCap)}
          sub={`${gpuLabel || result.metadata.system} total capacity`}
          flipped={flipped.gpu ?? false}
          onFlip={() => toggleFlip('gpu')}
          backContent={
            <>
              <div className={styles.backTitle}>GPU memory detail</div>
              <BackRow label="Total GPU" value={formatBytes(result.gpuCapacity.totalBytes)} />
              <BackRow label="KV cache" value={formatBytes(result.kvCache.totalBytes)} />
              <BackRow label="KV % of GPU" value={`${((result.kvCache.totalBytes / result.gpuCapacity.totalBytes) * 100).toFixed(1)}%`} />
            </>
          }
        />
      </div>

      <MemoryBreakdownSection result={result} />
      
    </div>
  )
}

interface TileCardProps {
  id: string
  dark?: boolean
  label: string
  value: string
  sub: string
  flipped: boolean
  onFlip: () => void
  backContent: React.ReactNode
}

function TileCard({ dark, label, value, sub, flipped, onFlip, backContent }: TileCardProps) {
  return (
    <div
      className={`${styles.tileCard} ${dark ? styles.tileDark : ''}`}
      role="button"
      tabIndex={0}
      aria-pressed={flipped}
      onClick={() => { if (!flipped) onFlip() }}
      onKeyDown={e => {
        if ((e.key === 'Enter' || e.key === ' ') && !flipped) {
          e.preventDefault()
          onFlip()
        }
      }}
    >
      {!flipped ? (
        <>
          <div className={styles.tileLabel}>{label}</div>
          <div className={styles.tileValue}>{value}</div>
          <div className={styles.tileSub}>{sub}</div>
          <span className={styles.seeMath}>&#8635; details</span>
        </>
      ) : (
        <>
          {backContent}
          <span
            className={styles.seeMath}
            role="button"
            tabIndex={0}
            onClick={e => { e.stopPropagation(); onFlip() }}
            onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); onFlip() } }}
          >
            &#8635; flip back
          </span>
        </>
      )}
    </div>
  )
}

function BackRow({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.backRow}>
      <span className={styles.backRowLabel}>{label}</span>
      <span className={styles.backRowValue}>{value}</span>
    </div>
  )
}

function MemoryBreakdownSection({ result }: { result: KvCacheCalcResult }) {
  const segments = [
    { label: 'KV cache', bytes: result.kvCache.totalBytes, color: BREAKDOWN_COLORS.kv },
    { label: 'Weights', bytes: result.memoryBreakdown.weightsBytes, color: BREAKDOWN_COLORS.weights },
    { label: 'Activations', bytes: result.memoryBreakdown.activationsBytes, color: BREAKDOWN_COLORS.activations },
    { label: 'Runtime', bytes: result.memoryBreakdown.runtimeOverheadBytes, color: BREAKDOWN_COLORS.runtime },
  ]

  if (result.memoryBreakdown.commOverheadBytes > 0) {
    segments.push({ label: 'Comm', bytes: result.memoryBreakdown.commOverheadBytes, color: BREAKDOWN_COLORS.comm })
  }

  const totalUsed = segments.reduce((s, seg) => s + seg.bytes, 0)
  const gpuTotal = result.gpuCapacity.totalBytes
  const free = Math.max(0, gpuTotal - totalUsed)

  if (totalUsed === 0) return null

  return (
    <div className={styles.breakdownSection}>
      <div className={styles.breakdownTitle}>GPU memory breakdown</div>
      <div className={styles.memoryBar}>
        {segments.map((seg, i) => {
          const pct = gpuTotal > 0 ? (seg.bytes / gpuTotal) * 100 : (seg.bytes / totalUsed) * 100
          if (pct < 0.5) return null
          return (
            <div
              key={i}
              className={styles.memorySegment}
              style={{ width: `${pct}%`, background: seg.color }}
              title={`${seg.label}: ${formatBytes(seg.bytes)} (${pct.toFixed(1)}%)`}
            >
              {pct > 8 ? seg.label : ''}
            </div>
          )
        })}
        {free > 0 && gpuTotal > 0 && (
          <div
            className={styles.memorySegment}
            style={{ width: `${(free / gpuTotal) * 100}%`, background: '#e0e0e0', color: '#3c3f42' }}
            title={`Free: ${formatBytes(free)}`}
          >
            {(free / gpuTotal) * 100 > 8 ? 'Free' : ''}
          </div>
        )}
      </div>
      <div className={styles.breakdownLegend}>
        {segments.map((seg, i) => (
          <span key={i}>
            <span className={styles.legendDot} style={{ background: seg.color }} />
            {seg.label}: {formatBytes(seg.bytes)}
          </span>
        ))}
        {free > 0 && gpuTotal > 0 && (
          <span>
            <span className={styles.legendDot} style={{ background: '#e0e0e0' }} />
            Free: {formatBytes(free)}
          </span>
        )}
      </div>
    </div>
  )
}

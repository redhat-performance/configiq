'use client'

import * as React from 'react'
import { Alert, Label, Spinner } from '@patternfly/react-core'
import CheckCircleIcon from '@patternfly/react-icons/dist/esm/icons/check-circle-icon'
import { formatBytes } from '@/lib/utils/format'
import { useCountUp } from '@/app/predict/performanceHelpers'
import { useCatalog } from '@/lib/hooks/useCatalog'
import { useTestedModels } from '@/lib/hooks/useTestedModels'
import { useSettings, type InferenceBackend } from '@/contexts/SettingsContext'
import { getAppConfig } from '@/lib/app-config'
import { ModelInput, type ModelStatus } from '@/components/ui/ModelInput';
import { ComboBox, type ComboBoxItem } from '@/components/ModelComboBox/ModelComboBox';
import { buildModelItems, needsHfConfig } from '@/lib/model-options';
import { fetchModelConfig } from '@/lib/huggingface/fetch-config';
import { GpuSystemInput } from '@/components/ui/GpuSystemInput'
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
  const { hydrated, hfToken, defaultModel: settingsDefaultModel, inferenceBackend, backendVersion: settingsBackendVersion } = useSettings()
  const { modelOptions: catalogModels, gpuOptions: catalogGpus, modelSpecs, backendOptions, isLoading: catalogLoading } = useCatalog()
  const { modelIds: testedModelIds, isAvailable: testedModelsAvailable } = useTestedModels()
  const MODEL_OPTIONS = catalogModels

  const [model, setModel] = React.useState('')
  const [system, setSystem] = React.useState(() => getAppConfig().defaultSystem)
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
  const [maxNumTokens, setMaxNumTokens] = React.useState(8192)
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
  const [advancedOpen, setAdvancedOpen] = React.useState(false)
  const [loading, setLoading] = React.useState(false)
  const [results, setResults] = React.useState<PhaseResult[]>([])
  const [error, setError] = React.useState<string | null>(null)
  const [debugOpen, setDebugOpen] = React.useState(false)
  const [debugRequest, setDebugRequest] = React.useState<Record<string, unknown> | Record<string, unknown>[] | null>(null)
  const [debugResponse, setDebugResponse] = React.useState<Record<string, unknown> | Record<string, unknown>[] | null>(null)
  const [debugStatus, setDebugStatus] = React.useState<number | null>(null)
  const [debugDuration, setDebugDuration] = React.useState<number | null>(null)

  const [maxNumTokensInput, setMaxNumTokensInput] = React.useState('8192')
  const [maxBatchSizeInput, setMaxBatchSizeInput] = React.useState('128')
  const [tpSizeInput, setTpSizeInput] = React.useState('1')
  const [ppSizeInput, setPpSizeInput] = React.useState('1')
  const [memFractionValueInput, setMemFractionValueInput] = React.useState('1.0')

  const invalidMaxNumTokens = maxNumTokensInput === '' || parseInt(maxNumTokensInput, 10) < 1;
  const invalidMaxBatchSize = maxBatchSizeInput === '' || parseInt(maxBatchSizeInput, 10) < 1;
  const invalidTpSize = tpSizeInput === '' || parseInt(tpSizeInput, 10) < 1;
  const invalidPpSize = ppSizeInput === '' || parseInt(ppSizeInput, 10) < 1;
  const invalidMemFractionValue = memFractionValueInput === '' || !Number.isFinite(Number(memFractionValueInput)) || Number(memFractionValueInput) < 0 || Number(memFractionValueInput) > 1;
  const invalidParallelism = servingMode === 'disagg'
    ? invalidPhaseParallel(prefillPar) || invalidPhaseParallel(decodePar)
    : invalidTpSize || invalidPpSize;

  const handleMaxNumTokensChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setMaxNumTokensInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setMaxNumTokens(n);
  };
  
  const handleMaxBatchSizeChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setMaxBatchSizeInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setMaxBatchSize(n);
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps -- hydrated gates config.json readiness
  const modelItems: ComboBoxItem[] = React.useMemo(() => buildModelItems(catalogModels, testedModelIds, modelSpecs), [catalogModels, testedModelIds, modelSpecs, hydrated]);

  // HF config for models AISimulators can't resolve from its catalog (incl. tested models
  // outside the catalog). Fetched on model change, sent to /api/memory on calc.
  const [hfConfig, setHfConfig] = React.useState<Record<string, unknown> | null>(null);

  React.useEffect(() => {
    setHfConfig(null);
    if (catalogLoading || !needsHfConfig(model, catalogModels) || !model.includes('/')) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      fetchModelConfig(model, hfToken).then(r => {
        // Ignore a response for a model the user has since moved away from.
        if (!cancelled && r.success && r.config) setHfConfig(r.config as Record<string, unknown>);
      });
    }, 500);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [model, hfToken, catalogModels, catalogLoading]);

  const handleTpSizeChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setTpSizeInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setTpSize(n);
  };

  const handlePpSizeChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setPpSizeInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setPpSize(n);
  };

  const handleMemFractionValueChange = (raw: string) => {
    const cleaned = raw.replace(/[^0-9.]/g, '');
    setMemFractionValueInput(cleaned);
    const n = Number(cleaned);
    if (Number.isFinite(n) && n >= 0 && n <= 1) setMemFractionValue(n);
  };

  const catalogMatch = MODEL_OPTIONS.includes(model)
  const kvModelStatus: ModelStatus = testedModelIds.includes(model)
    ? 'supported'
    : catalogMatch ? 'catalog'
    : catalogLoading ? 'fetching'
    : model ? 'idle' : 'idle'

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
          })
          const data = await res.json()
          return { label, status: res.status, data }
        }),
      )

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
      setDebugDuration(Math.round(performance.now() - t0))
      setError('Failed to connect to the server. Please try again.')
    } finally {
      setLoading(false)
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
      </div>

      {/* Input card */}
      <div className={styles.inputCard}>
        <div className={styles.inputRow}>
          <div className={styles.field}>
            <ComboBox
              id="kv-model"
              value={model}
              onChange={setModel}
              items={modelItems}
              placeholder="Type model name or select from dropdown..."
              allowCustom
              supportedModels={testedModelsAvailable ? testedModelIds : undefined}
            />
          </div>

          <GpuSystemInput id="kv-gpu" value={system} onChange={setSystem} gpuOptions={catalogGpus} />
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 16 }}>
          <button
            type="button"
            className={styles.calcBtn}
            onClick={handleCalculate} 
            disabled={loading || !model.trim() || invalidMaxNumTokens || invalidMaxBatchSize || invalidParallelism || invalidMemFractionValue}
          >
            {loading ? 'Calculating…' : 'Calculate'}
          </button>
        </div>

        {/* Advanced settings accordion */}
        <button
          type="button"
          className={styles.advancedToggle}
          onClick={() => setAdvancedOpen(prev => !prev)}
          aria-expanded={advancedOpen}
        >
          <span className={styles.advancedToggleIcon}>{advancedOpen ? '▾' : '▸'}</span>
          Advanced settings
          {!advancedOpen && (
            <span className={styles.advancedSummary}>
              {backend}{backendVersion ? ` v${backendVersion}` : ''} · tokens: {maxNumTokens.toLocaleString()} · batch: {maxBatchSize}
              {servingMode === 'disagg'
                ? ` · disagg · P: TP ${prefillPar.tp}/PP ${prefillPar.pp} · D: TP ${decodePar.tp}/PP ${decodePar.pp}`
                : ` · TP ${tpSize}${ppSize > 1 ? ` · PP ${ppSize}` : ''}${moeTpSize ? ` · MoE TP ${moeTpSize}` : ''}`}
            </span>
          )}
        </button>
        {advancedOpen && (
          <div className={styles.advancedBody}>
            {/* Serving config */}
            <div className={styles.advancedSectionLabel}>Serving config</div>
            <div className={styles.advancedRow4}>
              <div className={styles.field}>
                <label htmlFor="kv-backend" className={styles.fieldLabel}>Backend</label>
                <select
                  id="kv-backend"
                  value={backend}
                  onChange={e => setBackend(e.target.value)}
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
                  onChange={e => setBackendVersion(e.target.value)}
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

            {/* Serving mode */}
            <div className={styles.advancedSectionLabel}>Serving mode</div>
            <div className={styles.modeToggle}>
              <button
                type="button"
                className={`${styles.modeButton} ${servingMode === 'agg' ? styles.modeButtonActive : ''}`}
                onClick={() => { setServingMode('agg'); setResults([]); }}
                aria-pressed={servingMode === 'agg'}
              >
                Aggregated
              </button>
              <button
                type="button"
                className={`${styles.modeButton} ${servingMode === 'disagg' ? styles.modeButtonActive : ''}`}
                onClick={() => { setServingMode('disagg'); setResults([]); }}
                aria-pressed={servingMode === 'disagg'}
              >
                Disaggregated
              </button>
            </div>

            {/* Parallelism */}
            <div className={styles.advancedSectionLabel}>Parallelism</div>
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
                    onChange={e => setMoeTpSize(e.target.value.replace(/[^0-9]/g, ''))}
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
                    onChange={e => setMoeEpSize(e.target.value.replace(/[^0-9]/g, ''))}
                    placeholder="auto"
                    className={styles.numberInput}
                  />
                </div>
              </div>
            ) : (
              <>
                <PhaseParallelFields title="Prefill pool" par={prefillPar} onChange={setPrefillPar} idPrefix="kv-p" />
                <PhaseParallelFields title="Decode pool" par={decodePar} onChange={setDecodePar} idPrefix="kv-d" />
              </>
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
                  onChange={e => setMemFractionKind(e.target.value)}
                  className={styles.gpuSelect}
                >
                  <option value="of_total">of_total</option>
                  <option value="of_free">of_free</option>
                </select>
              </div>
            </div>
          </div>
        )}
      </div>

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

/** Four inputs (TP/PP/MoE TP/MoE EP) for one disagg pool. */
function PhaseParallelFields({ title, par, onChange, idPrefix }: {
  title: string
  par: PhaseParallelInput
  onChange: (p: PhaseParallelInput) => void
  idPrefix: string
}) {
  const digits = (v: string) => v.replace(/[^0-9]/g, '')
  const invalidTp = par.tp === '' || parseInt(par.tp, 10) < 1
  const invalidPp = par.pp === '' || parseInt(par.pp, 10) < 1
  return (
    <>
      <div className={styles.phaseFieldsLabel}>{title}</div>
      <div className={styles.advancedRow4}>
        <div className={styles.field}>
          <label htmlFor={`${idPrefix}-tp`} className={styles.fieldLabel}>TP size</label>
          <input
            type="number" id={`${idPrefix}-tp`} value={par.tp} min={1}
            onChange={e => onChange({ ...par, tp: digits(e.target.value) })}
            className={invalidTp ? styles.paramInputInvalid : styles.numberInput}
          />
        </div>
        <div className={styles.field}>
          <label htmlFor={`${idPrefix}-pp`} className={styles.fieldLabel}>PP size</label>
          <input
            type="number" id={`${idPrefix}-pp`} value={par.pp} min={1}
            onChange={e => onChange({ ...par, pp: digits(e.target.value) })}
            className={invalidPp ? styles.paramInputInvalid : styles.numberInput}
          />
        </div>
        <div className={styles.field}>
          <label htmlFor={`${idPrefix}-moe-tp`} className={styles.fieldLabel}>MoE TP size</label>
          <input
            type="text" id={`${idPrefix}-moe-tp`} value={par.moeTp} placeholder="auto"
            onChange={e => onChange({ ...par, moeTp: digits(e.target.value) })}
            className={styles.numberInput}
          />
        </div>
        <div className={styles.field}>
          <label htmlFor={`${idPrefix}-moe-ep`} className={styles.fieldLabel}>MoE EP size</label>
          <input
            type="text" id={`${idPrefix}-moe-ep`} value={par.moeEp} placeholder="auto"
            onChange={e => onChange({ ...par, moeEp: digits(e.target.value) })}
            className={styles.numberInput}
          />
        </div>
      </div>
    </>
  )
}

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
              <BackRow label="Raw bytes" value={result.kvCache.totalBytes.toLocaleString()} dark />
              <BackRow label="Total tokens" value={result.kvCache.totalTokens.toLocaleString()} dark />
              <BackRow label="Per token" value={`${result.kvCache.perTokenBytes.toLocaleString()} B`} dark />
              <BackRow label="Source" value={result.metadata.source} dark />
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

      <div className={styles.configSection}>
        <div className={styles.configTitle}>Request details{label ? ` — ${label}` : ''}</div>
        <div className={styles.configGrid}>
          <ConfigItem label="Model" value={result.metadata.modelPath} />
          <ConfigItem label="Backend" value={result.metadata.backendVersion ? `${result.metadata.backend} v${result.metadata.backendVersion}` : result.metadata.backend} />
          <ConfigItem label="GPU system" value={result.metadata.system} />
          <ConfigItem label="Max tokens" value={result.metadata.maxNumTokens.toLocaleString()} />
          <ConfigItem label="Batch size" value={result.metadata.maxBatchSize.toLocaleString()} />
          <ConfigItem label="TP / PP" value={`${result.metadata.tpSize} / ${result.metadata.ppSize}`} />
          {result.metadata.moeTpSize != null && <ConfigItem label="MoE TP" value={String(result.metadata.moeTpSize)} />}
          {result.metadata.moeEpSize != null && <ConfigItem label="MoE EP" value={String(result.metadata.moeEpSize)} />}
          <ConfigItem label="Mem fraction" value={`${result.metadata.memoryFractionValue} (${result.metadata.memoryFractionKind})`} />
          <ConfigItem label="Source" value={result.metadata.source} />
        </div>
      </div>
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

function BackRow({ label, value }: { label: string; value: string; dark?: boolean }) {
  return (
    <div className={styles.backRow}>
      <span className={styles.backRowLabel}>{label}</span>
      <span className={styles.backRowValue}>{value}</span>
    </div>
  )
}

function ConfigItem({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.configItem}>
      <span className={styles.configLabel}>{label}</span>
      <span className={styles.configValue}>{value}</span>
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

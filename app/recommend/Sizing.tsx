'use client';

import * as React from 'react';
import { Label, Button, Accordion, AccordionItem, AccordionToggle, AccordionContent } from '@patternfly/react-core';
import MicrochipIcon from '@patternfly/react-icons/dist/esm/icons/microchip-icon';
import MemoryIcon from '@patternfly/react-icons/dist/esm/icons/memory-icon';
import ClockIcon from '@patternfly/react-icons/dist/esm/icons/clock-icon';
import TachometerAltIcon from '@patternfly/react-icons/dist/esm/icons/tachometer-alt-icon';
import EyeIcon from '@patternfly/react-icons/dist/esm/icons/eye-icon';
import EyeSlashIcon from '@patternfly/react-icons/dist/esm/icons/eye-slash-icon';
import ExclamationTriangleIcon from '@patternfly/react-icons/dist/esm/icons/exclamation-triangle-icon';
import CheckCircleIcon from '@patternfly/react-icons/dist/esm/icons/check-circle-icon';
import DollarSignIcon from '@patternfly/react-icons/dist/esm/icons/dollar-sign-icon';
import { InfoStrip, InfoStripAction } from '@/components/ui/InfoStrip';
import { DebugPanel } from '@/components/DebugPanel/DebugPanel';

import styles from './Sizing.module.css';
import { fetchModelConfig } from '@/lib/huggingface/fetch-config';
import { useRecommend } from '@/contexts/RecommendContext';
import type { RecommendProgressEvent } from '@/lib/api/recommend';
import { isMoeConfig, type PhaseConfig } from '@/lib/api/recommend';
import { useCatalog } from '@/lib/hooks/useCatalog';
import { useTestedModels } from '@/lib/hooks/useTestedModels';
import { useSettings } from '@/contexts/SettingsContext';
import { useCostings, resolveCloudRate } from '@/lib/hooks/useCostings';
import { getAppConfig } from '@/lib/app-config';
import { DEFAULT_WORKLOAD, type WorkloadPreset } from '@/lib/workload-presets';
import { ModelInput } from '@/components/ui/ModelInput';
import { ComboBox, type ComboBoxItem } from '@/components/ModelComboBox/ModelComboBox';
import { buildModelItems, needsHfConfig } from '@/lib/model-options';
import { GpuSystemInput } from '@/components/ui/GpuSystemInput';
import { TestedPerformanceFeedback } from '@/components/ui/TestedPerformanceFeedback';
import { useTestedClassifier } from '@/lib/hooks/useTestedClassifier';
import type { CanonicalClassifierInput, ClassifierPhaseInput } from '@/lib/tested-models/types';

function modelSuggestions(): string {
  const names = getAppConfig().suggestedModelNames;
  return names.length > 0 ? names.join(', ') : 'Nemotron, DeepSeek V4, Gemma 4, Kimi';
}
import { GpuChipLoader } from '@/components/GpuChipLoader/GpuChipLoader';
import { Term } from '@/app/predict/performanceHelpers';
import { HOURS_PER_MONTH, AMORT_MONTHS_5YR } from '@/lib/utils/format';


// ─── FlipTile (reused from Quick Estimate pattern) ───────────────────────────

function FlipTile({ dark = false, front, back }: {
  dark?: boolean; front: React.ReactNode; back: React.ReactNode;
}) {
  const [flipped, setFlipped] = React.useState(false);
  return (
    <div
      className={`${styles.flip} ${dark ? styles.tileDark : ''} ${flipped ? styles.flipped : ''}`}
      role="button" tabIndex={0} aria-pressed={flipped}
      onClick={() => setFlipped(f => !f)}
      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setFlipped(f => !f); } }}
    >
      <div className={`${styles.flipFace} ${styles.flipFront}`}>
        {front}
        <span className={styles.seeMath}>↻ see math</span>
      </div>
      <div className={`${styles.flipFace} ${styles.flipBack}`}>
        {back}
        <span className={styles.seeMath}>↻ flip back</span>
      </div>
    </div>
  );
}

// ─── Phase (disagg pool) tile ────────────────────────────────────────────────

/**
 * Compact parallelism label for a worker/phase. Shows attention dims (TP/PP/DP)
 * and, for MoE, expert dims separately — EP (moe_ep) and ETP (moe_tp).
 */
function parallelLabel(p: {
  tensorParallelSize: number
  pipelineParallelSize: number
  dataParallelSize: number
  contextParallelSize?: number
  moeExpertParallelSize: number | null
  moeTensorParallelSize: number | null
}): string {
  const parts = [`TP${p.tensorParallelSize}`, `PP${p.pipelineParallelSize}`];
  if (p.dataParallelSize > 1) parts.push(`DP${p.dataParallelSize}`);
  if (p.contextParallelSize != null && p.contextParallelSize > 1) parts.push(`CP${p.contextParallelSize}`);
  if (isMoeConfig(p.moeTensorParallelSize, p.moeExpertParallelSize)) {
    if (p.moeExpertParallelSize != null && p.moeExpertParallelSize > 1) parts.push(`EP${p.moeExpertParallelSize}`);
    if (p.moeTensorParallelSize != null && p.moeTensorParallelSize > 1) parts.push(`ETP${p.moeTensorParallelSize}`);
  }
  return parts.join(' · ');
}

function PhaseTile({ name, icon, phase }: {
  name: string; icon: React.ReactNode; phase: PhaseConfig;
}) {
  const poolGpus = phase.workers * phase.gpusPerWorker;
  // gpus/worker dims that are > 1 (matches the SDK's tp·pp·dp·cp definition).
  const gpwFactors = [
    ['TP', phase.tensorParallelSize],
    ['PP', phase.pipelineParallelSize],
    ['DP', phase.dataParallelSize],
    ['CP', phase.contextParallelSize],
  ].filter(([, v]) => (v as number) > 1);
  const gpwMath = gpwFactors.length > 0
    ? gpwFactors.map(([, v]) => v).join(' × ')
    : '1';
  return (
    <FlipTile
      front={
        <>
          <span className={styles.tileLabel}>{icon} {name}</span>
          <span className={styles.tileValue}>
            {poolGpus}<span className={styles.tileUnit}>GPUs</span>
          </span>
          <span className={styles.tileSub}>
            {phase.workers} worker{phase.workers === 1 ? '' : 's'} × {phase.gpusPerWorker} GPU/worker · {parallelLabel(phase)}
            {phase.batchSize != null ? ` · bs ${phase.batchSize}` : ''}
          </span>
        </>
      }
      back={
        <>
          <div className={styles.backTitle}>{name} pool</div>
          <div className={styles.formula}>
            {parallelLabel(phase)}<br />
            gpus/worker = {gpwMath} = <span className={styles.em}>{phase.gpusPerWorker}</span><br />
            workers = <span className={styles.em}>{phase.workers}</span>
            {phase.batchSize != null && <> · bs = <span className={styles.em}>{phase.batchSize}</span></>}<br />
            pool = <span className={styles.em}>{phase.workers} × {phase.gpusPerWorker} = {poolGpus} GPUs</span>
          </div>
        </>
      }
    />
  );
}

// ─── useCountUp (reused from Quick Estimate) ─────────────────────────────────

function useCountUp(target: number, duration = 750, decimals = 0) {
  const [val, setVal] = React.useState(target);
  const raf = React.useRef<number>(0);
  React.useEffect(() => {
    if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      setVal(target); return;
    }
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min((now - start) / duration, 1);
      const e = 1 - Math.pow(1 - p, 4);
      setVal(target * e);
      if (p < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [target, duration]);
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

// ─── Friendly error messages ────────────────────────────────────────────────

function friendlyErrorTitle(code: string | null): string {
  switch (code) {
    case 'AISIM_TIMEOUT': return 'Request timed out';
    case 'AISIM_NO_CONFIGURATION': return 'No valid configuration found';
    case 'AISIM_UNAVAILABLE': return 'Sizing service unavailable';
    case 'AISIM_NOT_CONFIGURED': return 'Service not configured';
    case 'AISIM_INVALID_RESPONSE': return 'Unexpected response';
    case 'INVALID_REQUEST': return 'Invalid input';
    case 'NETWORK_ERROR': return 'Connection error';
    default: return 'Something went wrong';
  }
}

function friendlyErrorMessage(code: string | null, raw: string): string {
  switch (code) {
    case 'AISIM_TIMEOUT':
      return 'The AISimulators service took too long to respond. This can happen with complex configurations.';
    case 'AISIM_NO_CONFIGURATION':
      return 'No valid GPU configuration found for this model and hardware combination.';
    case 'AISIM_UNAVAILABLE':
      return 'The AISimulators service is temporarily unreachable. This is usually a transient issue.';
    case 'AISIM_NOT_CONFIGURED':
      return 'The AISimulators service URL is not configured.';
    case 'AISIM_INVALID_RESPONSE':
      return 'The sizing engine returned an unexpected response format.';
    case 'INVALID_REQUEST':
      return 'Some input values are missing or invalid. Please check your model name and parameters.';
    case 'NETWORK_ERROR':
      return 'Could not reach a REST API service or it took too long to respond.';
    default:
      return raw;
  }
}

function friendlyErrorHint(code: string | null): string {
  switch (code) {
    case 'AISIM_TIMEOUT':
      return 'Try again, or try a smaller model or simpler configuration.';
    case 'AISIM_NO_CONFIGURATION':
      return 'Try a different GPU system, or reduce the input token length (ISL).';
    case 'AISIM_UNAVAILABLE':
      return 'Wait a moment and try again.';
    case 'NETWORK_ERROR':
      return 'Check your connection and try again.';
    case 'INVALID_REQUEST':
      return 'Make sure the model name is a valid Hugging Face ID (e.g. meta-llama/Llama-3.1-70B-Instruct).';
    default:
      return 'If this persists, try a different model or GPU combination.';
  }
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function Sizing() {
  const { hydrated, hfToken, defaultModel: settingsDefaultModel, inferenceBackend, backendVersion, costingsEnabled, pricingSource, preferredCloudProvider } = useSettings();
  const costings = useCostings(costingsEnabled, pricingSource);
  const { modelOptions: catalogModels, gpuOptions: catalogGpus, modelSpecs, timeoutSeconds: gatewayTimeout, isLoading: catalogLoading } = useCatalog();
  const { pairs: testedPairs, modelIds: testedModelIds, isAvailable: testedModelsAvailable } = useTestedModels();
  const MODEL_OPTIONS = catalogModels;

  // eslint-disable-next-line react-hooks/exhaustive-deps -- hydrated gates config.json readiness
  const modelItems: ComboBoxItem[] = React.useMemo(() => buildModelItems(catalogModels, testedModelIds, modelSpecs), [catalogModels, testedModelIds, modelSpecs, hydrated]);

  // Input state
  const [model, setModel] = React.useState('');
  const [hfConfig, setHfConfig] = React.useState<Record<string, unknown> | null>(null);

  // Set model from settings after context has loaded from localStorage
  const modelFromSettings = React.useRef(false);
  React.useEffect(() => {
    if (!hydrated || modelFromSettings.current) return;
    modelFromSettings.current = true;
    setModel(settingsDefaultModel);
  }, [hydrated, settingsDefaultModel]);
  const [gpuSystem, setGpuSystem] = React.useState(() => getAppConfig().defaultSystem);
  const testedSystemsForModel = React.useMemo(
    () => new Set(testedPairs.filter(pair => pair.modelId === model).map(pair => pair.systemId)),
    [testedPairs, model],
  );
  const selectableGpus = React.useMemo(
    () => testedSystemsForModel.size > 0 ? catalogGpus.filter(gpu => testedSystemsForModel.has(gpu.systemId)) : catalogGpus,
    [catalogGpus, testedSystemsForModel],
  );
  const [isl, setIsl] = React.useState(2048);
  const [osl, setOsl] = React.useState(128);
  const [maxSeqLen, setMaxSeqLen] = React.useState<number | null>(null);
  const [prefillMaxSeqLen, setPrefillMaxSeqLen] = React.useState<number | null>(null);
  const [decodeMaxSeqLen, setDecodeMaxSeqLen] = React.useState<number | null>(null);
  const [ttft, setTtft] = React.useState(1000);
  const [tpot, setTpot] = React.useState(30);
  const [targetConcurrency, setTargetConcurrency] = React.useState(1);
  const [requestLatency, setRequestLatency] = React.useState<number | null>(null);
  const [prefix, setPrefix] = React.useState(0);

  // Model status + HF config
  const [modelStatus, setModelStatus] = React.useState<'idle' | 'supported' | 'catalog' | 'fetching' | 'fetched' | 'error'>('idle');

  // GPU sizer (persistent across navigation)
  const { params, isLoading, result, error, errorCode, elapsed, debugRequest, debugResponse, debugStatus, debugDuration, progressHistory, startSizing } = useRecommend();
  const [debugOpen, setDebugOpen] = React.useState(false);

  // Additional constraints accordion
  const [expanded, setExpanded] = React.useState<string[]>(['perf']);

  const [islInput, setIslInput] = React.useState('2048');
  const [oslInput, setOslInput] = React.useState('128');
  const [maxSeqLenInput, setMaxSeqLenInput] = React.useState('');
  const [prefillMaxSeqLenInput, setPrefillMaxSeqLenInput] = React.useState('');
  const [decodeMaxSeqLenInput, setDecodeMaxSeqLenInput] = React.useState('');
  const [ttftInput, setTtftInput] = React.useState('1000');
  const [tpotInput, setTpotInput] = React.useState('30');
  const [concurrencyInput, setConcurrencyInput] = React.useState('1');
  const [latencyInput, setLatencyInput] = React.useState('');
  const [prefixInput, setPrefixInput] = React.useState('0');

  const invalidISL = islInput === '' || parseInt(islInput, 10) < 1;
  const invalidOSL = oslInput === '' || parseInt(oslInput, 10) < 1;
  const invalidTTFT = ttftInput === '' || !Number.isFinite(Number(ttftInput)) || Number(ttftInput) <= 0;
  const invalidTpot = tpotInput === '' || !Number.isFinite(Number(tpotInput)) || Number(tpotInput) <= 0;
  const invalidConcurrency = concurrencyInput === '' || parseInt(concurrencyInput, 10) < 1;
  const invalidLatency = latencyInput !== '' && (!Number.isFinite(Number(latencyInput)) || Number(latencyInput) <= 0);
  const validSequenceLength = (value: string) => /^[1-9][0-9]*$/.test(value) && Number.isSafeInteger(Number(value));
  const invalidMaxSeqLen = maxSeqLenInput !== '' && !validSequenceLength(maxSeqLenInput);
  const invalidPrefillMaxSeqLen = prefillMaxSeqLenInput !== '' && !validSequenceLength(prefillMaxSeqLenInput);
  const invalidDecodeMaxSeqLen = decodeMaxSeqLenInput !== '' && !validSequenceLength(decodeMaxSeqLenInput);

  const handleSequenceLengthChange = (
    raw: string,
    setInput: (value: string) => void,
    setValue: (value: number | null) => void,
  ) => {
    const value = raw.trim();
    setInput(value);
    const n = Number(value);
    setValue(validSequenceLength(value) ? n : null);
  };

  const handleIslChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setIslInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setIsl(n);
  };

  const handleOslChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setOslInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setOsl(n);
  };

  const handleTtftChange = (raw: string) => {
    const cleaned = raw.replace(/[^0-9.]/g, '');
    setTtftInput(cleaned);
    const n = Number(cleaned);
    if (Number.isFinite(n) && n > 0) setTtft(n);
  };

  const handleTpotChange = (raw: string) => {
    const cleaned = raw.replace(/[^0-9.]/g, '');
    setTpotInput(cleaned);
    const n = Number(cleaned);
    if (Number.isFinite(n) && n > 0) setTpot(n);
  };

  const handleConcurrencyChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setConcurrencyInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setTargetConcurrency(n);
  };

  const handleLatencyChange = (raw: string) => {
    const cleaned = raw.replace(/[^0-9.]/g, '');
    setLatencyInput(cleaned);
    const n = Number(cleaned);
    setRequestLatency(cleaned === '' ? null : (Number.isFinite(n) && n > 0 ? n : null));
  };

  const handlePrefixChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setPrefixInput(digits);
    const n = parseInt(digits, 10);
    setPrefix(isNaN(n) ? 0 : n);
  };

  // Model status check + fetch HF config. Catalog models resolve server-side;
  // everything else (incl. tested models outside the catalog) needs its HF
  // config.json fetched and stored so it can be sent to AISimulators on calculate.
  React.useEffect(() => {
    if (catalogLoading) { setModelStatus('idle'); return; }
    setHfConfig(null);
    let cancelled = false;
    const timer = setTimeout(() => {
      if (!model.includes('/')) { setModelStatus('idle'); return; }
      const isTested = testedModelIds.includes(model);
      if (!needsHfConfig(model, MODEL_OPTIONS)) {
        setModelStatus(isTested ? 'supported' : 'catalog');
        return;
      }
      setModelStatus('fetching');
      fetchModelConfig(model, hfToken).then(r => {
        // Ignore a response for a model the user has since moved away from.
        if (cancelled) return;
        if (r.success && r.config) {
          setHfConfig(r.config as Record<string, unknown>);
          // Tested models keep their blue "supported" status even though we
          // fetched the config from HF; others show the neutral "fetched".
          setModelStatus(isTested ? 'supported' : 'fetched');
        } else {
          // Tested models stay blue — AISimulators can still resolve them from HF
          // server-side when we send no model_config.
          setModelStatus(isTested ? 'supported' : 'error');
        }
      });
    }, 500);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [model, hfToken, MODEL_OPTIONS, catalogLoading, hydrated, testedModelIds]);

  // Fetch live pricing

  const currentGpuOption = selectableGpus.find(g => g.systemId === gpuSystem) ?? selectableGpus[0] ?? null;

  React.useEffect(() => {
    if (!catalogLoading && selectableGpus.length > 0 && !selectableGpus.some(gpu => gpu.systemId === gpuSystem)) {
      setGpuSystem(selectableGpus[0].systemId);
    }
  }, [catalogLoading, gpuSystem, selectableGpus]);

  const classifierInput = React.useMemo<CanonicalClassifierInput | null>(() => {
    if (!result) return null;
    const phase = (value: PhaseConfig | null): ClassifierPhaseInput | null => value ? {
      tp_size: value.tensorParallelSize,
      pp_size: value.pipelineParallelSize,
      dp_size: value.dataParallelSize,
      cp_size: value.contextParallelSize,
      moe_tp_size: value.moeTensorParallelSize,
      moe_ep_size: value.moeExpertParallelSize,
      workers: value.workers,
      batch_size: value.batchSize,
    } : null;
    return {
      effective: {
        model_id: model,
        system_id: gpuSystem,
        backend: inferenceBackend,
        backend_version: backendVersion || null,
        isl,
        osl,
        max_seq_len: maxSeqLen,
        prefill_max_seq_len: prefillMaxSeqLen,
        decode_max_seq_len: decodeMaxSeqLen,
        concurrency: result.performance.concurrency,
        tp_size: result.recommendation.tensorParallelSize,
        pp_size: result.recommendation.pipelineParallelSize,
        dp_size: result.recommendation.dataParallelSize,
        cp_size: result.recommendation.contextParallelSize,
        moe_tp_size: result.recommendation.moeTensorParallelSize,
        moe_ep_size: result.recommendation.moeExpertParallelSize,
        shared_prefix_tokens: prefix,
        prefix_caching_enabled: prefix > 0 ? true : null,
        weight_precision: null,
        kv_cache_precision: null,
        moe_quant_mode: null,
        gpu_memory_utilization: null,
        max_num_seqs: null,
        chunked_prefill_enabled: null,
        serving_mode: result.mode,
        replicas: result.recommendation.replicasNeeded,
        prefill: phase(result.phases.prefill),
        decode: phase(result.phases.decode),
        encode: phase(result.phases.encode),
      },
      request: {
        request_kind: 'recommend',
        target_request_rate: params?.target_request_rate ?? null,
        target_concurrency: params?.target_concurrency ?? null,
        target_latency_ms: params?.request_latency ?? null,
        target_ttft_ms: params?.ttft ?? null,
        target_tpot_ms: params?.tpot ?? null,
        database_mode: null,
        top_n: null,
      },
      model_config: hfConfig,
    };
  }, [backendVersion, decodeMaxSeqLen, gpuSystem, hfConfig, inferenceBackend, isl, maxSeqLen, model, osl, params, prefillMaxSeqLen, prefix, result]);
  const { prediction: classifierPrediction, isLoading: classifierLoading } = useTestedClassifier(classifierInput);

  const [activePreset, setActivePreset] = React.useState<string>('default');

  const applyPreset = (p: WorkloadPreset) => {
    setIsl(p.isl); setIslInput(String(p.isl));
    setOsl(p.osl); setOslInput(String(p.osl));
    setTtft(p.ttft); setTtftInput(String(p.ttft));
    setTpot(p.tpot); setTpotInput(String(p.tpot));
    setTargetConcurrency(p.concurrency); setConcurrencyInput(String(p.concurrency));
    setPrefix(p.prefix); setPrefixInput(String(p.prefix));
    setActivePreset(p.key);
    setExpanded(prev => prev.includes('customize') ? prev : [...prev, 'customize']);
  };

  const resetToDefaults = () => {
    applyPreset(DEFAULT_WORKLOAD);
    setRequestLatency(null); setLatencyInput('');
    setMaxSeqLen(null); setMaxSeqLenInput('');
    setPrefillMaxSeqLen(null); setPrefillMaxSeqLenInput('');
    setDecodeMaxSeqLen(null); setDecodeMaxSeqLenInput('');
  };

  const handleCalculate = () => {
    startSizing({
      model_path: model, system: gpuSystem, isl, osl, ttft,
      tpot, target_concurrency: targetConcurrency, prefix,
      ...(maxSeqLen != null ? { max_seq_len: maxSeqLen } : {}),
      ...(prefillMaxSeqLen != null ? { prefill_max_seq_len: prefillMaxSeqLen } : {}),
      ...(decodeMaxSeqLen != null ? { decode_max_seq_len: decodeMaxSeqLen } : {}),
      ...(requestLatency != null ? { request_latency: requestLatency } : {}),
      backend: inferenceBackend,
      ...(backendVersion ? { backend_version: backendVersion } : {}),
      // Send the HF config for models AISimulators can't resolve from its catalog.
      ...(needsHfConfig(model, MODEL_OPTIONS) && hfConfig ? { model_config: hfConfig } : {}),
    });
  };

  // Local memory analysis using the same engine as Quick Estimate
  // Animated values
  const gpuCount = useCountUp(result?.recommendation.gpusNeeded ?? 0);
  const ttftMs = useCountUp(result?.performance.ttftLatencyMs ?? 0, 750, 0);
  const tpsVal = useCountUp(result?.throughput.tokensPerSecond ?? 0, 750, 0);
  const memVal = useCountUp(result?.memory.value ?? 0, 750, 1);

  const hwCost = costings.gpuHardwareCosts.get(gpuSystem)?.new_usd ?? null;
  const amortizedHwPerHour = hwCost != null ? hwCost / (AMORT_MONTHS_5YR * HOURS_PER_MONTH) : null;
  const resolvedCloudRate = resolveCloudRate(costings.gpuCloudRates.get(gpuSystem), preferredCloudProvider);
  const pricePerHour = resolvedCloudRate?.rate ?? amortizedHwPerHour ?? null;
  const rateBasis = resolvedCloudRate
    ? `${resolvedCloudRate.provider.replace('.', ' · ')} ${resolvedCloudRate.kind === 'spot' ? 'spot' : 'on-demand'}`
    : amortizedHwPerHour != null ? 'amortized hardware (5-year)'
    : '';
  // Cost is for the whole deployment, so use the aggregate GPU total
  // (replicas × gpus/replica), not per-replica/per-worker.
  const numGpus = result?.recommendation.gpusNeeded ?? 0;
  const monthlyCost = pricePerHour != null ? numGpus * pricePerHour * HOURS_PER_MONTH : null;

  return (
    <div className={styles.page}>
      {/* ─── Header ─── */}
      <div className={styles.header}>
        <h1 className={styles.pageTitle}>Recommend sizing</h1>
        <p className={styles.subtitle}>
          Start with just a model name — we fill the rest, then let you tune every assumption.
        </p>
      </div>

      {/* ─── Input card ─── */}
      <div className={`${styles.card} ${styles.inputCard}`}>
        {/* Model + GPU row */}
        <div className={styles.inputGrid}>
          <div>
            <ComboBox
              id="adv-model"
              value={model}
              onChange={setModel}
              items={modelItems}
              placeholder="e.g. meta-llama/Llama-3.1-70B-Instruct"
              allowCustom
                supportedModels={testedModelsAvailable ? testedModelIds : undefined}
              hfToken={hfToken}
            />
          </div>

          <GpuSystemInput id="adv-gpu" value={gpuSystem} onChange={setGpuSystem} gpuOptions={selectableGpus} />
        </div>

        {/* Calculate button */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 16, marginTop: 16 }}>
          <button
            className={styles.calcBtn}
            onClick={handleCalculate}
             disabled={isLoading || !model.includes('/') || invalidISL || invalidOSL || invalidTTFT || invalidTpot || invalidConcurrency || invalidLatency || invalidMaxSeqLen || invalidPrefillMaxSeqLen || invalidDecodeMaxSeqLen}
          >
            {isLoading ? 'Calculating...' : 'Calculate'}
          </button>
        </div>
      </div>

      {hydrated && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
          <span style={{ fontSize: '12px', fontWeight: 600, fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.06em', color: '#3c3f42', whiteSpace: 'nowrap' }}>Workload:</span>
          <Button variant={activePreset === 'default' ? 'secondary' : 'tertiary'} size="sm" onClick={resetToDefaults}>Default</Button>
          {getAppConfig().workloadPresets.map(p => (
            <Button key={p.key} variant={activePreset === p.key ? 'secondary' : 'tertiary'} size="sm" onClick={() => applyPreset(p)}>
              {p.label}
            </Button>
          ))}
        </div>
      )}

      <InfoStrip>
        Based on your configuration — ISL {isl}, OSL {osl}, TTFT target {(() => {
          const sec = (ttft / 1000).toFixed(3);
          return sec.endsWith('000') ? (ttft / 1000).toFixed(0) : parseFloat(sec).toString();
        })()}s{prefix > 0 ? `, prefix ${prefix.toLocaleString()} tokens` : ''}, concurrency {targetConcurrency}, TPOT {tpot} ms.
        {' '}<InfoStripAction onClick={() => setExpanded(expanded.includes('customize') ? expanded.filter(e => e !== 'customize') : [...expanded, 'customize'])}>
          Adjust? (edit fields below)
        </InfoStripAction>
      </InfoStrip>

      {/* ─── Customization section (ISL/OSL/TTFT + HF token + constraints) ─── */}
      {expanded.includes('customize') && (
        <div className={`${styles.card}`} style={{ marginBottom: 16 }}>
          <div className={styles.cardBody}>
            <div className={styles.paramGrid} style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
              <div>
                <label className={styles.fieldLabel}>Avg input tokens (ISL) <Term k="isl" /></label>
                <input
                  type="number"
                  className={invalidISL ? styles.paramInputInvalid : styles.paramInput}
                  value={islInput}
                  onChange={e => handleIslChange(e.target.value)}
                  min={1}
                />
              </div>
              <div>
                <label className={styles.fieldLabel}>Avg output tokens (OSL) <Term k="osl" /></label>
                <input
                  type="number"
                  className={invalidOSL ? styles.paramInputInvalid : styles.paramInput}
                  value={oslInput}
                  onChange={e => handleOslChange(e.target.value)}
                  min={1}
                />
              </div>
              <div>
                <label className={styles.fieldLabel}>Max TTFT (ms) <Term k="ttft" /></label>
                <input
                  type="number"
                  className={invalidTTFT ? styles.paramInputInvalid : styles.paramInput}
                  value={ttftInput}
                  onChange={e => handleTtftChange(e.target.value)}
                  min={0.1}
                  step={0.1}
                />
              </div>
              <div>
                <label className={styles.fieldLabel}>Shared prefix tokens <Term k="prefix" /></label>
                <input
                  type="number"
                  className={styles.paramInput}
                  value={prefixInput}
                  onChange={e => handlePrefixChange(e.target.value)}
                  min={0}
                />
              </div>
            </div>

            <div className={styles.paramGrid} style={{ marginTop: 12, gridTemplateColumns: 'repeat(3, 1fr)' }}>
              <div>
                <label className={styles.fieldLabel}>Target concurrency <Term k="concurrent" /></label>
                <input
                  type="number"
                  className={invalidConcurrency ? styles.paramInputInvalid : styles.paramInput}
                  value={concurrencyInput}
                  onChange={e => handleConcurrencyChange(e.target.value)}
                  min={1}
                />
              </div>
              <div>
                <label className={styles.fieldLabel}>Max TPOT (ms) <Term k="tpot" /></label>
                <input
                  type="number"
                  className={invalidTpot ? styles.paramInputInvalid : styles.paramInput}
                  value={tpotInput}
                  onChange={e => handleTpotChange(e.target.value)}
                  min={0.1}
                  step={0.1}
                />
              </div>
              <div>
                <label className={styles.fieldLabel}>Max E2E latency (ms) <Term k="requestLatency" /></label>
                <input
                  type="number"
                  className={invalidLatency ? styles.paramInputInvalid : styles.paramInput}
                  value={latencyInput}
                  onChange={e => handleLatencyChange(e.target.value)}
                  placeholder="Auto"
                  min={1}
                />
              </div>
            </div>

            <Accordion style={{ marginTop: 16 }}>
              <AccordionItem isExpanded={expanded.includes('context-window')}>
                <AccordionToggle
                  id="context-window-toggle"
                  onClick={() => setExpanded(
                    expanded.includes('context-window')
                      ? expanded.filter(e => e !== 'context-window')
                      : [...expanded, 'context-window']
                  )}
                >
                  <span style={{ fontWeight: 600 }}>Context window sizing</span>
                </AccordionToggle>
                <AccordionContent>
                  <div style={{ paddingTop: 12 }}>
                    <div className={styles.paramGrid} style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
                      <div>
                        <label className={styles.fieldLabel}>Max sequence length (tokens) <Term k="maxModelLen" /></label>
                        <input
                          type="text"
                          inputMode="numeric"
                          className={invalidMaxSeqLen ? styles.paramInputInvalid : styles.paramInput}
                          value={maxSeqLenInput}
                          onChange={e => handleSequenceLengthChange(e.target.value, setMaxSeqLenInput, setMaxSeqLen)}
                          min={1}
                          placeholder="ISL + OSL"
                          aria-invalid={invalidMaxSeqLen}
                        />
                      </div>
                      <div>
                        <label className={styles.fieldLabel}>Prefill length (tokens) <Term k="prefillMaxSeqLen" /></label>
                        <input
                          type="text"
                          inputMode="numeric"
                          className={invalidPrefillMaxSeqLen ? styles.paramInputInvalid : styles.paramInput}
                          value={prefillMaxSeqLenInput}
                          onChange={e => handleSequenceLengthChange(e.target.value, setPrefillMaxSeqLenInput, setPrefillMaxSeqLen)}
                          min={1}
                          placeholder="Optional override"
                          aria-invalid={invalidPrefillMaxSeqLen}
                        />
                      </div>
                      <div>
                        <label className={styles.fieldLabel}>Decode length (tokens) <Term k="decodeMaxSeqLen" /></label>
                        <input
                          type="text"
                          inputMode="numeric"
                          className={invalidDecodeMaxSeqLen ? styles.paramInputInvalid : styles.paramInput}
                          value={decodeMaxSeqLenInput}
                          onChange={e => handleSequenceLengthChange(e.target.value, setDecodeMaxSeqLenInput, setDecodeMaxSeqLen)}
                          min={1}
                          placeholder="Optional override"
                          aria-invalid={invalidDecodeMaxSeqLen}
                        />
                      </div>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        </div>
      )}

      {/* ─── Loading ─── */}
      {isLoading && (
        <div className={styles.card}>
          <GpuChipLoader elapsed={elapsed} timeoutSeconds={gatewayTimeout} progressMessages={recommendProgressMessages(progressHistory)} />
        </div>
      )}

      {/* ─── Error ─── */}
      {error && (
        <div className={styles.errorWrap}>
          <div className={styles.errorTitle}>
            <ExclamationTriangleIcon /> {friendlyErrorTitle(errorCode)}
          </div>
          <div className={styles.errorMsg}>{friendlyErrorMessage(errorCode, error)}</div>
          <div className={styles.errorHint}>{friendlyErrorHint(errorCode)}</div>
        </div>
      )}

      <TestedPerformanceFeedback prediction={classifierPrediction} isLoading={classifierLoading} />

      {/* ─── Result tiles ─── */}
      {result && (
        <>
          <div className={styles.tilesGrid}>
            {/* GPUs Required */}
            <FlipTile
              dark
              front={
                <>
                  <span className={styles.tileLabel}>
                    <MicrochipIcon /> GPUs required
                    <Label isCompact color={result.mode === 'disagg' ? 'purple' : 'blue'} style={{ marginLeft: 'auto' }}>
                      {result.mode === 'disagg' ? 'disagg' : 'agg'}
                    </Label>
                  </span>
                  <span className={styles.tileValue}>
                    {gpuCount}<span className={styles.tileUnit}>× {currentGpuOption.label}</span>
                  </span>
                  <span className={styles.tileSub}>
                    {result.mode === 'disagg'
                      ? `${result.recommendation.replicasNeeded} replicas × ${result.recommendation.gpusPerReplica} GPUs/replica · ${result.performance.concurrency} concurrent users`
                      : `${parallelLabel(result.recommendation)} · ${result.performance.concurrency} concurrent users`}
                  </span>
                </>
              }
              back={
                result.mode === 'disagg' ? (
                  <>
                    <div className={styles.backTitle}>Disagg topology (xPyD)</div>
                    <div className={styles.formula}>
                      {([
                        ['prefill', result.phases.prefill],
                        ['decode', result.phases.decode],
                        ['encode', result.phases.encode],
                      ] as const)
                        .filter(([, p]) => p != null)
                        .map(([label, p]) => (
                          <React.Fragment key={label}>
                            {label}: <span className={styles.em}>{p!.workers} × {p!.gpusPerWorker} = {p!.workers * p!.gpusPerWorker}</span> GPUs<br />
                          </React.Fragment>
                        ))}
                      gpus/replica ={' '}
                      <span className={styles.em}>
                        {([result.phases.prefill, result.phases.decode, result.phases.encode]
                          .filter((p): p is PhaseConfig => p != null)
                          .map(p => p.workers * p.gpusPerWorker)
                          .join(' + '))} = {result.recommendation.gpusPerReplica}
                      </span><br />
                      total = <span className={styles.em}>{result.recommendation.gpusPerReplica} × {result.recommendation.replicasNeeded} replicas = {result.recommendation.gpusNeeded} GPUs</span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className={styles.backTitle}>GPU topology</div>
                    <div className={styles.formula}>
                      tensor parallel = <span className={styles.em}>{result.recommendation.tensorParallelSize}</span><br />
                      pipeline parallel = <span className={styles.em}>{result.recommendation.pipelineParallelSize}</span><br />
                      data parallel = <span className={styles.em}>{result.recommendation.dataParallelSize}</span><br />
                      {result.recommendation.contextParallelSize > 1 &&
                        <>context parallel = <span className={styles.em}>{result.recommendation.contextParallelSize}</span><br /></>}
                      {result.recommendation.moeExpertParallelSize != null && result.recommendation.moeExpertParallelSize > 1 &&
                        <>expert parallel (EP) = <span className={styles.em}>{result.recommendation.moeExpertParallelSize}</span><br /></>}
                      {result.recommendation.moeTensorParallelSize != null && result.recommendation.moeTensorParallelSize > 1 &&
                        <>expert tensor parallel (ETP) = <span className={styles.em}>{result.recommendation.moeTensorParallelSize}</span><br /></>}
                      gpus/worker = <span className={styles.em}>{result.recommendation.gpusPerReplica}</span><br />
                      total = <span className={styles.em}>{result.recommendation.gpusPerReplica} × {result.recommendation.replicasNeeded} replicas = {result.recommendation.gpusNeeded} GPUs</span>
                    </div>
                  </>
                )
              }
            />

            {/* TTFT */}
            <FlipTile
              front={
                <>
                  <span className={styles.tileLabel}><ClockIcon /> TTFT</span>
                  <span className={styles.tileValue}>
                    {ttftMs}<span className={styles.tileUnit}>ms</span>
                  </span>
                  <span className={styles.tileSub}>
                    {result.performance.ttftLatencyMs <= ttft ? (
                      <Label color="green" isCompact icon={<CheckCircleIcon />}>meets target</Label>
                    ) : (
                      <Label color="orange" isCompact icon={<ExclamationTriangleIcon />}>above target</Label>
                    )}
                  </span>
                </>
              }
              back={
                <>
                  <div className={styles.backTitle}>Time to first token</div>
                  <div className={styles.formula}>
                    target: <span className={styles.em}>{ttft.toLocaleString()} ms</span><br />
                    estimated: <span className={styles.em}>{result.performance.ttftLatencyMs.toFixed(1)} ms</span><br />
                    headroom: <span className={styles.em}>{(ttft - result.performance.ttftLatencyMs).toFixed(1)} ms</span><br />
                    TPOT: <span className={styles.em}>{result.performance.tpotMs.toFixed(1)} ms</span>
                  </div>
                </>
              }
            />

            {/* Throughput */}
            <FlipTile
              front={
                <>
                  <span className={styles.tileLabel}><TachometerAltIcon /> Throughput</span>
                  <span className={styles.tileValue}>
                    {tpsVal}<span className={styles.tileUnit}>tok/s</span>
                  </span>
                  <span className={styles.tileSub}>
                    {result.throughput.tokensPerSecondPerGpu.toFixed(1)}/GPU · {result.throughput.tokensPerSecondPerUser.toFixed(2)}/user
                  </span>
                </>
              }
              back={
                <>
                  <div className={styles.backTitle}>Throughput breakdown</div>
                  <div className={styles.formula}>
                    total: <span className={styles.em}>{result.throughput.tokensPerSecond.toFixed(1)} tok/s</span><br />
                    per GPU: <span className={styles.em}>{result.throughput.tokensPerSecondPerGpu.toFixed(1)} tok/s</span><br />
                    per user: <span className={styles.em}>{result.throughput.tokensPerSecondPerUser.toFixed(2)} tok/s</span><br />
                    concurrency: <span className={styles.em}>{result.performance.concurrency}</span>
                  </div>
                </>
              }
            />

            {/* Peak memory per GPU */}
            <FlipTile
              front={
                <>
                  <span className={styles.tileLabel}><MemoryIcon /> Peak mem / GPU</span>
                  <span className={styles.tileValue}>
                    {memVal}<span className={styles.tileUnit}>GB</span>
                  </span>
                  <span className={styles.tileSub}>
                    {result.mode === 'disagg' ? 'worst case across pools' : 'peak usage per GPU'}
                  </span>
                </>
              }
              back={
                <>
                  <div className={styles.backTitle}>Memory per GPU</div>
                  <div className={styles.formula}>
                    {result.mode === 'disagg' ? (
                      <>
                        {([
                          ['prefill', result.phases.prefill],
                          ['decode', result.phases.decode],
                          ['encode', result.phases.encode],
                        ] as const)
                          .filter(([, p]) => p != null && p.memoryGb != null)
                          .map(([label, p]) => (
                            <React.Fragment key={label}>
                              {label}: <span className={styles.em}>{p!.memoryGb!.toFixed(1)} GB/GPU</span><br />
                            </React.Fragment>
                          ))}
                        peak = <span className={styles.em}>{result.memory.value.toFixed(1)} GB/GPU</span> (worst case)
                      </>
                    ) : (
                      <>
                        {result.memory.breakdown && (
                          <>
                            weights: <span className={styles.em}>{result.memory.breakdown.weightsGb.toFixed(1)} GB</span><br />
                            KV cache: <span className={styles.em}>{result.memory.breakdown.kvCacheGb.toFixed(1)} GB</span><br />
                            overhead: <span className={styles.em}>{result.memory.breakdown.overheadGb.toFixed(1)} GB</span><br />
                          </>
                        )}
                        peak: <span className={styles.em}>{result.memory.value.toFixed(1)} {result.memory.unit}</span> / GPU<br />
                      </>
                    )}
                  </div>
                </>
              }
            />

            {/* Est. monthly cost — only when costings is enabled and a rate is available */}
            {costingsEnabled && monthlyCost != null && pricePerHour != null && (
              <FlipTile
                front={
                  <>
                    <span className={styles.tileLabel}><DollarSignIcon /> Est. monthly cost</span>
                    <span className={styles.tileValue}>
                      ${Math.round(monthlyCost).toLocaleString()}<span className={styles.tileUnit}>/mo</span>
                    </span>
                    <span className={styles.tileSub}>
                      {numGpus} × ${pricePerHour.toFixed(2)}/hr{rateBasis && ` · ${rateBasis}`}
                    </span>
                  </>
                }
                back={
                  <>
                    <div className={styles.backTitle}>Monthly cost</div>
                    <div className={styles.formula}>
                      GPUs: <span className={styles.em}>{numGpus}</span><br />
                      rate: <span className={styles.em}>${pricePerHour.toFixed(2)}/GPU-hr</span><br />
                      hours/mo: <span className={styles.em}>{HOURS_PER_MONTH}</span><br />
                      total = <span className={styles.em}>${Math.round(monthlyCost).toLocaleString()}/mo</span>
                    </div>
                  </>
                }
              />
            )}
          </div>

          {result.mode === 'disagg' && (result.phases.prefill || result.phases.decode || result.phases.encode) && (
            <div className={styles.phaseSection}>
              <div className={styles.phaseHeading}>
                Disaggregated pools <span>per replica — {result.recommendation.gpusPerReplica} GPUs</span>
              </div>
              <div className={styles.phaseGrid}>
                {result.phases.prefill && <PhaseTile name="Prefill" icon={<MicrochipIcon />} phase={result.phases.prefill} />}
                {result.phases.decode && <PhaseTile name="Decode" icon={<MicrochipIcon />} phase={result.phases.decode} />}
                {result.phases.encode && <PhaseTile name="Encode" icon={<MicrochipIcon />} phase={result.phases.encode} />}
              </div>
            </div>
          )}

          {costingsEnabled && monthlyCost == null && (
            <div className={styles.card} style={{ marginBottom: 24, fontSize: '13px', color: '#54585c' }}>
              <DollarSignIcon /> Cloud rate and hardware cost unavailable for this GPU —
              pick a provider on the Sources page or choose a GPU with published rates
              to see the estimated monthly cost.
            </div>
          )}

          {/* ─── Estimated serving performance ─── */}
          <div className={styles.card} style={{ marginBottom: 24 }}>
            <Accordion>
              <AccordionItem isExpanded={expanded.includes('perf')}>
                <AccordionToggle
                  id="perf-toggle"
                  onClick={() => setExpanded(
                    expanded.includes('perf') ? expanded.filter(e => e !== 'perf') : [...expanded, 'perf']
                  )}
                >
                  <span style={{ fontWeight: 600 }}>Estimated serving performance</span>
                </AccordionToggle>
                <AccordionContent>
                  <div className={styles.cardBody}>
                    <div className={styles.paramGrid}>
                      <div>
                        <div className={styles.fieldLabel}>Request latency</div>
                        <div style={{ fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 700 }}>
                          {(result.performance.requestLatencyMs / 1000).toFixed(1)}s
                        </div>
                        <div style={{ fontSize: 13, color: '#3c3f42', marginTop: 4 }}>
                          End-to-end for {osl} output tokens
                        </div>
                      </div>
                      <div>
                        <div className={styles.fieldLabel}>Concurrency</div>
                        <div style={{ fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 700 }}>
                          {result.performance.concurrency}
                        </div>
                        <div style={{ fontSize: 13, color: '#3c3f42', marginTop: 4 }}>
                          Concurrent users supported
                        </div>
                      </div>
                      <div>
                        <div className={styles.fieldLabel}>TPOT</div>
                        <div style={{ fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 700 }}>
                          {result.performance.tpotMs.toFixed(1)} ms
                        </div>
                        <div style={{ fontSize: 13, color: '#3c3f42', marginTop: 4 }}>
                          Time per output token
                        </div>
                      </div>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>

          {/* Memory layout bar hidden — local estimates don't match API internals */}

          {/* ─── Warnings ─── */}
          {result.warnings.length > 0 && (
            <div className={styles.warningsList}>
              {result.warnings.map((w, i) => (
                <div key={i} className={styles.warningItem}>
                  <ExclamationTriangleIcon style={{ color: '#f0ab00', flexShrink: 0, marginTop: 1 }} />
                  <span>{w.message}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* ─── Debug panel ─── */}
      <DebugPanel
        request={debugRequest}
        response={debugResponse}
        status={debugStatus}
        duration={debugDuration}
        open={debugOpen}
        onToggle={setDebugOpen}
        endpoint="POST /api/recommend"
      />
    </div>
  );
}

// ─── Status chip ─────────────────────────────────────────────────────────────

function StatusChip({ status }: { status: string }) {
  if (status === 'idle') return null;
  return null; // Chip is rendered inside the input wrapper instead
}

function recommendProgressMessages(progressHistory: RecommendProgressEvent[]): string[] {
  return progressHistory.flatMap(progress => {
    switch (progress.type) {
      case 'search_started':
        return [`Searching up to ${progress.maxGpus} GPUs`];
      case 'window_started':
        return [progress.window.minGpus === progress.window.maxGpus
          ? `Evaluating ${progress.window.minGpus} GPU`
          : `Evaluating ${progress.window.minGpus}–${progress.window.maxGpus} GPUs`];
      case 'window_completed':
        return [progress.candidateGpus == null
          ? `${progress.window.minGpus}–${progress.window.maxGpus} GPUs — no qualifying configuration`
          : `${progress.window.minGpus}–${progress.window.maxGpus} GPUs — provisional candidate at ${progress.candidateGpus} GPUs`];
      case 'refining':
        return [`Refining below ${progress.candidateGpus} GPUs: checking ${progress.window.minGpus}–${progress.window.maxGpus}`];
      case 'completed':
        return [];
    }
  });
}

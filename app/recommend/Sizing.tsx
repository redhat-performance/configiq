'use client';

import * as React from 'react';
import { Label, Button, Accordion, AccordionItem, AccordionToggle, AccordionContent, FormSelect, FormSelectOption, ToggleGroup, ToggleGroupItem } from '@patternfly/react-core';
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
import { WarningPanel } from '@/components/ui/WarningPanel';
import { DebugPanel } from '@/components/DebugPanel/DebugPanel';

import styles from './Sizing.module.css';
import { fetchModelConfig } from '@/lib/huggingface/fetch-config';
import { useRecommend, type RecommendObjective, type RecommendParams, type SizingGoal } from '@/contexts/RecommendContext';
import type { RecommendSweepMeta } from '@/lib/api/recommend';
import { isMoeConfig, type PhaseConfig } from '@/lib/api/recommend';
import { FrontierPanel } from '@/components/frontier/FrontierPanel';
import { CapacityOptionsPanel } from '@/components/capacity/CapacityOptionsPanel';
import { useCatalog, engineVersionSupport } from '@/lib/hooks/useCatalog';
import { simulatorDisplayLabel } from '@/lib/simulator-labels';
import { useTestedModels } from '@/lib/hooks/useTestedModels';
import { testedSystemsFirst } from '@/lib/tested-models/registry-types';
import { useSettings } from '@/contexts/SettingsContext';
import { useRememberDefaultModel } from '@/lib/hooks/useRememberedDefaults';
import { useCostings, resolveCloudRate } from '@/lib/hooks/useCostings';
import { getAppConfig } from '@/lib/app-config';
import { DEFAULT_WORKLOAD, matchWorkloadPreset, workloadPresetOptions, type WorkloadPreset } from '@/lib/workload-presets';
import { matchConcurrencyPresetKey, type ConcurrencyPreset } from '@/lib/concurrency-presets';
import { ModelInput } from '@/components/ui/ModelInput';
import { ComboBox, type ComboBoxItem } from '@/components/ModelComboBox/ModelComboBox';
import { buildModelItems, needsHfConfig } from '@/lib/model-options';
import { GpuSystemInput } from '@/components/ui/GpuSystemInput';
import { SimulatorBadge } from '@/components/ui/SimulatorBadge'
import { TestedPerformanceFeedback } from '@/components/ui/TestedPerformanceFeedback';
import { useTestedClassifier } from '@/lib/hooks/useTestedClassifier';
import type { CanonicalClassifierInput, ClassifierPhaseInput } from '@/lib/tested-models/types';
import { resolveSearchFabric, submittedInputsMatch } from './inputs-match';
import { statusForModel, type ModelCheckStatus, type ModelStatusCheck } from '@/lib/model-status';
import { CopyLinkButton } from '@/components/help/CopyLinkButton';
import { TourButton } from '@/components/help/TourButton';
import { FABRIC_NONE } from '@/lib/deep-links/common';
import { recommendLink, type RecommendLinkValues } from '@/lib/deep-links/pages';
import { defineAccessors } from '@/lib/deep-links/shared-view';
import { useSharedView } from '@/lib/deep-links/useSharedView';
import { useLinkProvenance } from '@/lib/deep-links/useLinkProvenance';
import { SharedViewProvenance } from '@/components/ui/SharedViewProvenance';
import { GpuChipLoader } from '@/components/GpuChipLoader/GpuChipLoader';
import { Term } from '@/app/predict/performanceHelpers';
import { AMORT_MONTHS_5YR, HOURS_PER_MONTH, formatLatencyBound } from '@/lib/utils/format';

/** Offered rate a demand run starts from; the UI labels it as the default. */
const DEFAULT_DEMAND_RATE = 10;

/**
 * The link's own starting points. A field left at one of these is omitted from
 * a generated link, so the recipient falls back to the same value.
 */
const RECOMMEND_LINK_DEFAULTS: RecommendLinkValues = {
  goal: 'demand',
  isl: 2048,
  osl: 128,
  concurrency: 1,
  ttft: 1000,
  tpot: 30,
  demandRate: DEFAULT_DEMAND_RATE,
  prefix: 0,
};

/** Match a retained request back to the sizing goal that produced it. */
function goalFromParams(params: RecommendParams | null): SizingGoal {
  if (!params) return 'demand';
  const objective = params.objective
    ?? (params.sla_mode === 'none' ? 'throughput_demand' : 'min_gpus');
  if (objective === 'concurrency_capacity') return 'capacity';
  if (objective === 'throughput_demand') return 'demand';
  return 'latency';
}

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
    case 'SIMULATOR_MISMATCH': return 'Simulator mismatch';
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
      return raw || 'No valid GPU configuration found for this model and hardware combination.';
    case 'SIMULATOR_MISMATCH':
      return raw;
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
      return 'Try a different GPU system or adjust the requested configuration.';
    case 'SIMULATOR_MISMATCH':
      return 'The server did not confirm the requested simulator. The application will not substitute another simulator.';
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
  const { hydrated, hfToken, defaultModel: settingsDefaultModel, defaultSystem: settingsDefaultSystem, inferenceBackend, backendVersion, simulator, costingsEnabled, pricingSource, preferredCloudProvider, setDefaultSystem } = useSettings();
  const costings = useCostings(costingsEnabled, pricingSource);
  const { modelOptions: allCatalogModels, gpuOptions: allCatalogGpus, modelSpecs, networkOptions, simulatorSupport, simulators, objectives: catalogObjectives, timeoutSeconds: gatewayTimeout, isLoading: catalogLoading } = useCatalog();
  // A simulator may price only a subset of the catalog. Its override narrows
  // the pickers; no override means the top-level catalog is its support.
  // An unavailable simulator has an explicit empty support entry, so its pages
  // never fall back to another simulator's catalogue.
  const support = simulatorSupport[simulator];
  const simulatorUnavailable = support?.available === false;
  const catalogModels = support?.models ?? allCatalogModels;
  const catalogGpus = React.useMemo(
    () => (support
      ? allCatalogGpus.filter(option => support.systems.includes(option.systemId))
      : allCatalogGpus),
    [support, allCatalogGpus],
  );
  const engineBackend = support?.backends.find(backend => backend.id === inferenceBackend)
    ?? support?.backends[0];
  const engineSupport = engineVersionSupport(engineBackend, backendVersion || engineBackend?.defaultVersion);
  const disaggSupported = engineSupport?.disaggregated ?? true;
  // The simulator reports whether a P/D run needs a fabric; AISimulate's P/D
  // search takes none, so the fabric control stays hidden there.
  const fabricRequired = engineBackend?.disaggregationRequiresFabric ?? false;
  // Name the release in the "cannot disaggregate" explanation without ever
  // rendering an empty version.
  const modeledBackendLabel = [
    inferenceBackend === 'vllm' ? 'vLLM' : inferenceBackend,
    backendVersion || engineBackend?.defaultVersion,
  ].filter(Boolean).join(' ');
  const { pairs: testedPairs, validatedModelIds, isAvailable: testedModelsAvailable } = useTestedModels();
  const MODEL_OPTIONS = catalogModels;
  const [networkFabric, setNetworkFabric] = React.useState<string | null>(null);
  const availableNetworks = support?.networks ?? networkOptions;
  // The simulator names the fabric its P/D topology uses by default. Never
  // invent a fabric name: an empty catalog means aggregate-only search.
  const defaultNetworkFabric = support?.defaultNetwork ?? availableNetworks[0] ?? '';
  const selectedNetworkFabric = networkFabric ?? defaultNetworkFabric;

  // eslint-disable-next-line react-hooks/exhaustive-deps -- hydrated gates config.json readiness
  const modelItems: ComboBoxItem[] = React.useMemo(() => buildModelItems(catalogModels, modelSpecs, validatedModelIds), [catalogModels, modelSpecs, validatedModelIds, hydrated]);

  // Input state
  const [model, setModel] = React.useState('');
  // Persist the model only after a pick in this tool. Filter-driven
  // replacements must not overwrite the cross-tool default.
  const modelTouched = React.useRef(false);
  const [hfConfig, setHfConfig] = React.useState<Record<string, unknown> | null>(null);
  // A simulator that models a catalog also says which models it cannot price.
  // The default model may be one of them; surface that instead of replacing it.
  const modelUnsupported = model !== '' && (support?.modelsExhaustive ?? false) && !catalogModels.includes(model);

  // Set model from settings after context has loaded from localStorage.
  // Track readiness separately from the value: an intentionally empty saved
  // default is a valid, settled state.
  const modelFromSettings = React.useRef(false);
  const [modelFromSettingsApplied, setModelFromSettingsApplied] = React.useState(false);
  React.useEffect(() => {
    if (!hydrated || modelFromSettings.current) return;
    modelFromSettings.current = true;
    setModel(settingsDefaultModel);
    setModelFromSettingsApplied(true);
  }, [hydrated, settingsDefaultModel]);
  const [gpuSystem, setGpuSystem] = React.useState('');
  const [gpuReady, setGpuReady] = React.useState(false);
  // The linked system the shared simulator did not offer yet. It is re-applied
  // once the catalog for the accepted simulator arrives.
  const pendingLinkedSystem = React.useRef<string | null>(null);
  const testedSystemsForModel = React.useMemo(
    () => new Set(testedPairs.filter(pair => pair.modelId === model).map(pair => pair.systemId)),
    [testedPairs, model],
  );
  const selectableGpus = React.useMemo(
    () => testedSystemsFirst(catalogGpus, testedSystemsForModel),
    [catalogGpus, testedSystemsForModel],
  );

  // Start from the system remembered across tools. A saved system the current
  // simulator cannot offer falls back to the first selectable one without
  // replacing the stored preference.
  React.useEffect(() => {
    if (!hydrated || catalogLoading || gpuReady) return;
    const preferred = settingsDefaultSystem && selectableGpus.some(gpu => gpu.systemId === settingsDefaultSystem)
      ? settingsDefaultSystem
      : selectableGpus[0]?.systemId ?? '';
    setGpuSystem(preferred);
    setGpuReady(true);
  }, [hydrated, catalogLoading, gpuReady, settingsDefaultSystem, selectableGpus]);
  const [isl, setIsl] = React.useState(2048);
  const [osl, setOsl] = React.useState(128);
  const [maxSeqLen, setMaxSeqLen] = React.useState<number | null>(null);
  const [prefillMaxSeqLen, setPrefillMaxSeqLen] = React.useState<number | null>(null);
  const [decodeMaxSeqLen, setDecodeMaxSeqLen] = React.useState<number | null>(null);
  const [ttft, setTtft] = React.useState(1000);
  const [tpot, setTpot] = React.useState(30);
  // Start at a single user so the first estimate sizes the smallest
  // configuration; see d32b0dd and 0730b07. Users raise this with the presets.
  const [targetConcurrency, setTargetConcurrency] = React.useState(1);
  const [requestLatency, setRequestLatency] = React.useState<number | null>(null);
  const [prefix, setPrefix] = React.useState(0);

  // GPU sizer (persistent across navigation)
  const { params, runMode, isLoading, result: primaryResult, error, errorCode, elapsed, debugRequest, debugResponse, debugStatus, debugDuration, frontier, startSizing, startPareto, reset } = useRecommend();
  // Three sizing goals share one Calculate flow. `demand` asks how many GPUs
  // sustain an arrival rate; `latency` enforces the TTFT/TPOT (or E2E) SLA;
  // `capacity` asks how many GPUs hold N sequences at full context. Both
  // aggregate-only goals report performance without gating on it. Restore the
  // retained request's goal so returning to the page does not force a reset
  // just to reselect the goal that produced the visible result.
  const [goal, setGoal] = React.useState<SizingGoal>(() => goalFromParams(params));
  const [demandRate, setDemandRate] = React.useState(DEFAULT_DEMAND_RATE);
  const isDemand = goal === 'demand';
  const isCapacity = goal === 'capacity';
  // The simulator advertises the goals it can answer. Gate the selector on that
  // capability, never on the simulator's ID.
  const availableObjectives = support?.objectives ?? catalogObjectives;
  const capacitySupported = availableObjectives.includes('concurrency_capacity');
  const paretoSupported = availableObjectives.includes('pareto');
  // Demand and capacity both report a GPU count without a latency SLA, and
  // neither searches a topology the page can choose: demand's answer is the
  // deployment that sustains the rate, and capacity reports aggregated
  // configurations plus disaggregated comparison options. The page therefore
  // leaves the topology to the service for both goals.
  const aggregateOnlyGoal = isDemand || isCapacity;
  // The mode the user or a shared link chose, or null while nothing has chosen
  // one. A null choice derives from capability so this page keeps the
  // deployment search it made before this control existed: aggregated only when
  // the simulator cannot model prefill/decode or the goal has no mode to choose.
  const [servingModeChoice, setServingModeChoice] = React.useState<'agg' | 'disagg' | null>(null);
  // Whether the simulator can search a prefill/decode topology at all: it must
  // model one, the goal must offer the choice, and a P/D search needs a fabric,
  // so a catalog that offers none leaves nothing to search. Both inputs are
  // capability, never a simulator ID.
  const disaggCapable = disaggSupported
    && !aggregateOnlyGoal
    && (!fabricRequired || availableNetworks.length > 0);
  // The explicit "None" fabric is an aggregate-only request, so it leaves no P/D
  // topology either.
  const disaggSelectable = disaggCapable && (!fabricRequired || selectedNetworkFabric !== '');
  // Clamp display and submission; never overwrite the choice. An explicit mode
  // therefore survives a goal or simulator change and applies again as soon as
  // the form can honour it.
  const servingMode: 'agg' | 'disagg' = disaggSelectable ? servingModeChoice ?? 'disagg' : 'agg';
  // A Pareto sweep is not a sizing goal, so it is unavailable only when the
  // simulator cannot compute one or the workload has no throughput trade-off.
  // It is unavailable for capacity because capacity already reports P/D options.
  const paretoDisabledReason = isCapacity
    ? 'Pareto frontier is not available for concurrency-capacity sizing.'
    : !paretoSupported
      ? 'Pareto frontier is not available for the selected simulator.'
      : null;

  // Model status + HF config
  const [modelCheck, setModelCheck] = React.useState<ModelStatusCheck>({ model: '', status: 'idle' });
  // A status applies only to the model it was computed for. Until the current
  // model is checked, treat it as idle so a stale status cannot authorize
  // saving an unvalidated model as the default.
  const modelStatus = statusForModel(modelCheck, model);
  useRememberDefaultModel(
    model,
    modelTouched.current && !modelUnsupported &&
      (modelStatus === 'supported' || modelStatus === 'catalog' || modelStatus === 'fetched'),
  );

  // Capacity runs return ranked alternatives. The result panel shows whichever
  // option is promoted; every other view keeps the primary recommendation so
  // the search summary and provenance still describe what was submitted.
  const [selectedCapacityOptionId, setSelectedCapacityOptionId] = React.useState<string | null>(null);
  React.useEffect(() => {
    setSelectedCapacityOptionId(null);
  }, [primaryResult]);
  const capacityOptions = primaryResult?.capacityOptions ?? [];
  const selectedCapacityOption = capacityOptions.find(
    option => option.optionId === selectedCapacityOptionId,
  ) ?? null;
  const result = selectedCapacityOption?.result ?? primaryResult;
  // The result outlives a remount, but the input goal does not. Derive the
  // result's objective from the request that produced it so returning to
  // /recommend does not relabel a demand or capacity result as a latency one.
  const resultObjective = params?.objective
    ?? (params?.sla_mode === 'none' ? 'throughput_demand' : 'min_gpus');
  const resultIsDemand = resultObjective === 'throughput_demand';
  const resultIsCapacity = resultObjective === 'concurrency_capacity';
  const resultDemandRate = params?.target_request_rate ?? demandRate;
  const resultRequestLatency = params?.request_latency ?? null;
  const resultTtft = params?.ttft ?? ttft;
  const resultDemandOutputTokPerSecond = resultIsDemand
    ? resultDemandRate * (params?.osl ?? osl)
    : 0;
  const [debugOpen, setDebugOpen] = React.useState(false);
  const [inputsDirty, setInputsDirty] = React.useState(false);

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
  const [demandRateInput, setDemandRateInput] = React.useState(String(DEFAULT_DEMAND_RATE));

  const invalidISL = islInput === '' || parseInt(islInput, 10) < 1;
  const invalidOSL = oslInput === '' || parseInt(oslInput, 10) < 1;
  const invalidTTFT = ttftInput === '' || !Number.isFinite(Number(ttftInput)) || Number(ttftInput) <= 0;
  const invalidTpot = tpotInput === '' || !Number.isFinite(Number(tpotInput)) || Number(tpotInput) <= 0;
  const invalidConcurrency = concurrencyInput === '' || parseInt(concurrencyInput, 10) < 1;
  const invalidLatency = latencyInput !== '' && (!Number.isFinite(Number(latencyInput)) || Number(latencyInput) <= 0);
  const invalidDemandRate = demandRateInput === '' || !Number.isFinite(Number(demandRateInput)) || Number(demandRateInput) <= 0;
  const validSequenceLength = (value: string) => /^[1-9][0-9]*$/.test(value) && Number.isSafeInteger(Number(value));
  const invalidMaxSeqLen = maxSeqLenInput !== '' && !validSequenceLength(maxSeqLenInput);
  const invalidPrefillMaxSeqLen = prefillMaxSeqLenInput !== '' && !validSequenceLength(prefillMaxSeqLenInput);
  const invalidDecodeMaxSeqLen = decodeMaxSeqLenInput !== '' && !validSequenceLength(decodeMaxSeqLenInput);

  // A Pareto sweep keeps only the fields buildFrontierRequest sends. It has no
  // load target, so demand rate and concurrency never block it; in SLA-free
  // mode it has no latency bound; with an end-to-end bound it drops TTFT/TPOT.
  const invalidParetoInputs = isDemand
    ? false
    : requestLatency != null
      ? invalidLatency
      : invalidTTFT || invalidTpot;

  const markInputsDirty = () => {
    if (primaryResult || isLoading || frontier.status !== 'idle') setInputsDirty(true);
  };

  const handleSequenceLengthChange = (
    raw: string,
    setInput: (value: string) => void,
    setValue: (value: number | null) => void,
  ) => {
    const value = raw.trim();
    setInput(value);
    const n = Number(value);
    setValue(validSequenceLength(value) ? n : null);
    markInputsDirty();
  };

  const handleIslChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setIslInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setIsl(n);
    markInputsDirty();
  };

  const handleOslChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setOslInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setOsl(n);
    markInputsDirty();
  };

  const handleTtftChange = (raw: string) => {
    const cleaned = raw.replace(/[^0-9.]/g, '');
    setTtftInput(cleaned);
    const n = Number(cleaned);
    if (Number.isFinite(n) && n > 0) setTtft(n);
    markInputsDirty();
  };

  const handleTpotChange = (raw: string) => {
    const cleaned = raw.replace(/[^0-9.]/g, '');
    setTpotInput(cleaned);
    const n = Number(cleaned);
    if (Number.isFinite(n) && n > 0) setTpot(n);
    markInputsDirty();
  };

  const handleConcurrencyChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setConcurrencyInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setTargetConcurrency(n);
    markInputsDirty();
  };

  const handleLatencyChange = (raw: string) => {
    const cleaned = raw.replace(/[^0-9.]/g, '');
    setLatencyInput(cleaned);
    const n = Number(cleaned);
    setRequestLatency(cleaned === '' ? null : (Number.isFinite(n) && n > 0 ? n : null));
    markInputsDirty();
  };

  const handlePrefixChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setPrefixInput(digits);
    const n = parseInt(digits, 10);
    setPrefix(isNaN(n) ? 0 : n);
    markInputsDirty();
  };

  const handleDemandRateChange = (raw: string) => {
    const cleaned = raw.replace(/[^0-9.]/g, '');
    setDemandRateInput(cleaned);
    const n = Number(cleaned);
    if (Number.isFinite(n) && n > 0) setDemandRate(n);
    markInputsDirty();
  };

  // Model status check + fetch HF config. Catalog models resolve server-side;
  // everything else (incl. tested models outside the catalog) needs its HF
  // config.json fetched and stored so it can be sent to AISimulators on calculate.
  React.useEffect(() => {
    const reportStatus = (status: ModelCheckStatus) => setModelCheck({ model, status });
    if (catalogLoading) { reportStatus('idle'); return; }
    setHfConfig(null);
    let cancelled = false;
    const timer = setTimeout(() => {
      if (!model.includes('/')) { reportStatus('idle'); return; }
      const isTested = validatedModelIds.includes(model);
      if (!needsHfConfig(model, MODEL_OPTIONS)) {
        reportStatus(isTested ? 'supported' : 'catalog');
        return;
      }
      reportStatus('fetching');
      fetchModelConfig(model, hfToken).then(r => {
        // Ignore a response for a model the user has since moved away from.
        if (cancelled) return;
        if (r.success && r.config) {
          setHfConfig(r.config as Record<string, unknown>);
          // Tested models keep their blue "supported" status even though we
          // fetched the config from HF; others show the neutral "fetched".
          reportStatus(isTested ? 'supported' : 'fetched');
        } else {
          // Tested models stay blue — AISimulators can still resolve them from HF
          // server-side when we send no model_config.
          reportStatus(isTested ? 'supported' : 'error');
        }
      });
    }, 500);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [model, hfToken, MODEL_OPTIONS, catalogLoading, hydrated, validatedModelIds]);

  // Fetch live pricing

  const currentGpuOption = selectableGpus.find(g => g.systemId === gpuSystem) ?? selectableGpus[0] ?? null;

  React.useEffect(() => {
    if (!gpuReady || selectableGpus.length === 0) return;
    if (!selectableGpus.some(gpu => gpu.systemId === gpuSystem)) {
      setGpuSystem(selectableGpus[0].systemId);
    }
  }, [gpuReady, gpuSystem, selectableGpus]);

  // Accepting a shared simulator narrows the catalog, and the effect above
  // replaces a system that simulator does not offer. Put the linked system back
  // once it is selectable; declare this after the replacement so it wins.
  React.useEffect(() => {
    const pending = pendingLinkedSystem.current;
    if (!pending) return;
    if (!selectableGpus.some(option => option.systemId === pending)) return;
    pendingLinkedSystem.current = null;
    setGpuSystem(pending);
    markInputsDirty();
    // The catalog narrowing is what matters here, not a keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectableGpus]);

  const classifierInput = React.useMemo<CanonicalClassifierInput | null>(() => {
    if (!result || !params) return null;
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
        model_id: params.model_path,
        system_id: params.system,
        backend: params.backend ?? inferenceBackend,
        backend_version: params.backend_version ?? null,
        isl: params.isl,
        osl: params.osl,
        max_seq_len: params.max_seq_len ?? null,
        prefill_max_seq_len: params.prefill_max_seq_len ?? null,
        decode_max_seq_len: params.decode_max_seq_len ?? null,
        concurrency: result.performance.concurrency,
        tp_size: result.recommendation.tensorParallelSize,
        pp_size: result.recommendation.pipelineParallelSize,
        dp_size: result.recommendation.dataParallelSize,
        cp_size: result.recommendation.contextParallelSize,
        moe_tp_size: result.recommendation.moeTensorParallelSize,
        moe_ep_size: result.recommendation.moeExpertParallelSize,
        shared_prefix_tokens: params.prefix ?? 0,
        prefix_caching_enabled: (params.prefix ?? 0) > 0 ? true : null,
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
        target_request_rate: params.target_request_rate ?? null,
        target_concurrency: params.target_concurrency ?? null,
        target_latency_ms: params.request_latency ?? null,
        target_ttft_ms: params.ttft ?? null,
        target_tpot_ms: params.tpot ?? null,
        database_mode: null,
        top_n: null,
      },
      model_config: params.model_config ?? null,
    };
  }, [inferenceBackend, params, result]);
  const { prediction: classifierPrediction, isLoading: classifierLoading } = useTestedClassifier(classifierInput);

  // Selection follows the actual value, so clicking a preset, typing matching
  // lengths, or opening a shared view all light the same chip. Keeping the
  // choice in its own state let a link restore the lengths while the chip went
  // on claiming "Default". A cleared field keeps its last accepted number, so
  // invalid input must select nothing.
  const activePreset = invalidISL || invalidOSL
    ? ''
    : (matchWorkloadPreset(
        workloadPresetOptions(getAppConfig().workloadPresets),
        isl,
        osl,
      )?.key ?? '');
  // Selection follows the actual value, so typing a matching concurrency
  // selects the preset and changing it deselects without extra state.
  const activeConcurrencyPreset = matchConcurrencyPresetKey(getAppConfig().concurrencyPresets, concurrencyInput);

  const applyPreset = (p: Pick<WorkloadPreset, 'key' | 'isl' | 'osl'>) => {
    reset();
    setIsl(p.isl); setIslInput(String(p.isl));
    setOsl(p.osl); setOslInput(String(p.osl));
    setInputsDirty(false);
    setExpanded(prev => prev.includes('customize') ? prev : [...prev, 'customize']);
  };

  const resetToDefaults = () => {
    applyPreset(DEFAULT_WORKLOAD);
    setRequestLatency(null); setLatencyInput('');
    setMaxSeqLen(null); setMaxSeqLenInput('');
    setPrefillMaxSeqLen(null); setPrefillMaxSeqLenInput('');
    setDecodeMaxSeqLen(null); setDecodeMaxSeqLenInput('');
  };

  const applyConcurrencyPreset = (p: ConcurrencyPreset) => {
    setTargetConcurrency(p.concurrency); setConcurrencyInput(String(p.concurrency));
    markInputsDirty();
  };

  // One row per link field: how this view exports it and how an incoming link
  // applies it. The table is total over the fields this page owns, so a new
  // contract field fails to compile until it is bound on both sides.
  const linkFields = defineAccessors(recommendLink, {
    model: {
      read: () => model || undefined,
      write: (value) => {
        setModel(value);
        // The link changed the form, not this reader's saved default.
        modelTouched.current = false;
      },
    },
    system: { read: () => gpuSystem || undefined, write: setGpuSystem },
    goal: { read: () => goal, write: setGoal },
    isl: {
      read: () => isl,
      write: (value) => { setIsl(value); setIslInput(String(value)); },
    },
    osl: {
      read: () => osl,
      write: (value) => { setOsl(value); setOslInput(String(value)); },
    },
    // Latency and capacity sizing both submit a target concurrency, so it
    // belongs in every non-demand link; omitting it restored concurrency 1 on
    // a fresh page.
    concurrency: {
      read: () => (isDemand ? undefined : targetConcurrency),
      write: (value) => { setTargetConcurrency(value); setConcurrencyInput(String(value)); },
    },
    ttft: {
      read: () => (goal === 'latency' && requestLatency === null ? ttft : undefined),
      write: (value) => { setTtft(value); setTtftInput(String(value)); },
    },
    tpot: {
      read: () => (goal === 'latency' && requestLatency === null ? tpot : undefined),
      write: (value) => { setTpot(value); setTpotInput(String(value)); },
    },
    e2eLatency: {
      read: () => (goal === 'latency' && requestLatency !== null ? requestLatency : undefined),
      write: (value) => { setRequestLatency(value); setLatencyInput(String(value)); },
    },
    demandRate: {
      read: () => (goal === 'demand' ? demandRate : undefined),
      write: (value) => { setDemandRate(value); setDemandRateInput(String(value)); },
    },
    prefix: {
      read: () => prefix,
      write: (value) => { setPrefix(value); setPrefixInput(String(value)); },
    },
    maxSeqLen: {
      read: () => maxSeqLen ?? undefined,
      write: (value) => { setMaxSeqLen(value); setMaxSeqLenInput(String(value)); },
    },
    servingMode: {
      read: () => servingMode,
      // A link names the topology the author was viewing, so applying it is an
      // explicit choice even while the current simulator cannot honour it.
      write: (value) => setServingModeChoice(value),
    },
    prefillMaxSeqLen: {
      read: () => (servingMode === 'disagg' ? prefillMaxSeqLen ?? undefined : undefined),
      write: (value, values) => {
        setPrefillMaxSeqLen(value);
        setPrefillMaxSeqLenInput(String(value));
        // Pool lengths only exist in a prefill/decode view, so a link written
        // before the mode field described one. A link that names the mode has
        // already been applied and must keep its own choice.
        if (values.servingMode == null) setServingModeChoice('disagg');
      },
    },
    decodeMaxSeqLen: {
      read: () => (servingMode === 'disagg' ? decodeMaxSeqLen ?? undefined : undefined),
      write: (value, values) => {
        setDecodeMaxSeqLen(value);
        setDecodeMaxSeqLenInput(String(value));
        if (values.servingMode == null) setServingModeChoice('disagg');
      },
    },
    // The field has three meanings and the link must carry the one the page
    // submits: omitted for a goal or backend that makes no topology claim, the
    // name for a P/D search, and sentinel-encoded "none" for an explicit
    // aggregate-only request. Reading the raw selection instead would export a
    // remembered fabric name for a request that sends null.
    fabric: {
      read: () => {
        const submitted = resolveSearchFabric(
          fabricRequired,
          aggregateOnlyGoal,
          disaggSupported,
          selectedNetworkFabric,
          servingMode,
        );
        if (submitted === undefined) return undefined;
        return submitted === null || submitted === '' ? FABRIC_NONE : submitted;
      },
      write: (value, values) => {
        setNetworkFabric(value === FABRIC_NONE ? '' : value);
        // A named fabric is the topology a pre-mode link described, and "None"
        // was that link's aggregate-only request. Applied in field order, so a
        // fabric write settles an inferred pool-length mode.
        if (values.servingMode == null) {
          setServingModeChoice(value === FABRIC_NONE ? 'agg' : 'disagg');
        }
      },
    },
  });

  // A shared view applies after settings, the catalog, and the starting model
  // have settled. URL values change this form only; nothing is persisted, and
  // opening a link never submits a request. The provenance trio is applied for
  // the session by `useLinkProvenance` instead of by this form.
  const shared = useSharedView(recommendLink, {
    fields: linkFields,
    ready: hydrated && gpuReady && modelFromSettingsApplied && !catalogLoading,
    defaults: RECOMMEND_LINK_DEFAULTS,
    provenance: {
      simulator: simulator || undefined,
      backend: inferenceBackend,
      backendVersion: backendVersion || undefined,
    },
    // A retained result no longer matches the form the link just described.
    onApplied: () => {
      // A new link supersedes any system still waiting on an accepted
      // simulator; carrying it over would rewrite a different view.
      pendingLinkedSystem.current = null;
      markInputsDirty();
    },
  });
  const linkWarnings = shared.warnings;

  // A link generated with another simulator is never switched silently: this
  // page shows the shared values and offers the switch. A matching simulator
  // applies the shared backend and version to this view only.
  const { sharedSimulator, acceptSharedSimulator } = useLinkProvenance(
    () => shared.parsed?.values ?? null,
    hydrated && gpuReady && modelFromSettingsApplied && !catalogLoading,
  );

  const useSharedSimulator = () => {
    // The catalog narrows to the shared simulator on the next render, which
    // replaces a system that simulator does not offer. Keep the linked system
    // to re-apply once it is selectable again, and mark any visible output
    // stale because a different simulator will produce the next result.
    pendingLinkedSystem.current = shared.parsed?.values?.system ?? null;
    markInputsDirty();
    acceptSharedSimulator();
  };

  // The recommend provider keeps the submitted request and its result across
  // navigation, but this page's form starts from scratch. Compare each restored
  // or newly submitted request with the visible form so a stale result is never
  // presented as matching the inputs that produced it.
  const checkedSubmittedParams = React.useRef<typeof params>(null);
  const hasDisplayedResult = primaryResult != null || runMode === 'pareto';
  React.useEffect(() => {
    if (!hydrated || !modelFromSettingsApplied || catalogLoading || !params || !hasDisplayedResult) return;
    if (checkedSubmittedParams.current === params) return;
    checkedSubmittedParams.current = params;
    const currentObjective = isCapacity
      ? 'concurrency_capacity'
      : isDemand
        ? 'throughput_demand'
        : 'min_gpus';
    const matches = submittedInputsMatch(params, {
      model,
      system: gpuSystem,
      isl,
      osl,
      objective: currentObjective,
      ttft,
      tpot,
      targetConcurrency,
      demandRate,
      requestLatency,
      prefix,
      maxSeqLen,
      prefillMaxSeqLen: servingMode === 'disagg' ? prefillMaxSeqLen : null,
      decodeMaxSeqLen: servingMode === 'disagg' ? decodeMaxSeqLen : null,
      effectiveFabric: resolveSearchFabric(
        fabricRequired,
        aggregateOnlyGoal,
        disaggSupported,
        selectedNetworkFabric,
        servingMode,
      ),
      servingMode,
    });
    if (!matches) setInputsDirty(true);
  }, [hydrated, modelFromSettingsApplied, catalogLoading, params, hasDisplayedResult, model, gpuSystem, isl, osl, ttft, tpot, targetConcurrency, demandRate, prefix, maxSeqLen, prefillMaxSeqLen, decodeMaxSeqLen, servingMode, requestLatency, isCapacity, isDemand, aggregateOnlyGoal, fabricRequired, disaggSupported, selectedNetworkFabric]);

  const sizingObjective: RecommendObjective = isCapacity
    ? 'concurrency_capacity'
    : isDemand
      ? 'throughput_demand'
      : 'min_gpus';

  // One builder for both buttons: a Pareto sweep is the same request minus the
  // mode. Both read the same fabric, so the control, the sweep, and the copied
  // link describe one topology. Building them separately let the paths drift.
  const buildRecommendParams = ({ pareto = false }: { pareto?: boolean } = {}): RecommendParams => {
    const fabric = resolveSearchFabric(
      fabricRequired,
      aggregateOnlyGoal,
      disaggSupported,
      selectedNetworkFabric,
      servingMode,
    );
    return {
      model_path: model,
      system: gpuSystem,
      isl,
      osl,
      sla_mode: aggregateOnlyGoal ? 'none' : 'enforce',
      objective: sizingObjective,
      ...(isCapacity
        ? { target_concurrency: targetConcurrency }
        : isDemand
          ? { target_request_rate: demandRate }
          : {
              target_concurrency: targetConcurrency,
              // An end-to-end bound replaces the decomposed TTFT/TPOT bounds; the
              // request schema rejects sending both shapes at once.
              ...(requestLatency != null
                ? { request_latency: requestLatency }
                : { ttft, tpot }),
            }),
      prefix,
      ...(maxSeqLen != null ? { max_seq_len: maxSeqLen } : {}),
      ...(servingMode === 'disagg' && prefillMaxSeqLen != null ? { prefill_max_seq_len: prefillMaxSeqLen } : {}),
      ...(servingMode === 'disagg' && decodeMaxSeqLen != null ? { decode_max_seq_len: decodeMaxSeqLen } : {}),
      backend: inferenceBackend,
      ...(backendVersion ? { backend_version: backendVersion } : {}),
      // Send the HF config for models AISimulators can't resolve from its catalog.
      ...(needsHfConfig(model, MODEL_OPTIONS) && hfConfig ? { model_config: hfConfig } : {}),
      simulator,
      // The simulator reports whether it answers one bounded search or streams
      // GPU windows. This is a proxy hint and is never forwarded upstream.
      search_strategy: support?.recommendSearch ?? 'windowed',
      // Omitted for an aggregate-only goal and for a backend that models no
      // fabric; null asks for aggregate candidates only; a name is the topology
      // the search should model.
      ...(fabric === undefined ? {} : { fabric }),
      // The selected serving mode, for a goal that can act on it. Demand and
      // capacity report the deployment they sized, so a mode here would state a
      // narrowing the response does not honour. A sweep searches both modes.
      ...(pareto || aggregateOnlyGoal ? {} : { mode: servingMode }),
    };
  };

  const handleCalculate = () => {
    setInputsDirty(false);
    startSizing(buildRecommendParams());
  };

  const handlePareto = () => {
    setInputsDirty(false);
    // The sweep reads the same deployment view as sizing: a P/D view names the
    // fabric, while an aggregated view asks for aggregate candidates only. It
    // omits the mode, which a sweep rejects.
    startPareto(buildRecommendParams({ pareto: true }));
  };

  // Local memory analysis using the same engine as Quick Estimate
  // Animated values
  const gpuCount = useCountUp(result?.recommendation.gpusNeeded ?? 0);
  const ttftMs = useCountUp(result?.performance.ttftLatencyMs ?? 0, 750, 0);
  const e2eLatencyMs = useCountUp(result?.performance.requestLatencyMs ?? 0, 750, 0);
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
  const demandOutputTokPerSecond = isDemand ? demandRate * osl : 0;
  // Capacity reserves the peak context a resident sequence can reach. Prefix
  // caching does not shrink that, so it is not subtracted.
  const capacityContextTokens = maxSeqLen ?? isl + osl;

  return (
    <div className={styles.page}>
      {/* ─── Header ─── */}
      <div className={styles.header}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16 }}>
          <div>
            <h1 className={styles.pageTitle}>Recommend sizing</h1>
            <p className={styles.subtitle}>
              Start with just a model name — we fill the rest, then let you tune every assumption.
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 'none' }}>
            <CopyLinkButton getUrl={shared.build} />
            <TourButton />
          </div>
        </div>
      </div>

      {linkWarnings.length > 0 && <WarningPanel warnings={linkWarnings} />}

      {sharedSimulator && (
        <SharedViewProvenance
          sharedSimulator={sharedSimulator}
          currentSimulator={simulator}
          simulators={simulators}
          onUseShared={useSharedSimulator}
        />
      )}

      {/* ─── Input card ─── */}
      <div className={`${styles.card} ${styles.inputCard}`} data-tour="model">
        {/* Model + GPU row */}
        <div className={styles.inputGrid}>
          <div>
            <ComboBox
              id="adv-model"
              value={model}
              onChange={(value, source) => { modelTouched.current = source === 'user'; setModel(value); markInputsDirty(); }}
              items={modelItems}
              placeholder="e.g. meta-llama/Llama-3.1-70B-Instruct"
              allowCustom
                supportedModels={testedModelsAvailable ? validatedModelIds : undefined}
              hfToken={hfToken}
            />
          </div>

          <GpuSystemInput id="adv-gpu" value={gpuSystem} onChange={(value) => { setGpuSystem(value); markInputsDirty(); if (value) setDefaultSystem(value); }} gpuOptions={selectableGpus} testedSystemIds={[...testedSystemsForModel]} />
        </div>

        {simulatorUnavailable && (
          <div style={{ marginTop: 12, fontSize: 13, color: '#a30000' }}>
            The capabilities for <strong>{support ? simulatorDisplayLabel(support.id, support.label) : 'the selected simulator'}</strong> could not be read,
            so it cannot size workloads. Choose another simulator in Settings.
          </div>
        )}

        {modelUnsupported && (
          <div style={{ marginTop: 12, fontSize: 13, color: '#a30000' }}>
            <strong>{model}</strong> is not in the {support ? simulatorDisplayLabel(support.id, support.label) : 'selected simulator'} model catalog,
            which lists every model that simulator can price. Choose a catalog model or switch simulators in Settings.
          </div>
        )}

        {fabricRequired && disaggCapable && (
          <div style={{ marginTop: 14, maxWidth: 360 }}>
            <label className={styles.fieldLabel} htmlFor="recommend-network-fabric">Inter-node fabric</label>
            <FormSelect
              id="recommend-network-fabric"
              // Show what the request will submit. An aggregated view sends an
              // explicit null even when a fabric name is still remembered, so
              // displaying the remembered name would claim a topology the
              // request does not use.
              value={servingMode === 'disagg' ? selectedNetworkFabric : ''}
              aria-label="Inter-node fabric"
              isDisabled={availableNetworks.length === 0}
              onChange={(_, value) => {
                setNetworkFabric(value);
                // A named fabric is a prefill/decode topology; "None" is the
                // aggregate-only request. Record the explicit choice so a later
                // capability change does not silently reverse it.
                setServingModeChoice(value === '' ? 'agg' : 'disagg');
                markInputsDirty();
              }}
            >
              <FormSelectOption value="" label="None (aggregate candidates only)" />
              {availableNetworks.map(network => (
                <FormSelectOption key={network} value={network} label={network} />
              ))}
            </FormSelect>
            <div style={{ fontSize: 13, color: '#3c3f42', marginTop: 4 }}>
              {servingMode === 'disagg'
                ? networkFabric === null && defaultNetworkFabric
                  ? `${defaultNetworkFabric} is the simulator's default; choose another fabric or None for aggregate-only search.`
                  : 'The simulator searches aggregate and P/D candidates using this fabric.'
                : 'The aggregated view sends no fabric, so sizing and the Pareto frontier both ask for aggregated candidates only.'}
            </div>
          </div>
        )}

        {/* Calculate and Pareto are alternative actions, not a chain. */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 12, marginTop: 16 }} data-tour="calculate">
          <Button
            variant="secondary"
            onClick={handlePareto}
            isDisabled={
              isLoading ||
              simulatorUnavailable ||
              !paretoSupported ||
              isCapacity ||
              modelUnsupported ||
              !model.includes('/') ||
              invalidISL ||
              invalidOSL ||
              invalidMaxSeqLen ||
              (servingMode === 'disagg' && (invalidPrefillMaxSeqLen || invalidDecodeMaxSeqLen)) ||
              catalogLoading ||
              invalidParetoInputs
            }
          >
            {isLoading && runMode === 'pareto' ? 'Running Pareto...' : 'Pareto frontier'}
          </Button>
          <Button
            variant="primary"
            onClick={handleCalculate}
            isDisabled={isLoading || simulatorUnavailable || modelUnsupported || !model.includes('/') || invalidISL || invalidOSL || invalidMaxSeqLen || (servingMode === 'disagg' && (invalidPrefillMaxSeqLen || invalidDecodeMaxSeqLen)) || catalogLoading || (isDemand ? invalidDemandRate : isCapacity ? invalidConcurrency : (invalidTTFT || invalidTpot || invalidConcurrency || invalidLatency))}
          >
            {isLoading && runMode === 'sizing' ? 'Calculating...' : 'Calculate'}
          </Button>
        </div>
        {paretoDisabledReason && (
          <div style={{ fontSize: 13, color: '#3c3f42', marginTop: 6, textAlign: 'right' }}>
            {paretoDisabledReason}
          </div>
        )}
      </div>

      {inputsDirty && (
        <WarningPanel
          warnings={[
            runMode === 'pareto'
              ? 'Inputs have changed since the last Pareto sweep. Click Pareto frontier to update the results.'
              : 'Inputs have changed since the last submitted calculation. Click Calculate to update results.',
          ]}
        />
      )}

      {hydrated && (
        <div data-tour="workload-presets" style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
          <span style={{ fontSize: '12px', fontWeight: 600, fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.06em', color: '#3c3f42', whiteSpace: 'nowrap' }}>Workload:</span>
          <Button className="gc-preset-button" variant={activePreset === DEFAULT_WORKLOAD.key ? 'secondary' : 'tertiary'} size="sm" aria-pressed={activePreset === DEFAULT_WORKLOAD.key} onClick={resetToDefaults}>Default</Button>
          {getAppConfig().workloadPresets.map(p => (
            <Button key={p.key} className="gc-preset-button" variant={activePreset === p.key ? 'secondary' : 'tertiary'} size="sm" aria-pressed={activePreset === p.key} onClick={() => applyPreset(p)}>
              {p.label}
            </Button>
          ))}
        </div>
      )}

      {/* ---------- concurrency presets ---------- */}
      {!isDemand && hydrated && getAppConfig().concurrencyPresets.length > 0 && (
        <div data-tour="concurrency-presets" style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
          <span style={{ fontSize: '12px', fontWeight: 600, fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.06em', color: '#3c3f42', whiteSpace: 'nowrap' }}>Concurrency:</span>
          {getAppConfig().concurrencyPresets.map(p => (
            <Button key={p.key} className="gc-preset-button" variant={activeConcurrencyPreset === p.key ? 'secondary' : 'tertiary'} size="sm" aria-pressed={activeConcurrencyPreset === p.key} onClick={() => applyConcurrencyPreset(p)}>
              {p.label}
            </Button>
          ))}
        </div>
      )}

      {/* ---------- sizing goal ---------- */}
      <div className={styles.goalRow} data-tour="goal">
        <span className={styles.goalLabel}>Sizing goal:</span>
        <ToggleGroup aria-label="Sizing goal">
          <ToggleGroupItem
            text="Latency SLO"
            isSelected={goal === 'latency'}
            onChange={() => {
              if (goal !== 'latency') { reset(); setGoal('latency'); }
            }}
          />
          <ToggleGroupItem
            text="Throughput demand"
            isSelected={goal === 'demand'}
            onChange={() => {
              if (goal !== 'demand') { reset(); setGoal('demand'); }
            }}
          />
          {capacitySupported && (
            <ToggleGroupItem
              text="Concurrency capacity"
              isSelected={goal === 'capacity'}
              onChange={() => {
                if (goal !== 'capacity') { reset(); setGoal('capacity'); }
              }}
            />
          )}
        </ToggleGroup>
        <span className={styles.goalHint}>
          {isCapacity
            ? 'Find the fewest GPUs whose weights and KV cache hold a number of concurrent sequences. Latency and throughput are reported, not gated.'
            : isDemand
              ? 'Find the fewest GPUs that sustain an arrival rate. No latency target is applied.'
              : 'Find the fewest GPUs that meet TTFT/TPOT (or end-to-end) latency targets.'}
        </span>
      </div>

      <InfoStrip data-tour="targets">
        {isCapacity ? (
          <>
            Based on your configuration — ISL: {isl}, OSL: {osl}, {targetConcurrency} concurrent {targetConcurrency === 1 ? 'sequence' : 'sequences'} at {capacityContextTokens.toLocaleString()} tokens each, no latency target.
          </>
        ) : isDemand ? (
          <>
            Based on your configuration — ISL: {isl}, OSL: {osl}, Demand: {demandRate} req/s
            {demandRate === DEFAULT_DEMAND_RATE && (
              <span style={{ color: 'var(--t2)' }}> (default — adjust)</span>
            )}
            {' '}(≈ {Math.round(demandOutputTokPerSecond).toLocaleString()} output tok/s at OSL {osl}), no latency target.
          </>
        ) : requestLatency != null ? (
          <>
            Based on your configuration — ISL: {isl}, OSL: {osl}, E2E target: {formatLatencyBound(requestLatency)}s
            {prefix > 0 ? `, prefix ${prefix.toLocaleString()} tokens` : ''}, {targetConcurrency} concurrent {targetConcurrency === 1 ? 'user' : 'users'}.
          </>
        ) : (
          <>
            Based on your configuration — ISL: {isl}, OSL: {osl}, Maximum TTFT: {formatLatencyBound(ttft)}s
            {prefix > 0 ? `, prefix ${prefix.toLocaleString()} tokens` : ''}, Maximum TPOT: {tpot} ms, {targetConcurrency} concurrent {targetConcurrency === 1 ? 'user' : 'users'}.
          </>
        )}
        {!aggregateOnlyGoal && (
          <>
            {' '}Deployment view: {servingMode === 'disagg' ? 'disaggregated' : 'aggregated'}.
          </>
        )}
        {' '}<InfoStripAction onClick={() => setExpanded(expanded.includes('customize') ? expanded.filter(e => e !== 'customize') : [...expanded, 'customize'])}>
          Adjust? (edit fields below)
        </InfoStripAction>
      </InfoStrip>

      {/* ─── Customization section (ISL/OSL/TTFT + HF token + constraints) ─── */}
      {expanded.includes('customize') && (
        <div className={`${styles.card}`} style={{ marginBottom: 16 }}>
          <div className={styles.cardBody}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 12, fontFamily: 'var(--sans)', fontSize: 14, fontWeight: 600, color: 'var(--t)' }}>
              Deployment view
              {!aggregateOnlyGoal && (
                <>
                  <Button
                    className="gc-preset-button"
                    variant={servingMode === 'agg' ? 'secondary' : 'tertiary'}
                    size="sm"
                    aria-pressed={servingMode === 'agg'}
                    onClick={() => { setServingModeChoice('agg'); markInputsDirty(); }}
                  >
                    Aggregated
                  </Button>
                  <Button
                    className="gc-preset-button"
                    variant={servingMode === 'disagg' ? 'secondary' : 'tertiary'}
                    size="sm"
                    aria-pressed={servingMode === 'disagg'}
                    isDisabled={!disaggSelectable}
                    onClick={() => {
                      if (!disaggSelectable) return;
                      setServingModeChoice('disagg');
                      markInputsDirty();
                    }}
                  >
                    Disaggregated
                  </Button>
                </>
              )}
              {!aggregateOnlyGoal && servingMode === 'disagg' && (
                <span style={{ fontSize: 13, fontWeight: 400, color: '#3c3f42' }}>Prefill and decode have separate inputs in this view.</span>
              )}
              {!aggregateOnlyGoal && !disaggSelectable && (
                <span style={{ fontSize: 13, fontWeight: 400, color: '#3c3f42' }}>
                  {!disaggSupported
                    ? `The selected simulator does not model prefill/decode disaggregation for ${modeledBackendLabel || 'the selected backend'}.`
                    : availableNetworks.length === 0
                      ? 'No network fabric is available to model a prefill/decode search.'
                      : 'Choose a network fabric to view prefill/decode inputs.'}
                </span>
              )}
              {aggregateOnlyGoal && (
                <span style={{ fontSize: 13, fontWeight: 400, color: '#3c3f42' }}>
                  {isCapacity
                    ? 'Capacity sizing reports aggregated and disaggregated options; it does not search one serving mode.'
                    : 'Demand sizing reports the deployment that sustains the arrival rate; it does not take a serving mode.'}
                </span>
              )}
            </div>
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
                <label className={styles.fieldLabel}>
                  {isCapacity
                    ? <>Concurrent sequences <Term k="concurrent" /></>
                    : isDemand
                      ? 'Demand (req/s)'
                      : <>Max TTFT (ms) <Term k="ttft" /></>}
                </label>
                {isCapacity ? (
                  <>
                    <input
                      type="number"
                      className={invalidConcurrency ? styles.paramInputInvalid : styles.paramInput}
                      value={concurrencyInput}
                      onChange={e => handleConcurrencyChange(e.target.value)}
                      min={1}
                    />
                    <div className={styles.helperText}>Sequences held at full context at the same time.</div>
                  </>
                ) : isDemand ? (
                  <>
                    <input
                      type="number"
                      className={invalidDemandRate ? styles.paramInputInvalid : styles.paramInput}
                      value={demandRateInput}
                      onChange={e => handleDemandRateChange(e.target.value)}
                      min={0.1}
                      step={0.1}
                      placeholder="10"
                    />
                    <div className={styles.helperText}>Default 10 req/s — adjust for your workload.</div>
                  </>
                ) : (
                  <input
                    type="number"
                    className={invalidTTFT ? styles.paramInputInvalid : styles.paramInput}
                    value={ttftInput}
                    onChange={e => handleTtftChange(e.target.value)}
                    min={0.1}
                    step={0.1}
                  />
                )}
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

            {isCapacity ? (
              <div className={styles.paramGrid} style={{ marginTop: 12, gridTemplateColumns: 'repeat(3, 1fr)' }}>
                <div>
                  <div className={styles.fieldLabel}>Context per sequence</div>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 700 }}>
                    {capacityContextTokens.toLocaleString()} tokens
                  </div>
                  <div className={styles.helperText}>
                    {maxSeqLen != null ? 'Max sequence length' : 'ISL + OSL, with no prefix-cache credit'}
                  </div>
                </div>
                <div>
                  <div className={styles.fieldLabel}>Sequences to hold</div>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 700 }}>
                    {targetConcurrency.toLocaleString()}
                  </div>
                  <div className={styles.helperText}>
                    Weights plus KV cache must hold all of them concurrently.
                  </div>
                </div>
                <div>
                  <div className={styles.fieldLabel}>Latency targets</div>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 700 }}>
                    Not applied
                  </div>
                  <div className={styles.helperText}>
                    Concurrency capacity sizes for memory residency only. Resulting latency and throughput are reported, not gated.
                  </div>
                </div>
              </div>
            ) : isDemand ? (
              <div className={styles.paramGrid} style={{ marginTop: 12, gridTemplateColumns: 'repeat(2, 1fr)' }}>
                <div>
                  <div className={styles.fieldLabel}>Derived output throughput</div>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 700 }}>
                    {Math.round(demandOutputTokPerSecond).toLocaleString()} tok/s
                  </div>
                  <div className={styles.helperText}>
                    {demandRate} req/s × {osl} output tokens per request
                  </div>
                </div>
                <div>
                  <div className={styles.fieldLabel}>Latency targets</div>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 700 }}>
                    Not applied
                  </div>
                  <div className={styles.helperText}>
                    Throughput demand mode sizes for the arrival rate only. Switch to Latency SLO to enforce TTFT, TPOT, or end-to-end targets.
                  </div>
                </div>
              </div>
            ) : (
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
            )}

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
                      {servingMode === 'disagg' && (
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
                      )}
                      {servingMode === 'disagg' && (
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
                      )}
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        </div>
      )}

      {/* ─── Loading ─── */}
      {isLoading && runMode === 'sizing' && (
        <div className={styles.card}>
          <GpuChipLoader elapsed={elapsed} timeoutSeconds={gatewayTimeout} />
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

      <TestedPerformanceFeedback
        prediction={classifierPrediction}
        isLoading={classifierLoading}
        modelId={params?.model_path ?? model}
        systemId={params?.system ?? gpuSystem}
        headerRight={result ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginLeft: 'auto' }}>
            <SimulatorBadge simulator={result.simulator} />
            <span style={{ fontSize: 13, color: '#3c3f42' }}>Sizing recommendation</span>
          </div>
        ) : undefined}
      >
        {primaryResult && (primaryResult.searchSummary || primaryResult.sweep?.windowMinGpus != null || !primaryResult.memoryAvailable) && (
          <div style={{ fontSize: 13, color: '#3c3f42' }}>
            {primaryResult.searchSummary && (
              <span>
                Search: {primaryResult.searchSummary.strategy.replaceAll('_', ' ')} · {primaryResult.simulator} · tested {primaryResult.searchSummary.evaluated_configurations ?? primaryResult.searchSummary.evaluated_gpus.length} configurations across {primaryResult.searchSummary.evaluated_gpus.join(', ') || 'none'} GPUs · ceiling {primaryResult.searchSummary.effective_max_gpus} (configured {primaryResult.searchSummary.configured_max_gpus}, hardware {primaryResult.searchSummary.hardware_max_gpus}) · {primaryResult.searchSummary.time_budget_seconds}s{primaryResult.searchSummary.budget_exhausted ? ' time budget reached' : ''}.
              </span>
            )}
            {primaryResult.sweep?.windowMinGpus != null && primaryResult.sweep.windowMaxGpus != null && (
              <div style={{ marginTop: primaryResult.searchSummary ? 4 : 0 }}>
                {sweepCoverageLabel(primaryResult.sweep)}
              </div>
            )}
            {!primaryResult.memoryAvailable && (
              <div style={{ marginTop: primaryResult.searchSummary || primaryResult.sweep?.windowMinGpus != null ? 4 : 0 }}>
                This simulator does not provide memory estimates; the memory tile is hidden.
              </div>
            )}
          </div>
        )}
      </TestedPerformanceFeedback>

      {/* ─── Result tiles ─── */}
      {result && (
        <>
          <div className={styles.tilesGrid} data-tour="results">
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
                  <span className={styles.tileValue} data-testid="recommend-gpus">
                    {gpuCount}<span className={styles.tileUnit}>× {currentGpuOption.label}</span>
                  </span>
                  <span className={styles.tileSub}>
                    {result.mode === 'disagg'
                      ? `${result.recommendation.replicasNeeded} ${result.recommendation.replicasNeeded === 1 ? 'replica' : 'replicas'} × ${result.recommendation.gpusPerReplica} GPUs/replica · ${result.performance.concurrency} concurrent users`
                      : `${parallelLabel(result.recommendation)} · ${
                          resultIsDemand
                            ? `${result.demand?.observedConcurrency != null ? result.demand.observedConcurrency.toFixed(1) : '—'} observed concurrent`
                            : resultIsCapacity
                              ? `${result.capacity?.capacitySequences ?? result.performance.concurrency} sequences fit · ${result.capacity?.targetConcurrency ?? result.performance.concurrency} requested`
                              : `${result.performance.concurrency} concurrent users`
                        }`}
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
                      total = <span className={styles.em}>{result.recommendation.gpusPerReplica} × {result.recommendation.replicasNeeded} {result.recommendation.replicasNeeded === 1 ? 'replica' : 'replicas'} = {result.recommendation.gpusNeeded} GPUs</span>
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
                      total = <span className={styles.em}>{result.recommendation.gpusPerReplica} × {result.recommendation.replicasNeeded} {result.recommendation.replicasNeeded === 1 ? 'replica' : 'replicas'} = {result.recommendation.gpusNeeded} GPUs</span>
                    </div>
                  </>
                )
              }
            />

            {/* Capacity outcome, demand outcome, or TTFT */}
            {resultIsCapacity ? (
              <FlipTile
                front={
                  <>
                    <span className={styles.tileLabel}><MemoryIcon /> Capacity</span>
                    <span className={styles.tileValue}>
                      {result.capacity?.capacitySequences ?? '—'}<span className={styles.tileUnit}>seq</span>
                    </span>
                    <span className={styles.tileSub}>
                      {result.capacity?.fits ? (
                        <Label color="green" isCompact icon={<CheckCircleIcon />}>holds target</Label>
                      ) : (
                        <Label color="orange" isCompact icon={<ExclamationTriangleIcon />}>
                          short by {result.capacity?.shortfallSequences ?? '—'}
                        </Label>
                      )}
                      <span> · target {result.capacity?.targetConcurrency ?? targetConcurrency} sequences</span>
                    </span>
                  </>
                }
                back={
                  <>
                    <div className={styles.backTitle}>Concurrency capacity</div>
                    <div className={styles.formula}>
                      target: <span className={styles.em}>{result.capacity?.targetConcurrency ?? targetConcurrency} sequences</span><br />
                      context per sequence: <span className={styles.em}>
                        {(result.capacity?.contextPerSequenceTokens ?? capacityContextTokens).toLocaleString()} tokens
                      </span><br />
                      KV capacity: <span className={styles.em}>
                        {(result.capacity?.capacityTokens ?? 0).toLocaleString()} tokens
                      </span><br />
                      fits: <span className={styles.em}>{result.capacity?.capacitySequences ?? '—'} sequences</span><br />
                      {result.capacity?.shortfallTokens != null && (
                        <>shortfall: <span className={styles.em}>{result.capacity.shortfallTokens.toLocaleString()} tokens</span><br /></>
                      )}
                      {result.capacity?.minimumGpus != null && (
                        <>fewest GPUs for one sequence: <span className={styles.em}>{result.capacity.minimumGpus}</span><br /></>
                      )}
                      replicas: <span className={styles.em}>{result.capacity?.replicas ?? result.recommendation.replicasNeeded}</span>
                      {result.capacity && result.capacity.attentionDp > 1 && (
                        <> · attention DP: <span className={styles.em}>{result.capacity.attentionDp}</span></>
                      )}
                    </div>
                    <p className={styles.formulaNote}>
                      Capacity is decided by the KV-cache estimate for the selected aggregated shape.
                      Performance is informational and never gates the fit.
                    </p>
                  </>
                }
              />
            ) : resultIsDemand ? (
              <FlipTile
                front={
                  <>
                    <span className={styles.tileLabel}><TachometerAltIcon /> Demand</span>
                    <span className={styles.tileValue}>
                      {result.demand?.achievedRps != null
                        ? result.demand.achievedRps.toFixed(1)
                        : '—'}<span className={styles.tileUnit}>req/s</span>
                    </span>
                    <span className={styles.tileSub}>
                      {result.demand?.met === false ? (
                        <>
                          <Label color="orange" isCompact icon={<ExclamationTriangleIcon />}>
                            below target
                          </Label>
                          <span> · target {result.demand?.targetRps ?? resultDemandRate} req/s</span>
                        </>
                      ) : result.demand?.met ? (
                        <Label color="green" isCompact icon={<CheckCircleIcon />}>meets target</Label>
                      ) : (
                        <span>target {resultDemandRate} req/s</span>
                      )}
                    </span>
                  </>
                }
                back={
                  <>
                    <div className={styles.backTitle}>Throughput demand</div>
                    <div className={styles.formula}>
                      target: <span className={styles.em}>{result.demand?.targetRps ?? resultDemandRate} req/s</span><br />
                      sustained: <span className={styles.em}>
                        {result.demand?.achievedRps != null ? `${result.demand.achievedRps.toFixed(2)} req/s` : 'not measured'}
                      </span><br />
                      best observed: <span className={styles.em}>
                        {result.demand?.bestObservedRps != null ? `${result.demand.bestObservedRps.toFixed(2)} req/s` : '—'}
                      </span><br />
                      observed concurrency: <span className={styles.em}>
                        {result.demand?.observedConcurrency != null ? result.demand.observedConcurrency.toFixed(1) : '—'}
                      </span><br />
                      arrival window: <span className={styles.em}>
                        {result.demand?.windowSeconds != null ? `${result.demand.windowSeconds.toFixed(2)} s` : '—'}
                      </span>
                    </div>
                    <p className={styles.formulaNote}>
                      Sustained rate excludes the final response drain, so it is directly comparable to
                      the offered rate.
                    </p>
                  </>
                }
              />
            ) : resultRequestLatency != null ? (
              <FlipTile
                front={
                  <>
                    <span className={styles.tileLabel}><ClockIcon /> E2E latency</span>
                    <span className={styles.tileValue}>
                      {e2eLatencyMs.toLocaleString()}<span className={styles.tileUnit}>ms</span>
                    </span>
                    <span className={styles.tileSub}>
                      {result.performance.requestLatencyMs <= resultRequestLatency ? (
                        <Label color="green" isCompact icon={<CheckCircleIcon />}>meets target</Label>
                      ) : (
                        <Label color="orange" isCompact icon={<ExclamationTriangleIcon />}>above target</Label>
                      )}
                    </span>
                  </>
                }
                back={
                  <>
                    <div className={styles.backTitle}>End-to-end latency</div>
                    <div className={styles.formula}>
                      target: <span className={styles.em}>{resultRequestLatency.toLocaleString()} ms</span><br />
                      estimated: <span className={styles.em}>{result.performance.requestLatencyMs.toFixed(1)} ms</span><br />
                      headroom: <span className={styles.em}>{(resultRequestLatency - result.performance.requestLatencyMs).toFixed(1)} ms</span>
                    </div>
                  </>
                }
              />
            ) : (
              <FlipTile
                front={
                  <>
                    <span className={styles.tileLabel}><ClockIcon /> TTFT</span>
                    <span className={styles.tileValue} data-testid="recommend-ttft">
                      {ttftMs}<span className={styles.tileUnit}>ms</span>
                    </span>
                    <span className={styles.tileSub}>
                      {result.performance.ttftLatencyMs <= resultTtft ? (
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
                      target: <span className={styles.em}>{resultTtft.toLocaleString()} ms</span><br />
                      estimated: <span className={styles.em}>{result.performance.ttftLatencyMs.toFixed(1)} ms</span><br />
                      headroom: <span className={styles.em}>{(resultTtft - result.performance.ttftLatencyMs).toFixed(1)} ms</span><br />
                      TPOT: <span className={styles.em}>{result.performance.tpotMs.toFixed(1)} ms</span>
                    </div>
                  </>
                }
              />
            )}

            {/* Throughput (hidden when a capacity run had no performance simulation) */}
            {(!resultIsCapacity || result.capacity?.performanceAvailable) && (
            <FlipTile
              front={
                <>
                  <span className={styles.tileLabel}><TachometerAltIcon /> Throughput</span>
                  <span className={styles.tileValue} data-testid="recommend-throughput">
                    {tpsVal}<span className={styles.tileUnit}>tok/s</span>
                  </span>
                  <span className={styles.tileSub}>
                    {resultIsDemand
                      ? `${result.throughput.tokensPerSecondPerGpu.toFixed(1)}/GPU · ≈ ${Math.round(resultDemandOutputTokPerSecond).toLocaleString()} demanded tok/s`
                      : `${result.throughput.tokensPerSecondPerGpu.toFixed(1)}/GPU · ${result.throughput.tokensPerSecondPerUser.toFixed(2)}/user`}
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
            )}

            {/* Peak memory per GPU (only when supplied by the selected simulator) */}
            {result.memoryAvailable && <FlipTile
              front={
                <>
                  <span className={styles.tileLabel}><MemoryIcon /> Peak mem / GPU</span>
                  <span className={styles.tileValue} data-testid="recommend-peak-memory">
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
            />}

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
              pick a provider on the Cost sources page or choose a GPU with published rates
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
                    {resultIsCapacity && !result.capacity?.performanceAvailable ? (
                      <div style={{ fontSize: 13, color: '#3c3f42' }}>
                        Performance simulation was unavailable for this configuration
                        {result.capacity?.performanceUnavailableReason
                          ? `: ${result.capacity.performanceUnavailableReason}`
                          : ''}. The capacity result does not depend on it.
                      </div>
                    ) : (
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
                    )}
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

      {/* ─── Capacity comparison (promote any option to the result above) ─── */}
      {resultIsCapacity && capacityOptions.length > 0 && (
        <CapacityOptionsPanel
          options={capacityOptions}
          selectedOptionId={selectedCapacityOption?.optionId ?? null}
          gpuLabel={currentGpuOption?.label ?? gpuSystem}
          onSelect={setSelectedCapacityOptionId}
        />
      )}

      {/* ─── Pareto frontier (standalone action, exclusive of the sizing result) ─── */}
      {runMode === 'pareto' && (
        <FrontierPanel
          state={frontier}
          gpuLabel={
            selectableGpus.find(option => option.systemId === params?.system)?.label
            ?? params?.system
            ?? currentGpuOption?.label
            ?? gpuSystem
          }
          onRetry={handlePareto}
          elapsed={elapsed}
          timeoutSeconds={gatewayTimeout}
        />
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

function sweepCoverageLabel(sweep: RecommendSweepMeta): string {
  const min = sweep.windowMinGpus ?? 1
  const max = sweep.windowMaxGpus ?? min
  const range = min === max ? `${min} GPU${min === 1 ? '' : 's'}` : `${min}–${max} GPUs`
  const evaluated = sweep.evaluated ?? 0
  const budget = sweep.budgetExhausted ? ' · time budget reached' : ''
  // Report the deployments the run actually covered. The form's deployment
  // view is an input, not a promise: a search can recommend an aggregated
  // deployment after also evaluating P/D, or stop before reaching it.
  const modes = sweep.modesSearched?.length
    ? ` · searched ${sweep.modesSearched
        .map(mode => (mode === 'disagg' ? 'disaggregated' : 'aggregated'))
        .join(' and ')}`
    : ''
  return `Searched ${range} · ${evaluated} configuration${evaluated === 1 ? '' : 's'} evaluated${budget}${modes}`
}

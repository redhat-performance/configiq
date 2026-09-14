'use client';

import * as React from 'react';
import {
  Button,
  TextInput,
  FormSelect, FormSelectOption,
  Switch,
  Label,
  Accordion, AccordionItem, AccordionToggle, AccordionContent,
} from '@patternfly/react-core';
import StarIcon from '@patternfly/react-icons/dist/esm/icons/star-icon';
import OutlinedStarIcon from '@patternfly/react-icons/dist/esm/icons/outlined-star-icon';
import CheckCircleIcon from '@patternfly/react-icons/dist/esm/icons/check-circle-icon';
import ExclamationTriangleIcon from '@patternfly/react-icons/dist/esm/icons/exclamation-triangle-icon';
import MicrochipIcon from '@patternfly/react-icons/dist/esm/icons/microchip-icon';
import MemoryIcon from '@patternfly/react-icons/dist/esm/icons/memory-icon';
import DollarSignIcon from '@patternfly/react-icons/dist/esm/icons/dollar-sign-icon';
import LayerGroupIcon from '@patternfly/react-icons/dist/esm/icons/layer-group-icon';
import InfoCircleIcon from '@patternfly/react-icons/dist/esm/icons/info-circle-icon';
import EyeIcon from '@patternfly/react-icons/dist/esm/icons/eye-icon';
import EyeSlashIcon from '@patternfly/react-icons/dist/esm/icons/eye-slash-icon';
import styles from './PerformanceEstimate.module.css';
import { Term, FlipTile, Sparkline, useCountUp } from './quickEstimateHelpers';
import { ProductTour, type TourStep } from '@/components/ProductTour';
import { SaveEstimateModal } from './SaveEstimateModal';
import { fetchModelConfig, type HFModelConfig } from '@/lib/huggingface/fetch-config';
import { saveEstimate, getSavedEstimateCount } from '@/lib/saved-estimates';
import { fetchEstimateAsInferenceResult, EstimateError } from '@/lib/api/estimate-adapter';
import { InfoStrip, InfoStripAction } from '@/components/ui/InfoStrip';
import { ModelInput, type ModelStatus } from '@/components/ui/ModelInput';
import { ComboBox, type ComboBoxItem } from '@/components/ModelComboBox/ModelComboBox';
import { buildModelItems, needsHfConfig } from '@/lib/model-options';
import { GpuSystemInput } from '@/components/ui/GpuSystemInput';
import { useAicCatalog } from '@/lib/hooks/useAicCatalog';
import { GpuChipLoader } from '@/components/GpuChipLoader/GpuChipLoader';
import { useSettings } from '@/contexts/SettingsContext';
import { useCostings, resolveCloudRate } from '@/lib/hooks/useCostings';
import { getAppConfig } from '@/lib/app-config';
import { DEFAULT_WORKLOAD, type WorkloadPreset } from '@/lib/workload-presets';
import type { EstimatePhase, InferenceConfigResult } from '@/lib/gpu-math/inference-config';
import Link from 'next/link';
import { HOURS_PER_MONTH, AMORT_MONTHS_3YR, AMORT_MONTHS_5YR } from '@/lib/utils/format';
import { parsePerformancePrefill } from './performance-prefill';

function modelSuggestions(): string {
  return getAppConfig().suggestedModelNames.join(', ');
}

/** String-backed parallelism inputs for one disagg pool (prefill/decode). */
interface PerfPhaseInput {
  tp: string
  pp: string
  workers: string
  batch: string
}

function parsePerfPhase(p: PerfPhaseInput) {
  const n = (v: string, min = 1) => Math.max(min, parseInt(v, 10) || min);
  return { tp: n(p.tp), pp: n(p.pp), workers: n(p.workers), batch: n(p.batch) };
}

/** Four inputs (TP/PP/Workers/Batch) for one disagg pool. */
function PerfPhaseFields({ title, cfg, onChange }: {
  title: string;
  cfg: PerfPhaseInput;
  onChange: (c: PerfPhaseInput) => void;
}) {
  const digits = (v: string) => v.replace(/[^0-9]/g, '');
  const field = (label: string, key: keyof PerfPhaseInput) => (
    <div className={styles.accField}>
      <label className={styles.accFieldLabel}>{label}</label>
      <TextInput
        type="number"
        value={cfg[key]}
        aria-label={`${title} ${label}`}
        onChange={(_, v) => onChange({ ...cfg, [key]: digits(v) })}
      />
    </div>
  );
  return (
    <div>
      <div style={{ fontSize: '13px', fontWeight: 600, fontFamily: 'var(--mono)', color: '#3c3f42', marginBottom: 8 }}>{title}</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 12 }}>
        {field('TP', 'tp')}
        {field('PP', 'pp')}
        {field('Workers', 'workers')}
        {field('Batch', 'batch')}
      </div>
    </div>
  );
}


const QUICK_ESTIMATE_TOUR: TourStep[] = [
  {
    target: '[data-tour="model"]',
    title: 'Start with a model',
    description: 'Type any Hugging Face model ID or pick from popular models. We\'ll auto-detect the specs and fill in smart defaults.',
    position: 'bottom'
  },
  {
    target: '[data-tour="warning"]',
    title: 'Default assumptions',
    description: 'Quick estimates start with common defaults. Click "Adjust" to match your actual workload and traffic patterns.',
    position: 'bottom'
  },
  {
    target: '[data-tour="result-tile-gpus"]',
    title: 'Your results at a glance',
    description: 'These tiles show GPU count, memory requirements, and monthly cost. Click any tile to see the math behind it.',
    position: 'right'
  },
  {
    target: '[data-tour="assumptions"]',
    title: 'Fine-tune your workload',
    description: 'Expand these sections to adjust traffic patterns, sequence lengths, and hardware settings. Results update live as you edit.',
    position: 'top'
  }
];

export default function QuickEstimate() {
  console.log('🔵 QuickEstimate component mounting');
  const { hydrated, hfToken, defaultModel: settingsDefaultModel, inferenceBackend, backendVersion, costingsEnabled, preferredCloudProvider, pricingSource } = useSettings();
  const costings = useCostings(costingsEnabled, pricingSource);
  const { gpuOptions: aicGpus, modelOptions: aicModels, modelSpecs, timeoutSeconds: aicTimeout, isLoading: catalogLoading } = useAicCatalog();

  const [model, setModel] = React.useState('');
  const [gpu, setGpu] = React.useState(() => getAppConfig().defaultSystem);
  const [prefillChecked, setPrefillChecked] = React.useState(false);
  const modelWasPrefilled = React.useRef(false);

  React.useEffect(() => {
    const prefill = parsePerformancePrefill(globalThis.location?.search || '');
    modelWasPrefilled.current = Boolean(prefill.model);
    if (prefill.model) setModel(prefill.model);
    if (prefill.system) setGpu(prefill.system);
    setPrefillChecked(true);
  }, []);

  // eslint-disable-next-line react-hooks/exhaustive-deps -- hydrated gates config.json readiness
  const modelItems: ComboBoxItem[] = React.useMemo(() => buildModelItems(aicModels), [aicModels, hydrated]);

  // Set model from settings after context has loaded from localStorage
  const modelFromSettings = React.useRef(false);
  React.useEffect(() => {
    if (!hydrated || !prefillChecked || modelFromSettings.current) return;
    modelFromSettings.current = true;
    if (!modelWasPrefilled.current) setModel(settingsDefaultModel);
  }, [hydrated, prefillChecked, settingsDefaultModel]);

  // If defaultSystem not in catalog, fall back to first available
  // but only if it's the initial default, not a prefilled value
  const gpuWasPrefilled = React.useRef(false);
  React.useEffect(() => {
    const prefill = parsePerformancePrefill(globalThis.location?.search || '');
    gpuWasPrefilled.current = Boolean(prefill.system);
  }, []);

  React.useEffect(() => {
    if (prefillChecked && aicGpus.length > 0 && !aicGpus.find(g => g.systemId === gpu) && !gpuWasPrefilled.current) {
      setGpu(aicGpus[0].systemId);
    }
  }, [aicGpus, gpu, prefillChecked]);

  const [fav, setFav] = React.useState(false);
  const [expanded, setExpanded] = React.useState<string[]>([]);
  const [showApi, setShowApi] = React.useState(false);
  const [showTour, setShowTour] = React.useState(false);
  const [tourSeen, setTourSeen] = React.useState(false);

  // 🧪 TEST: Inference config engine integration
  const [testResult, setTestResult] = React.useState<InferenceConfigResult | null>(null);
  const [testError, setTestError] = React.useState<string | null>(null);
  const [testErrorCode, setTestErrorCode] = React.useState<string | null>(null);

  // HuggingFace config fetching
  const [hfConfig, setHfConfig] = React.useState<HFModelConfig | null>(null);
  const [isFetchingConfig, setIsFetchingConfig] = React.useState(false);
  const [isUsingFallback, setIsUsingFallback] = React.useState(false);
  const [fallbackReason, setFallbackReason] = React.useState<string>('');

  const modelStatus: ModelStatus = getAppConfig().testedModels.includes(model)
    ? 'supported'
    : aicModels.includes(model)
    ? 'catalog'
    : isFetchingConfig || catalogLoading
    ? 'fetching'
    : hfConfig
    ? 'fetched'
    : testError
    ? 'error'
    : 'idle';

  // Collapsible state for "Why this GPU count?" card
  const [whyGpuExpanded, setWhyGpuExpanded] = React.useState(false);

  // Collapsible state for "Want to change assumptions?" section
  const [assumptionsExpanded, setAssumptionsExpanded] = React.useState(false);
  const [assumptionsHighlight, setAssumptionsHighlight] = React.useState(false);
  const assumptionsRef = React.useRef<HTMLDivElement>(null);

  // Manual override states
  const [parallelismOverride, setParallelismOverride] = React.useState(false);
  const [parallelismManualTP, setParallelismManualTP] = React.useState<number | null>(null);
  const [parallelismManualReplicas, setParallelismManualReplicas] = React.useState<number | null>(null);
  const [vllmOverride, setVllmOverride] = React.useState(false);
  const [vllmManualMaxNumSeqs, setVllmManualMaxNumSeqs] = React.useState<number | null>(null);
  const [vllmManualMaxModelLen, setVllmManualMaxModelLen] = React.useState<number | null>(null);
  const [vllmManualChunkedPrefill, setVllmManualChunkedPrefill] = React.useState<boolean | null>(null);
  const [vllmManualPrefixCaching, setVllmManualPrefixCaching] = React.useState<boolean | null>(null);
  const [vllmManualGpuUtil, setVllmManualGpuUtil] = React.useState<number | null>(null);

  // Save estimate modal
  const [showSaveModal, setShowSaveModal] = React.useState(false);
  const [savedCount, setSavedCount] = React.useState(0);
  const [showToast, setShowToast] = React.useState(false);
  const [toastMessage, setToastMessage] = React.useState('');

  // Interactive controls
  const [testConcurrentUsers, setTestConcurrentUsers] = React.useState(1);
  const [testISL, setTestISL] = React.useState(2048);
  const [testOSL, setTestOSL] = React.useState(128);
  const [testPrefix, setTestPrefix] = React.useState(0);
  const [testTpSize, setTestTpSize] = React.useState(1);
  const [calcTrigger, setCalcTrigger] = React.useState(0);
  const [elapsed, setElapsed] = React.useState(0);
  const [testWeightPrecision, setTestWeightPrecision] = React.useState<'FP16' | 'FP8' | 'INT8' | 'INT4' | 'MXFP4' | 'NVFP4'>('FP16');
  const [testKVCachePrecision, setTestKVCachePrecision] = React.useState<'FP16' | 'FP8' | 'NVFP4'>('FP16');
  const [testMoeQuantMode, setTestMoeQuantMode] = React.useState<'w4a16_mxfp4' | 'w4a8_mxfp4_mxfp8' | 'w4a16_mxfp4_cutlass' | 'w4a8_mxfp4_mxfp8_trtllm'>('w4a16_mxfp4');
  const [testPpSize, setTestPpSize] = React.useState(1);

  // Disagg: prefill and decode pools with independent parallelism / batch.
  const [servingMode, setServingMode] = React.useState<'agg' | 'disagg'>('agg');
  const [prefillCfg, setPrefillCfg] = React.useState<PerfPhaseInput>({ tp: '1', pp: '1', workers: '1', batch: '1' });
  const [decodeCfg, setDecodeCfg] = React.useState<PerfPhaseInput>({ tp: '1', pp: '1', workers: '1', batch: '64' });

  const [islInput, setIslInput] = React.useState('2048');
  const [oslInput, setOslInput] = React.useState('128');
  const [concurrentUsersInput, setConcurrentUsersInput] = React.useState('1');
  const [prefixInput, setPrefixInput] = React.useState('0');
  const [tpSizeInput, setTpSizeInput] = React.useState('1');
  const [ppSizeInput, setPpSizeInput] = React.useState('1');

  const invalidISL = islInput === '' || parseInt(islInput, 10) < 1;
  const invalidOSL = oslInput === '' || parseInt(oslInput, 10) < 1;
  const invalidUsers = concurrentUsersInput === '' || parseInt(concurrentUsersInput, 10) < 1;
  const invalidTpSize = tpSizeInput === '' || parseInt(tpSizeInput, 10) < 1;
  const invalidPpSize = ppSizeInput === '' || parseInt(ppSizeInput, 10) < 1;

  const handleIslChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setIslInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setTestISL(n);
  };

  const handleOslChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setOslInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setTestOSL(n);
  };

  const handleConcurrentUsersChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setConcurrentUsersInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setTestConcurrentUsers(n);
  };

  const handleTpSizeChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setTpSizeInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setTestTpSize(n);
  };

  const handlePpSizeChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setPpSizeInput(digits);
    const n = parseInt(digits, 10);
    if (!isNaN(n) && n >= 1) setTestPpSize(n);
  };

  const handlePrefixChange = (raw: string) => {
    const digits = raw.replace(/[^0-9]/g, '');
    setPrefixInput(digits);
    const n = parseInt(digits, 10);
    setTestPrefix(isNaN(n) ? 0 : n);
  };

  const [activePreset, setActivePreset] = React.useState<string>('default');

  const applyPreset = (p: Pick<WorkloadPreset, 'key' | 'isl' | 'osl' | 'concurrency' | 'prefix'>) => {
    setTestISL(p.isl); setIslInput(String(p.isl));
    setTestOSL(p.osl); setOslInput(String(p.osl));
    setTestConcurrentUsers(p.concurrency); setConcurrentUsersInput(String(p.concurrency));
    setTestPrefix(p.prefix); setPrefixInput(String(p.prefix));
    setActivePreset(p.key);
  };

  const resetToDefaults = () => applyPreset(DEFAULT_WORKLOAD);

  const [isCalculating, setIsCalculating] = React.useState(false);

  // Fetch HF config when model changes
  React.useEffect(() => {
    setIsUsingFallback(false);
    setFallbackReason('');
    // Clear any prior model's config so it can't leak into the next estimate.
    setHfConfig(null);

    // Skip HF fetch only for catalog models — AIC resolves those itself.
    // Tested models outside the catalog still need their HF config sent along.
    if (!needsHfConfig(model, aicModels)) {
      setIsFetchingConfig(false);
      return;
    }

    let cancelled = false;
    const fetchConfig = async () => {
      setIsFetchingConfig(true);
      console.log('🔄 Fetching config from HuggingFace for:', model);
      console.log('🔑 HF Token:', hfToken ? `Provided (${hfToken.substring(0, 7)}...)` : 'Not provided');

      const result = await fetchModelConfig(model, hfToken);

      // Ignore a response for a model the user has since moved away from — the
      // superseding effect run owns the loading + config state.
      if (cancelled) return;

      if (result.success && result.config) {
        setHfConfig(result.config);
        setTestError(null);
        setIsUsingFallback(false);
        console.log('✅ Fetched HF config:', result.config);
      } else {
        // Fetch failed - will use fallback estimation
        setHfConfig(null);
        setIsUsingFallback(true);
        setFallbackReason(result.error || 'Unknown error');
        console.warn('⚠️ Failed to fetch HF config, will use estimation:', result.error);
      }

      setIsFetchingConfig(false);
    };

    // Debounce to avoid fetching while user is typing
    const timer = setTimeout(fetchConfig, 500);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [model, hfToken, hydrated, aicModels]);

  // Auto-run calculation when inputs change — calls AIC /recommend API
  React.useEffect(() => {
    const aicGpu = aicGpus.find(g => g.systemId === gpu);
    const systemId = aicGpu?.systemId ?? null;
    if (!systemId || !model || catalogLoading) return;

    if (calcTrigger === 0) return; // don't auto-run on mount

    let cancelled = false;
    setIsCalculating(true);
    setElapsed(0);
    const elapsedTimer = setInterval(() => setElapsed(e => e + 1), 1000);

    const timer = setTimeout(async () => {
      try {
        const spec = modelSpecs.get(model)
        // Check both catalog metadata and HF config for MoE detection
        const catalogExperts = spec?.num_experts ?? 0
        const hfExperts = (hfConfig?.num_experts as number) ?? (hfConfig?.num_local_experts as number) ?? 0
        const isMoe = catalogExperts > 1 || hfExperts > 1

        const result = await fetchEstimateAsInferenceResult({
          model_path: model,
          system: systemId,
          isl: testISL,
          osl: testOSL,
          batch_size: testConcurrentUsers,
          tp_size: testTpSize,
          pp_size: testPpSize,
          backend: inferenceBackend,
          prefix: testPrefix > 0 ? testPrefix : undefined,
          vram_gb: currentAicGpu?.vramGb ?? null,
          gpu_memory_utilization: currentAicGpu?.gpuMemoryUtilization,
          backend_version: backendVersion || undefined,
          // Only send the config for models AIC can't resolve itself; guards
          // against a stale config from a previously selected model.
          hf_model_config: needsHfConfig(model, aicModels) ? (hfConfig as Record<string, unknown> | null) : null,
          kvcache_quant_mode: testKVCachePrecision === 'FP8' ? 'fp8' :
                             testKVCachePrecision === 'NVFP4' ? 'nvfp4' : null,
          gemm_quant_mode: testWeightPrecision === 'FP8' ? 'fp8' :
                          testWeightPrecision === 'INT8' ? 'int8_wo' :
                          testWeightPrecision === 'INT4' ? 'int4_wo' :
                          testWeightPrecision === 'MXFP4' ? 'mxfp4' :
                          testWeightPrecision === 'NVFP4' ? 'nvfp4' : null,
          moe_quant_mode: isMoe ? testMoeQuantMode : undefined,
          ...(isMoe && { moe_ep_size: testTpSize }),
          ...(servingMode === 'disagg' && {
            mode: 'disagg' as const,
            prefill: parsePerfPhase(prefillCfg),
            decode: parsePerfPhase(decodeCfg),
          }),
        });
        if (!cancelled) {
          setTestResult(result);
          setTestError(null);
          setTestErrorCode(null);
        }
      } catch (error) {
        if (!cancelled) {
          if (error instanceof EstimateError) {
            setTestErrorCode(error.code)
            setTestError(error.message)
          } else {
            setTestErrorCode('AIC_UNAVAILABLE')
            setTestError(error instanceof Error ? error.message : String(error))
          }
        }
      } finally {
        if (!cancelled) {
          setIsCalculating(false);
          clearInterval(elapsedTimer);
        }
      }
    }, 0);

    return () => { cancelled = true; clearTimeout(timer); clearInterval(elapsedTimer); };
    // Intentionally keyed on calcTrigger alone: the Calculate button increments
    // it to snapshot the current inputs and fire one /recommend request. Listing
    // the individual inputs here would auto-run the API on every keystroke, which
    // is exactly the behaviour the explicit-Calculate flow avoids.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [calcTrigger]);


  // Check if user has seen the tour before
  React.useEffect(() => {
    const hasSeenTour = localStorage.getItem('qe-tour-seen');
    if (hasSeenTour) {
      setTourSeen(true);
    } else {
      // Show tour after a brief delay on first visit
      const timer = setTimeout(() => setShowTour(true), 1000);
      return () => clearTimeout(timer);
    }
  }, []);

  // Reset overrides when major inputs change (model or GPU selection)
  React.useEffect(() => {
    if (parallelismOverride || vllmOverride) {
      setParallelismOverride(false);
      setParallelismManualTP(null);
      setParallelismManualReplicas(null);
      setVllmOverride(false);
      setVllmManualMaxNumSeqs(null);
      setVllmManualMaxModelLen(null);
      setVllmManualChunkedPrefill(null);
      setVllmManualPrefixCaching(null);
      setVllmManualGpuUtil(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, gpu]);

  const toggleAcc = (id: string) => {
    setExpanded((e) => (e.includes(id) ? e.filter((x) => x !== id) : [...e, id]));

    // Scroll to the accordion section after a brief delay
    setTimeout(() => {
      const element = document.getElementById(`acc-${id}`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 100);
  };

  const handleCustomizeClick = () => {
    // Expand assumptions section if collapsed
    if (!assumptionsExpanded) {
      setAssumptionsExpanded(true);
    }

    // Scroll to assumptions section
    setTimeout(() => {
      if (assumptionsRef.current) {
        assumptionsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 100);

    // Add highlight pulse animation
    setAssumptionsHighlight(true);
    setTimeout(() => setAssumptionsHighlight(false), 2000);
  };

  // Debounced search query

  // Nullable memory fields — null when VRAM is unknown (catalog not yet loaded)
  const memUsablePerGpu = testResult?.memory_analysis.usable_hbm_per_gpu ?? 0
  const memTotalVram = testResult?.memory_analysis.total_vram_gb ?? 0
  const memKvBudget = testResult?.memory_analysis.kv_cache_budget_gb ?? 0

  // Disagg detail (present only when the estimate ran in disagg mode).
  const disagg = testResult?.mode === 'disagg' ? testResult.disagg ?? null : null;
  const isDisagg = disagg != null;

  // animated headline numbers
  // Calculate real values from inference engine. Disagg = sum of each pool's
  // workers × gpus/worker; agg = tp × replicas × pp.
  const realGpuCount = testResult ?
    (isDisagg
      ? disagg.prefill.workers * disagg.prefill.gpusPerWorker + disagg.decode.workers * disagg.decode.gpusPerWorker
      : testResult.memory_analysis.tp_size * testResult.memory_analysis.replicas * (testResult.parallelism_strategy.pp_size || 1)) :
    0;

  const realWeightGB = testResult ?
    testResult.memory_analysis.weight_gb :
    0;

  // Get actual weight precision being used (detected from HF config or user-selected)
  const actualWeightPrecision = React.useMemo(() => {
    if (!hfConfig || !hfConfig.quantization_config) {
      return testWeightPrecision; // No HF config, use user selection
    }

    const qconfig = hfConfig.quantization_config as any;
    const quantMethod = qconfig.quant_method || qconfig.quant_type || qconfig.quantization_algo;

    if (!quantMethod || quantMethod === 'none') {
      return testWeightPrecision; // No quantization, use user selection
    }

    // Map quantization method to display format
    const method = quantMethod.toLowerCase();
    if (method.includes('fp8') || method === 'fp8') return 'FP8';
    if (method.includes('int8') || method === 'int8') return 'INT8';
    if (method.includes('int4') || method === 'int4') return 'INT4';
    if (method.includes('mxfp4') || method === 'mxfp4') return 'MXFP4';
    if (method === 'gptq' || method === 'awq') {
      // Check bits field for GPTQ/AWQ
      const bits = qconfig.bits || qconfig.num_bits || 4;
      if (bits === 8) return 'INT8';
      if (bits === 4) return 'INT4';
      return 'INT4'; // Default to INT4 for GPTQ/AWQ
    }
    if (method === 'bnb' || method.includes('bitsandbytes')) return 'INT4';

    return testWeightPrecision; // Unknown, fall back to user selection
  }, [hfConfig, testWeightPrecision]);

  const realKVPerReqMB = testResult && testResult.memory_analysis.kv_cache_used_gb ?
    (testResult.memory_analysis.kv_cache_used_gb / testConcurrentUsers) * 1000 : // Convert GB to MB
    0;

  // Use live pricing if available, fallback to estimated pricing from hardware cost
  const currentAicGpu = aicGpus.find(g => g.systemId === gpu);
  const gpuLabel = currentAicGpu?.label ?? '';
  const hwCostEntry = costings.gpuHardwareCosts.get(gpu)
  const catalogGpuForPricing = hwCostEntry?.new_usd != null
    ? { hardware_cost_usd: hwCostEntry.new_usd, name: currentAicGpu?.label ?? gpu }
    : null

  // Map GPU to pricing key for live pricing worker (pending Costings REST API)
  const gpuPricingKey = gpuLabel.includes('H200') ? 'H200' :
                        gpuLabel.includes('H100') ? 'H100' :
                        gpuLabel.includes('A100') ? 'A100' :
                        gpuLabel.includes('L40S') ? 'L40S' :
                        gpuLabel.includes('MI300X') ? 'MI300X' : gpuLabel;

  const resolvedCloudRate = resolveCloudRate(costings.gpuCloudRates.get(gpu), preferredCloudProvider)
  const gpuPricePerHour: number | null = resolvedCloudRate?.rate ?? null;
  const cloudRateLabel = resolvedCloudRate
    ? `${resolvedCloudRate.provider.replace('.', ' · ')} ${resolvedCloudRate.kind === 'spot' ? 'spot' : 'on-demand'}`
    : '';

  const realMonthlyCost = testResult && gpuPricePerHour != null ?
    realGpuCount * gpuPricePerHour * HOURS_PER_MONTH :
    null;
  const realMonthlyCostForDisplay = realMonthlyCost ?? 0;

  const gpus = useCountUp(realGpuCount);
  const weight = useCountUp(realWeightGB);
  const kv = useCountUp(realKVPerReqMB);
  const cost = useCountUp(realMonthlyCostForDisplay);

  const handleTourComplete = () => {
    setShowTour(false);
    setTourSeen(true);
    localStorage.setItem('qe-tour-seen', 'true');
  };

  const handleTakeTour = () => {
    setShowTour(true);
  };

  // Load saved count on mount
  React.useEffect(() => {
    setSavedCount(getSavedEstimateCount());
  }, []);

  // Generate auto name for save
  const generateAutoName = () => {
    const modelName = model.split('/').pop() || model;
    const gpuName = gpu.replace('NVIDIA ', '').replace('AMD ', '');
    return `${modelName} · ${gpuName} · ${testConcurrentUsers} users`;
  };

  const handleSaveEstimate = (data: { name: string; tags: string; notes: string }) => {
    if (!testResult || !catalogGpuForPricing || gpuPricePerHour == null) return;

    const kvPerUserGB = (testResult.memory_analysis.kv_cache_used_gb || 0) / testConcurrentUsers;
    const kvMBPerToken = (kvPerUserGB * 1000) / (testISL + testOSL);

    saveEstimate({
      name: data.name,
      tags: data.tags,
      notes: data.notes,
      model,
      gpu,
      inputs: {
        isl: testISL,
        osl: testOSL,
        concurrentUsers: testConcurrentUsers,
        weightPrecision: testWeightPrecision,
        kvCachePrecision: testKVCachePrecision,
      },
      results: {
        gpusRequired: realGpuCount,
        tpSize: testResult.memory_analysis.tp_size,
        ppSize: testResult.parallelism_strategy.pp_size,
        replicas: testResult.memory_analysis.replicas,
        weightMemoryGB: testResult.memory_analysis.weight_gb,
        kvCachePerUserGB: kvPerUserGB,
        kvCacheTotalGB: testResult.memory_analysis.kv_cache_used_gb || 0,
        kvCacheMBPerToken: kvMBPerToken,
        kvCategory: testResult.memory_analysis.kv_category || 'KV-1',
        kvCategoryLabel: testResult.memory_analysis.kv_category_label || 'Standard Dense',
        cloudCostMonthly: realMonthlyCost ?? 0,
        cloudCost5Year: (realMonthlyCost ?? 0) * 60,
        selfHostedCostMonthly: (catalogGpuForPricing.hardware_cost_usd * realGpuCount) / AMORT_MONTHS_5YR,
        selfHostedCost5Year: catalogGpuForPricing.hardware_cost_usd * realGpuCount,
      },
    });

    setSavedCount(getSavedEstimateCount());
    setToastMessage('saved');
    setShowToast(true);
    setTimeout(() => setShowToast(false), 5000);
  };

  // Build the /api/estimate request body from current form state and GPU-specific values
  const buildEstimateRequestBody = React.useCallback(() => {
    const spec = modelSpecs.get(model);
    const catalogExperts = spec?.num_experts ?? 0;
    const hfExperts = (hfConfig?.num_experts as number) ??
      (hfConfig?.num_local_experts as number) ?? 0;
    const isMoe = catalogExperts > 1 || hfExperts > 1;

    return {
      model_path: model || '(select model)',
      system: gpu || '(select GPU)',
      backend: inferenceBackend,
      isl: testISL,
      osl: testOSL,
      batch_size: testConcurrentUsers,
      tp_size: testResult?.memory_analysis.tp_size ?? testTpSize,
      pp_size: testResult?.parallelism_strategy.pp_size ?? testPpSize,
      vram_gb: currentAicGpu?.vramGb ?? null,
      gpu_memory_utilization: currentAicGpu?.gpuMemoryUtilization,
      ...(testPrefix > 0 && { prefix: testPrefix }),
      ...(backendVersion && { backend_version: backendVersion }),
      ...(testWeightPrecision === 'FP8' && { gemm_quant_mode: 'fp8' }),
      ...(testWeightPrecision === 'INT8' && { gemm_quant_mode: 'int8_wo' }),
      ...(testWeightPrecision === 'INT4' && { gemm_quant_mode: 'int4_wo' }),
      ...(testWeightPrecision === 'MXFP4' && { gemm_quant_mode: 'mxfp4' }),
      ...(testWeightPrecision === 'NVFP4' && { gemm_quant_mode: 'nvfp4' }),
      ...(testKVCachePrecision === 'FP8' && { kvcache_quant_mode: 'fp8' }),
      ...(testKVCachePrecision === 'NVFP4' && { kvcache_quant_mode: 'nvfp4' }),
      hf_model_config: needsHfConfig(model, aicModels)
        ? (hfConfig as Record<string, unknown> | null)
        : null,
      moe_quant_mode: isMoe ? testMoeQuantMode : undefined,
      ...(isMoe && { moe_ep_size: testTpSize }),
      ...(servingMode === 'disagg' && {
        mode: 'disagg',
        prefill_tp_size: parsePerfPhase(prefillCfg).tp,
        prefill_pp_size: parsePerfPhase(prefillCfg).pp,
        prefill_num_workers: parsePerfPhase(prefillCfg).workers,
        prefill_batch_size: parsePerfPhase(prefillCfg).batch,
        decode_tp_size: parsePerfPhase(decodeCfg).tp,
        decode_pp_size: parsePerfPhase(decodeCfg).pp,
        decode_num_workers: parsePerfPhase(decodeCfg).workers,
        decode_batch_size: parsePerfPhase(decodeCfg).batch,
      })
    };
  }, [model, gpu, inferenceBackend, testISL, testOSL, testConcurrentUsers, testResult, testTpSize, testPpSize, currentAicGpu, testPrefix, backendVersion, testWeightPrecision, testKVCachePrecision, servingMode, prefillCfg, decodeCfg, modelSpecs, hfConfig, aicModels, testMoeQuantMode]);

  // Copy API request body to clipboard
  const handleCopyAPIRequest = async () => {
    if (!testResult) return;

    try {
      await navigator.clipboard.writeText(JSON.stringify(buildEstimateRequestBody(), null, 2));
      setToastMessage('api-copied');
      setShowToast(true);
      setTimeout(() => setShowToast(false), 3000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  // Copy CLI command to clipboard
  const handleCopyCLICommand = async () => {
    if (!testResult) return;

    // Only FP16/FP8 are valid --dtype values; quantized modes use --quantization
    const dtypeValue = testWeightPrecision === 'FP16' ? 'float16' :
                       testWeightPrecision === 'FP8' ? 'fp8' : 'auto';

    // Map quantization modes: NVFP4 requires modelopt_fp4, others use backend value
    let quantValue = testResult.vllm_config.quantization;
    if (testWeightPrecision === 'NVFP4' && (!quantValue || quantValue === 'auto')) {
      quantValue = 'modelopt_fp4';
    }

    const quantFlag = (testWeightPrecision === 'INT4' || testWeightPrecision === 'INT8' || testWeightPrecision === 'MXFP4' || testWeightPrecision === 'NVFP4')
      ? ` \\\n  --quantization ${quantValue}`
      : '';

    let cliCommand: string;
    if (isDisagg) {
      // A single vllm serve can't express disagg; emit one command per pool and
      // note the KV-transfer connector needed to wire prefill -> decode.
      const poolCmd = (role: string, ph: EstimatePhase) => {
        const ppFlag = ph.pp_size > 1 ? ` \\\n  --pipeline-parallel-size ${ph.pp_size}` : '';
        return `# ${role} pool — ${ph.workers} worker(s), TP${ph.tp_size}${ph.pp_size > 1 ? ` PP${ph.pp_size}` : ''}, batch ${ph.batch_size}\n` +
          `vllm serve ${model} \\\n` +
          `  --tensor-parallel-size ${ph.tp_size}${ppFlag} \\\n` +
          `  --gpu-memory-utilization 0.90 \\\n` +
          `  --dtype ${dtypeValue}${quantFlag} \\\n` +
          `  --kv-cache-dtype ${testKVCachePrecision.toLowerCase()}`;
      };
      cliCommand =
        `# Disaggregated serving: run the prefill and decode pools separately and\n` +
        `# connect them with a KV-transfer connector (e.g. LMCache / NIXL / Dynamo).\n` +
        `# Scale each pool to the worker count shown.\n\n` +
        `${poolCmd('Prefill', disagg.prefill)}\n\n` +
        `${poolCmd('Decode', disagg.decode)}`;
    } else {
      const ppFlag = testResult.parallelism_strategy.pp_size > 1
        ? ` \\\n  --pipeline-parallel-size ${testResult.parallelism_strategy.pp_size}`
        : '';
      cliCommand = `vllm serve ${model} \\
  --tensor-parallel-size ${testResult.memory_analysis.tp_size}${ppFlag} \\
  --max-model-len auto \\
  --gpu-memory-utilization 0.90 \\
  --dtype ${dtypeValue}${quantFlag} \\
  --kv-cache-dtype ${testKVCachePrecision.toLowerCase()} \\
  --max-num-seqs ${testResult.vllm_config?.max_num_seqs || 256}${testResult.vllm_config?.enable_chunked_prefill ? ' \\\n  --enable-chunked-prefill' : ''}`;
    }

    try {
      await navigator.clipboard.writeText(cliCommand);
      setToastMessage('cli-copied');
      setShowToast(true);
      setTimeout(() => setShowToast(false), 3000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  // Export to Google Sheets (downloads as CSV)
  const handleExportToSheets = () => {
    if (!testResult || !catalogGpuForPricing) return;

    // Prepare data in CSV format
    const headers = [
      'Model', 'GPU', 'GPUs Required', 'TP Size', 'PP Size', 'Replicas',
      'ISL', 'OSL', 'Concurrent Users',
      'Weight Precision', 'KV Cache Precision',
      'Weight Memory (GB)', 'KV Cache Total (GB)', 'KV Category',
      'Cloud Cost (Monthly)', 'Cloud Cost (5yr)',
      'Self-Hosted Cost (Monthly)', 'Self-Hosted Cost (5yr)'
    ];

    const values = [
      model, gpu, realGpuCount, testResult.memory_analysis.tp_size, testResult.parallelism_strategy.pp_size, testResult.memory_analysis.replicas,
      testISL, testOSL, testConcurrentUsers,
      testWeightPrecision, testKVCachePrecision,
      testResult.memory_analysis.weight_gb.toFixed(1),
      (testResult.memory_analysis.kv_cache_used_gb || 0).toFixed(1),
      testResult.memory_analysis.kv_category_label || 'Standard Dense',
      `$${(realMonthlyCost ?? 0).toLocaleString()}`, `$${((realMonthlyCost ?? 0) * AMORT_MONTHS_5YR).toLocaleString()}`,
      `$${((catalogGpuForPricing.hardware_cost_usd * realGpuCount) / AMORT_MONTHS_5YR).toFixed(0)}`,
      `$${(catalogGpuForPricing.hardware_cost_usd * realGpuCount).toLocaleString()}`
    ];

    const csvContent = headers.join(',') + '\n' + values.map(v => `"${v}"`).join(',');

    // Download as CSV (can be imported into Google Sheets)
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `configiq-estimate-${Date.now()}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    setToastMessage('exported');
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  // Validation warnings for manual overrides
  const getValidationWarnings = (): string[] => {
    const warnings: string[] = [];

    if (parallelismOverride) {
      if (parallelismManualTP !== null) {
        // TP must be power of 2
        if (parallelismManualTP <= 0) {
          warnings.push('Parallelism: Tensor parallel size must be > 0');
        } else if ((parallelismManualTP & (parallelismManualTP - 1)) !== 0) {
          warnings.push('Parallelism: Tensor parallel size should be a power of 2 (1, 2, 4, 8, 16)');
        }
      }
      if (parallelismManualReplicas !== null && parallelismManualReplicas <= 0) {
        warnings.push('Parallelism: Replica count must be > 0');
      }
    }

    if (vllmOverride) {
      if (vllmManualMaxNumSeqs !== null && vllmManualMaxNumSeqs <= 0) {
        warnings.push('vLLM: max_num_seqs must be > 0');
      }
      if (vllmManualMaxModelLen !== null && vllmManualMaxModelLen <= 0) {
        warnings.push('vLLM: max_model_len must be > 0');
      }
      if (vllmManualGpuUtil !== null && (vllmManualGpuUtil < 50 || vllmManualGpuUtil > 95)) {
        warnings.push('vLLM: gpu_memory_utilization should be between 50-95%');
      }
    }

    return warnings;
  };

  const validationWarnings = getValidationWarnings();

  // Build accordion sections dynamically from current state
  const buildAccordionSections = () => {
    const spec = modelSpecs.get(model);
    // Check both catalog metadata and HF config for MoE detection
    const catalogExperts = spec?.num_experts ?? 0;
    const hfExperts = (hfConfig?.num_experts as number) ?? (hfConfig?.num_local_experts as number) ?? 0;
    const isMoeModel = catalogExperts > 1 || hfExperts > 1;

    return [
    {
      id: 'workload', title: 'Workload',
      summary: [
        { k: 'ISL', v: `${testISL}` },
        { k: 'OSL', v: `${testOSL}` },
        { k: 'users', v: `${testConcurrentUsers}` },
        { k: 'prefix', v: `${testPrefix}` },
      ],
      fields: [
        {
          label: 'Input sequence length (ISL)',
          value: islInput,
          term: 'isl',
          type: 'number' as const,
          invalid: invalidISL,
          onChange: (val: string) => handleIslChange(val)
        },
        {
          label: 'Output sequence length (OSL)',
          value: oslInput,
          term: 'osl',
          type: 'number' as const,
          invalid: invalidOSL,
          onChange: (val: string) => handleOslChange(val)
        },
        {
          label: 'Concurrent users',
          value: concurrentUsersInput,
          term: 'concurrent',
          type: 'number' as const,
          invalid: invalidUsers,
          onChange: (val: string) => handleConcurrentUsersChange(val)
        },
        {
          label: 'Shared prefix tokens',
          value: prefixInput,
          term: 'prefix',
          type: 'number' as const,
          invalid: false,
          onChange: (val: string) => handlePrefixChange(val)
        },
      ],
    },
    {
      id: 'memory', title: 'Precision & memory',
      summary: [
        { k: 'weights', v: actualWeightPrecision },
        { k: 'KV', v: testKVCachePrecision },
        ...(isMoeModel ? [{
          k: 'MoE',
          v: testMoeQuantMode === 'w4a16_mxfp4' ? 'W4A16' :
             testMoeQuantMode === 'w4a8_mxfp4_mxfp8' ? 'W4A8' :
             testMoeQuantMode === 'w4a16_mxfp4_cutlass' ? 'CUTLASS' : 'TRT-LLM'
        }] : [])
      ],
      fields: [
        {
          label: actualWeightPrecision !== testWeightPrecision ?
            'Weight precision (overridden by model quantization_config)' : 'Weight precision',
          value: testWeightPrecision,
          type: 'select' as const,
          term: 'weightPrecision',
          options: ['FP16', 'FP8', 'INT8', 'INT4', 'MXFP4', 'NVFP4'] as const,
          onChange: (val: string) => {
            if (val === 'FP16' || val === 'FP8' || val === 'INT8' || val === 'INT4' || val === 'MXFP4' || val === 'NVFP4') {
              setTestWeightPrecision(val);
            }
          }
        },
        {
          label: 'KV cache precision',
          value: testKVCachePrecision,
          type: 'select' as const,
          term: 'kvCachePrecision',
          options: ['FP16', 'FP8', 'NVFP4'] as const,
          onChange: (val: string) => {
            if (val === 'FP16' || val === 'FP8' || val === 'NVFP4') {
              setTestKVCachePrecision(val);
            }
          }
        },
        ...(isMoeModel ? [{
          label: 'MoE quantization',
          value: testMoeQuantMode,
          type: 'select' as const,
          term: 'moeQuantization',
          options: [
            { value: 'w4a16_mxfp4', label: 'W4A16 MXFP4' },
            { value: 'w4a8_mxfp4_mxfp8', label: 'W4A8 MXFP4+FP8' },
            { value: 'w4a16_mxfp4_cutlass', label: 'W4A16 MXFP4 (CUTLASS)' },
            { value: 'w4a8_mxfp4_mxfp8_trtllm', label: 'W4A8 MXFP4+FP8 (TRT-LLM)' }
          ],
          onChange: (val: string) => {
            if (val === 'w4a16_mxfp4' || val === 'w4a8_mxfp4_mxfp8' ||
                val === 'w4a16_mxfp4_cutlass' || val === 'w4a8_mxfp4_mxfp8_trtllm') {
              setTestMoeQuantMode(val);
            }
          },
          help: 'W4A16: best quality, H100+. W4A8: ~2× faster on B200. CUTLASS: H100-optimized. TRT-LLM: B200 TRT-LLM variant.'
        }] : []),
      ],
    },
    {
      id: 'parallel',
      title: 'Parallelism',
      badge: parallelismOverride ? 'Manual override' : 'Auto-computed',
      badgeColor: parallelismOverride ? 'orange' : 'blue',
      hasOverride: true,
      isOverridden: parallelismOverride,
      onOverrideToggle: () => {
        if (parallelismOverride) {
          // Reset to auto
          setParallelismOverride(false);
          setParallelismManualTP(null);
          setParallelismManualReplicas(null);
        } else {
          // Enable manual override - initialize with current computed values
          setParallelismOverride(true);
          if (testResult) {
            setParallelismManualTP(testResult.memory_analysis.tp_size);
            setParallelismManualReplicas(testResult.memory_analysis.replicas);
          }
        }
      },
      summary: [
        { k: 'TP', v: `${testTpSize}` },
        { k: 'PP', v: `${testPpSize}` },
      ],
      fields: [
        {
          label: 'Tensor parallel size (TP)',
          value: tpSizeInput,
          term: 'tensorParallel',
          readonly: false,
          type: 'number' as const,
          invalid: invalidTpSize,
          onChange: (val: string) => handleTpSizeChange(val),
        },
        {
          label: 'Pipeline parallel size (PP)',
          value: ppSizeInput,
          term: 'pipelineParallel',
          readonly: false,
          type: 'number' as const,
          invalid: invalidPpSize,
          onChange: (val: string) => handlePpSizeChange(val),
        },
        {
          label: 'Total GPUs',
          value: testResult ? `${testResult.memory_analysis.tp_size * testResult.memory_analysis.replicas * testResult.parallelism_strategy.pp_size}` : `${parseInt(tpSizeInput || '1') * parseInt(ppSizeInput || '1')}`,
          readonly: true,
        },
      ],
    },
    {
      id: 'engine',
      title: 'vLLM config',
      badge: vllmOverride ? 'Manual override' : 'Auto-computed',
      badgeColor: vllmOverride ? 'orange' : 'blue',
      hasOverride: true,
      isOverridden: vllmOverride,
      onOverrideToggle: () => {
        if (vllmOverride) {
          // Reset to auto
          setVllmOverride(false);
          setVllmManualMaxNumSeqs(null);
          setVllmManualMaxModelLen(null);
          setVllmManualChunkedPrefill(null);
          setVllmManualPrefixCaching(null);
          setVllmManualGpuUtil(null);
        } else {
          // Enable manual override - initialize with current computed values
          setVllmOverride(true);
          if (testResult) {
            setVllmManualMaxNumSeqs(testResult.vllm_config.max_num_seqs);
            setVllmManualMaxModelLen(testResult.vllm_config.max_model_len);
            setVllmManualChunkedPrefill(testResult.vllm_config.enable_chunked_prefill);
            setVllmManualPrefixCaching(testResult.vllm_config.enable_prefix_caching);
            setVllmManualGpuUtil(Math.round(testResult.vllm_config.gpu_memory_utilization * 100));
          }
        }
      },
      summary: [
        { k: 'max_num_seqs', v: vllmOverride && vllmManualMaxNumSeqs !== null ? `${vllmManualMaxNumSeqs}` : testResult ? `${testResult.vllm_config.max_num_seqs}` : '—' },
        { k: 'chunked', v: vllmOverride && vllmManualChunkedPrefill !== null ? (vllmManualChunkedPrefill ? 'on' : 'off') : testResult ? (testResult.vllm_config.enable_chunked_prefill ? 'on' : 'off') : '—' }
      ],
      fields: [
        {
          label: 'max_num_seqs',
          value: vllmOverride && vllmManualMaxNumSeqs !== null ? `${vllmManualMaxNumSeqs}` : testResult ? `${testResult.vllm_config.max_num_seqs}` : '—',
          term: 'maxNumSeqs',
          readonly: !vllmOverride,
          type: vllmOverride ? 'number' as const : undefined,
          onChange: vllmOverride ? (val: string) => setVllmManualMaxNumSeqs(parseInt(val) || 1) : undefined
        },
        {
          label: 'max_model_len',
          value: vllmOverride && vllmManualMaxModelLen !== null ? `${vllmManualMaxModelLen}` : testResult ? `${testResult.vllm_config.max_model_len}` : '—',
          term: 'maxModelLen',
          readonly: !vllmOverride,
          type: vllmOverride ? 'number' as const : undefined,
          onChange: vllmOverride ? (val: string) => setVllmManualMaxModelLen(parseInt(val) || 1) : undefined
        },
        {
          label: 'enable_chunked_prefill',
          value: vllmOverride && vllmManualChunkedPrefill !== null ? (vllmManualChunkedPrefill ? 'Yes' : 'No') : testResult ? (testResult.vllm_config.enable_chunked_prefill ? 'Yes' : 'No') : '—',
          term: 'chunkedPrefill',
          readonly: !vllmOverride,
          type: vllmOverride ? 'select' as const : undefined,
          options: vllmOverride ? ['Yes', 'No'] : undefined,
          onChange: vllmOverride ? (val: string) => setVllmManualChunkedPrefill(val === 'Yes') : undefined
        },
        {
          label: 'enable_prefix_caching',
          value: vllmOverride && vllmManualPrefixCaching !== null ? (vllmManualPrefixCaching ? 'Yes' : 'No') : testResult ? (testResult.vllm_config.enable_prefix_caching ? 'Yes' : 'No') : '—',
          term: 'prefixCaching',
          readonly: !vllmOverride,
          type: vllmOverride ? 'select' as const : undefined,
          options: vllmOverride ? ['Yes', 'No'] : undefined,
          onChange: vllmOverride ? (val: string) => setVllmManualPrefixCaching(val === 'Yes') : undefined
        },
        {
          label: 'gpu_memory_utilization',
          value: vllmOverride && vllmManualGpuUtil !== null ? `${vllmManualGpuUtil}%` : testResult ? `${(testResult.vllm_config.gpu_memory_utilization * 100).toFixed(0)}%` : '—',
          term: 'gpuUtil',
          readonly: !vllmOverride,
          type: vllmOverride ? 'range' as const : undefined,
          min: vllmOverride ? 50 : undefined,
          max: vllmOverride ? 95 : undefined,
          step: vllmOverride ? 5 : undefined,
          rangeValue: vllmOverride && vllmManualGpuUtil !== null ? vllmManualGpuUtil : undefined,
          onChange: vllmOverride ? (val: number) => setVllmManualGpuUtil(val) : undefined
        },
      ],
    },
  ];
  };

  return (
    <div className={styles.page}>
      {showTour && (
        <ProductTour
          steps={QUICK_ESTIMATE_TOUR}
          tourId="qe"
          onComplete={handleTourComplete}
        />
      )}
      {/* ---------- header ---------- */}
      <div className={styles.header}>
        <div className={styles.headRow}>
          <div>
            <h1 className={styles.pageTitle}>Performance estimate</h1>
            <p className={styles.subtitle}>See time to first token, throughput, and memory estimates in seconds — refine as needed.</p>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <div style={{ position: 'relative' }}>
              <Button
                variant="link"
                onClick={handleTakeTour}
                style={{ fontSize: '14px' }}
              >
                Take a tour
              </Button>
              {!tourSeen && <div className={styles.tourBeacon} />}
            </div>
            <Button
              variant="plain"
              aria-label={fav ? 'Remove from favorites' : 'Add to favorites'}
              onClick={() => setFav((f) => !f)}
              icon={fav ? <StarIcon /> : <OutlinedStarIcon />}
            />
          </div>
        </div>
      </div>


      {/* ---------- input row ---------- */}
      <div className={`${styles.card} ${styles.inputCard}`} data-tour="model">
        <div className={styles.inputRow}>
          {/* Column 1: Model field */}
          <div>
            <ComboBox
              id="qe-model"
              value={model}
              onChange={setModel}
              items={modelItems}
              placeholder="Type model name or select from dropdown..."
              allowCustom
              supportedModels={getAppConfig().testedModels}
              hfToken={hfToken}
            />
          </div>

          {/* Column 2: GPU target */}
          <GpuSystemInput id="qe-gpu" value={gpu} onChange={setGpu} gpuOptions={aicGpus} />

        </div>
        
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 16 }}>
          <Button
            variant="primary"
            size="lg"
            onClick={() => { setTestResult(null); setCalcTrigger(t => t + 1); }}
            isDisabled={isCalculating || !gpu || !model || catalogLoading || invalidISL || invalidOSL || invalidUsers || invalidTpSize || invalidPpSize}
          >
            {isCalculating ? 'Calculating...' : 'Calculate'}
          </Button>
        </div>

      </div>

      {/* ---------- serving mode (agg / disagg) ---------- */}
      <div className={`${styles.card}`} style={{ padding: '14px 18px', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <span style={{ fontSize: '12px', fontWeight: 600, fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.06em', color: '#3c3f42', whiteSpace: 'nowrap' }}>Serving mode:</span>
          <Button variant={servingMode === 'agg' ? 'secondary' : 'tertiary'} size="sm" onClick={() => { setServingMode('agg'); setTestResult(null); }}>Aggregated</Button>
          <Button variant={servingMode === 'disagg' ? 'secondary' : 'tertiary'} size="sm" onClick={() => { setServingMode('disagg'); setTestResult(null); }}>Disaggregated</Button>
          {servingMode === 'disagg' && (
            <span style={{ fontSize: '12px', color: '#54585c' }}>Prefill and decode run on separate GPU pools.</span>
          )}
        </div>
        {servingMode === 'disagg' && (
          <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
            <PerfPhaseFields title="Prefill pool" cfg={prefillCfg} onChange={setPrefillCfg} />
            <PerfPhaseFields title="Decode pool" cfg={decodeCfg} onChange={setDecodeCfg} />
          </div>
        )}
      </div>

      {/* ---------- workload presets ---------- */}
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

      <InfoStrip data-tour="warning">
        Based on your configuration — ISL {testISL}, OSL {testOSL}, {testKVCachePrecision} KV cache,
        {' '}{testConcurrentUsers} concurrent users.
        {' '}<InfoStripAction onClick={handleCustomizeClick}>Adjust? (see &lsquo;Want to change assumptions?&rsquo; below)</InfoStripAction>
      </InfoStrip>


      {/* ---------- fallback warning ---------- */}
      {isUsingFallback && !testError && (
        <div style={{
          padding: '16px 20px',
          marginBottom: '20px',
          background: '#e7f4ff',
          border: '2px solid #0066cc',
          borderRadius: '8px',
          display: 'flex',
          gap: '12px',
          alignItems: 'flex-start'
        }}>
          <span style={{ fontSize: '24px', flexShrink: 0 }}>ℹ️</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: '600', color: '#004085', marginBottom: '8px', fontSize: '16px' }}>
              Using estimated architecture
            </div>
            <div style={{ fontSize: '14px', color: '#004085', lineHeight: '1.6', marginBottom: '8px' }}>
              Could not fetch model configuration from HuggingFace. Using estimated values based on model size.
              Results may be less accurate.
            </div>
            <div style={{ fontSize: '13px', color: '#004085' }}>
              <strong>Reason:</strong> {fallbackReason}
            </div>
            {!hfToken && fallbackReason.includes('gated') && (
              <div style={{ fontSize: '13px', color: '#004085', marginTop: '8px' }}>
                💡 <strong>Tip:</strong> Add a HuggingFace token above for accurate results with gated models.
              </div>
            )}
          </div>
        </div>
      )}

      {/* ---------- validation warnings ---------- */}
      {validationWarnings.length > 0 && (
        <div style={{
          padding: '16px 20px',
          marginBottom: '20px',
          background: '#fff8e1',
          border: '2px solid #f0ab00',
          borderRadius: '8px',
          display: 'flex',
          gap: '12px',
          alignItems: 'flex-start'
        }}>
          <ExclamationTriangleIcon style={{ fontSize: '24px', color: '#f0ab00', flexShrink: 0 }} />
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: '600', color: '#795600', marginBottom: '8px', fontSize: '16px' }}>
              Manual override validation warnings
            </div>
            <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '14px', color: '#795600', lineHeight: '1.6' }}>
              {validationWarnings.map((warning, i) => (
                <li key={i}>{warning}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* ---------- error display ---------- */}
      {testError && (
        <div style={{
          padding: '16px 20px',
          marginBottom: '20px',
          background: '#fff3cd',
          border: '2px solid #ffc107',
          borderRadius: '8px',
          display: 'flex',
          gap: '12px',
          alignItems: 'flex-start'
        }}>
          <span style={{ fontSize: '24px', flexShrink: 0 }}>⚠️</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: '600', color: '#856404', marginBottom: '8px', fontSize: '16px' }}>
              {testErrorCode === 'OOM' ? 'Not enough GPU memory'
                : testErrorCode === 'AUTH_REQUIRED' ? 'Authentication required'
                : testErrorCode === 'MODEL_NOT_FOUND' ? 'Model not found'
                : testErrorCode === 'MOE_PARAMS_REQUIRED' ? 'MoE model — expert parallelism required'
                : testErrorCode === 'AIC_TIMEOUT' ? 'Request timed out'
                : testErrorCode === 'AIC_UNAVAILABLE' ? 'Sizing service unavailable'
                : 'Estimate failed'}
            </div>

            {testErrorCode === 'OOM' ? (
              <div style={{ fontSize: '14px', color: '#664d03', lineHeight: '1.6' }}>
                <p style={{ margin: '0 0 10px' }}>This model requires more GPU memory than available with the current tensor parallel size. Try one of:</p>
                <ul style={{ margin: '0', paddingLeft: '20px', lineHeight: '1.8' }}>
                  <li><strong>Increase tensor parallel size</strong> in the options above — double it and try again</li>
                  <li><strong>Use a quantized variant</strong> — look for FP8 or INT4 versions of this model on HuggingFace</li>
                  <li><strong>Select a larger GPU system</strong> — switch to a system with more VRAM per GPU</li>
                </ul>
              </div>
            ) : testErrorCode === 'AUTH_REQUIRED' ? (
              <div style={{ fontSize: '14px', color: '#664d03', lineHeight: '1.6' }}>
                <p style={{ margin: '0 0 10px' }}>This model is gated and requires a HuggingFace token:</p>
                <ul style={{ margin: '0', paddingLeft: '20px', lineHeight: '1.8' }}>
                  <li>Add your token in the <strong>HuggingFace token</strong> field above</li>
                  <li>Get a token at <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener" style={{ color: '#0066cc' }}>huggingface.co/settings/tokens</a></li>
                  <li>Accept the model&apos;s license on HuggingFace first</li>
                </ul>
              </div>
            ) : testErrorCode === 'MOE_PARAMS_REQUIRED' ? (
              <div style={{ fontSize: '14px', color: '#664d03', lineHeight: '1.6' }}>
                This is a Mixture-of-Experts (MoE) model. Add your HuggingFace token above so the model config can be loaded — expert parallelism will then be configured automatically.
              </div>
            ) : testErrorCode === 'AIC_TIMEOUT' ? (
              <div style={{ fontSize: '14px', color: '#664d03', lineHeight: '1.6' }}>
                The sizing engine took too long to respond. Try again, or use a smaller model or simpler configuration.
              </div>
            ) : testErrorCode === 'MODEL_NOT_FOUND' ? (
              <div style={{ fontSize: '14px', color: '#664d03', lineHeight: '1.6' }}>
                <p style={{ margin: '0 0 10px' }}>This model wasn&apos;t found in the catalog. Check that:</p>
                <ul style={{ margin: '0', paddingLeft: '20px', lineHeight: '1.8' }}>
                  <li>The HuggingFace ID is correct and case-sensitive (e.g. <code>meta-llama/Llama-3.1-8B</code>)</li>
                  <li>It&apos;s not a GGUF repo — use the base model instead</li>
                  <li>Popular supported models: {modelSuggestions()}</li>
                </ul>
              </div>
            ) : (
              <div style={{ fontSize: '14px', color: '#664d03', lineHeight: '1.6' }}>
                {testError}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ---------- result tiles ---------- */}
      {!testResult && !isCalculating && !testError}
      {isCalculating && (
        <div className={styles.card}>
          <GpuChipLoader elapsed={elapsed} timeoutSeconds={aicTimeout} />
        </div>
      )}
      {(testResult || isCalculating) && (
      <div className={styles.tilesGrid} style={{ opacity: isCalculating ? 0.5 : 1, transition: 'opacity 0.2s' }}>
        <div data-tour="result-tile-gpus">
          <FlipTile
            dark
          front={
            <>
              <span className={styles.tileLabel}><MicrochipIcon /> GPUs required</span>
              <span className={styles.tileValue}>{Math.round(gpus)}<span className={styles.tileUnit}>× {gpu}</span></span>
              <span className={styles.tileSub}>
                {testResult ? (
                  isDisagg ? (
                    <>disagg · prefill {disagg.prefill.workers}×{disagg.prefill.gpusPerWorker} + decode {disagg.decode.workers}×{disagg.decode.gpusPerWorker} · {testConcurrentUsers} concurrent users</>
                  ) : (
                    <>TP={testResult.memory_analysis.tp_size}{testResult.parallelism_strategy.pp_size > 1 ? ` × PP=${testResult.parallelism_strategy.pp_size}` : ''} × {testResult.memory_analysis.replicas} replica{testResult.memory_analysis.replicas > 1 ? 's' : ''} · {testConcurrentUsers} concurrent users</>
                  )
                ) : (
                  <>Configure workload below to see results</>
                )}
              </span>
            </>
          }
          back={
            <>
              <div className={styles.backTitle}>How we got {Math.round(gpus)}</div>
              <div className={styles.formula}>
                {testResult && isDisagg ? (
                  <>
                    prefill = <span className={styles.em}>{disagg.prefill.workers} × {disagg.prefill.gpusPerWorker} = {disagg.prefill.workers * disagg.prefill.gpusPerWorker}</span> GPUs<br />
                    decode = <span className={styles.em}>{disagg.decode.workers} × {disagg.decode.gpusPerWorker} = {disagg.decode.workers * disagg.decode.gpusPerWorker}</span> GPUs<br />
                    total = <span className={styles.em}>{disagg.prefill.workers * disagg.prefill.gpusPerWorker} + {disagg.decode.workers * disagg.decode.gpusPerWorker} = {Math.round(gpus)} GPUs</span>
                  </>
                ) : testResult ? (
                  <>
                    weight memory = <span className={styles.em}>{testResult.memory_analysis.weight_gb.toFixed(1)} GB</span><br />
                    usable / GPU = <span className={styles.em}>{memUsablePerGpu.toFixed(0)} GB</span><br />
                    TP size = ⌈{testResult.memory_analysis.weight_gb.toFixed(0)} ÷ {memUsablePerGpu.toFixed(0)}⌉ = <span className={styles.em}>{testResult.memory_analysis.tp_size}</span><br />
                    {testResult.parallelism_strategy.pp_size > 1 && (
                      <>PP size = <span className={styles.em}>{testResult.parallelism_strategy.pp_size}</span><br /></>
                    )}
                    replicas = {testResult.memory_analysis.replicas}<br />
                    total = {testResult.parallelism_strategy.pp_size > 1 ? `${testResult.memory_analysis.tp_size}×${testResult.parallelism_strategy.pp_size}×${testResult.memory_analysis.replicas}` : `${testResult.memory_analysis.tp_size}×${testResult.memory_analysis.replicas}`} = <span className={styles.em}>{Math.round(gpus)} GPUs</span>
                  </>
                ) : (
                  <>
                    total memory = <span className={styles.em}>20 GB</span><br />
                    usable / GPU = <span className={styles.em}>72 GB</span> (90% of 80)<br />
                    ⌈20 ÷ 72⌉ = <span className={styles.em}>1 GPU</span><br />
                    peak 3× → range up to 2
                  </>
                )}
              </div>
            </>
          }
        />
        </div>

        {!isDisagg && (
        <div>
          <FlipTile
            front={
              <>
                <span className={styles.tileLabel}><MemoryIcon /> Weight memory <Term k="weightMemory" /></span>
                <span className={styles.tileValue}>{Math.round(weight)}<span className={styles.tileUnit}>GB</span></span>
                <span className={styles.tileSub}>
                  {testResult ? (
                    <>{model.split('/')[1] || model} · {actualWeightPrecision}</>
                  ) : (
                    <>{model.split('/')[1] || model}</>
                  )}
                </span>
              </>
            }
            back={
              <>
                <div className={styles.backTitle}>Weight memory</div>
              <div className={styles.formula}>
                {testResult ? (
                  <>
                    precision = <span className={styles.em}>{actualWeightPrecision}</span><br />
                    bytes/param = <span className={styles.em}>
                      {actualWeightPrecision === 'FP16' ? '2' :
                       actualWeightPrecision === 'FP8' ? '1' :
                       actualWeightPrecision === 'INT8' ? '1' :
                       actualWeightPrecision === 'MXFP4' ? '0.5' : '0.5'}
                    </span><br />
                    params × bytes/param<br />
                    = <span className={styles.em}>{testResult.memory_analysis.weight_gb.toFixed(1)} GB</span>
                  </>
                ) : (
                  <>
                    params × bytes/param<br />
                    <span className={styles.em}>8B</span> × <span className={styles.em}>2</span> (BF16)<br />
                    = <span className={styles.em}>16 GB</span>
                  </>
                )}
              </div>
            </>
          }
        />
        </div>
        )}

        {isDisagg && ([['Prefill', disagg.prefill], ['Decode', disagg.decode]] as const).map(([name, ph]) => (
          <div key={name}>
            <FlipTile
              front={
                <>
                  <span className={styles.tileLabel}><MicrochipIcon /> {name} pool</span>
                  <span className={styles.tileValue}>{ph.workers * ph.gpusPerWorker}<span className={styles.tileUnit}>GPUs</span></span>
                  <span className={styles.tileSub}>
                    {ph.workers} × {ph.gpusPerWorker}/worker · TP{ph.tp_size}{ph.pp_size > 1 ? ` · PP${ph.pp_size}` : ''} · bs {ph.batch_size}
                  </span>
                </>
              }
              back={
                <>
                  <div className={styles.backTitle}>{name} pool</div>
                  <div className={styles.formula}>
                    gpus/worker = {ph.tp_size}{ph.pp_size > 1 ? ` × ${ph.pp_size}` : ''} = <span className={styles.em}>{ph.gpusPerWorker}</span><br />
                    workers = <span className={styles.em}>{ph.workers}</span> · bs = <span className={styles.em}>{ph.batch_size}</span><br />
                    {ph.memory_gb != null && <>peak mem/GPU = <span className={styles.em}>{ph.memory_gb.toFixed(1)} GB</span><br /></>}
                    pool = <span className={styles.em}>{ph.workers} × {ph.gpusPerWorker} = {ph.workers * ph.gpusPerWorker} GPUs</span>
                  </div>
                </>
              }
            />
          </div>
        ))}

        {!isDisagg && (
        <div>
          <FlipTile
            front={
              <>
                <span className={styles.tileLabel}><LayerGroupIcon /> KV cache / req <Term k="kvPerReq" /></span>
                <span className={styles.tileValue}>{Math.round(kv)}<span className={styles.tileUnit}>MB</span></span>
                {testResult ? (
                  <span className={styles.tileSub}>
                    {testKVCachePrecision} · {testISL + testOSL} tokens/req · {testConcurrentUsers} users
                  </span>
                ) : (
                  <span className={styles.tileSub}>
                    {testISL + testOSL} tokens/req · {testConcurrentUsers} users
                  </span>
                )}
              </>
            }
            back={
              <>
                <div className={styles.backTitle}>KV cache / request</div>
                <div className={styles.formula}>
                  {testResult && testResult.memory_analysis.kv_cache_used_gb ? (
                    <>
                      total KV used = <span className={styles.em}>{testResult.memory_analysis.kv_cache_used_gb.toFixed(1)} GB</span><br />
                      concurrent users = <span className={styles.em}>{testConcurrentUsers}</span><br />
                      KV / req = {testResult.memory_analysis.kv_cache_used_gb.toFixed(1)} ÷ {testConcurrentUsers}<br />
                      = <span className={styles.em}>{((testResult.memory_analysis.kv_cache_used_gb / testConcurrentUsers) * 1000).toFixed(0)} MB</span><br />
                      precision: <span className={styles.em}>{testKVCachePrecision}</span>
                    </>
                  ) : (
                    <>
                      2 × layers × kv_heads ×<br />
                      head_dim × bytes × tokens<br />
                      2×<span className={styles.em}>32</span>×<span className={styles.em}>8</span>×<span className={styles.em}>128</span>×2 = 128 KB/tok<br />
                      × <span className={styles.em}>150</span> tokens = 19 MB
                    </>
                  )}
                </div>
              </>
            }
          />
        </div>
        )}

        {costingsEnabled && gpuPricePerHour == null && !costings.isLoading && (
          <div style={{
            border: '1px solid #d2d2d2', borderRadius: '6px', padding: '14px',
            fontSize: '13px', fontFamily: 'var(--font-mono)', color: '#54585c',
          }}>
            <DollarSignIcon /> Cloud rate unavailable for this GPU — pick a
            provider on the Sources page or choose a GPU with published rates to
            see the cost comparison.
          </div>
        )}

        {costingsEnabled && gpuPricePerHour != null && (
        <div>
          <FlipTile
            front={
            <>
              <span className={styles.tileLabel}><DollarSignIcon /> MONTHLY COST</span>
              <div style={{ display: 'flex', gap: '12px', marginTop: '12px' }}>
                {/* Cloud pricing tile */}
                <div style={{
                  flex: 1,
                  border: '1px solid #d2d2d2',
                  borderRadius: '6px',
                  padding: '14px',
                  transition: 'transform 200ms ease-out',
                  cursor: 'pointer'
                }}
                onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.06)'}
                onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="#ee0000">
                      <path d="M8 0C3.6 0 0 3.6 0 8s3.6 8 8 8 8-3.6 8-8-3.6-8-8-8zm0 14c-3.3 0-6-2.7-6-6s2.7-6 6-6 6 2.7 6 6-2.7 6-6 6z"/>
                      <circle cx="8" cy="8" r="3"/>
                    </svg>
                    <span style={{ fontSize: '13px', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.04em', color: '#ee0000', fontWeight: 600 }}>CLOUD</span>
                    <Term k="cloudPricing" />
                  </div>
                  <div style={{ fontSize: '28px', fontFamily: 'var(--font-display)', fontWeight: 700, color: '#151515', marginBottom: '4px' }}>
                    ${((Math.round(gpus) * gpuPricePerHour * HOURS_PER_MONTH) / 1000).toFixed(1)}K/mo
                  </div>
                  <div style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: '#3c3f42' }}>
                    {cloudRateLabel && `${cloudRateLabel} · `}${((Math.round(gpus) * gpuPricePerHour * HOURS_PER_MONTH * AMORT_MONTHS_5YR) / 1000).toFixed(0)}K over 5yr
                  </div>
                </div>

                {/* Self-hosted pricing tile — only when a hardware cost is known,
                    otherwise a $0 tile would imply self-hosting is free. */}
                {catalogGpuForPricing && (
                <div style={{
                  flex: 1,
                  border: '1px solid #d2d2d2',
                  borderRadius: '6px',
                  padding: '14px',
                  transition: 'transform 200ms ease-out',
                  cursor: 'pointer'
                }}
                onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.06)'}
                onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="#151515">
                      <rect x="2" y="3" width="12" height="2" rx="1"/>
                      <rect x="2" y="7" width="12" height="2" rx="1"/>
                      <rect x="2" y="11" width="12" height="2" rx="1"/>
                    </svg>
                    <span style={{ fontSize: '13px', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.04em', color: '#151515', fontWeight: 600 }}>SELF-HOSTED</span>
                    <Term k="selfHosted" />
                  </div>
                  <div style={{ fontSize: '28px', fontFamily: 'var(--font-display)', fontWeight: 700, color: '#151515', marginBottom: '4px' }}>
                    ${(catalogGpuForPricing.hardware_cost_usd * Math.round(gpus) / AMORT_MONTHS_5YR / 1000).toFixed(1)}K/mo
                  </div>
                  <div style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: '#3c3f42' }}>
                    5yr amort · ${(catalogGpuForPricing.hardware_cost_usd * Math.round(gpus) / 1000).toFixed(0)}K total
                  </div>
                </div>
                )}
              </div>

              {/* Cost-difference label — only meaningful when both cloud and
                  self-hosted costs are known. Names whichever option is cheaper
                  rather than always claiming self-hosted "saves" (which would show
                  a negative saving when self-hosted is the pricier option). */}
              {catalogGpuForPricing && (() => {
                const cloudMonthly = Math.round(gpus) * gpuPricePerHour * HOURS_PER_MONTH
                const selfHostedMonthly = (catalogGpuForPricing.hardware_cost_usd * Math.round(gpus)) / AMORT_MONTHS_5YR
                const diff = cloudMonthly - selfHostedMonthly
                const cheaper = diff >= 0 ? 'Self-hosted' : 'Cloud'
                const color = diff >= 0 ? '#3d7317' : '#8250df'
                return (
                  <div style={{ marginTop: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <div style={{ width: '8px', height: '8px', background: color, borderRadius: '2px' }}></div>
                    <span style={{ fontSize: '13px', fontFamily: 'var(--font-display)', fontWeight: 600, color }}>
                      {cheaper} saves ${(Math.abs(diff) / 1000).toFixed(1)}K/mo
                    </span>
                  </div>
                )
              })()}
            </>
          }
          back={
            <>
              <div className={styles.backTitle}>How we calculated this</div>
              <div className={styles.formula}>
                <strong style={{ color: '#3c3f42', fontSize: '12px' }}>Cloud:</strong><br />
                {Math.round(gpus)} GPUs × <span className={styles.em}>${gpuPricePerHour.toFixed(2)}/gpu-hr</span> × <span className={styles.em}>730 hrs</span><br />
                = <span className={styles.em}>${((Math.round(gpus) * gpuPricePerHour * HOURS_PER_MONTH) / 1000).toFixed(1)}K/mo</span><br />
                <br />
                {catalogGpuForPricing && (
                <>
                <strong style={{ color: '#3c3f42', fontSize: '12px' }}>Self-hosted:</strong><br />
                ${(catalogGpuForPricing.hardware_cost_usd * Math.round(gpus) / 1000).toFixed(0)}K ÷ <span className={styles.em}>60 months</span><br />
                = <span className={styles.em}>${(catalogGpuForPricing.hardware_cost_usd * Math.round(gpus) / AMORT_MONTHS_5YR / 1000).toFixed(1)}K/mo</span><br />
                <span style={{ fontSize: '11.5px', color: '#3c3f42' }}>(hardware amortization only)</span><br />
                <br />
                </>
                )}
                <strong style={{ color: '#3c3f42', fontSize: '12px' }}>5-year totals:</strong><br />
                Cloud: <span className={styles.em}>${((Math.round(gpus) * gpuPricePerHour * HOURS_PER_MONTH * AMORT_MONTHS_5YR) / 1000).toFixed(0)}K</span><br />
                {catalogGpuForPricing && (
                <>Hardware: <span className={styles.em}>${(catalogGpuForPricing.hardware_cost_usd * Math.round(gpus) / 1000).toFixed(0)}K</span><br /></>
                )}
                <br />
                <div style={{ background: 'rgba(255, 193, 7, 0.1)', padding: '8px', borderRadius: '4px', marginTop: '8px' }}>
                  <span style={{ fontSize: '11.5px', color: '#995c00', lineHeight: '1.5' }}>
                    ⚠️ Self-hosted excludes: power (~$X/mo), cooling, staff, networking. Typical full TCO adds 40–80% to this number.
                  </span>
                </div>
              </div>
            </>
          }
        />
        </div>
        )}
      </div>
      )}

      {/* ---------- Why this GPU count (agg only) ---------- */}
      {testResult && !isDisagg && (
        <div className={styles.card} style={{ marginBottom: 20 }}>
          <div
            className={styles.cardHead}
            onClick={() => setWhyGpuExpanded(!whyGpuExpanded)}
            style={{
              cursor: 'pointer',
              transition: 'background 150ms',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '16px 20px'
            }}
            onMouseEnter={(e) => e.currentTarget.style.background = '#f5f5f5'}
            onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
          >
            <span className={styles.cardTitle}>Why this GPU count?</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {testResult.memory_analysis.kv_category && (
                <span style={{
                  fontSize: '11.5px',
                  fontFamily: 'var(--font-mono)',
                  background: '#f5f5f5',
                  border: '1px solid #d2d2d2',
                  borderRadius: '4px',
                  padding: '3px 8px',
                  color: '#3c3f42',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}>
                  KV cache: {testResult.memory_analysis.kv_category} · {testResult.memory_analysis.kv_category_label}
                  <Term k="kvCategory" />
                </span>
              )}
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                style={{
                  transition: 'transform 200ms',
                  transform: whyGpuExpanded ? 'rotate(0deg)' : 'rotate(-90deg)'
                }}
              >
                <path d="M4 6 L8 10 L12 6" stroke="#3c3f42" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateRows: whyGpuExpanded ? '1fr' : '0fr',
            transition: 'grid-template-rows 250ms ease-out'
          }}>
            <div style={{ overflow: 'hidden' }}>
              <div className={styles.cardBody}>
                <div style={{ display: 'grid', gap: '12px' }}>
                  <div style={{ padding: '12px', background: '#f5f5f5', borderRadius: '4px' }}>
                    <strong style={{ color: '#0066cc' }}>Memory Breakdown</strong>
                    <div style={{ marginTop: '8px', fontSize: '13px', lineHeight: '1.6' }}>
                      • Weight memory: <strong>{testResult.memory_analysis.weight_gb.toFixed(1)} GB</strong> ({actualWeightPrecision})<br/>
                      • Weight per GPU: <strong>{testResult.memory_analysis.weight_gb_per_gpu.toFixed(1)} GB</strong><br/>
                      • Usable per GPU: <strong>{memUsablePerGpu.toFixed(0)} GB</strong> (90% of {gpu.includes('H200') ? '141' : '80'} GB)<br/>
                      • Tensor Parallel size: <strong>{testResult.memory_analysis.tp_size}</strong> {testResult.memory_analysis.weight_gb > memUsablePerGpu ? '(required - weights don\'t fit in 1 GPU)' : '(weights fit, but using for replicas)'}<br/>
                      {testResult.parallelism_strategy.pp_size > 1 && (
                        <>• Pipeline Parallel size: <strong>{testResult.parallelism_strategy.pp_size}</strong> (splits model layers across pipeline stages)<br/></>
                      )}
                    </div>
                  </div>

                  <div style={{ padding: '12px', background: '#fffbf0', borderRadius: '4px' }}>
                    <strong style={{ color: '#995c00' }}>Workload Sizing</strong>
                    <div style={{ marginTop: '8px', fontSize: '13px', lineHeight: '1.6' }}>
                      • KV cache used: <strong>{testResult.memory_analysis.kv_cache_used_gb?.toFixed(1) || '—'} GB</strong> ({testKVCachePrecision}, {testConcurrentUsers} users)<br/>
                      • KV cache budget: <strong>{memKvBudget.toFixed(1)} GB</strong> available<br/>
                      • max_num_seqs: <strong>{testResult.vllm_config.max_num_seqs}</strong><br/>
                      • Replicas: <strong>{testResult.memory_analysis.replicas}</strong> (for throughput/redundancy)
                    </div>
                  </div>

                  <div style={{ padding: '12px', background: '#f0f9ff', borderRadius: '4px' }}>
                    <strong style={{ color: '#0066cc' }}>Bottleneck Analysis</strong>
                    <div style={{ marginTop: '8px', fontSize: '13px', lineHeight: '1.6' }}>
                      • Primary bottleneck: <strong>{testResult.bottleneck_analysis.primary}</strong><br/>
                      • Risk: {testResult.bottleneck_analysis.risk}<br/>
                      {testResult.bottleneck_analysis.fix_suggestions.length > 0 && (
                        <>• Suggestions: {testResult.bottleneck_analysis.fix_suggestions.join(', ')}</>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ---------- Memory Layout (agg only) ---------- */}
      {testResult && !isDisagg && (
        <div className={styles.card} style={{ marginBottom: 20 }}>
          <div className={styles.cardHead}>
            <span className={styles.cardTitle}>Memory layout per GPU</span>
            <span className={styles.cardHint}>{memUsablePerGpu.toFixed(0)} GB usable · {testResult.memory_analysis.tp_size * (testResult.parallelism_strategy.pp_size || 1)} GPU{testResult.memory_analysis.tp_size * (testResult.parallelism_strategy.pp_size || 1) > 1 ? 's' : ''} per model instance</span>
          </div>
          <div className={styles.cardBody}>
            <div style={{ marginBottom: '12px' }}>
              <div style={{
                display: 'flex',
                height: '40px',
                borderRadius: '4px',
                overflow: 'hidden',
                border: '1px solid #ddd'
              }}>
                {/* Weights */}
                <div style={{
                  width: `${(testResult.memory_analysis.weight_gb_per_gpu / memUsablePerGpu) * 100}%`,
                  background: '#0066cc',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fff',
                  fontSize: '12px',
                  fontWeight: '600'
                }}>
                  Weights
                </div>
                {/* KV Cache */}
                <div style={{
                  width: `${((testResult.memory_analysis.kv_cache_used_gb || 0) / (testResult.memory_analysis.tp_size * (testResult.parallelism_strategy.pp_size || 1)) / memUsablePerGpu) * 100}%`,
                  background: '#f59e0b',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fff',
                  fontSize: '12px',
                  fontWeight: '600'
                }}>
                  KV Cache
                </div>
                {/* Reserved/Overhead */}
                <div style={{
                  flex: 1,
                  background: '#e0e0e0',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#666',
                  fontSize: '12px'
                }}>
                  Reserved
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '16px', fontSize: '13px' }}>
              <span>
                <span style={{ display: 'inline-block', width: '12px', height: '12px', background: '#0066cc', marginRight: '6px', borderRadius: '2px' }}></span>
                Weights: <strong>{testResult.memory_analysis.weight_gb_per_gpu.toFixed(1)} GB</strong>
              </span>
              <span>
                <span style={{ display: 'inline-block', width: '12px', height: '12px', background: '#f59e0b', marginRight: '6px', borderRadius: '2px' }}></span>
                KV Cache: <strong>{((testResult.memory_analysis.kv_cache_used_gb || 0) / (testResult.memory_analysis.tp_size * (testResult.parallelism_strategy.pp_size || 1))).toFixed(1)} GB</strong>
              </span>
              <span>
                <span style={{ display: 'inline-block', width: '12px', height: '12px', background: '#e0e0e0', marginRight: '6px', borderRadius: '2px' }}></span>
                Reserved: <strong>{(memTotalVram - memUsablePerGpu).toFixed(1)} GB</strong>
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ---------- assumptions ---------- */}
      <div
        ref={assumptionsRef}
        className={`${styles.assumptionsHead} ${assumptionsHighlight ? styles.assumptionsHighlight : ''}`}
        data-tour="assumptions"
        onClick={() => setAssumptionsExpanded(!assumptionsExpanded)}
        style={{
          cursor: 'pointer',
          transition: 'background 150ms',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 0',
          borderRadius: '4px'
        }}
        onMouseEnter={(e) => !assumptionsHighlight && (e.currentTarget.style.background = '#f5f5f5')}
        onMouseLeave={(e) => !assumptionsHighlight && (e.currentTarget.style.background = 'transparent')}
      >
        <span className={styles.assumptionsTitle}>Want to change assumptions?</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Button variant="link" isInline onClick={(e) => { e.stopPropagation(); /* Reset logic */ }}>Reset to defaults</Button>
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            style={{
              transition: 'transform 200ms',
              transform: assumptionsExpanded ? 'rotate(0deg)' : 'rotate(-90deg)'
            }}
          >
            <path d="M4 6 L8 10 L12 6" stroke="#3c3f42" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateRows: assumptionsExpanded ? '1fr' : '0fr',
        transition: 'grid-template-rows 250ms ease-out'
      }}>
        <div style={{ overflow: 'hidden' }}>
          <p className={styles.assumptionsSub} style={{ marginBottom: 12 }}>
            Every number above comes from these. Open a section to tune it — closed sections show their current values.
          </p>

          <Accordion asDefinitionList={false}>
            {buildAccordionSections().map((sec) => (
              <AccordionItem key={sec.id}>
                <AccordionToggle
                  id={`acc-${sec.id}`}
                  isExpanded={expanded.includes(sec.id)}
                  onClick={() => toggleAcc(sec.id)}
                >
                  <span style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', width: '100%' }}>
                    <span className={styles.cardTitle} style={{ fontSize: 16 }}>{sec.title}</span>
                    {'badge' in sec && sec.badge ? <Label isCompact color={sec.badgeColor as any}>{sec.badge}</Label> : null}
                    {'hasOverride' in sec && sec.hasOverride && (
                      <span
                        role="button"
                        tabIndex={0}
                        onClick={(e) => {
                          e.stopPropagation();
                          sec.onOverrideToggle?.();
                        }}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.stopPropagation(); sec.onOverrideToggle?.(); } }}
                        style={{ fontSize: '13px', marginLeft: 'auto', padding: '4px 8px', color: 'var(--gc-link)', cursor: 'pointer' }}
                      >
                        {sec.isOverridden ? '↺ Reset to auto' : '✎ Override'}
                      </span>
                    )}
                    {!expanded.includes(sec.id) && (
                      <span className={styles.accSummary}>
                        {sec.summary.map((p) => (
                          <span key={p.k}><span className="k">{p.k}</span> {p.v}</span>
                        ))}
                      </span>
                    )}
                  </span>
                </AccordionToggle>
                <AccordionContent isHidden={!expanded.includes(sec.id)}>
                  <div className={styles.accGrid}>
                    {sec.fields.map((f: any) => (
                      <div key={f.label} className={styles.accField}>
                        <label className={styles.accFieldLabel}>
                          {f.label}{'term' in f && f.term ? <Term k={f.term as any} /> : null}
                        </label>
                        {f.readonly ? (
                          <TextInput value={String(f.value)} aria-label={f.label} isDisabled />
                        ) : f.type === 'select' ? (
                          <>
                            <FormSelect
                              value={String(f.value)}
                              aria-label={f.label}
                              onChange={(_, val) => f.onChange?.(val)}
                            >
                              {(f.options || [f.value]).map((o: any) => {
                                const optValue = typeof o === 'string' ? o : o.value;
                                const optLabel = typeof o === 'string' ? o : o.label;
                                return <FormSelectOption key={optValue} value={optValue} label={optLabel} />;
                              })}
                            </FormSelect>
                            {f.help && (
                              <div style={{ fontSize: '12px', color: '#6a6e73', marginTop: '4px', lineHeight: '1.5' }}>
                                {f.help}
                              </div>
                            )}
                          </>
                        ) : f.type === 'range' ? (
                          <div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                              <span style={{ fontSize: '13px', fontWeight: '600', fontFamily: 'var(--mono)' }}>
                                {f.rangeValue !== undefined ? `${f.rangeValue}%` : f.value}
                              </span>
                            </div>
                            <input
                              type="range"
                              min={f.min}
                              max={f.max}
                              step={f.step}
                              value={f.rangeValue !== undefined ? f.rangeValue : (typeof f.value === 'string' ? parseInt(f.value) : f.value)}
                              onChange={(e) => f.onChange?.(Number(e.target.value))}
                              style={{ width: '100%' }}
                            />
                          </div>
                        ) : f.type === 'number' ? (
                          <TextInput
                            value={String(f.value)}
                            aria-label={f.label}
                            type="number"
                            validated={f.invalid ? 'error' : 'default'}
                            onChange={(_, val) => f.onChange?.(val)}
                          />
                        ) : (
                          <TextInput
                            value={String(f.value)}
                            aria-label={f.label}
                            onChange={(_, val) => f.onChange?.(val)}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </div>
      </div>

      {/* ---------- footer actions ---------- */}
      <div className={styles.footerRow}>
        <Button variant="secondary" onClick={handleCopyAPIRequest} isDisabled={!testResult}>
          Copy API request
        </Button>
        <Button variant="secondary" onClick={handleCopyCLICommand} isDisabled={!testResult}>
          Copy CLI command
        </Button>
        <Button
          variant="secondary"
          onClick={handleExportToSheets}
          isDisabled={!testResult || !catalogGpuForPricing || gpuPricePerHour == null || isDisagg}
          title={isDisagg ? 'Export currently supports aggregated results only' : undefined}
        >
          Export to Sheets
        </Button>
        <span className={styles.footerSpacer} />
        <Button
          variant="primary"
          onClick={() => setShowSaveModal(true)}
          isDisabled={!testResult || !catalogGpuForPricing || gpuPricePerHour == null || isDisagg}
          title={isDisagg ? 'Saving currently supports aggregated results only' : undefined}
        >
          Save estimate{savedCount > 0 && ` (${savedCount})`}
        </Button>
      </div>
      {isDisagg && (
        <div style={{ marginTop: 6, fontSize: '12px', color: '#54585c' }}>
          Save and export currently support aggregated results only.
        </div>
      )}

      {/* Save estimate modal */}
      <SaveEstimateModal
        isOpen={showSaveModal}
        onClose={() => setShowSaveModal(false)}
        onSave={handleSaveEstimate}
        defaultName={generateAutoName()}
      />

      {/* Toast notification */}
      {showToast && (
        <div style={{
          position: 'fixed',
          bottom: '24px',
          right: '24px',
          background: '#151515',
          color: '#fff',
          padding: '16px 20px',
          borderRadius: '6px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          zIndex: 9999,
          fontFamily: 'var(--font-sans)',
          fontSize: '14px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          <CheckCircleIcon style={{ color: '#3d7317' }} />
          <span>
            {toastMessage === 'saved' && (
              <>Estimate saved — <Link href="/compare" style={{ color: '#4da6ff', textDecoration: 'underline' }}>view in Compare →</Link></>
            )}
            {toastMessage === 'api-copied' && 'API request copied to clipboard'}
            {toastMessage === 'cli-copied' && 'CLI command copied to clipboard'}
            {toastMessage === 'exported' && 'CSV file downloaded — import into Google Sheets'}
          </span>
        </div>
      )}
      <div className={styles.apiPreview}>
        <Button variant="link" isInline onClick={() => setShowApi((s) => !s)}>
          {showApi ? 'Hide' : 'Preview'} API request body
        </Button>
        {showApi && (
          <pre className={styles.apiBody}>
            {JSON.stringify(buildEstimateRequestBody(), null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}

/* ---------- constraint row (unused - kept for future implementation) ---------- */
type Status = 'ok' | 'watch' | 'bottleneck';
function ConstraintRow({ label, detail, status, term }: { label: string; detail: string; status: Status; term?: string }) {
  const dot = status === 'ok' ? styles.conOk : status === 'watch' ? styles.conWatch : styles.conBottleneck;
  const pill = status === 'ok' ? styles.pillOk : status === 'watch' ? styles.pillWatch : styles.pillBottleneck;
  const text = status === 'ok' ? 'OK' : status === 'watch' ? 'Watch' : 'Bottleneck';
  return (
    <div className={styles.constraint}>
      <span className={`${styles.conStatus} ${dot}`} />
      <span className={styles.conLabel}>{label}{term ? <Term k={term as any} /> : null}</span>
      <span className={styles.conDetail}>{detail}</span>
      <span className={`${styles.conPill} ${pill}`}>{text}</span>
    </div>
  );
}

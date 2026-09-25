import type { WorkloadPreset } from './workload-presets';

export interface AppConfig {
  defaultModel: string;
  defaultSystem: string;
  defaultOpenModel: string;
  defaultFrontierModel: string;
  defaultBackend: string;
  testedModels: string[];
  huggingFaceModels: string[];
  suggestedModelNames: string[];
  modelRequestUrl: string;
  workloadPresets: WorkloadPreset[];
}

// Hardcoded fallback — used if /config.json fails to load
const FALLBACK: AppConfig = {
  defaultModel: 'Qwen/Qwen3-32B',
  defaultSystem: 'h200_sxm',
  defaultOpenModel: 'Qwen/Qwen3-32B',
  defaultFrontierModel: 'claude-fable-5',
  defaultBackend: 'vllm',
  testedModels: [],
  huggingFaceModels: [],
  suggestedModelNames: [],
  modelRequestUrl: '',
  workloadPresets: [],
};

let cached: AppConfig | null = null;

export async function loadAppConfig(): Promise<AppConfig> {
  if (cached) return cached;
  try {
    const res = await fetch('/config.json');
    if (!res.ok) throw new Error(`config.json fetch failed (${res.status})`);
    const data = await res.json() as Partial<AppConfig>;
    cached = {
      defaultModel: data.defaultModel ?? FALLBACK.defaultModel,
      defaultSystem: data.defaultSystem ?? FALLBACK.defaultSystem,
      defaultOpenModel: data.defaultOpenModel ?? FALLBACK.defaultOpenModel,
      defaultFrontierModel: data.defaultFrontierModel ?? FALLBACK.defaultFrontierModel,
      defaultBackend: data.defaultBackend ?? FALLBACK.defaultBackend,
      testedModels: Array.isArray(data.testedModels)
        ? data.testedModels.filter((model): model is string => typeof model === 'string')
        : FALLBACK.testedModels,
      huggingFaceModels: data.huggingFaceModels ?? FALLBACK.huggingFaceModels,
      suggestedModelNames: data.suggestedModelNames ?? FALLBACK.suggestedModelNames,
      modelRequestUrl: data.modelRequestUrl ?? FALLBACK.modelRequestUrl,
      workloadPresets: data.workloadPresets ?? FALLBACK.workloadPresets,
    };
  } catch {
    cached = FALLBACK;
  }
  return cached;
}

// Synchronous read after loadAppConfig() has been awaited — returns fallback until then
export function getAppConfig(): AppConfig {
  return cached ?? FALLBACK;
}

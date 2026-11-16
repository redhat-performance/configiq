import appConfig from '../public/config.json'
import { DEFAULT_WORKLOAD, type WorkloadPreset } from '../lib/workload-presets'

// The live UI matrix checks each of these tested GPU systems explicitly.
export const coveredSystems = [
  'a100_sxm',
  'b200_sxm',
  'b300_sxm',
  'b60',
  'gb200',
  'gb300',
  'h100_sxm',
  'h200_sxm',
  'l40s',
  'rtx_pro_6000_server',
] as const

export const workloads: WorkloadPreset[] = [DEFAULT_WORKLOAD, ...appConfig.workloadPresets]

// These are test inputs for models too large for TP=1, not sizing formulas.
export const modelParallelism: Record<string, number> = {
  'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block': 8,
  'deepseek-ai/DeepSeek-V4-Pro': 16,
  'RedHatAI/GLM-5.2-FP8': 16,
  'moonshotai/Kimi-K2.5': 8,
  'moonshotai/Kimi-K3': 32,
}

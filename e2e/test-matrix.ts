import appConfig from '../public/config.json'
import { DEFAULT_WORKLOAD, type WorkloadPreset } from '../lib/workload-presets'
import type { ConcurrencyPreset } from '../lib/concurrency-presets'

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

// Workload presets set ISL/OSL only. Exercise each visible concurrency choice
// independently where the page sends a concurrency target.
export const matrixConcurrencyPresets: readonly ConcurrencyPreset[] = appConfig.concurrencyPresets
if (matrixConcurrencyPresets.length === 0) throw new Error('No concurrency presets in config.json')
export const baselineConcurrencyPreset: ConcurrencyPreset = (() => {
  const preset = matrixConcurrencyPresets.find(candidate => candidate.key === 'small')
  if (!preset) throw new Error('The small concurrency preset is missing from config.json')
  return preset
})()

export const scheduledMatrix = process.env.PLAYWRIGHT_MATRIX_MODE === 'scheduled'
const rotation = Number(process.env.PLAYWRIGHT_MATRIX_ROTATION ?? '0')
if (!Number.isSafeInteger(rotation) || rotation < 0) {
  throw new Error('PLAYWRIGHT_MATRIX_ROTATION must be a non-negative integer')
}

// Throughput demand has a request-rate input instead of a concurrency preset.
export const matrixDemandRate = 10

// Keep the nightly dev run small while exercising all tested models and workloads.
// The selected systems produced completed responses in the prior live report,
// except Nemotron Ultra, which still needs a successful live pairing confirmed.
const nightlySystemByModel: Record<string, (typeof coveredSystems)[number]> = {
  'google/gemma-4-31B-it': 'h200_sxm',
  'RedHatAI/gemma-4-26B-A4B-it': 'h100_sxm',
  'RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic': 'h100_sxm',
  'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8': 'h200_sxm',
  'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block': 'gb200',
  'deepseek-ai/DeepSeek-V4-Pro': 'b200_sxm',
  'openai/gpt-oss-120b': 'h100_sxm',
  'RedHatAI/GLM-5.2-FP8': 'gb200',
  'moonshotai/Kimi-K2.5': 'b200_sxm',
  'moonshotai/Kimi-K3': 'gb200',
}

export function systemsForModel(model: string): readonly (typeof coveredSystems)[number][] {
  if (process.env.PLAYWRIGHT_NIGHTLY_SAMPLE !== '1') return coveredSystems
  const system = nightlySystemByModel[model]
  if (!system) throw new Error(`No nightly GPU system selected for tested model ${model}`)
  return [system]
}

type SystemId = (typeof coveredSystems)[number]

export interface MatrixCoordinates {
  workload: WorkloadPreset
  systemId: SystemId
}

// The scheduled extras cover each model once per control. Advancing the run
// number rotates those controls through workloads and, on XYZ, GPU systems.
export function scheduledCoordinates(model: string, scenarioIndex: number): MatrixCoordinates {
  const modelIndex = appConfig.approvedModels.indexOf(model)
  if (modelIndex < 0) throw new Error(`Unknown tested model ${model}`)
  const systems = systemsForModel(model)
  const position = rotation + modelIndex + scenarioIndex
  return {
    workload: workloads[position % workloads.length],
    systemId: systems[position % systems.length],
  }
}

export function concurrencyCasesForModel(model: string, scenarioOffset: number): Array<MatrixCoordinates & { concurrency: ConcurrencyPreset }> {
  const systems = systemsForModel(model)
  const baseline = workloads.flatMap(workload => systems.map(systemId => ({ workload, systemId, concurrency: baselineConcurrencyPreset })))
  const extras = matrixConcurrencyPresets.filter(preset => preset.key !== baselineConcurrencyPreset.key)
  if (scheduledMatrix) {
    return [
      ...baseline,
      ...extras.map((concurrency, index) => ({ ...scheduledCoordinates(model, scenarioOffset + index), concurrency })),
    ]
  }
  return workloads.flatMap(workload => systems.flatMap(systemId => matrixConcurrencyPresets.map(concurrency => ({ workload, systemId, concurrency }))))
}

// These are test inputs for models too large for TP=1, not sizing formulas.
export const modelParallelism: Record<string, number> = {
  'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block': 8,
  'deepseek-ai/DeepSeek-V4-Pro': 16,
  'RedHatAI/GLM-5.2-FP8': 16,
  'moonshotai/Kimi-K2.5': 8,
  'moonshotai/Kimi-K3': 32,
}

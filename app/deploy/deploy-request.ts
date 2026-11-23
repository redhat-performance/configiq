import type { RecommendParams } from '@/contexts/RecommendContext'
import type { WorkloadPreset } from '@/lib/workload-presets'

/**
 * The deploy page's form state. `ttftTarget`/`tpotTarget` and `gpusFree` are
 * display-only: with `sla_mode: 'none'` the run is shaped by the load alone,
 * latency targets compare against the answer after the fact, and the free-GPU
 * count is the user's cluster fact, never a request field.
 */
export interface DeployFormState {
  model: string
  /** AISimulators system id, not the display label. */
  systemId: string
  workload: WorkloadPreset
  loadUnit: 'users' | 'rps'
  loadValue: number
  ttftTarget: number
  tpotTarget: number
  gpusFree: number
  backend: string
  backendVersion?: string
  simulator?: string
}

/** Form state → the /recommend request. One mapping, tested, used by every run. */
export function buildDeployParams(form: DeployFormState): RecommendParams {
  const params: RecommendParams = {
    model_path: form.model,
    system: form.systemId,
    isl: form.workload.isl,
    osl: form.workload.osl,
    sla_mode: 'none',
    backend: form.backend,
    ...(form.workload.prefix != null ? { prefix: form.workload.prefix } : {}),
    ...(form.backendVersion ? { backend_version: form.backendVersion } : {}),
    ...(form.simulator ? { simulator: form.simulator } : {}),
    ...(form.loadUnit === 'users'
      ? { objective: 'concurrency_capacity', target_concurrency: form.loadValue }
      : { objective: 'throughput_demand', target_request_rate: form.loadValue }),
  }
  return params
}

export interface DeployDefaultsInput {
  modelOptions: string[]
  gpuOptions: Array<{ systemId: string; label: string }>
  /** Models the validation registry marks validated, when it loaded. */
  validatedModelIds: string[]
  /** Validated model/system pairs, when the registry loaded. */
  pairs: Array<{ modelId: string; systemId: string }>
}

/**
 * First-page-load defaults: a validated model over a plain catalog one, and a
 * system that model was validated on. The catalog is the authority — a registry
 * model the catalog no longer lists is not offered.
 */
export function pickDeployDefaults(input: DeployDefaultsInput): { model: string; systemId: string } | null {
  const validated = input.validatedModelIds.find(id => input.modelOptions.includes(id))
  const model = validated ?? input.modelOptions[0]
  if (!model) return null
  const systemId = input.pairs.find(pair => pair.modelId === model)?.systemId
    ?? input.gpuOptions[0]?.systemId
  if (!systemId) return null
  return { model, systemId }
}

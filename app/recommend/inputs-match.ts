import type { RecommendObjective } from '@/contexts/RecommendContext'

/**
 * The subset of a submitted recommend request that can make the visible form
 * disagree with the result it produced. Kept structural so the provider's
 * `RecommendParams` can be passed without exporting its implementation type.
 */
export interface SubmittedRecommendParams {
  model_path: string
  system: string
  isl: number
  osl: number
  max_seq_len?: number
  prefill_max_seq_len?: number
  decode_max_seq_len?: number
  sla_mode: 'enforce' | 'none'
  objective?: RecommendObjective
  ttft?: number
  tpot?: number
  target_concurrency?: number
  target_request_rate?: number
  request_latency?: number
  prefix?: number
  fabric?: string | null
}

/** The visible form values the dirty check compares against the request. */
export interface CurrentRecommendInputs {
  model: string
  system: string
  isl: number
  osl: number
  objective: RecommendObjective
  ttft: number
  tpot: number
  targetConcurrency: number
  demandRate: number
  requestLatency: number | null
  prefix: number
  maxSeqLen: number | null
  prefillMaxSeqLen: number | null
  decodeMaxSeqLen: number | null
  /** `undefined` means fabric is not required for this goal; `null` means omitted on purpose. */
  effectiveFabric: string | null | undefined
}

/**
 * Mirror `handleCalculate`'s fabric decision. An aggregate-only goal omits the
 * fabric even when the selected backend normally requires one, so the dirty
 * check must use `undefined` for that case rather than a resolved fabric name.
 */
export function resolveEffectiveFabric(
  fabricRequired: boolean,
  aggregateOnlyGoal: boolean,
  disaggSupported: boolean,
  selectedFabric: string | null,
): string | null | undefined {
  if (!fabricRequired || aggregateOnlyGoal) return undefined
  return disaggSupported ? selectedFabric || null : null
}

/**
 * Whether the current form still describes the request that produced the
 * displayed result.
 *
 * Only fields the submitted objective actually constrains are compared. In
 * particular, an end-to-end bound replaces TTFT/TPOT, so those two must not
 * mark the form dirty when `request_latency` was what the request sent.
 */
export function submittedInputsMatch(
  params: SubmittedRecommendParams,
  inputs: CurrentRecommendInputs,
): boolean {
  const submittedObjective: RecommendObjective =
    params.objective ?? (params.sla_mode === 'none' ? 'throughput_demand' : 'min_gpus')

  if (submittedObjective !== inputs.objective) return false
  if (params.model_path !== inputs.model) return false
  if (params.system !== inputs.system) return false
  if (params.isl !== inputs.isl) return false
  if (params.osl !== inputs.osl) return false
  if ((params.prefix ?? 0) !== inputs.prefix) return false
  if ((params.max_seq_len ?? null) !== inputs.maxSeqLen) return false
  if ((params.prefill_max_seq_len ?? null) !== inputs.prefillMaxSeqLen) return false
  if ((params.decode_max_seq_len ?? null) !== inputs.decodeMaxSeqLen) return false
  if (params.fabric !== inputs.effectiveFabric) return false

  switch (submittedObjective) {
    case 'throughput_demand':
      return (params.target_request_rate ?? null) === inputs.demandRate
    case 'concurrency_capacity':
      return (params.target_concurrency ?? 32) === inputs.targetConcurrency
    case 'min_gpus': {
      const submittedLatency = params.request_latency ?? null
      // An end-to-end bound replaces the decomposed TTFT/TPOT bounds. Compare
      // the shape that was actually sent, including a switch between the two.
      if (submittedLatency != null || inputs.requestLatency != null) {
        return submittedLatency === inputs.requestLatency &&
          (params.target_concurrency ?? 32) === inputs.targetConcurrency
      }
      return params.ttft === inputs.ttft &&
        (params.tpot ?? 30) === inputs.tpot &&
        (params.target_concurrency ?? 32) === inputs.targetConcurrency
    }
    default:
      return false
  }
}

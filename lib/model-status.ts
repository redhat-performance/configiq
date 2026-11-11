/** Result of checking whether the selected model can be priced. */
export type ModelCheckStatus =
  | 'idle'
  | 'supported'
  | 'catalog'
  | 'fetching'
  | 'fetched'
  | 'error'

export interface ModelStatusCheck {
  /** Model the status was computed for. */
  model: string
  status: ModelCheckStatus
}

/**
 * A status only describes the model it was computed for. A model change must
 * not inherit the previous model's validated status while the new lookup is
 * still pending.
 */
export function statusForModel(check: ModelStatusCheck, model: string): ModelCheckStatus {
  return check.model === model ? check.status : 'idle'
}

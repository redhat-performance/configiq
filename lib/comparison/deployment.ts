import type { ComparisonEntry } from './types'

/**
 * Why a saved entry cannot render a deployment artifact yet, or null when it
 * can.
 *
 * An entry saved before deployment artifacts existed has no `deployment` field
 * at all; one saved since but from a run that reported no engine limits has
 * `null`. The first is fixable by saving the result again, the second is not,
 * so they get different explanations.
 */
export function comparisonDeployDisabledReason(entry: ComparisonEntry): string | null {
  if (entry.deployment === undefined) {
    return 'Saved before deployment artifacts were available. Reopen the result and save it again.'
  }
  if (entry.deployment == null) {
    return 'This result did not report the engine limits a deployment is rendered from.'
  }
  return null
}

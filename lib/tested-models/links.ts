/**
 * Canonical tested-model URLs. Build these centrally so feedback, dashboard
 * cards, and redirects cannot drift apart.
 */

export const TESTED_MODELS_BASE = '/tested-models'

export type TestedModelsAnchor = 'boundary' | 'coverage' | 'metrics'

export function testedPairHref(pairId: string, anchor?: TestedModelsAnchor): string {
  const path = `${TESTED_MODELS_BASE}/${encodeURIComponent(pairId)}`
  return anchor ? `${path}#${anchor}` : path
}

export function testedModelsFilterHref(filters: { modelId?: string; systemId?: string }): string {
  const params = new URLSearchParams()
  if (filters.modelId) params.set('model', filters.modelId)
  if (filters.systemId) params.set('system', filters.systemId)
  const query = params.toString()
  return query ? `${TESTED_MODELS_BASE}?${query}` : TESTED_MODELS_BASE
}

/** Legacy `?pair=` links redirect to the canonical pair route. */
export function legacyPairRedirectTarget(pairId: string): string {
  return testedPairHref(pairId)
}

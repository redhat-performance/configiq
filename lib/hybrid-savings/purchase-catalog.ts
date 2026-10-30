/** Complete-server estimates supplied by the API; no frontend purchase prices. */
export interface ServerPurchaseConfiguration {
  id: string
  systemId: string
  gpuCount: number
  purchasePrice: number
  installationCost: number | null
  sourceLabel: string
  sourceUrl: string | null
  sourceDate: string | null
  indicative: boolean
  collectionKind: 'automated' | 'curated'
  retrievedAt: string | null
  stale: boolean
}
export function hasServerPurchaseConfiguration(
  configurations: readonly ServerPurchaseConfiguration[], systemId: string,
): boolean {
  return configurations.some(configuration => configuration.systemId === systemId)
}
export function resolveCompatibleServerPurchaseConfigurations(
  configurations: readonly ServerPurchaseConfiguration[], systemId: string, gpusPerReplica: number,
): ServerPurchaseConfiguration[] {
  if (!Number.isFinite(gpusPerReplica) || gpusPerReplica < 0.5 || Math.abs(Math.round(gpusPerReplica) - gpusPerReplica) > 0.1) return []
  return configurations.filter(configuration => configuration.systemId === systemId && configuration.gpuCount >= Math.round(gpusPerReplica))
    .sort((left, right) => left.gpuCount - right.gpuCount || left.purchasePrice - right.purchasePrice)
}

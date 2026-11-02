import { notFound } from 'next/navigation'
import { readTestedModelsRegistry } from '@/lib/tested-models/registry'
import type { PublicRegistryPair, RegistryPair, ValidationGateThresholds } from '@/lib/tested-models/registry-types'
import { TestedModelPair } from './TestedModelPair'

export const dynamic = 'force-dynamic'

interface TestedModelPairPageProps {
  params: Promise<{ pairId: string }>
}

export default async function TestedModelPairPage({ params }: TestedModelPairPageProps) {
  const { pairId } = await params
  let pair: RegistryPair | undefined
  let gate: ValidationGateThresholds | null = null
  try {
    const registry = await readTestedModelsRegistry()
    pair = registry.pairs.find(candidate => candidate.id === pairId)
    gate = registry.gate
  } catch {
    notFound()
  }
  if (!pair) notFound()

  const publicPair: PublicRegistryPair = {
    id: pair.id,
    modelId: pair.modelId,
    systemId: pair.systemId,
    status: pair.status,
    coverage: pair.coverage,
    diagnostics: pair.diagnostics,
    metrics: pair.metrics,
  }

  return <TestedModelPair pair={publicPair} gate={gate} />
}

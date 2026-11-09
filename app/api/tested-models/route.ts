import { NextResponse } from 'next/server'
import { readTestedModelsRegistry } from '@/lib/tested-models/registry'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const registry = await readTestedModelsRegistry()
    return NextResponse.json(
      {
        status: 'available',
        schemaVersion: registry.schemaVersion,
        generatedAt: registry.generatedAt,
        dataset: registry.dataset ?? null,
        gate: registry.gate,
        approvedModels: registry.approvedModels,
        pairs: registry.pairs.map(pair => ({
          id: pair.id,
          modelId: pair.modelId,
          systemId: pair.systemId,
          status: pair.status,
          coverage: pair.coverage,
          dashboardPath: pair.dashboardPath,
          gate: pair.gate,
          diagnostics: pair.diagnostics,
          metrics: pair.metrics,
          thresholds: pair.thresholds,
          legacy: pair.legacy,
        })),
      },
      { headers: { 'Cache-Control': 'no-store' } },
    )
  } catch {
    return NextResponse.json(
      { status: 'unavailable', schemaVersion: null, generatedAt: null, gate: null, approvedModels: [], pairs: [] },
      { headers: { 'Cache-Control': 'no-store' } },
    )
  }
}

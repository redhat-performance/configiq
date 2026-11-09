import { NextResponse } from 'next/server'
import { backendFamily, buildFeatureVector } from '@/lib/tested-models/feature-vector'
import type { CanonicalClassifierInput, ClassifierStatus } from '@/lib/tested-models/types'
import { modelStatus, readTestedModelsRegistry, type RegistryPair } from '@/lib/tested-models/registry'
import { testedModelsFilterHref, testedPairHref } from '@/lib/tested-models/links'

export const dynamic = 'force-dynamic'

function statusForProbability(probability: number, thresholds: RegistryPair['thresholds']): ClassifierStatus {
  const parsed = thresholds !== null && typeof thresholds === 'object' ? thresholds as Record<string, unknown> : {}
  const within = typeof parsed.within_region_probability === 'number' ? parsed.within_region_probability : 0.85
  const uncertain = typeof parsed.uncertain_probability === 'number' ? parsed.uncertain_probability : 0.5
  if (probability >= within) return 'within_validated_range'
  if (probability >= uncertain) return 'near_validated_boundary'
  return 'outside_validated_range'
}

function unavailable(reasons: string[], status: ClassifierStatus = 'validation_unavailable', extra: Record<string, unknown> = {}) {
  return NextResponse.json({
    status,
    confidence: null,
    pair_id: null,
    classifier: null,
    reasons,
    dashboard_href: null,
    boundary_href: null,
    ...extra,
  })
}

export async function POST(request: Request) {
  let input: CanonicalClassifierInput
  try {
    input = await request.json() as CanonicalClassifierInput
  } catch {
    return unavailable(['The classification request was not valid JSON.'])
  }

  let registry
  try {
    registry = await readTestedModelsRegistry()
  } catch {
    return unavailable(['The tested-model registry is unavailable. Tested status is unknown, not untested.'])
  }

  const modelId = input?.effective?.model_id
  const systemId = input?.effective?.system_id
  if (!modelId || !systemId) {
    return unavailable(['The request did not include a model and GPU system.'])
  }

  const pair = registry.pairs.find(candidate => candidate.modelId === modelId && candidate.systemId === systemId)
  if (!pair) {
    const status = modelStatus(modelId, registry)
    if (status === 'approved_no_data') {
      return unavailable(
        ['This model is approved for testing, but no measurements have been published yet.'],
        'no_measurements',
        { dashboard_href: testedModelsFilterHref({ modelId }) },
      )
    }
    if (status === 'measured_no_envelope' || status === 'validated') {
      return unavailable(
        [`Measurements exist for ${modelId}, but not for ${systemId}. No classifier verdict is available for this pair.`],
        'no_measurements',
        { dashboard_href: testedModelsFilterHref({ modelId }) },
      )
    }
    return unavailable(['No tested measurements are available for this model and GPU system.'])
  }

  const dashboardHref = testedPairHref(pair.id)
  const boundaryHref = testedPairHref(pair.id, 'boundary')
  const requestedBackend = backendFamily(input.effective.backend)
  const measuredBackends = pair.coverage?.backends ?? []
  if (requestedBackend !== 'unknown' && measuredBackends.length > 0 && !measuredBackends.includes(requestedBackend)) {
    return unavailable(
      [
        `Measurements for ${modelId} on ${systemId} cover ${measuredBackends.join(', ')}, not ${requestedBackend}. No classifier verdict is available for this backend.`,
      ],
      'no_measurements',
      { pair_id: pair.id, dashboard_href: dashboardHref, boundary_href: boundaryHref },
    )
  }
  if (pair.status !== 'validated') {
    return unavailable(
      [
        'Measurements exist for this model and GPU system, but no classifier passes the validation gate, so no verdict is shown.',
      ],
      'measured_no_envelope',
      { pair_id: pair.id, dashboard_href: dashboardHref, boundary_href: boundaryHref },
    )
  }

  try {
    const tritonUrl = process.env.TRITON_INFERENCE_URL ?? 'http://host.containers.internal:8000'
    const vector = buildFeatureVector(input, pair.featureSchema!)
    const response = await fetch(`${tritonUrl}/v2/models/${encodeURIComponent(pair.tritonModel!)}/infer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({
        inputs: [{ name: 'input__0', shape: [1, vector.length], datatype: 'FP32', data: vector }],
        outputs: [{ name: 'output__0' }],
      }),
      signal: AbortSignal.timeout(5000),
    })
    if (!response.ok) throw new Error(`Triton returned ${response.status}`)
    const result = await response.json() as { outputs?: Array<{ data?: unknown[] }> }
    const data = result.outputs?.[0]?.data
    const probability = Array.isArray(data) && typeof data[0] === 'number' ? data[0] : null
    if (probability == null || !Number.isFinite(probability)) throw new Error('Triton returned no within-region probability')

    return NextResponse.json({
      status: statusForProbability(probability, pair.thresholds),
      confidence: probability,
      pair_id: pair.id,
      classifier: pair.classifier,
      reasons: [],
      dashboard_href: dashboardHref,
      boundary_href: boundaryHref,
    })
  } catch {
    return unavailable(
      ['The validated classifier could not be evaluated. No verdict is available for this request.'],
      'validation_unavailable',
      { pair_id: pair.id, dashboard_href: dashboardHref, boundary_href: boundaryHref },
    )
  }
}

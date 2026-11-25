'use client'

import { Label } from '@patternfly/react-core'
import Link from 'next/link'
import type { ReactNode } from 'react'
import type {
  CanonicalClassifierInput,
  ClassifierStatus,
  ClassifierPrediction,
} from '@/lib/tested-models/types'
import { testedPairHref, testedModelsFilterHref } from '@/lib/tested-models/links'
import { buildValidated } from '@/lib/deep-links/common'
import { testedModelLink } from '@/lib/deep-links/tested-models'

interface TestedPerformanceFeedbackProps {
  prediction: ClassifierPrediction | null
  isLoading?: boolean
  modelId?: string
  systemId?: string
  /** The canonical input the classifier ran on, used to describe the shared prediction. */
  predictionInput?: CanonicalClassifierInput | null
  /** Predicted aggregate output throughput, plotted against the measured curve. */
  predictionThroughput?: number | null
  /** Result provenance shown right-aligned on the status row, e.g. the simulator badge. */
  headerRight?: ReactNode
  /** Supporting detail shown below the validation message, e.g. search summary. */
  children?: ReactNode
}

const STATUS_DETAILS: Record<ClassifierStatus, { label: string; color: 'green' | 'orange' | 'red' | 'blue' | 'grey' }> = {
  within_validated_range: { label: 'Within validated range', color: 'green' },
  near_validated_boundary: { label: 'Near validated boundary', color: 'orange' },
  outside_validated_range: { label: 'Outside validated range', color: 'red' },
  measured_no_envelope: { label: 'Measured, no validated envelope', color: 'blue' },
  no_measurements: { label: 'No measurements yet', color: 'grey' },
  validation_unavailable: { label: 'Validation unavailable', color: 'grey' },
}

const BORDER_COLOR: Record<'green' | 'orange' | 'red' | 'blue' | 'grey', string> = {
  green: '#3e8635',
  orange: '#f0ab00',
  red: '#c9190b',
  blue: '#0066cc',
  grey: '#8a8d90',
}

const LINK_BACKENDS = new Set(['vllm', 'tensorrt-llm', 'sglang'])

/**
 * The measured-envelope link carrying this prediction as a snapshot.
 *
 * The snapshot is built from the configuration the classifier actually ran on,
 * so a later edit to the form cannot change what the shared link describes. It
 * is a description, not a request: the pair page plots it and never re-runs a
 * simulator. Any value the contract cannot accept makes the link fall back to
 * the plain envelope link instead of carrying a partial snapshot.
 */
function predictionSnapshotHref(
  prediction: ClassifierPrediction,
  input: CanonicalClassifierInput | null | undefined,
  throughput: number | null | undefined,
): string | null {
  if (!prediction.pair_id || !input) return null
  if (typeof throughput !== 'number' || !Number.isFinite(throughput) || throughput <= 0) return null
  const effective = input.effective
  if (effective.concurrency < 1) return null
  const tp = effective.tp_size || 1
  const snapshot = buildValidated(testedModelLink, {
    // The workload selection stays unset: the destination picks the
    // prediction's measured workload when one exists and its own default
    // otherwise. Naming the prediction's unmeasured ISL/OSL here would open an
    // empty chart.
    tp,
    predIsl: effective.isl,
    predOsl: effective.osl,
    predConcurrency: effective.concurrency,
    predTp: tp,
    predThroughput: throughput,
    predReplicas: effective.replicas ?? undefined,
    predPp: effective.pp_size || undefined,
    predStatus: prediction.status,
    predConfidence: prediction.confidence ?? undefined,
    predBackend: LINK_BACKENDS.has(effective.backend)
      ? effective.backend as 'vllm' | 'tensorrt-llm' | 'sglang'
      : undefined,
    predBackendVersion: effective.backend_version ?? undefined,
    predServingMode: effective.serving_mode,
  })
  // The overlay needs the point coordinates. If the contract rejected any of
  // them, showing no overlay is better than plotting a partial snapshot.
  const required = [
    snapshot.values.predIsl,
    snapshot.values.predOsl,
    snapshot.values.predConcurrency,
    snapshot.values.predTp,
    snapshot.values.predThroughput,
  ]
  if (required.some(value => value === undefined)) return null
  return testedModelLink.build(snapshot.values, testedPairHref(prediction.pair_id))
}

export function TestedPerformanceFeedback({
  prediction,
  isLoading = false,
  modelId,
  systemId,
  predictionInput,
  predictionThroughput,
  headerRight,
  children,
}: TestedPerformanceFeedbackProps) {
  if (!prediction && !isLoading && !headerRight && !children) return null
  if (!prediction && isLoading) {
    if (!headerRight && !children) {
      return <div style={{ marginBottom: 20, color: '#3c3f42', fontSize: 13 }}>Checking the tested performance envelope...</div>
    }
    return (
      <div style={{ marginBottom: 20, padding: '14px 18px', border: '1px solid #d2d2d2', borderLeft: '4px solid #8a8d90', background: '#fff' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
          <span style={{ color: '#3c3f42', fontSize: 13 }}>Checking the tested performance envelope...</span>
          {headerRight}
        </div>
        {children && <div style={{ marginTop: 8 }}>{children}</div>}
      </div>
    )
  }
  if (!prediction) {
    return (
      <div style={{ marginBottom: 20, padding: '14px 18px', border: '1px solid #d2d2d2', background: '#fff' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
          {headerRight}
        </div>
        {children && <div style={{ marginTop: 8 }}>{children}</div>}
      </div>
    )
  }

  const detail = STATUS_DETAILS[prediction.status]
  if (!detail) return null
  const confidence = prediction.confidence == null ? null : `${Math.round(prediction.confidence * 100)}% confidence`

  const message = prediction.status === 'outside_validated_range'
    ? 'This configuration is outside the measured performance range for this model on this GPU system.'
    : prediction.status === 'near_validated_boundary'
      ? 'This configuration is near a measured performance boundary. Consider a lower concurrency or shorter context.'
      : prediction.status === 'within_validated_range'
        ? 'This configuration is inside the measured performance range for this model on this GPU system.'
        : prediction.status === 'measured_no_envelope'
          ? 'Measurements exist for this model and GPU system, but no classifier passes the validation gate. The graph is available; no verdict is shown.'
          : prediction.status === 'no_measurements'
            ? 'This model has no published measurements for the selected GPU system yet. No verdict is shown.'
            : 'The tested-model registry or classifier could not be evaluated. Tested status is unknown, not "untested".'

  const fallbackHref = prediction.pair_id
    ? testedPairHref(prediction.pair_id)
    : modelId
      ? testedModelsFilterHref({ modelId, systemId })
      : null
  const snapshotHref = predictionSnapshotHref(prediction, predictionInput, predictionThroughput)
  const href = snapshotHref ?? prediction.dashboard_href ?? fallbackHref
  const linkLabel = snapshotHref
    ? 'View your prediction on the measured envelope'
    : prediction.status === 'measured_no_envelope' || prediction.status === 'no_measurements'
      ? 'View tested coverage'
      : 'View the measured performance envelope'

  return (
    <div style={{ marginBottom: 20, padding: '14px 18px', border: '1px solid #d2d2d2', borderLeft: `4px solid ${BORDER_COLOR[detail.color]}`, background: '#fff' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10, marginBottom: 6 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Label color={detail.color} isCompact>{detail.label}</Label>
          {confidence && <span style={{ color: '#3c3f42', fontSize: 12 }}>{confidence}</span>}
        </div>
        {headerRight}
      </div>
      <div style={{ color: '#3c3f42', fontSize: 14 }}>{message}</div>
      {prediction.reasons.length > 0 && prediction.status !== 'within_validated_range' && (
        <ul style={{ margin: '8px 0 0', paddingLeft: 18, color: '#3c3f42', fontSize: 13 }}>
          {prediction.reasons.map(reason => <li key={reason}>{reason}</li>)}
        </ul>
      )}
      {children && <div style={{ marginTop: 8 }}>{children}</div>}
      {href && (
        <Link href={href} style={{ display: 'inline-block', marginTop: 8, color: '#0066cc', fontSize: 13 }}>
          {linkLabel}
        </Link>
      )}
    </div>
  )
}

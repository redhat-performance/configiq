'use client'

import { Label } from '@patternfly/react-core'
import Link from 'next/link'
import type { ClassifierStatus, ClassifierPrediction } from '@/lib/tested-models/types'
import { testedPairHref, testedModelsFilterHref } from '@/lib/tested-models/links'

interface TestedPerformanceFeedbackProps {
  prediction: ClassifierPrediction | null
  isLoading?: boolean
  modelId?: string
  systemId?: string
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

export function TestedPerformanceFeedback({ prediction, isLoading = false, modelId, systemId }: TestedPerformanceFeedbackProps) {
  if (!prediction && !isLoading) return null
  if (!prediction && isLoading) {
    return <div style={{ marginBottom: 20, color: '#3c3f42', fontSize: 13 }}>Checking the tested performance envelope...</div>
  }
  if (!prediction) return null

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
  const href = prediction.dashboard_href ?? fallbackHref
  const linkLabel = prediction.status === 'measured_no_envelope' || prediction.status === 'no_measurements'
    ? 'View tested coverage'
    : 'View the measured performance envelope'

  return (
    <div style={{ marginBottom: 20, padding: '14px 18px', border: '1px solid #d2d2d2', borderLeft: `4px solid ${BORDER_COLOR[detail.color]}`, background: '#fff' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
        <Label color={detail.color} isCompact>{detail.label}</Label>
        {confidence && <span style={{ color: '#3c3f42', fontSize: 12 }}>{confidence}</span>}
      </div>
      <div style={{ color: '#3c3f42', fontSize: 14 }}>{message}</div>
      {prediction.reasons.length > 0 && prediction.status !== 'within_validated_range' && (
        <ul style={{ margin: '8px 0 0', paddingLeft: 18, color: '#3c3f42', fontSize: 13 }}>
          {prediction.reasons.map(reason => <li key={reason}>{reason}</li>)}
        </ul>
      )}
      {href && (
        <Link href={href} style={{ display: 'inline-block', marginTop: 8, color: '#0066cc', fontSize: 13 }}>
          {linkLabel}
        </Link>
      )}
    </div>
  )
}

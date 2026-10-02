'use client'

import { Label } from '@patternfly/react-core'
import Link from 'next/link'
import type { ClassifierPrediction } from '@/lib/tested-models/types'

interface TestedPerformanceFeedbackProps {
  prediction: ClassifierPrediction | null
  isLoading?: boolean
}

const STATUS_DETAILS = {
  within_validated_range: { label: 'Within validated range', color: 'green' as const },
  near_validated_boundary: { label: 'Near validated boundary', color: 'orange' as const },
  outside_validated_range: { label: 'Outside validated range', color: 'red' as const },
  validation_unavailable: { label: 'Validation unavailable', color: 'grey' as const },
}

export function TestedPerformanceFeedback({ prediction, isLoading = false }: TestedPerformanceFeedbackProps) {
  if (!prediction && !isLoading) return null
  if (!prediction && isLoading) {
    return <div style={{ marginBottom: 20, color: '#54585c', fontSize: 13 }}>Checking the tested performance envelope...</div>
  }
  if (!prediction || prediction.status === 'validation_unavailable') return null

  const detail = STATUS_DETAILS[prediction.status] ?? STATUS_DETAILS.validation_unavailable
  if (!STATUS_DETAILS[prediction.status]) return null
  const confidence = prediction.confidence == null ? null : `${Math.round(prediction.confidence * 100)}% confidence`
  const message = prediction.status === 'outside_validated_range'
    ? 'This configuration is outside the measured performance range for this model on this GPU system.'
    : prediction.status === 'near_validated_boundary'
      ? 'This configuration is near a measured performance boundary. Consider a lower concurrency or shorter context.'
      : 'This configuration is inside the measured performance range for this model on this GPU system.'

  return (
    <div style={{ marginBottom: 20, padding: '14px 18px', border: '1px solid #d2d2d2', borderLeft: `4px solid ${detail.color === 'green' ? '#3e8635' : detail.color === 'orange' ? '#f0ab00' : '#c9190b'}`, background: '#fff' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
        <Label color={detail.color} isCompact>{detail.label}</Label>
        {confidence && <span style={{ color: '#54585c', fontSize: 12 }}>{confidence}</span>}
      </div>
      <div style={{ color: '#3c3f42', fontSize: 14 }}>{message}</div>
      {prediction.pair_id && (
        <Link href={`/tested-models?pair=${encodeURIComponent(prediction.pair_id)}`} style={{ display: 'inline-block', marginTop: 8, color: '#0066cc', fontSize: 13 }}>
          View the measured performance envelope
        </Link>
      )}
    </div>
  )
}

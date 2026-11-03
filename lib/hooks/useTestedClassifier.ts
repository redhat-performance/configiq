'use client'

import * as React from 'react'
import type { CanonicalClassifierInput, ClassifierPrediction } from '@/lib/tested-models/types'

export function useTestedClassifier(input: CanonicalClassifierInput | null): {
  prediction: ClassifierPrediction | null
  isLoading: boolean
} {
  const [prediction, setPrediction] = React.useState<ClassifierPrediction | null>(null)
  const [isLoading, setIsLoading] = React.useState(false)
  const serializedInput = input ? JSON.stringify(input) : null

  React.useEffect(() => {
    if (!serializedInput) {
      setPrediction(null)
      setIsLoading(false)
      return
    }
    let cancelled = false
    setIsLoading(true)
    fetch('/api/tested-models/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: serializedInput,
      cache: 'no-store',
    })
      .then(response => response.json() as Promise<ClassifierPrediction>)
      .then(result => {
        if (!cancelled) setPrediction(result)
      })
      .catch(() => {
        if (!cancelled) setPrediction({
          status: 'validation_unavailable',
          confidence: null,
          pair_id: null,
          classifier: null,
          reasons: ['Performance-envelope classification is unavailable.'],
        })
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })
    return () => { cancelled = true }
  }, [serializedInput])

  return { prediction, isLoading }
}

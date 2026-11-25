'use client'

import * as React from 'react'
import type { CanonicalClassifierInput, ClassifierPrediction } from '@/lib/tested-models/types'

interface ClassifiedState {
  /** The serialized input the prediction was produced for. */
  key: string | null
  prediction: ClassifierPrediction | null
}

export function useTestedClassifier(input: CanonicalClassifierInput | null): {
  prediction: ClassifierPrediction | null
  isLoading: boolean
} {
  const [state, setState] = React.useState<ClassifiedState>({ key: null, prediction: null })
  const [isLoading, setIsLoading] = React.useState(false)
  const serializedInput = input ? JSON.stringify(input) : null

  React.useEffect(() => {
    if (!serializedInput) {
      setState({ key: null, prediction: null })
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
        if (!cancelled) setState({ key: serializedInput, prediction: result })
      })
      .catch(() => {
        if (!cancelled) {
          setState({
            key: serializedInput,
            prediction: {
              status: 'validation_unavailable',
              confidence: null,
              pair_id: null,
              classifier: null,
              reasons: ['Performance-envelope classification is unavailable.'],
            },
          })
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })
    return () => { cancelled = true }
  }, [serializedInput])

  // A prediction applies only to the input it was submitted for. During the
  // render after a selection changes, the previous response is still in state
  // but no longer describes the form, so hide it before the new request starts.
  const matches = serializedInput !== null && state.key === serializedInput
  return {
    prediction: matches ? state.prediction : null,
    isLoading: serializedInput !== null && (!matches || isLoading),
  }
}

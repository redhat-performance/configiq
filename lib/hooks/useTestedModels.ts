'use client'

import * as React from 'react'

export interface TestedModelPair {
  id: string
  modelId: string
  systemId: string
  metrics?: unknown
  thresholds?: unknown
}

interface TestedModelsResponse {
  status?: string
  pairs?: TestedModelPair[]
}

export interface TestedModelsCatalog {
  pairs: TestedModelPair[]
  modelIds: string[]
  isLoading: boolean
  isAvailable: boolean
}

export function useTestedModels(): TestedModelsCatalog {
  const [pairs, setPairs] = React.useState<TestedModelPair[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [isAvailable, setIsAvailable] = React.useState(false)

  React.useEffect(() => {
    let cancelled = false
    fetch('/api/tested-models', { cache: 'no-store' })
      .then(response => response.json() as Promise<TestedModelsResponse>)
      .then(data => {
        if (cancelled) return
        const nextPairs = Array.isArray(data.pairs) ? data.pairs : []
        setPairs(nextPairs)
        setIsAvailable(data.status === 'available')
      })
      .catch(() => {
        if (!cancelled) {
          setPairs([])
          setIsAvailable(false)
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })
    return () => { cancelled = true }
  }, [])

  return {
    pairs,
    modelIds: [...new Set(pairs.map(pair => pair.modelId))],
    isLoading,
    isAvailable,
  }
}

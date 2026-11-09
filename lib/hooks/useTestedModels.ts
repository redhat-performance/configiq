'use client'

import * as React from 'react'
import {
  modelStatus as deriveModelStatus,
  validatedModelIds,
  type RegistryDatasetProvenance,
  type RegistryGateResult,
  type RegistryPair,
  type TestedModelStatus,
  type ValidationGateThresholds,
} from '@/lib/tested-models/registry-types'

interface TestedModelsResponse {
  status?: string
  schemaVersion?: number | null
  generatedAt?: string | null
  dataset?: RegistryDatasetProvenance | null
  gate?: ValidationGateThresholds | null
  approvedModels?: string[]
  pairs?: RegistryPair[]
}

export interface TestedModelsCatalog {
  pairs: RegistryPair[]
  /** Models with at least one validated pair. Used for tested-only filters. */
  modelIds: string[]
  validatedModelIds: string[]
  measuredModelIds: string[]
  approvedModels: string[]
  /** Registry-derived status for one model; undefined while the registry loads. */
  modelStatus: (modelId: string) => TestedModelStatus | null
  gate: ValidationGateThresholds | null
  schemaVersion: number | null
  /** Dataset revision the deployed bundle was built from, when recorded. */
  dataset: RegistryDatasetProvenance | null
  isLoading: boolean
  isAvailable: boolean
}

export function useTestedModels(): TestedModelsCatalog {
  const [pairs, setPairs] = React.useState<RegistryPair[]>([])
  const [approvedModels, setApprovedModels] = React.useState<string[]>([])
  const [gate, setGate] = React.useState<ValidationGateThresholds | null>(null)
  const [schemaVersion, setSchemaVersion] = React.useState<number | null>(null)
  const [dataset, setDataset] = React.useState<RegistryDatasetProvenance | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const [isAvailable, setIsAvailable] = React.useState(false)

  React.useEffect(() => {
    let cancelled = false
    fetch('/api/tested-models', { cache: 'no-store' })
      .then(response => response.json() as Promise<TestedModelsResponse>)
      .then(data => {
        if (cancelled) return
        const available = data.status === 'available'
        setPairs(available && Array.isArray(data.pairs) ? data.pairs : [])
        setApprovedModels(available && Array.isArray(data.approvedModels) ? data.approvedModels : [])
        setGate(available && data.gate ? data.gate : null)
        setSchemaVersion(available ? data.schemaVersion ?? null : null)
        setDataset(available && data.dataset ? data.dataset : null)
        setIsAvailable(available)
      })
      .catch(() => {
        if (!cancelled) {
          setPairs([])
          setApprovedModels([])
          setGate(null)
          setSchemaVersion(null)
          setDataset(null)
          setIsAvailable(false)
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })
    return () => { cancelled = true }
  }, [])

  const validatedIds = React.useMemo(() => validatedModelIds({ pairs }), [pairs])
  const measuredIds = React.useMemo(
    () => [...new Set(pairs.map(pair => pair.modelId))],
    [pairs],
  )
  const modelStatus = React.useCallback(
    (modelId: string): TestedModelStatus | null => {
      if (!isAvailable) return null
      return deriveModelStatus(modelId, { pairs, approvedModels })
    },
    [approvedModels, isAvailable, pairs],
  )

  return {
    pairs,
    modelIds: validatedIds,
    validatedModelIds: validatedIds,
    measuredModelIds: measuredIds,
    approvedModels,
    modelStatus,
    gate,
    schemaVersion,
    dataset,
    isLoading,
    isAvailable,
  }
}

export type { RegistryGateResult }

'use client'

import * as React from 'react'
import {
  DEFAULT_COST_ASSUMPTIONS, isValidCostAssumption, parseCostAssumptionOverrides,
  type CostAssumptionKey, type CostAssumptionOverrides, type PlanningCostAssumptions,
} from '@/lib/costing-assumptions'

export const COST_ASSUMPTIONS_STORAGE_KEY = 'configiq_cost_assumptions_v1'

interface CostAssumptionsState {
  assumptions: PlanningCostAssumptions
  overrides: CostAssumptionOverrides
  hydrated: boolean
  storageError: boolean
  updateAssumption: (key: CostAssumptionKey, value: number) => void
  resetAssumptions: () => void
}

const CostAssumptionsContext = React.createContext<CostAssumptionsState | null>(null)

export function CostAssumptionsProvider({ children }: { children: React.ReactNode }) {
  const [overrides, setOverrides] = React.useState<CostAssumptionOverrides>({})
  const [hydrated, setHydrated] = React.useState(false)
  const [storageError, setStorageError] = React.useState(false)

  React.useEffect(() => {
    function readStorage() {
      try {
        const raw = localStorage.getItem(COST_ASSUMPTIONS_STORAGE_KEY)
        setOverrides(parseCostAssumptionOverrides(raw ? JSON.parse(raw) : null))
      } catch {
        setOverrides({})
        setStorageError(true)
      }
    }
    readStorage()
    setHydrated(true)
    const sync = (event: StorageEvent) => {
      if (event.storageArea === localStorage && (event.key === COST_ASSUMPTIONS_STORAGE_KEY || event.key === null)) readStorage()
    }
    window.addEventListener('storage', sync)
    return () => window.removeEventListener('storage', sync)
  }, [])

  React.useEffect(() => {
    if (!hydrated) return
    try {
      localStorage.setItem(COST_ASSUMPTIONS_STORAGE_KEY, JSON.stringify(overrides))
    } catch {
      setStorageError(true)
    }
  }, [hydrated, overrides])

  const updateAssumption = React.useCallback((key: CostAssumptionKey, value: number) => {
    if (isValidCostAssumption(key, value)) setOverrides(previous => ({ ...previous, [key]: value }))
  }, [])
  const resetAssumptions = React.useCallback(() => setOverrides({}), [])
  const value = React.useMemo(() => ({
    assumptions: { ...DEFAULT_COST_ASSUMPTIONS, ...overrides }, overrides, hydrated, storageError,
    updateAssumption, resetAssumptions,
  }), [overrides, hydrated, storageError, updateAssumption, resetAssumptions])

  return <CostAssumptionsContext.Provider value={value}>{children}</CostAssumptionsContext.Provider>
}

export function useCostAssumptions(): CostAssumptionsState {
  const value = React.useContext(CostAssumptionsContext)
  if (!value) throw new Error('useCostAssumptions requires CostAssumptionsProvider')
  return value
}

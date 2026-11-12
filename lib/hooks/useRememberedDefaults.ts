'use client'

import * as React from 'react'
import { useSettings } from '@/contexts/SettingsContext'

/**
 * Persist a validated model as the cross-tool default.
 *
 * Pages call this only after the catalog, tested registry, or Hugging Face
 * config has confirmed the model can be priced. A partially typed or mistyped
 * ID therefore never replaces the saved default.
 */
export function useRememberDefaultModel(model: string, isValid: boolean): void {
  const { hydrated, defaultModel, setDefaultModel } = useSettings()

  React.useEffect(() => {
    if (!hydrated || !isValid || !model || model === defaultModel) return
    setDefaultModel(model)
  }, [hydrated, isValid, model, defaultModel, setDefaultModel])
}

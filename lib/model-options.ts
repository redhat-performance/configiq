import type { ComboBoxItem } from '@/components/ModelComboBox/ModelComboBox'
import { getAppConfig } from '@/lib/app-config'

/**
 * Build the model dropdown items from the three model sources, deduped:
 *   1. AISimulate catalog (from the /models REST endpoint)   → "in catalog" (green)
 *   2. Tested models (config.testedModels)                   → "tested" (blue)
 *   3. Hugging Face models (config.huggingFaceModels)        → "hugging face" (gold)
 *
 * A model can belong to more than one source; the flags are set independently
 * and the badge is chosen by priority (tested > catalog > hugging face) at
 * render time. Order: catalog first, then tested-not-in-catalog, then any
 * remaining HF-listed models.
 */
export function buildModelItems(catalogModels: string[]): ComboBoxItem[] {
  const config = getAppConfig()
  const catalog = new Set(catalogModels)
  const tested = new Set(config.testedModels)
  const hf = new Set(config.huggingFaceModels)

  const seen = new Set<string>()
  const items: ComboBoxItem[] = []
  for (const m of [...catalogModels, ...config.testedModels, ...config.huggingFaceModels]) {
    if (seen.has(m)) continue
    seen.add(m)
    const slash = m.indexOf('/')
    items.push({
      value: m,
      label: m,
      group: slash > 0 ? m.slice(0, slash) : '',
      isTested: tested.has(m),
      inCatalog: catalog.has(m),
      isHuggingFace: hf.has(m),
    })
  }
  return items
}

/**
 * Whether a model needs its config.json fetched from Hugging Face and sent to
 * aisimulators as `model_config`. True for any model the catalog can't resolve
 * on its own — including tested models that live outside the catalog.
 */
export function needsHfConfig(model: string, catalogModels: string[]): boolean {
  return !!model && !catalogModels.includes(model)
}

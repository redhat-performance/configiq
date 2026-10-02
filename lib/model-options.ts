import type { ComboBoxItem } from '@/components/ModelComboBox/ModelComboBox'
import type { ModelSpec } from '@/lib/hooks/useCatalog'
import { getAppConfig } from '@/lib/app-config'
import { modelFacets } from '@/lib/model-metadata'

/**
 * Build the model dropdown items from the three model sources, deduped:
 *   1. AISimulate catalog (from the /models REST endpoint)   → "in catalog" (green)
 *   2. Tested models (live tested-model registry)             → "tested" (blue)
 *   3. Hugging Face models (config.huggingFaceModels)        → "hugging face" (gold)
 *
 * A model can belong to more than one source; the flags are set independently
 * and the badge is chosen by priority (tested > catalog > hugging face) at
 * render time. Order: catalog first, then tested-not-in-catalog, then any
 * remaining HF-listed models.
 *
 * Each item also carries its derived facets (vendor, family, parameter count,
 * quantization, type) so the combo box can search and label on them. `specs`
 * is optional: models outside the catalog have no spec, and every facet
 * degrades to an ID-derived value or null without one.
 */
export function buildModelItems(catalogModels: string[], testedModelIds?: string[], specs?: Map<string, ModelSpec>): ComboBoxItem[] {
  const config = getAppConfig()
  const catalog = new Set(catalogModels)
  const testedValues = testedModelIds ?? []
  const tested = new Set(testedValues)
  const hf = new Set(config.huggingFaceModels)

  const seen = new Set<string>()
  const items: ComboBoxItem[] = []
  for (const m of [...catalogModels, ...testedValues, ...config.huggingFaceModels]) {
    if (seen.has(m)) continue
    seen.add(m)
    const slash = m.indexOf('/')
    items.push({
      ...modelFacets(m, {
        spec: specs?.get(m),
        testedModels: testedValues,
        catalogModels: catalog,
        huggingFaceModels: config.huggingFaceModels,
      }),
      value: m,
      label: m,
      group: slash > 0 ? m.slice(0, slash) : '',
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

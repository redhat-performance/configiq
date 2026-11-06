import type { ComboBoxItem } from '@/components/ModelComboBox/ModelComboBox'
import type { GpuOption, ModelSpec } from '@/lib/hooks/useCatalog'
import { modelFacets } from '@/lib/model-metadata'

/**
 * GPU display fallbacks for systems that the running simulator catalog does
 * not list. Only the label is supplied; specs stay null rather than inventing
 * memory or performance values the catalog never reported.
 */
const FALLBACK_GPU_LABELS: Record<string, string> = {
  mi300x: 'AMD Instinct MI300X',
  mi355x: 'AMD Instinct MI355X',
}

export function gpuSystemLabel(systemId: string, gpuOptions: GpuOption[]): string {
  const option = gpuOptions.find(gpu => gpu.systemId === systemId)
  return option?.label ?? FALLBACK_GPU_LABELS[systemId] ?? systemId
}

/** Catalog options for exactly these systems, preserving the caller's order. */
export function gpuOptionsForSystems(systemIds: string[], gpuOptions: GpuOption[]): GpuOption[] {
  return systemIds.map(systemId => {
    const option = gpuOptions.find(gpu => gpu.systemId === systemId)
    if (option) return option
    return {
      systemId,
      label: gpuSystemLabel(systemId, gpuOptions),
      vendor: null,
      architecture: null,
      vramGb: null,
      bandwidthTbps: null,
      tflopsBf16: null,
      tdpWatts: null,
      gpusPerNode: null,
    }
  })
}

export function gpuSystemSpecs(systemId: string, gpuOptions: GpuOption[]): string {
  const option = gpuOptions.find(gpu => gpu.systemId === systemId)
  if (!option) return ''
  const parts: string[] = []
  if (option.vramGb !== null) parts.push(`${option.vramGb} GB`)
  if (option.bandwidthTbps !== null) parts.push(`${option.bandwidthTbps} TB/s`)
  if (option.tflopsBf16 !== null) parts.push(`${Math.round(option.tflopsBf16)} TFLOPS BF16`)
  return parts.join(' · ')
}

/** Compact parameter count: 0.6B, 32B, 2.4T. */
function sizeChip(paramsB: number | null): string | null {
  if (paramsB === null) return null
  if (paramsB >= 1_000) return `${Number((paramsB / 1_000).toFixed(2))}T`
  if (paramsB >= 0.1) return `${Number(paramsB.toFixed(1))}B`
  return `${Math.round(paramsB * 1_000)}M`
}

/** Compact context window: 128K, 1M. */
function contextChip(contextLength: number | null): string | null {
  if (!contextLength) return null
  if (contextLength >= 1_048_576) return `${Number((contextLength / 1_048_576).toFixed(1))}M ctx`
  if (contextLength >= 1_024) return `${Math.round(contextLength / 1_024)}K ctx`
  return `${contextLength} ctx`
}

/** Human-readable facts about a model, matching the ComboBox row chips. */
export function modelFacetChips(modelId: string, spec?: ModelSpec): string[] {
  const facets = modelFacets(modelId, { spec })
  const chips: string[] = []
  const size = sizeChip(facets.paramsB)
  if (size) {
    chips.push(facets.activeParamsB !== null
      ? `${size} (${sizeChip(facets.activeParamsB)} active)`
      : size)
  }
  if (facets.types.includes('moe')) chips.push('MoE')
  if (facets.types.includes('vision')) chips.push('Vision')
  const context = contextChip(facets.contextLength)
  if (context) chips.push(context)
  if (facets.quantization) chips.push(facets.quantization)
  return chips
}

/**
 * Model options for the tested-model filter. Only registry models are listed;
 * catalog metadata is used for facets and provenance badges, not to add
 * unrelated models to the filter.
 */
export function registryModelItems(
  modelIds: string[],
  specs: Map<string, ModelSpec>,
  validatedModels: string[],
  catalogModels: string[] = [],
): ComboBoxItem[] {
  return [...new Set(modelIds)].sort().map(modelId => {
    const slash = modelId.indexOf('/')
    return {
      ...modelFacets(modelId, {
        spec: specs.get(modelId),
        testedModels: validatedModels,
        catalogModels,
      }),
      value: modelId,
      label: modelId,
      group: slash > 0 ? modelId.slice(0, slash) : '',
    }
  })
}

export function shortRevision(revision: string): string {
  return revision.slice(0, 12)
}

import type { GpuOption } from '@/lib/hooks/useCatalog'
import { acceleratorNamesMatch } from './benchmark-resolve'

/**
 * Reconcile measured accelerator labels with the catalog's GPU systems.
 *
 * A benchmark dataset names the part it measured (`H200`), while the catalog
 * names the system it prices (`NVIDIA H200`, id `h200_sxm`). Pages that show
 * measured GPUs through a catalog-backed control need to line the two up
 * without inventing a part the catalog does not contain.
 */

/**
 * The catalog GPU a measured accelerator label names, or null.
 *
 * An exact label match wins, because the catalog's own spelling is the one
 * that should survive a round trip. Otherwise the token-bounded comparison
 * lets `H200` find `NVIDIA H200` without letting `B200` find `GB200`.
 */
export function findCatalogGpu(accelerator: string, gpus: readonly GpuOption[]): GpuOption | null {
  const normalized = accelerator.trim().toLocaleLowerCase()
  if (!normalized) return null
  return gpus.find(gpu => gpu.label.toLocaleLowerCase() === normalized)
    ?? gpus.find(gpu => acceleratorNamesMatch(gpu.label, accelerator))
    ?? null
}

/**
 * The measured accelerator label that best answers a requested one.
 *
 * Prefers the same part spelled the same way, then the same part named
 * differently. Falls back to the first available label so a caller seeding a
 * default always has something valid to show.
 */
export function matchAccelerator(options: readonly string[], requested: string): string {
  const normalized = requested.trim().toLocaleLowerCase()
  if (!normalized) return options[0] ?? ''
  return options.find(option => option.toLocaleLowerCase() === normalized)
    ?? options.find(option => acceleratorNamesMatch(option, requested))
    ?? options[0]
    ?? ''
}

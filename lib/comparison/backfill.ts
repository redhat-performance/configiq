import { AMORT_MONTHS_5YR } from '../utils/format'
import { resolveCloudRate, type CloudRates, type HardwareCost } from '../hooks/useCostings'
import { comparisonCost, type CostFigures } from './cost'
import type { ComparisonDocument, ComparisonEntry, ComparisonFigures } from './types'

/**
 * Cost figures a saved snapshot can be completed from prices after the fact.
 *
 * A result page records what it could see at save time. When costings were off,
 * or the price catalogue had not arrived yet, the four money figures stayed
 * null and the saved result looked free. This fills only those gaps from the
 * current catalogue; a figure the page actually computed is never replaced.
 */

export interface ComparisonCostSources {
  gpuCloudRates: Map<string, Record<string, CloudRates>>
  gpuHardwareCosts: Map<string, HardwareCost>
  /** Provider the reader prefers, as the result pages resolve their rate. */
  preferredProvider?: string | null
}

export interface ComparisonBackfill {
  document: ComparisonDocument
  /** Entries that gained at least one figure; zero when nothing was missing. */
  filled: number
}

const COST_FIELDS = [
  'cloudMonthlyUsd',
  'cloudFiveYearUsd',
  'selfHostedMonthlyUsd',
  'selfHostedFiveYearUsd',
] as const satisfies readonly (keyof ComparisonFigures)[]

/** True when a snapshot is missing at least one money figure. */
export function missingCost(entry: ComparisonEntry): boolean {
  return COST_FIELDS.some(field => entry.figures[field] == null)
}

function entryCost(entry: ComparisonEntry, sources: ComparisonCostSources): CostFigures {
  const rate = resolveCloudRate(
    sources.gpuCloudRates.get(entry.system),
    sources.preferredProvider ?? null,
  )
  return comparisonCost({
    gpus: entry.figures.gpus,
    cloudRatePerHour: rate?.rate ?? null,
    hardwareCostUsd: sources.gpuHardwareCosts.get(entry.system)?.new_usd ?? null,
  })
}

function filledFigures(
  entry: ComparisonEntry,
  sources: ComparisonCostSources,
): ComparisonFigures {
  const computed = entryCost(entry, sources)
  const figures: ComparisonFigures = { ...entry.figures }

  // Each pair is one quantity in two windows. When one side is already saved,
  // the other follows it rather than the catalogue, so a row cannot show a
  // monthly figure and a five-year total that disagree.
  figures.cloudMonthlyUsd = figures.cloudMonthlyUsd
    ?? (figures.cloudFiveYearUsd == null
      ? computed.cloudMonthlyUsd
      : figures.cloudFiveYearUsd / AMORT_MONTHS_5YR)
  figures.cloudFiveYearUsd = figures.cloudFiveYearUsd
    ?? (figures.cloudMonthlyUsd == null ? null : figures.cloudMonthlyUsd * AMORT_MONTHS_5YR)

  figures.selfHostedFiveYearUsd = figures.selfHostedFiveYearUsd
    ?? (figures.selfHostedMonthlyUsd == null
      ? computed.selfHostedFiveYearUsd
      : figures.selfHostedMonthlyUsd * AMORT_MONTHS_5YR)
  figures.selfHostedMonthlyUsd = figures.selfHostedMonthlyUsd
    ?? (figures.selfHostedFiveYearUsd == null
      ? null
      : figures.selfHostedFiveYearUsd / AMORT_MONTHS_5YR)

  return figures
}

/**
 * Complete missing cost figures from the price catalogue.
 *
 * The returned document is a new object only when something changed, so a
 * caller can persist it without writing on every render. An entry with no
 * catalogue price stays null rather than being given a zero.
 */
export function backfillComparisonCosts(
  document: ComparisonDocument,
  sources: ComparisonCostSources,
): ComparisonBackfill {
  let filled = 0
  const entries = document.entries.map(entry => {
    if (!missingCost(entry)) return entry
    const figures = filledFigures(entry, sources)
    if (COST_FIELDS.every(field => figures[field] === entry.figures[field])) return entry
    filled += 1
    return { ...entry, figures }
  })

  return {
    document: filled === 0 ? document : { ...document, entries },
    filled,
  }
}

/**
 * Copy completed cost figures onto the set that is in storage now.
 *
 * The set this page loaded can be out of date by the time prices arrive: another
 * tab may have saved or removed a result. Writing the loaded set back would
 * restore what was removed or drop what was added, so the filled figures are
 * matched to the current entries by id and everything else is left as stored.
 * Entries that appeared after the load are untouched and gain their own figures
 * on the next pass.
 */
export function mergeCostFigures(
  current: ComparisonDocument,
  filled: ComparisonDocument,
): ComparisonDocument {
  const completed = new Map(filled.entries.map(entry => [entry.id, entry]))
  let changed = false

  const entries = current.entries.map(entry => {
    const source = completed.get(entry.id)
    if (source == null) return entry
    const figures: ComparisonFigures = { ...entry.figures }
    for (const field of COST_FIELDS) {
      if (figures[field] == null && source.figures[field] != null) {
        figures[field] = source.figures[field]
      }
    }
    if (COST_FIELDS.every(field => figures[field] === entry.figures[field])) return entry
    changed = true
    return { ...entry, figures }
  })

  return changed ? { ...current, entries } : current
}

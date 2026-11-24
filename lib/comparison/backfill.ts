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
  figures.cloudMonthlyUsd = figures.cloudMonthlyUsd ?? computed.cloudMonthlyUsd
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

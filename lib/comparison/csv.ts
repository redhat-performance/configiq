import { COMPARISON_METRICS } from './metrics'
import type { ComparisonDocument, ComparisonEntry } from './types'

/**
 * CSV export: one metric per row, one saved result per column.
 *
 * Raw figures are written rather than the table's formatted strings, so a
 * spreadsheet can compute with them. Labels come from the same metric list the
 * table renders, so the export and the screen cannot disagree about a unit.
 */

/** Identity columns that precede the metrics. */
const IDENTITY_ROWS: readonly { label: string; value: (entry: ComparisonEntry) => string }[] = [
  { label: 'Saved at', value: entry => entry.savedAt },
  { label: 'Link', value: entry => entry.link },
]

/**
 * A leading one of these makes a spreadsheet treat the cell as a formula.
 * A name or link can come from an imported file, so text that starts with one
 * is prefixed with an apostrophe, which spreadsheets read as literal text.
 * Numbers are written unaltered so a negative figure stays numeric.
 */
const FORMULA_LEADING = /^[=+\-@\t\r\n]/

function csvCell(value: string | number | null): string {
  if (value == null) return ''
  if (typeof value === 'number') return String(value)
  const text = FORMULA_LEADING.test(value) ? `'${value}` : value
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

/** Render the comparison set as CSV text, with a trailing newline. */
export function comparisonCsv(document: ComparisonDocument): string {
  const header = ['Metric', ...document.entries.map(entry => entry.name)]
  const lines = [header.map(csvCell).join(',')]

  for (const row of IDENTITY_ROWS) {
    lines.push([row.label, ...document.entries.map(entry => row.value(entry))].map(csvCell).join(','))
  }

  for (const metric of COMPARISON_METRICS) {
    lines.push(
      [metric.label, ...document.entries.map(entry => metric.value(entry))]
        .map(csvCell)
        .join(','),
    )
  }

  return `${lines.join('\n')}\n`
}

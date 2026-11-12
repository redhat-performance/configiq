import {
  UNREPORTED_VERSION,
  backendVersionsMatch,
  benchmarkFilterValue,
  benchmarkFilterValues,
  filterRows,
  normalizeBackendVersion,
} from './benchmark-dataset'
import {
  BENCHMARK_FILTER_KEYS,
  type BenchmarkFilterKey,
  type BenchmarkFilters,
  type DatasetRow,
} from './benchmark-types'

/**
 * A filter value the server replaced so the selection stays renderable.
 *
 * The comparison page shows these instead of clearing sibling inputs, because
 * a cleared input forces the user to rebuild a valid selection by hand.
 */
export interface BenchmarkAdjustment {
  field: BenchmarkFilterKey
  from: string
  to: string
  reason: string
}

/**
 * A protected choice that the benchmark data cannot satisfy.
 *
 * Model, backend, and version are never rewritten: the model is the user's
 * subject, and backend/version are global Settings. The page renders an empty
 * state that points back to the owning control instead.
 */
export interface BenchmarkUnavailable {
  field: BenchmarkFilterKey
  message: string
}

export interface BenchmarkResolution {
  filters: BenchmarkFilters
  adjustments: BenchmarkAdjustment[]
  unavailable: BenchmarkUnavailable | null
  /**
   * Set when the caller left the version blank, which means "the newest
   * benchmarked release for this backend". Not a repair: the caller asked for
   * the benchmark's own default rather than a pin.
   */
  versionFallback: { value: string; message: string } | null
}

/** Filter keys the resolver may repair; the others are never rewritten. */
const REPAIRABLE_KEYS = ['accelerator', 'isl', 'osl', 'tp'] as const

function numeric(value: string): number | null {
  const parsed = Number(value)
  return value.trim() !== '' && Number.isFinite(parsed) ? parsed : null
}

/** Pick the measured value closest to `value`, preferring an exact match. */
export function nearestMeasuredValue(options: string[], value: string): string | null {
  if (options.length === 0) return null
  if (options.includes(value)) return value
  const target = numeric(value)
  if (target === null) return options[0]
  let best = options[0]
  let bestDistance = Infinity
  for (const option of options) {
    const candidate = numeric(option)
    if (candidate === null) continue
    const distance = Math.abs(candidate - target)
    // A tie keeps the larger measured value so the selection moves toward the
    // next supported step instead of dropping below what was asked for.
    if (distance < bestDistance || (distance === bestDistance && candidate > (numeric(best) ?? candidate))) {
      best = option
      bestDistance = distance
    }
  }
  return best
}

/** True when `char` does not continue an accelerator name token. */
function isNameBoundary(char: string): boolean {
  return char === '' || !/[a-z0-9]/.test(char)
}

function containsNameToken(haystack: string, needle: string): boolean {
  let index = haystack.indexOf(needle)
  while (index !== -1) {
    const before = index === 0 ? '' : haystack[index - 1]
    const after = index + needle.length >= haystack.length ? '' : haystack[index + needle.length]
    if (isNameBoundary(before) && isNameBoundary(after)) return true
    index = haystack.indexOf(needle, index + 1)
  }
  return false
}

/**
 * Whether two accelerator labels name the same part.
 *
 * Matching is token-bounded so a short name such as `B200` cannot match
 * `GB200`, while display prefixes such as `NVIDIA H200` still match `H200`.
 */
export function acceleratorNamesMatch(left: string, right: string): boolean {
  const leftName = left.trim().toLocaleLowerCase()
  const rightName = right.trim().toLocaleLowerCase()
  if (!leftName || !rightName) return false
  return containsNameToken(leftName, rightName) || containsNameToken(rightName, leftName)
}

/**
 * Pick a measured accelerator that can stand in for the requested one.
 *
 * Prefer the same part named differently (`H200` vs `NVIDIA H200`), then the
 * accelerator with the most benchmark rows for this model, so the repaired
 * selection still shows a useful curve.
 */
function nearestAccelerator(rows: DatasetRow[], requested: string): string | null {
  const options = benchmarkFilterValues(rows, 'accelerator')
  if (options.length === 0) return null
  const normalized = requested.trim().toLowerCase()
  const direct = options.find(option => option.toLowerCase() === normalized)
  if (direct) return direct
  const equivalent = options.find(option => acceleratorNamesMatch(option, requested))
  if (equivalent) return equivalent
  const counts = new Map<string, number>()
  for (const row of rows) {
    const value = benchmarkFilterValue(row, 'accelerator')
    if (value) counts.set(value, (counts.get(value) ?? 0) + 1)
  }
  return [...options].sort((left, right) => (counts.get(right) ?? 0) - (counts.get(left) ?? 0))[0]
}

/** Newest measured release in `values`, ignoring rows that named none. */
function newestMeasuredVersion(values: string[]): string | null {
  const measured = values.filter(value => value && value !== UNREPORTED_VERSION)
  if (measured.length === 0) return null
  return [...measured].sort((left, right) =>
    right.localeCompare(left, undefined, { numeric: true, sensitivity: 'base' }),
  )[0]
}

function repairReason(field: BenchmarkFilterKey, from: string, to: string, subject: string): string {
  switch (field) {
    case 'accelerator':
      return `${subject} has no benchmark data on ${from}; using ${to}, which it does have.`
    case 'isl':
      return `No benchmark rows at ${from} input tokens; using the nearest measured length, ${to}.`
    case 'osl':
      return `No benchmark rows at ${from} output tokens; using the nearest measured length, ${to}.`
    case 'tp':
      return `No benchmark rows at TP ${from}; using the nearest measured size, TP ${to}.`
    default:
      return `${from} is not measured; using ${to}.`
  }
}

/**
 * Resolve a requested benchmark selection against the loaded rows.
 *
 * Model, backend, and backend version are validated but never rewritten.
 * Accelerator and numeric workload dimensions are repaired to the nearest
 * measured value so changing one input never clears the rest.
 */
export function resolveBenchmarkFilters(rows: DatasetRow[], requested: BenchmarkFilters): BenchmarkResolution {
  const filters: BenchmarkFilters = { ...requested }
  const adjustments: BenchmarkAdjustment[] = []
  let versionFallback: BenchmarkResolution['versionFallback'] = null

  if (!filters.model) return { filters, adjustments, unavailable: null, versionFallback }

  const rowsForModel = rows.filter(row => benchmarkFilterValue(row, 'model') === filters.model)
  if (rowsForModel.length === 0) {
    return {
      filters,
      adjustments,
      versionFallback,
      unavailable: {
        field: 'model',
        message: `${filters.model} has no benchmark rows in this dataset. Choose another model.`,
      },
    }
  }

  // Settings offers catalog literals (`0.31.0`), while benchmark rows keep the
  // release label measured (`v0.31.0`). Canonicalize the pin once so an exact
  // user choice is not reported as missing.
  if (filters.backend && filters.backendVersion) {
    filters.backendVersion = normalizeBackendVersion(filters.backend, filters.backendVersion)
  }

  // Repair the GPU against rows that satisfy the Settings backend/release pin.
  // Otherwise a repair can land on a GPU that lacks the requested release and
  // produce an unavailable state even though another GPU measures it.
  const rowsForPin = rowsForModel.filter(row => {
    if (filters.backend && benchmarkFilterValue(row, 'backend') !== filters.backend) return false
    if (filters.backendVersion
      && !backendVersionsMatch(filters.backendVersion, benchmarkFilterValue(row, 'backendVersion'))) return false
    return true
  })
  const repairPool = rowsForPin.length > 0 ? rowsForPin : rowsForModel

  if (filters.accelerator) {
    const repaired = nearestAccelerator(repairPool, filters.accelerator)
    if (repaired && repaired !== filters.accelerator) {
      const pin = filters.backendVersion
        ? `${filters.backend ? `${filters.backend} ` : ''}${filters.backendVersion}`
        : filters.backend
      adjustments.push({
        field: 'accelerator',
        from: filters.accelerator,
        to: repaired,
        reason: pin
          ? `${filters.model} has no ${pin} benchmark rows on ${filters.accelerator}; using ${repaired}, which does.`
          : repairReason('accelerator', filters.accelerator, repaired, filters.model),
      })
      filters.accelerator = repaired
    }
  }

  // Backend and version come from Settings. They are matched exactly and are
  // never substituted, so a mismatch is reported rather than repaired.
  const rowsForSystem = filterRows(rowsForModel, { accelerator: filters.accelerator })
  const rowsForBackend = filters.backend
    ? filterRows(rowsForSystem, { backend: filters.backend })
    : rowsForSystem
  if (filters.backend) {
    const backendValues = benchmarkFilterValues(rowsForSystem, 'backend')
    if (backendValues.length > 0 && !backendValues.includes(filters.backend)) {
      const measured = backendValues.join(', ')
      return {
        filters,
        adjustments,
        versionFallback,
        unavailable: {
          field: 'backend',
          message: `No benchmark rows use ${filters.backend} on ${filters.accelerator || 'this GPU'}. `
            + `The benchmark covers ${measured}. Change the serving backend in Settings.`,
        },
      }
    }

    if (!filters.backendVersion) {
      // A blank version means "the benchmark's newest release for this
      // backend", which is a documented default rather than a repair.
      const versionValues = benchmarkFilterValues(rowsForBackend, 'backendVersion')
      const newest = newestMeasuredVersion(versionValues)
      const fallback = newest ?? (versionValues.includes(UNREPORTED_VERSION) ? UNREPORTED_VERSION : null)
      if (fallback) {
        versionFallback = {
          value: fallback,
          message: newest
            ? `Settings leaves the version at the simulator default, so the comparison uses ${newest}, `
              + `the newest release measured for ${filters.backend} on ${filters.accelerator || 'this GPU'}.`
            : `The benchmark did not report a release for ${filters.backend} on `
              + `${filters.accelerator || 'this GPU'}; version parity cannot be confirmed.`,
        }
        filters.backendVersion = fallback
      }
    }
  }

  if (filters.backendVersion) {
    const versionValues = benchmarkFilterValues(rowsForBackend, 'backendVersion')
    const requestedVersion = filters.backendVersion
    if (versionValues.length > 0
      && !versionValues.some(value => backendVersionsMatch(requestedVersion, value))) {
      return {
        filters,
        adjustments,
        versionFallback,
        unavailable: {
          field: 'backendVersion',
          message: `No benchmark rows use ${filters.backend || 'this backend'} ${filters.backendVersion} on `
            + `${filters.accelerator || 'this GPU'}. The benchmark covers ${versionValues.join(', ')}. `
            + 'Clear or change the version in Settings.',
        },
      }
    }
  }

  // Repair each workload dimension in order. The candidate pool starts from the
  // protected choices and is narrowed after each repair, so an invalid sibling
  // value cannot hide the rows that would have repaired this one.
  let pool = rowsForBackend
  if (filters.backendVersion) pool = filterRows(pool, { backendVersion: filters.backendVersion })
  for (const key of REPAIRABLE_KEYS) {
    if (key === 'accelerator' || !filters[key]) continue
    const repaired = nearestMeasuredValue(benchmarkFilterValues(pool, key), filters[key])
    if (!repaired) continue
    if (repaired !== filters[key]) {
      adjustments.push({
        field: key,
        from: filters[key],
        to: repaired,
        reason: repairReason(key, filters[key], repaired, filters.model),
      })
      filters[key] = repaired
    }
    pool = filterRows(pool, { [key]: repaired })
  }

  return { filters, adjustments, unavailable: null, versionFallback }
}

/**
 * Options the comparison page renders.
 *
 * Models always list the whole dataset so the model is never narrowed away by
 * a stale GPU. Accelerators list every system measured for the chosen model.
 * The remaining keys list what the resolved selection leaves available.
 */
export function benchmarkOptions(
  rows: DatasetRow[],
  filters: BenchmarkFilters,
  allOptions: Record<BenchmarkFilterKey, string[]>,
): Record<BenchmarkFilterKey, string[]> {
  const rowsForModel = filters.model
    ? rows.filter(row => benchmarkFilterValue(row, 'model') === filters.model)
    : rows
  const options = { ...allOptions }
  options.model = benchmarkFilterValues(rows, 'model')
  options.accelerator = benchmarkFilterValues(rowsForModel, 'accelerator')
  for (const key of BENCHMARK_FILTER_KEYS) {
    if (!Array.isArray(options[key])) options[key] = []
  }
  return options
}

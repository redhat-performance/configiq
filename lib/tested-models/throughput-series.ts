/**
 * Selecting and comparing measured throughput curves.
 *
 * A dashboard payload contains one curve per measured sweep: a fixed workload
 * (ISL/OSL/turn count), tensor-parallel width, and run configuration. The chart
 * draws one line per TP, so these helpers pick a comparable sweep across TP
 * widths and match a prediction snapshot to the measured points.
 *
 * Everything here is pure. The page owns fetching, rendering, and state.
 */

import { normalizeBackendVersion } from './feature-vector'

export interface ThroughputPoint {
  concurrency: number
  measuredConcurrency: number | null
  throughput: number
  throughputMin: number
  throughputMax: number
  latency: number | null
  observations: number
}

export interface ThroughputCurve {
  isl: number
  osl: number
  turns: number
  tp: number
  sweepId: string
  backend: string | null
  backendVersions: string[]
  configLabel: string
  points: ThroughputPoint[]
}

export interface Workload {
  isl: number
  osl: number
  turns: number
}

export interface WorkloadOption extends Workload {
  key: string
  label: string
  levels: number
  observations: number
  tps: number[]
}

export interface SeriesLine {
  tp: number
  curve: ThroughputCurve
}

export interface SelectedSeries {
  lines: SeriesLine[]
  primarySweepId: string | null
  /** TPs whose line could not use the primary sweep, so it is not like-for-like. */
  mixedConfigurations: number[]
}

export interface PredictionSnapshot {
  isl: number
  osl: number
  concurrency: number
  tp: number
  throughput: number
  status: string | null
  confidence: number | null
  backend: string | null
  backendVersion: string | null
  servingMode: 'agg' | 'disagg' | null
}

export interface InterpolatedPoint {
  /** Measured (or linearly interpolated) throughput at the predicted concurrency. */
  measured: number
  below: ThroughputPoint
  above: ThroughputPoint
  /** True when the predicted concurrency is exactly a measured level. */
  exact: boolean
  /** Set when the predicted concurrency lies outside the measured range. */
  clamped: 'below' | 'above' | null
}

export interface RangeCheck {
  id: string
  label: string
  ok: boolean
  detail: string
}

export interface PredictionComparison {
  workloadMeasured: boolean
  tpMeasured: boolean
  curve: ThroughputCurve | null
  at: InterpolatedPoint | null
  deltaPct: number | null
  observedMin: number | null
  observedMax: number | null
  checks: RangeCheck[]
}

export function workloadKey(workload: Workload): string {
  return `${workload.isl}|${workload.osl}|${workload.turns}`
}

export function turnsLabel(turns: number): string {
  return turns > 1 ? `${turns}-turn` : 'single-turn'
}

export function workloadLabel(workload: Workload): string {
  return `ISL ${workload.isl.toLocaleString()} · OSL ${workload.osl.toLocaleString()} · ${turnsLabel(workload.turns)}`
}

export function isSameWorkload(left: Workload, right: Workload): boolean {
  return left.isl === right.isl && left.osl === right.osl && left.turns === right.turns
}

const workloadOf = (curve: ThroughputCurve): Workload => ({
  isl: curve.isl,
  osl: curve.osl,
  turns: curve.turns,
})

function observationsOf(curve: ThroughputCurve): number {
  return curve.points.reduce((total, point) => total + point.observations, 0)
}

/**
 * Distinct measured workloads, with the coverage that the selector shows.
 *
 * Single-turn workloads sort first because a single-request prediction is only
 * comparable to one; multi-turn runs stay selectable and labelled.
 */
export function listWorkloads(curves: readonly ThroughputCurve[]): WorkloadOption[] {
  const grouped = new Map<string, {
    workload: Workload
    levels: Set<number>
    observations: number
    tps: Set<number>
  }>()
  for (const curve of curves) {
    const key = workloadKey(curve)
    const entry = grouped.get(key) ?? {
      workload: workloadOf(curve),
      levels: new Set<number>(),
      observations: 0,
      tps: new Set<number>(),
    }
    for (const point of curve.points) entry.levels.add(point.concurrency)
    entry.observations += observationsOf(curve)
    entry.tps.add(curve.tp)
    grouped.set(key, entry)
  }
  return [...grouped.entries()]
    .map(([key, entry]) => ({
      ...entry.workload,
      key,
      label: workloadLabel(entry.workload),
      levels: entry.levels.size,
      observations: entry.observations,
      tps: [...entry.tps].sort((left, right) => left - right),
    }))
    .sort((left, right) => (
      left.turns - right.turns
      || left.isl - right.isl
      || left.osl - right.osl
    ))
}

/**
 * The workload the chart opens on.
 *
 * Prefer a single-turn run, then the workload with the most concurrency levels
 * (the most informative curve), then the most observations. The final ISL/OSL
 * tie-breaks keep the choice stable across payload rebuilds.
 */
export function selectDefaultWorkload(curves: readonly ThroughputCurve[]): WorkloadOption | null {
  const options = listWorkloads(curves)
  if (options.length === 0) return null
  return [...options].sort((left, right) => (
    Number(right.turns === 1) - Number(left.turns === 1)
    || right.levels - left.levels
    || right.observations - left.observations
    || left.isl - right.isl
    || left.osl - right.osl
  ))[0]
}

interface SweepSummary {
  id: string
  tps: Set<number>
  levels: number
  observations: number
}

/**
 * One line per TP for the selected workload.
 *
 * The primary sweep is the run configuration that covers the most TPs, so the
 * lines stay comparable. When a TP was not measured under that configuration,
 * its best own sweep is used and reported in `mixedConfigurations` rather than
 * silently drawn as if it were the same run.
 */
export function selectSeries(
  curves: readonly ThroughputCurve[],
  workload: Workload,
): SelectedSeries {
  const matching = curves.filter(curve => isSameWorkload(workloadOf(curve), workload))
  if (matching.length === 0) {
    return { lines: [], primarySweepId: null, mixedConfigurations: [] }
  }

  const sweeps = new Map<string, SweepSummary>()
  for (const curve of matching) {
    const entry = sweeps.get(curve.sweepId) ?? {
      id: curve.sweepId,
      tps: new Set<number>(),
      levels: 0,
      observations: 0,
    }
    entry.tps.add(curve.tp)
    entry.levels = Math.max(entry.levels, curve.points.length)
    entry.observations += observationsOf(curve)
    sweeps.set(curve.sweepId, entry)
  }

  const primary = [...sweeps.values()].sort((left, right) => (
    right.tps.size - left.tps.size
    || right.levels - left.levels
    || right.observations - left.observations
    || left.id.localeCompare(right.id)
  ))[0]

  const lines: SeriesLine[] = []
  const mixedConfigurations: number[] = []
  const tps = [...new Set(matching.map(curve => curve.tp))].sort((left, right) => left - right)
  for (const tp of tps) {
    const candidates = matching.filter(curve => curve.tp === tp)
    const preferred = candidates.find(curve => curve.sweepId === primary.id)
    if (preferred) {
      lines.push({ tp, curve: preferred })
      continue
    }
    const fallback = [...candidates].sort((left, right) => (
      right.points.length - left.points.length
      || observationsOf(right) - observationsOf(left)
      || left.sweepId.localeCompare(right.sweepId)
    ))[0]
    if (fallback) {
      lines.push({ tp, curve: fallback })
      mixedConfigurations.push(tp)
    }
  }

  return { lines, primarySweepId: primary.id, mixedConfigurations }
}

/**
 * The measured workload a prediction can be compared with.
 *
 * Only a single-turn run describes one request at the predicted ISL/OSL.
 * Multi-turn aggregates are a different measurement and are never substituted.
 */
export function matchPredictionWorkload(
  curves: readonly ThroughputCurve[],
  prediction: Pick<PredictionSnapshot, 'isl' | 'osl'>,
): WorkloadOption | null {
  return listWorkloads(curves).find(option => (
    option.turns === 1 && option.isl === prediction.isl && option.osl === prediction.osl
  )) ?? null
}

/** Linear interpolation between the bracketing measured concurrency levels. */
export function interpolateThroughput(
  curve: ThroughputCurve,
  concurrency: number,
): InterpolatedPoint | null {
  const points = [...curve.points].sort((left, right) => left.concurrency - right.concurrency)
  if (points.length === 0) return null
  // An exact measured level is not an extrapolation, including the endpoints.
  const exact = points.find(point => point.concurrency === concurrency)
  if (exact) {
    return { measured: exact.throughput, below: exact, above: exact, exact: true, clamped: null }
  }
  const first = points[0]
  const last = points[points.length - 1]
  if (concurrency < first.concurrency) {
    return { measured: first.throughput, below: first, above: first, exact: false, clamped: 'below' }
  }
  if (concurrency > last.concurrency) {
    return { measured: last.throughput, below: last, above: last, exact: false, clamped: 'above' }
  }
  for (let index = 1; index < points.length; index += 1) {
    const below = points[index - 1]
    const above = points[index]
    if (concurrency <= above.concurrency) {
      const span = above.concurrency - below.concurrency
      const fraction = span === 0 ? 0 : (concurrency - below.concurrency) / span
      return {
        measured: below.throughput + fraction * (above.throughput - below.throughput),
        below,
        above,
        exact: false,
        clamped: null,
      }
    }
  }
  return { measured: last.throughput, below: last, above: last, exact: false, clamped: null }
}

function formatNumber(value: number): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: 0 })
}

function formatPercent(value: number): string {
  const rounded = Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(1)
  return `${value > 0 ? '+' : ''}${rounded}%`
}

/**
 * Explain where a prediction sits relative to the measured curve.
 *
 * `series` is the chart's current selection, so the comparison always describes
 * the lines the reader is looking at. A missing TP or workload is reported, not
 * papered over with a neighbouring line.
 */
export function comparePredictionToMeasured(
  series: SelectedSeries,
  prediction: PredictionSnapshot,
): PredictionComparison {
  const line = series.lines.find(candidate => candidate.tp === prediction.tp)
  const curve = line?.curve ?? null
  if (!curve) {
    return {
      workloadMeasured: series.lines.length > 0,
      tpMeasured: false,
      curve: null,
      at: null,
      deltaPct: null,
      observedMin: null,
      observedMax: null,
      checks: [{
        id: 'tp',
        label: `Tensor parallel ${prediction.tp}`,
        ok: false,
        detail: series.lines.length > 0
          ? `This workload was measured at TP ${series.lines.map(item => item.tp).join(', ')}, not TP ${prediction.tp}.`
          : 'This workload was not measured for this model and GPU system.',
      }],
    }
  }

  const at = interpolateThroughput(curve, prediction.concurrency)
  const observedMin = Math.min(...curve.points.map(point => point.throughputMin))
  const observedMax = Math.max(...curve.points.map(point => point.throughputMax))
  const concurrencyLevels = curve.points.map(point => point.concurrency)
  const minConcurrency = Math.min(...concurrencyLevels)
  const maxConcurrency = Math.max(...concurrencyLevels)
  const deltaPct = at && at.measured > 0
    ? ((prediction.throughput - at.measured) / at.measured) * 100
    : null

  const predictedFamily = prediction.backend || prediction.backendVersion
    ? normalizeBackendVersion(prediction.backend, prediction.backendVersion)
    : 'unknown'
  const measuredFamilies = new Set(
    curve.backendVersions.map(version => normalizeBackendVersion(curve.backend, version)),
  )
  const versionMeasured = measuredFamilies.has(predictedFamily)
  const tokenBudget = (prediction.isl + prediction.osl) * Math.max(prediction.concurrency, 1)
  const measuredTokenBudgets = curve.points.map(
    point => (prediction.isl + prediction.osl) * Math.max(point.concurrency, 1),
  )
  const minTokenBudget = Math.min(...measuredTokenBudgets)
  const maxTokenBudget = Math.max(...measuredTokenBudgets)

  const checks: RangeCheck[] = [
    {
      id: 'tp',
      label: `Tensor parallel ${prediction.tp}`,
      ok: true,
      detail: `Measured at TP ${prediction.tp} under ${curve.backendVersions.join(', ') || 'an unlabelled build'}.`,
    },
    {
      id: 'concurrency',
      label: 'Concurrency',
      ok: prediction.concurrency >= minConcurrency && prediction.concurrency <= maxConcurrency,
      detail: prediction.concurrency >= minConcurrency && prediction.concurrency <= maxConcurrency
        ? `${formatNumber(prediction.concurrency)} is inside the measured ${formatNumber(minConcurrency)}–${formatNumber(maxConcurrency)} range.`
        : `${formatNumber(prediction.concurrency)} is outside the measured ${formatNumber(minConcurrency)}–${formatNumber(maxConcurrency)} range.`,
    },
    {
      id: 'token-budget',
      label: 'Token budget',
      ok: tokenBudget >= minTokenBudget && tokenBudget <= maxTokenBudget,
      detail: `${formatNumber(tokenBudget)} tokens (ISL+OSL × concurrency) against a measured ${formatNumber(minTokenBudget)}–${formatNumber(maxTokenBudget)}.`,
    },
    {
      id: 'version',
      label: 'Backend version',
      ok: versionMeasured,
      detail: versionMeasured
        ? `${predictedFamily} was measured for this pair.`
        : `${predictedFamily} is not among the measured families (${[...measuredFamilies].join(', ')}).`,
    },
    {
      id: 'throughput',
      label: 'Predicted throughput',
      ok: prediction.throughput <= observedMax,
      detail: `${formatNumber(prediction.throughput)} tok/s against an observed maximum of ${formatNumber(observedMax)} tok/s.`,
    },
  ]

  if (prediction.servingMode === 'disagg') {
    checks.push({
      id: 'serving-mode',
      label: 'Serving mode',
      ok: false,
      detail: 'The measurements do not record aggregated versus disaggregated serving, so a disaggregated prediction cannot be matched exactly.',
    })
  }

  if (deltaPct !== null) {
    const measured = at?.measured ?? 0
    checks.push({
      id: 'delta',
      label: 'Compared with the measured curve',
      ok: Math.abs(deltaPct) <= 15,
      detail: `Prediction is ${formatPercent(deltaPct)} versus the measured ${formatNumber(measured)} tok/s at concurrency ${formatNumber(prediction.concurrency)}.`,
    })
  }

  return {
    workloadMeasured: true,
    tpMeasured: true,
    curve,
    at,
    deltaPct,
    observedMin,
    observedMax,
    checks,
  }
}

import { simulatorDisplayLabel } from '../simulator-labels'
import type { ComparisonEntry } from './types'

export type ComparisonSection = 'deployment' | 'performance' | 'memory' | 'cost'

export interface ComparisonMetric {
  id: string
  section: ComparisonSection
  label: string
  /** Numeric metrics can show a delta against the chosen baseline. */
  kind: 'number' | 'text'
  value: (entry: ComparisonEntry) => number | string | null
  /** Decimal places for a numeric value. */
  digits?: number
  /** Replaces the default number formatting, such as a compact currency. */
  formatter?: (value: number) => string
}

function phaseLabel(phase: { tp: number; pp: number; workers: number }): string {
  const parallel = phase.pp > 1 ? `tp${phase.tp}·pp${phase.pp}` : `tp${phase.tp}`;
  return `${phase.workers}× ${parallel}`;
}

export function topologyLabel(entry: ComparisonEntry): string {
  const { topology } = entry
  if (entry.mode === 'disagg' && topology.prefill && topology.decode) {
    return `${phaseLabel(topology.prefill)} prefill + ${phaseLabel(topology.decode)} decode`;
  }
  if (topology.tp == null) return 'not reported'
  const replicas = topology.replicas ?? 1
  const parallel = (topology.pp ?? 1) > 1 ? `tp${topology.tp}·pp${topology.pp}` : `tp${topology.tp}`
  return `${parallel} · ${replicas} replica${replicas === 1 ? '' : 's'}`
}

export function workloadLabel(entry: ComparisonEntry): string {
  const { isl, osl, concurrency } = entry.workload
  return `ISL ${isl} · OSL ${osl} · ${concurrency} users`
}

export function precisionLabel(entry: ComparisonEntry): string {
  return `${entry.workload.weightPrecision} weights · ${entry.workload.kvPrecision} KV`
}

export function backendLabel(entry: ComparisonEntry): string {
  if (entry.backend == null) return 'not reported'
  return entry.backendVersion == null ? entry.backend : `${entry.backend} ${entry.backendVersion}`
}

export function modeLabel(entry: ComparisonEntry): string {
  return entry.mode === 'disagg' ? 'disaggregated' : 'aggregated'
}

export function simulatorLabel(entry: ComparisonEntry): string {
  return simulatorDisplayLabel(entry.simulator, entry.simulator)
}

export function formatNumber(value: number, digits = 1): string {
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

/** Compact currency, matching the result tiles: $12.3K, $1.2M. */
export function formatUsd(value: number): string {
  const sign = value < 0 ? '-' : ''
  const absolute = Math.abs(value)
  if (absolute >= 1_000_000) return `${sign}$${formatNumber(absolute / 1_000_000, 1)}M`
  if (absolute >= 1_000) return `${sign}$${formatNumber(absolute / 1_000, 1)}K`
  return `${sign}$${formatNumber(absolute, 0)}`
}

export function formatMetric(metric: ComparisonMetric, value: number | string | null): string {
  if (value == null) return '—'
  if (typeof value === 'string') return value
  return metric.formatter ? metric.formatter(value) : formatNumber(value, metric.digits ?? 1)
}

/**
 * The table's rows, in reading order.
 *
 * Text rows describe the deployment; numeric rows carry the measurements the
 * comparison exists for. Each numeric row is also a CSV row, so the table and
 * the export cannot disagree about units or labels.
 */
export const COMPARISON_METRICS: readonly ComparisonMetric[] = [
  { id: 'model', section: 'deployment', label: 'Model', kind: 'text', value: entry => entry.model },
  { id: 'gpu', section: 'deployment', label: 'GPU system', kind: 'text', value: entry => entry.systemLabel },
  { id: 'simulator', section: 'deployment', label: 'Simulator', kind: 'text', value: simulatorLabel },
  { id: 'engine', section: 'deployment', label: 'Engine version', kind: 'text', value: entry => entry.engineVersion ?? 'not reported' },
  { id: 'backend', section: 'deployment', label: 'Backend', kind: 'text', value: backendLabel },
  { id: 'mode', section: 'deployment', label: 'Serving mode', kind: 'text', value: modeLabel },
  { id: 'workload', section: 'deployment', label: 'Workload', kind: 'text', value: workloadLabel },
  { id: 'precision', section: 'deployment', label: 'Precision', kind: 'text', value: precisionLabel },
  { id: 'topology', section: 'deployment', label: 'Topology', kind: 'text', value: topologyLabel },
  { id: 'gpus', section: 'performance', label: 'GPUs', kind: 'number', digits: 0, value: entry => entry.figures.gpus },
  { id: 'ttft', section: 'performance', label: 'TTFT (ms)', kind: 'number', digits: 1, value: entry => entry.figures.ttftMs },
  { id: 'tpot', section: 'performance', label: 'TPOT (ms)', kind: 'number', digits: 2, value: entry => entry.figures.tpotMs },
  { id: 'latency', section: 'performance', label: 'Request latency (ms)', kind: 'number', digits: 1, value: entry => entry.figures.requestLatencyMs },
  { id: 'throughput', section: 'performance', label: 'Throughput (tok/s)', kind: 'number', digits: 0, value: entry => entry.figures.tokensPerSecond },
  { id: 'per-gpu', section: 'performance', label: 'Throughput / GPU (tok/s)', kind: 'number', digits: 1, value: entry => entry.figures.tokensPerSecondPerGpu },
  { id: 'per-user', section: 'performance', label: 'Throughput / user (tok/s)', kind: 'number', digits: 1, value: entry => entry.figures.tokensPerSecondPerUser },
  { id: 'weights', section: 'memory', label: 'Weight memory (GB)', kind: 'number', digits: 1, value: entry => entry.figures.weightMemoryGb },
  { id: 'kv-request', section: 'memory', label: 'KV cache / request (MB)', kind: 'number', digits: 1, value: entry => entry.figures.kvCachePerRequestMb },
  { id: 'kv-used', section: 'memory', label: 'KV cache used (GB)', kind: 'number', digits: 1, value: entry => entry.figures.kvCacheUsedGb },
  { id: 'vram', section: 'memory', label: 'Total VRAM (GB)', kind: 'number', digits: 1, value: entry => entry.figures.totalVramGb },
  { id: 'cloud-monthly', section: 'cost', label: 'Cloud cost / month', kind: 'number', formatter: formatUsd, value: entry => entry.figures.cloudMonthlyUsd },
  { id: 'cloud-five-year', section: 'cost', label: 'Cloud cost / 5 years', kind: 'number', formatter: formatUsd, value: entry => entry.figures.cloudFiveYearUsd },
  { id: 'self-hosted-monthly', section: 'cost', label: 'On-prem cost / month', kind: 'number', formatter: formatUsd, value: entry => entry.figures.selfHostedMonthlyUsd },
  { id: 'self-hosted-five-year', section: 'cost', label: 'On-prem cost / 5 years', kind: 'number', formatter: formatUsd, value: entry => entry.figures.selfHostedFiveYearUsd },
]

export const COMPARISON_SECTIONS: readonly { id: ComparisonSection; label: string }[] = [
  { id: 'deployment', label: 'Deployment' },
  { id: 'performance', label: 'Performance' },
  { id: 'memory', label: 'Memory' },
  { id: 'cost', label: 'Cost' },
]

export function metricsFor(section: ComparisonSection): ComparisonMetric[] {
  return COMPARISON_METRICS.filter(metric => metric.section === section)
}

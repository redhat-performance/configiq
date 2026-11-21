import { AMORT_MONTHS_5YR, HOURS_PER_MONTH } from '../utils/format'
import type { ComparisonFigures } from './types'

export interface CostInputs {
  /** Total GPUs the result requires; the saved snapshot's `figures.gpus`. */
  gpus: number
  /** Selected cloud rate in USD per GPU-hour, or null when no rate is known. */
  cloudRatePerHour: number | null
  /** Hardware purchase price in USD per GPU, or null when no cost is known. */
  hardwareCostUsd: number | null
}

export type CostFigures = Pick<
  ComparisonFigures,
  'cloudMonthlyUsd' | 'cloudFiveYearUsd' | 'selfHostedMonthlyUsd' | 'selfHostedFiveYearUsd'
>

function positive(value: number | null): value is number {
  return value != null && Number.isFinite(value) && value >= 0
}

/**
 * Cost figures for a saved snapshot, from the same month length and
 * amortization window the result page uses.
 *
 * Keeping the arithmetic here means two rows cannot disagree about how many
 * hours a month has or how long the hardware is amortized over. A missing rate
 * or price stays null rather than being filled in as zero.
 */
export function comparisonCost({ gpus, cloudRatePerHour, hardwareCostUsd }: CostInputs): CostFigures {
  const gpuCount = positive(gpus) ? gpus : 0

  const cloudMonthlyUsd = positive(cloudRatePerHour)
    ? gpuCount * cloudRatePerHour * HOURS_PER_MONTH
    : null
  const hardwareTotalUsd = positive(hardwareCostUsd) ? gpuCount * hardwareCostUsd : null

  return {
    cloudMonthlyUsd,
    cloudFiveYearUsd: cloudMonthlyUsd == null ? null : cloudMonthlyUsd * AMORT_MONTHS_5YR,
    selfHostedMonthlyUsd: hardwareTotalUsd == null ? null : hardwareTotalUsd / AMORT_MONTHS_5YR,
    selfHostedFiveYearUsd: hardwareTotalUsd,
  }
}

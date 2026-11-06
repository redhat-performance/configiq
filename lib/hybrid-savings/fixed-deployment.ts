import {
  hostedCostAtVolume,
  infrastructureCandidateKey,
  ownedCostForCandidate,
  rentedCostForCandidate,
  workloadFacts,
  workloadOutputShare,
  type CostAssumptions,
  type CostOption,
  type HostedPrice,
  type HybridWorkload,
  type InfrastructureCandidate,
  type InfrastructurePath,
} from './calc'

/**
 * Fixed-deployment exploration for Hybrid Savings.
 *
 * The workload forecast resizes hardware as demand changes. The fixed view
 * instead freezes the whole deployment actually chosen for the current
 * workload and re-prices it (and the hosted API) across the demand range it
 * could serve, so unit costs fall only as the fixed deployment fills up.
 */

export interface FixedDeploymentCapture {
  path: InfrastructurePath
  /** Stable candidate identity at capture time. */
  candidateKey: string
  systemId: string
  gpusPerReplica: number
  /** Whole instances (rented) or servers (owned) in the frozen deployment. */
  wholeUnits: number
  billedGpus: number
  /** Complete replica slots the frozen deployment provides. */
  availableReplicaSlots: number
  /** Combined-token planning capacity of the frozen deployment. */
  planningTokenCapacity: number
  /**
   * Signature of every input the capture depends on. When the live signature
   * differs, the frozen configuration is stale and the view must exit.
   */
  basisSignature: string
}

export interface FixedDeploymentSample {
  /** Planning capacity used, as a percentage of the frozen deployment. */
  capacityUsedPct: number
  /** Combined billed tokens this sample represents. */
  combinedTokens: number
  /** Requested output tokens this sample represents. */
  outputTokens: number
  hosted: CostOption | null
  hardware: CostOption | null
}

/**
 * Capture the deployment actually chosen for the current workload. Returns
 * null when the option has no freezable hardware deployment.
 */
export function captureFixedDeployment(
  path: InfrastructurePath,
  option: CostOption,
  basisSignature: string,
): FixedDeploymentCapture | null {
  const deployment = option.deployment
  if (!option.candidate || !deployment || deployment.availableReplicaSlots <= 0) {
    return null
  }
  return {
    path,
    candidateKey: infrastructureCandidateKey(option.candidate, path),
    systemId: option.candidate.systemId,
    gpusPerReplica: option.candidate.gpusPerReplica,
    wholeUnits: deployment.wholeUnits,
    billedGpus: deployment.billedGpus,
    availableReplicaSlots: deployment.availableReplicaSlots,
    planningTokenCapacity: deployment.planningTokenCapacity,
    basisSignature,
  }
}

/**
 * Sample the frozen deployment across its usable demand range: every whole
 * percentage of planning capacity plus the exact current-workload
 * percentage. Hosted and hardware are priced against identical demand, and
 * the hardware layout never resizes.
 */
export function fixedDeploymentSamples(
  workload: HybridWorkload,
  hostedPrice: HostedPrice,
  assumptions: CostAssumptions,
  candidate: InfrastructureCandidate,
  capture: FixedDeploymentCapture,
): FixedDeploymentSample[] {
  const capacity = capture.planningTokenCapacity
  if (!Number.isFinite(capacity) || capacity <= 0) return []

  const outputShare = workloadOutputShare(workload)
  const currentTokens = workloadFacts(workload).monthlyTokens
  const currentPct = currentTokens > 0 && capacity > 0
    ? currentTokens / capacity * 100
    : 0

  const percentages = new Set<number>()
  for (let pct = 0; pct <= 100; pct += 1) percentages.add(pct)
  percentages.add(currentPct)

  return [...percentages]
    .filter(pct => Number.isFinite(pct) && pct >= 0 && pct <= 100)
    .sort((left, right) => left - right)
    .map(pct => {
      const combinedTokens = capacity * pct / 100
      const hardware = capture.path === 'rented'
        ? rentedCostForCandidate(workload, candidate, assumptions, combinedTokens, {
          availableReplicaSlots: capture.availableReplicaSlots,
        })
        : ownedCostForCandidate(workload, candidate, assumptions, combinedTokens, {
          availableReplicaSlots: capture.availableReplicaSlots,
        })
      return {
        capacityUsedPct: pct,
        combinedTokens,
        outputTokens: combinedTokens * outputShare,
        hosted: hostedCostAtVolume(workload, hostedPrice, assumptions, combinedTokens),
        hardware,
      }
    })
}

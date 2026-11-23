import type { RecommendResult } from '@/lib/api/recommend'
import type { DeployFormState } from './deploy-request'

/**
 * The review panel's artifact (docs/08 §4): the `LLMInferenceService` the user
 * would own, followed by the plan config the planner would write. Rendered for
 * review and copy only — nothing here is applied to a cluster. Field shapes
 * follow the interface spec; the objectives annotation key is PROPOSED and
 * moves to `spec.objectives` once accepted upstream.
 */
export function buildDeployYaml(form: DeployFormState, result: RecommendResult): string {
  const name = serviceName(form.model)
  const load = form.loadUnit === 'users'
    ? `concurrentRequests: ${form.loadValue}   # sent as target_concurrency`
    : `requestsPerSecond: ${form.loadValue}   # sent as target_request_rate`
  const { replicasNeeded, tensorParallelSize } = result.recommendation

  return `apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: ${name}
  annotations:
    # PROPOSED key and schema. Moves to spec.objectives once accepted upstream.
    inference.redhat.com/objectives: |
      ${load}
      ttftMean: ${form.ttftTarget}ms
      tpotMean: ${form.tpotTarget}ms
      tokens: {input: ${form.workload.isl}, output: ${form.workload.osl}}
      plan: manual            # manual | auto
spec:
  model:
    uri: hf://${form.model}
  baseRefs:
    - name: ${name}-plan  # written by the planner; listed once, never edited by hand
  router:
    scheduler: {}
---
# The planner writes and owns this config; deleting the service deletes it.
# Values below are the recommendation this page is showing you.
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceServiceConfig
metadata:
  name: ${name}-plan
  ownerReferences: [{kind: LLMInferenceService, name: ${name}}]
  annotations:
    inference.redhat.com/plan: |
      source: {simulator: ${result.simulator}, runId: ${result.requestId}}
      predicted: {ttftMean: ${roundMs(result.performance.ttftLatencyMs)}ms, tpotMean: ${roundMs(result.performance.tpotMs)}ms}
      verified: null          # filled by the benchmark run
spec:
  replicas: ${replicasNeeded}
  parallelism:
    tensor: ${tensorParallelSize}
`
}

/** Qwen/Qwen3-8B → qwen3-8b (a DNS-safe service name, lowercased and dashed). */
export function serviceName(model: string): string {
  const base = model.includes('/') ? model.slice(model.lastIndexOf('/') + 1) : model
  return base
    .toLowerCase()
    .replace(/[^a-z0-9.-]+/g, '-')
    .replace(/^[.-]+|[.-]+$/g, '')
    || 'llm-service'
}

function roundMs(value: number): string {
  return Number.isFinite(value) ? String(Math.round(value * 10) / 10) : 'null'
}

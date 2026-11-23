import { describe, expect, it } from 'vitest'
import type { RecommendResult } from '@/lib/api/recommend'
import type { WorkloadPreset } from '@/lib/workload-presets'
import { buildDeployYaml } from './deploy-yaml'
import type { DeployFormState } from './deploy-request'

const workload: WorkloadPreset = { key: 'chat', label: 'Chat', isl: 1024, osl: 1024 }

function form(overrides: Partial<DeployFormState> = {}): DeployFormState {
  return {
    model: 'Qwen/Qwen3-8B',
    systemId: 'h200_sxm',
    workload,
    loadUnit: 'users',
    loadValue: 64,
    ttftTarget: 400,
    tpotTarget: 30,
    gpusFree: 8,
    backend: 'vllm',
    backendVersion: '0.30.0',
    simulator: 'aisimulate',
    ...overrides,
  }
}

function resultFixture(): RecommendResult {
  return {
    requestId: 'req-42',
    status: 'completed',
    simulator: 'aisimulate',
    mode: 'agg',
    recommendation: {
      gpusNeeded: 8, gpusPerReplica: 8, totalGpus: 8, replicasNeeded: 1,
      tensorParallelSize: 8, pipelineParallelSize: 1, dataParallelSize: 1,
      contextParallelSize: 1, moeTensorParallelSize: null, moeExpertParallelSize: null,
      batchSize: null,
    },
    phases: { prefill: null, decode: null, encode: null },
    performance: { ttftLatencyMs: 251, tpotMs: 6.9, requestLatencyMs: 7100, concurrency: 64 },
    throughput: { tokensPerSecond: 4400, tokensPerSecondPerGpu: 550, tokensPerSecondPerUser: 68 },
    memory: { value: 120, unit: 'GB', breakdown: null },
    memoryAvailable: true,
    metadata: {
      modelPath: 'Qwen/Qwen3-8B', system: 'h200_sxm', inputTokens: 1024,
      outputTokens: 1024, targetTtftMs: null, durationMs: 4000,
    },
    warnings: [],
  }
}

describe('buildDeployYaml', () => {
  it('renders the LLMInferenceService with the objectives annotation', () => {
    const yaml = buildDeployYaml(form(), resultFixture())
    expect(yaml).toContain('apiVersion: serving.kserve.io/v1alpha2')
    expect(yaml).toContain('kind: LLMInferenceService')
    expect(yaml).toContain('name: qwen3-8b')
    expect(yaml).toContain('inference.redhat.com/objectives')
    expect(yaml).toContain('concurrentRequests: 64')
    expect(yaml).toContain('ttftMean: 400ms')
    expect(yaml).toContain('tpotMean: 30ms')
    expect(yaml).toContain('tokens: {input: 1024, output: 1024}')
    expect(yaml).toContain('uri: hf://Qwen/Qwen3-8B')
    expect(yaml).toContain('PROPOSED')
  })

  it('uses request rate for a demand run', () => {
    const yaml = buildDeployYaml(form({ loadUnit: 'rps', loadValue: 5 }), resultFixture())
    expect(yaml).toContain('requestsPerSecond: 5')
    expect(yaml).not.toContain('concurrentRequests')
  })

  it('renders the plan config the planner would own, with the run as its source', () => {
    const yaml = buildDeployYaml(form(), resultFixture())
    expect(yaml).toContain('kind: LLMInferenceServiceConfig')
    expect(yaml).toContain('name: qwen3-8b-plan')
    expect(yaml).toContain('ownerReferences')
    expect(yaml).toContain('replicas: 1')
    expect(yaml).toContain('tensor: 8')
    expect(yaml).toContain('simulator: aisimulate')
    expect(yaml).toContain('runId: req-42')
    expect(yaml).toContain('ttftMean: 251ms')
    expect(yaml).toContain('tpotMean: 6.9ms')
  })

  it('slugs the service name from the model path', () => {
    const yaml = buildDeployYaml(form({ model: 'meta-llama/Llama-3.1-8B-Instruct' }), resultFixture())
    expect(yaml).toContain('name: llama-3.1-8b-instruct')
    expect(yaml).toContain('name: llama-3.1-8b-instruct-plan')
    expect(yaml).toContain('uri: hf://meta-llama/Llama-3.1-8B-Instruct')
  })
})

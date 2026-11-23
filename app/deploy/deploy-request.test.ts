import { describe, expect, it } from 'vitest'
import { buildDeployParams, pickDeployDefaults, type DeployFormState } from './deploy-request'
import type { WorkloadPreset } from '@/lib/workload-presets'

const preset: WorkloadPreset = { key: 'chat', label: 'Chat', isl: 1024, osl: 1024 }

function form(overrides: Partial<DeployFormState> = {}): DeployFormState {
  return {
    model: 'Qwen/Qwen3-8B',
    systemId: 'h200_sxm',
    workload: preset,
    loadUnit: 'users',
    loadValue: 64,
    ttftTarget: 400,
    tpotTarget: 30,
    gpusFree: 8,
    backend: 'vllm',
    backendVersion: undefined,
    simulator: 'aisimulate',
    ...overrides,
  }
}

describe('buildDeployParams', () => {
  it('maps concurrent users to a capacity run with no SLA bounds', () => {
    const params = buildDeployParams(form())
    expect(params.objective).toBe('concurrency_capacity')
    expect(params.target_concurrency).toBe(64)
    expect(params.sla_mode).toBe('none')
    // Latency targets are display-only comparisons on the answer, never sent:
    // sla_mode none rejects bounds, and the answer must not be shaped by them.
    expect(params).not.toHaveProperty('ttft')
    expect(params).not.toHaveProperty('tpot')
    expect(params.model_path).toBe('Qwen/Qwen3-8B')
    expect(params.system).toBe('h200_sxm')
    expect(params.isl).toBe(1024)
    expect(params.osl).toBe(1024)
  })

  it('maps requests per second to a demand run', () => {
    const params = buildDeployParams(form({ loadUnit: 'rps', loadValue: 5 }))
    expect(params.objective).toBe('throughput_demand')
    expect(params.target_request_rate).toBe(5)
    expect(params).not.toHaveProperty('target_concurrency')
    expect(params.sla_mode).toBe('none')
  })

  it('carries the preset prefix and the backend identity', () => {
    const params = buildDeployParams(form({
      workload: { ...preset, prefix: 200 },
      backendVersion: '0.30.0',
    }))
    expect(params.prefix).toBe(200)
    expect(params.backend).toBe('vllm')
    expect(params.backend_version).toBe('0.30.0')
    expect(params.simulator).toBe('aisimulate')
  })

  it('never sends the free-GPU count', () => {
    const params = buildDeployParams(form({ gpusFree: 3 }))
    expect(JSON.stringify(params)).not.toContain('gpusFree')
    expect(JSON.stringify(params)).not.toContain('gpus_free')
  })
})

describe('pickDeployDefaults', () => {
  const gpus = [
    { systemId: 'a100_sxm', label: 'NVIDIA A100' },
    { systemId: 'h200_sxm', label: 'NVIDIA H200' },
  ]

  it('prefers a validated model and a system tested for it', () => {
    const defaults = pickDeployDefaults({
      modelOptions: ['Qwen/Qwen3-8B', 'meta-llama/Llama-3.1-8B'],
      gpuOptions: gpus,
      validatedModelIds: ['meta-llama/Llama-3.1-8B'],
      pairs: [{ modelId: 'meta-llama/Llama-3.1-8B', systemId: 'h200_sxm' }],
    })
    expect(defaults).toEqual({ model: 'meta-llama/Llama-3.1-8B', systemId: 'h200_sxm' })
  })

  it('falls back to the first catalog entries when nothing is validated', () => {
    const defaults = pickDeployDefaults({
      modelOptions: ['Qwen/Qwen3-8B'],
      gpuOptions: gpus.slice(0, 1),
      validatedModelIds: [],
      pairs: [],
    })
    expect(defaults).toEqual({ model: 'Qwen/Qwen3-8B', systemId: 'a100_sxm' })
  })

  it('ignores a validated model the catalog does not list', () => {
    const defaults = pickDeployDefaults({
      modelOptions: ['Qwen/Qwen3-8B'],
      gpuOptions: gpus.slice(0, 1),
      validatedModelIds: ['some/other-model'],
      pairs: [],
    })
    expect(defaults?.model).toBe('Qwen/Qwen3-8B')
  })

  it('returns null when the catalog is empty', () => {
    expect(pickDeployDefaults({ modelOptions: [], gpuOptions: [], validatedModelIds: [], pairs: [] })).toBeNull()
  })
})

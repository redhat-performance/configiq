import { describe, expect, it } from 'vitest'
import type { GpuOption, ModelSpec } from '@/lib/hooks/useCatalog'
import {
  gpuOptionsForSystems,
  gpuSystemLabel,
  gpuSystemSpecs,
  modelFacetChips,
  registryModelItems,
  shortRevision,
} from './display'

const GPUS: GpuOption[] = [
  {
    systemId: 'h200_sxm',
    label: 'NVIDIA H200',
    vendor: 'nvidia',
    architecture: 'hopper',
    vramGb: 141,
    bandwidthTbps: 4.8,
    tflopsBf16: 989,
    tdpWatts: 700,
    gpusPerNode: 8,
  },
]

const SPEC: ModelSpec = {
  id: 'org/model',
  num_experts: null,
  num_experts_per_tok: null,
  context_length: 131_072,
  num_attn_heads: 32,
  num_kv_heads: 8,
  architecture: 'LlamaForCausalLM',
}

describe('GPU labels', () => {
  it('prefers the catalog label and falls back to a known display name', () => {
    expect(gpuSystemLabel('h200_sxm', GPUS)).toBe('NVIDIA H200')
    expect(gpuSystemLabel('mi355x', GPUS)).toBe('AMD Instinct MI355X')
    expect(gpuSystemLabel('custom_gpu', GPUS)).toBe('custom_gpu')
  })

  it('builds catalog options for registry systems without inventing specs', () => {
    const options = gpuOptionsForSystems(['mi355x', 'h200_sxm'], GPUS)
    expect(options.map(option => option.systemId)).toEqual(['mi355x', 'h200_sxm'])
    expect(options[0]).toMatchObject({
      label: 'AMD Instinct MI355X',
      vramGb: null,
      bandwidthTbps: null,
      tflopsBf16: null,
    })
    expect(options[1].vramGb).toBe(141)
  })

  it('summarizes only specs the catalog actually reported', () => {
    expect(gpuSystemSpecs('h200_sxm', GPUS)).toBe('141 GB · 4.8 TB/s · 989 TFLOPS BF16')
    expect(gpuSystemSpecs('mi355x', GPUS)).toBe('')
  })
})

describe('model facet chips', () => {
  it('derives size, MoE and quantization from the checkpoint id', () => {
    expect(modelFacetChips('nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8')).toEqual([
      '120B (12B active)',
      'MoE',
      'FP8',
    ])
  })

  it('uses the catalog spec for context length', () => {
    expect(modelFacetChips('org/model', SPEC)).toContain('128K ctx')
  })
})

describe('registry model items', () => {
  it('lists only registry models and marks validated ones', () => {
    const items = registryModelItems(
      ['org/model', 'org/other', 'org/model'],
      new Map([['org/model', SPEC]]),
      ['org/model'],
      ['org/model', 'org/catalog-only'],
    )

    expect(items.map(item => item.value)).toEqual(['org/model', 'org/other'])
    expect(items[0]).toMatchObject({
      group: 'org',
      isTested: true,
      inCatalog: true,
      contextLength: 131_072,
    })
    expect(items[1]).toMatchObject({ isTested: false, inCatalog: false })
  })

  it('formats a short immutable revision', () => {
    expect(shortRevision('e45d9e1114b2d029fdca0db7bfe7873a0f5286af')).toBe('e45d9e1114b2')
  })
})

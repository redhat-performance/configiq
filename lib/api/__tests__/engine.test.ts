import { describe, expect, it } from 'vitest'

import { engineRoles, parseEngineSection, type EngineField, type EngineRole } from '../engine'

function field(value: boolean | number | string | null, source: EngineField['source'] = 'evaluated'): EngineField {
  return { value, source }
}

function role(overrides: Partial<EngineRole> = {}): EngineRole {
  return {
    replicas: field(1),
    tensor_parallel_size: field(2),
    pipeline_parallel_size: field(1),
    data_parallel_size: field(1),
    context_parallel_size: field(1),
    decode_context_parallel_size: field(1),
    moe_tensor_parallel_size: field(2),
    moe_expert_parallel_size: field(1),
    max_model_len: field(5000),
    max_num_seqs: field(128),
    max_num_batched_tokens: field(8192),
    block_size: field(64),
    gpu_memory_utilization: field(0.9),
    enable_prefix_caching: field(true),
    enable_chunked_prefill: field(null, 'not-modeled'),
    kv_cache_dtype: field(null, 'not-modeled'),
    quantization: field(null, 'not-modeled'),
    ...overrides,
  }
}

function section(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    model: 'Qwen/Qwen3-32B',
    backend: 'vllm',
    backend_version: '0.31.0',
    mode: 'agg',
    agg: role(),
    prefill: null,
    decode: null,
    ...overrides,
  }
}

describe('parseEngineSection', () => {
  it('keeps the value and the reason the service reported', () => {
    const parsed = parseEngineSection(section())

    expect(parsed).not.toBeNull()
    expect(parsed?.model).toBe('Qwen/Qwen3-32B')
    expect(parsed?.backend).toBe('vllm')
    expect(parsed?.backend_version).toBe('0.31.0')
    expect(parsed?.mode).toBe('agg')
    expect(parsed?.agg?.tensor_parallel_size).toEqual({ value: 2, source: 'evaluated' })
    expect(parsed?.agg?.enable_chunked_prefill).toEqual({ value: null, source: 'not-modeled' })
    expect(parsed?.prefill).toBeNull()
    expect(parsed?.decode).toBeNull()
  })

  it('keeps both pools of a disaggregated section', () => {
    const parsed = parseEngineSection(section({
      mode: 'disagg',
      agg: null,
      prefill: role({ tensor_parallel_size: field(4) }),
      decode: role({ tensor_parallel_size: field(2), replicas: field(3) }),
    }))

    expect(parsed?.mode).toBe('disagg')
    expect(parsed?.agg).toBeNull()
    expect(parsed?.prefill?.tensor_parallel_size.value).toBe(4)
    expect(parsed?.decode?.tensor_parallel_size.value).toBe(2)
    expect(parsed?.decode?.replicas.value).toBe(3)
  })

  it('treats a missing or malformed limit as not modeled instead of inventing one', () => {
    const raw = role() as unknown as Record<string, unknown>
    delete raw.block_size
    raw.max_num_seqs = { value: 128, source: 'guessed' }

    const parsed = parseEngineSection(section({ agg: raw }))

    expect(parsed?.agg?.block_size).toEqual({ value: null, source: 'not-modeled' })
    expect(parsed?.agg?.max_num_seqs).toEqual({ value: null, source: 'not-modeled' })
    // A field the response did report is untouched.
    expect(parsed?.agg?.tensor_parallel_size).toEqual({ value: 2, source: 'evaluated' })
  })

  it('rejects a value that is not an engine section', () => {
    expect(parseEngineSection(null)).toBeNull()
    expect(parseEngineSection([])).toBeNull()
    expect(parseEngineSection('engine')).toBeNull()
    expect(parseEngineSection(section({ model: '' }))).toBeNull()
    expect(parseEngineSection(section({ backend: '' }))).toBeNull()
    expect(parseEngineSection(section({ mode: 'split' }))).toBeNull()
  })

  it('reports an absent backend version as null, not a guessed one', () => {
    expect(parseEngineSection(section({ backend_version: null }))?.backend_version).toBeNull()
    expect(parseEngineSection(section({ backend_version: '' }))?.backend_version).toBeNull()
  })
})

describe('engineRoles', () => {
  it('lists only the roles the section reported, in display order', () => {
    const agg = parseEngineSection(section())
    const disagg = parseEngineSection(section({
      mode: 'disagg',
      agg: null,
      prefill: role(),
      decode: role(),
    }))

    expect(agg == null ? [] : engineRoles(agg).map(row => row.role)).toEqual(['agg'])
    expect(disagg == null ? [] : engineRoles(disagg).map(row => row.role)).toEqual(['prefill', 'decode'])
  })
})

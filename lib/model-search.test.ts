import { describe, expect, it } from 'vitest'
import type { ModelSpec } from '@/lib/hooks/useCatalog'
import { modelFacets, type ModelFacets } from './model-metadata'
import { scoreModel, searchModels, tokenizeQuery } from './model-search'

function facets(id: string, spec?: Partial<ModelSpec>, tested: string[] = []): ModelFacets {
  return modelFacets(id, {
    spec: spec
      ? {
          id,
          num_experts: null,
          num_experts_per_tok: null,
          context_length: null,
          num_attn_heads: null,
          num_kv_heads: null,
          architecture: null,
          ...spec,
        }
      : undefined,
    testedModels: tested,
  })
}

const CORPUS: ModelFacets[] = [
  facets('Qwen/Qwen3-32B'),
  facets('Qwen/Qwen3-32B-FP8'),
  facets('Qwen/Qwen3-8B'),
  facets('Qwen/Qwen3-235B-A22B'),
  facets('Qwen/Qwen3-VL-8B', { architecture: 'Qwen3VLForConditionalGeneration' }),
  facets('meta-llama/Llama-3.3-70B-Instruct'),
  facets('nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8'),
  facets('RedHatAI/GLM-5.3-MXFP4'),
  facets('deepseek-ai/DeepSeek-R1'),
]

function ids(results: ModelFacets[]): string[] {
  return results.map(r => r.id)
}

describe('tokenizeQuery', () => {
  it('splits on whitespace and commas and lowercases', () => {
    expect(tokenizeQuery('  70B, Vision ')).toEqual(['70b', 'vision'])
    expect(tokenizeQuery('   ')).toEqual([])
  })
})

describe('searching by parameter count', () => {
  it('finds models by size even when the size is not a substring of the query', () => {
    expect(ids(searchModels(CORPUS, '70b'))).toContain('meta-llama/Llama-3.3-70B-Instruct')
  })

  it('matches a bare number as a size', () => {
    expect(ids(searchModels(CORPUS, '32'))).toContain('Qwen/Qwen3-32B')
  })

  it('does not confuse nearby sizes', () => {
    const results = ids(searchModels(CORPUS, '8b'))
    expect(results).toContain('Qwen/Qwen3-8B')
    expect(results).not.toContain('Qwen/Qwen3-32B')
  })

  it('matches MoE models on their active parameter count', () => {
    expect(ids(searchModels(CORPUS, '22b'))).toContain('Qwen/Qwen3-235B-A22B')
  })

  it('understands trillion-scale sizes', () => {
    expect(scoreModel(facets('moonshotai/Kimi-K3'), '2.78t')).toBeGreaterThan(0)
  })
})

describe('searching by type', () => {
  it('finds vision models by the word vision', () => {
    const results = ids(searchModels(CORPUS, 'vision'))
    expect(results).toContain('Qwen/Qwen3-VL-8B')
    expect(results).not.toContain('Qwen/Qwen3-32B')
  })

  it('finds mixture-of-experts models by moe', () => {
    const results = ids(searchModels(CORPUS, 'moe'))
    expect(results).toContain('Qwen/Qwen3-235B-A22B')
    expect(results).toContain('nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8')
    expect(results).not.toContain('Qwen/Qwen3-8B')
  })

  it('treats dense as the absence of experts', () => {
    const results = ids(searchModels(CORPUS, 'dense'))
    expect(results).toContain('Qwen/Qwen3-32B')
    expect(results).not.toContain('Qwen/Qwen3-235B-A22B')
  })
})

describe('searching by quantization', () => {
  it('finds a checkpoint by its exact format', () => {
    expect(ids(searchModels(CORPUS, 'mxfp4'))).toEqual(['RedHatAI/GLM-5.3-MXFP4'])
  })

  it('finds quantized checkpoints by bit-width shorthand', () => {
    const results = ids(searchModels(CORPUS, '8bit'))
    expect(results).toContain('Qwen/Qwen3-32B-FP8')
    expect(results).not.toContain('Qwen/Qwen3-32B')
  })

  it('finds any quantized checkpoint', () => {
    const results = ids(searchModels(CORPUS, 'quantized'))
    expect(results).toContain('Qwen/Qwen3-32B-FP8')
    expect(results).toContain('RedHatAI/GLM-5.3-MXFP4')
    expect(results).not.toContain('Qwen/Qwen3-8B')
  })
})

describe('searching by vendor and family', () => {
  it('finds models by vendor', () => {
    expect(ids(searchModels(CORPUS, 'nvidia'))).toEqual(['nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8'])
  })

  it('finds models by family name fragment', () => {
    expect(ids(searchModels(CORPUS, 'nemotron'))).toContain('nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8')
  })
})

describe('typo tolerance', () => {
  it('still matches with a single-character typo', () => {
    expect(ids(searchModels(CORPUS, 'nemotrom'))).toContain('nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8')
    expect(ids(searchModels(CORPUS, 'deepsek'))).toContain('deepseek-ai/DeepSeek-R1')
  })

  it('does not fuzz very short terms into nonsense', () => {
    expect(scoreModel(facets('Qwen/Qwen3-32B'), 'xyz')).toBe(0)
  })
})

describe('multi-term queries', () => {
  it('requires every term to match', () => {
    const results = ids(searchModels(CORPUS, 'qwen 8b'))
    expect(results).toContain('Qwen/Qwen3-8B')
    expect(results).toContain('Qwen/Qwen3-VL-8B')
    expect(results).not.toContain('Qwen/Qwen3-32B')
  })

  it('narrows by type as well as size', () => {
    expect(ids(searchModels(CORPUS, '8b vision'))).toEqual(['Qwen/Qwen3-VL-8B'])
  })

  it('returns nothing when the terms cannot both be satisfied', () => {
    expect(searchModels(CORPUS, '70b vision')).toHaveLength(0)
  })
})

describe('ranking', () => {
  it('puts an exact id first', () => {
    expect(ids(searchModels(CORPUS, 'qwen/qwen3-32b'))[0]).toBe('Qwen/Qwen3-32B')
  })

  it('ranks the base model above its quantized variant for a family query', () => {
    const results = ids(searchModels(CORPUS, 'qwen3-32b'))
    expect(results.indexOf('Qwen/Qwen3-32B')).toBeLessThan(results.indexOf('Qwen/Qwen3-32B-FP8'))
  })

  it('prefers a strong text match over a tested-status nudge', () => {
    const corpus = [
      facets('org/Unrelated-8B', undefined, ['org/Unrelated-8B']),
      facets('org/Nemotron-8B'),
    ]
    expect(ids(searchModels(corpus, 'nemotron'))[0]).toBe('org/Nemotron-8B')
  })

  it('is deterministic for equally scored models', () => {
    const once = ids(searchModels(CORPUS, 'qwen'))
    const twice = ids(searchModels([...CORPUS].reverse(), 'qwen'))
    expect(once).toEqual(twice)
  })
})

describe('empty query', () => {
  it('returns the input untouched so browse order is preserved', () => {
    expect(searchModels(CORPUS, '')).toBe(CORPUS)
    expect(searchModels(CORPUS, '   ')).toBe(CORPUS)
  })

  it('scores nothing for an empty query', () => {
    expect(scoreModel(facets('Qwen/Qwen3-32B'), '')).toBe(0)
  })
})

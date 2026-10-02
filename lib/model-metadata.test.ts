import { describe, expect, it } from 'vitest'
import type { ModelSpec } from '@/lib/hooks/useCatalog'
import {
  isModelListedAsTested,
  modelActiveParameterBillions,
  modelFacets,
  modelFamily,
  modelParameterBillions,
  modelQuantization,
  modelSizeLabel,
  modelTierLabel,
  modelTypes,
  modelVendor,
} from './model-metadata'

function spec(partial: Partial<ModelSpec>): ModelSpec {
  return {
    id: 'test',
    num_experts: null,
    num_experts_per_tok: null,
    context_length: null,
    num_attn_heads: null,
    num_kv_heads: null,
    architecture: null,
    ...partial,
  }
}

describe('model parameter and tier labels', () => {
  it('reads parameter sizes encoded in checkpoint names', () => {
    expect(modelSizeLabel('Qwen/Qwen3-0.6B')).toBe('0.6B parameters')
    expect(modelSizeLabel('Qwen/Qwen3-32B-FP8')).toBe('32B parameters')
    expect(modelSizeLabel('Qwen/Qwen3-235B-A22B')).toBe('235B parameters')
  })

  it('supplies parameter sizes for catalogue names without a size suffix', () => {
    expect(modelSizeLabel('MiniMaxAI/MiniMax-M3')).toBe('427B parameters')
    expect(modelSizeLabel('deepseek-ai/DeepSeek-R1')).toBe('684.5B parameters')
    expect(modelSizeLabel('deepseek-ai/DeepSeek-V4-Pro')).toBe('1.6T parameters')
    expect(modelSizeLabel('moonshotai/Kimi-K3')).toBe('2.78T parameters')
    expect(modelSizeLabel('Qwen/Qwen3.8-2.4T-A95B')).toBe('2.4T parameters')
    expect(modelSizeLabel('zai-org/GLM-5.3')).toBe('753.9B parameters')
  })

  it('uses total rather than active parameters for Llama 4 MoE checkpoints', () => {
    expect(modelParameterBillions('meta-llama/Llama-4-Maverick-17B-128E-Instruct')).toBe(401.6)
    expect(modelParameterBillions('meta-llama/Llama-4-Scout-17B-16E-Instruct')).toBe(108.6)
  })

  it('uses the actual parameter size for DeepSeek-derived checkpoints', () => {
    expect(modelSizeLabel('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')).toBe('32B parameters')
    expect(modelTierLabel('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')).toBe('Medium model')
    expect(modelSizeLabel('deepseek-ai/DeepSeek-R1-Distill-Llama-8B')).toBe('8B parameters')
    expect(modelSizeLabel('deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')).toBe('8B parameters')
    expect(modelTierLabel('deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')).toBe('Small model')
    expect(modelSizeLabel('deepseek-ai/DeepSeek-R1-0528-Distill-Qwen3-8B')).toBe('8B parameters')
    expect(modelSizeLabel('deepseek-ai/DeepSeek-V3.1-Distill-Qwen-32B')).toBe('32B parameters')
    expect(modelSizeLabel('deepseek-ai/DeepSeek-R1-0528')).toBe('684.5B parameters')
    expect(modelSizeLabel('deepseek-ai/DeepSeek-V3.1')).toBe('685.4B parameters')
  })

  it('classifies representative small, medium and large models', () => {
    expect(modelTierLabel('Qwen/Qwen3-8B')).toBe('Small model')
    expect(modelTierLabel('Qwen/Qwen3-32B')).toBe('Medium model')
    expect(modelTierLabel('Qwen/Qwen3-235B-A22B')).toBe('Large model')
  })

  it('matches tested configurations without depending on casing or whitespace', () => {
    const testedModels = [' moonshotai/Kimi-K3 ', 'openai/gpt-oss-120b']

    expect(isModelListedAsTested('MoonshotAI/kimi-k3', testedModels)).toBe(true)
    expect(isModelListedAsTested('Qwen/Qwen3-8B', testedModels)).toBe(false)
  })
})

describe('quantization detection', () => {
  it('reads the quantization format off the checkpoint name', () => {
    expect(modelQuantization('Qwen/Qwen3-32B-FP8')).toBe('FP8')
    expect(modelQuantization('RedHatAI/GLM-5.3-MXFP4')).toBe('MXFP4')
    expect(modelQuantization('RedHatAI/Qwen3.8-27B-INT4')).toBe('INT4')
    expect(modelQuantization('RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic')).toBe('FP8')
    expect(modelQuantization('Qwen/Qwen3-32B-FP8-Static-PerTensor')).toBe('FP8')
    expect(modelQuantization('RedHatAI/Llama-3.3-70B-quantized.w4a16')).toBe('W4A16')
  })

  it('prefers the weight precision when a checkpoint names two formats', () => {
    expect(modelQuantization('RedHatAI/GLM-5.2-NVFP4-FP8')).toBe('NVFP4')
  })

  it('returns null for unquantized checkpoints', () => {
    expect(modelQuantization('Qwen/Qwen3-32B')).toBeNull()
    expect(modelQuantization('deepseek-ai/DeepSeek-R1')).toBeNull()
  })

  it('does not mistake a version number for a precision tag', () => {
    expect(modelQuantization('meta-llama/Llama-4-Scout-17B-16E-Instruct')).toBeNull()
    expect(modelQuantization('Qwen/Qwen3-4B')).toBeNull()
  })
})

describe('family derivation', () => {
  it('collapses quantized variants onto one family', () => {
    expect(modelFamily('Qwen/Qwen3-32B')).toBe('Qwen3-32B')
    expect(modelFamily('Qwen/Qwen3-32B-FP8')).toBe('Qwen3-32B')
    expect(modelFamily('Qwen/Qwen3-32B-FP8-Static-PerTensor')).toBe('Qwen3-32B')
    expect(modelFamily('RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic')).toBe('gemma-4-26B-A4B-it')
    expect(modelFamily('RedHatAI/gemma-4-26B-A4B-it')).toBe('gemma-4-26B-A4B-it')
  })

  it('keeps instruct and base variants distinct — they are different models', () => {
    expect(modelFamily('meta-llama/Llama-4-Scout-17B-16E-Instruct')).toBe('Llama-4-Scout-17B-16E-Instruct')
    expect(modelFamily('deepseek-ai/DeepSeek-V3.1-Base')).toBe('DeepSeek-V3.1-Base')
  })

  it('preserves dots in version numbers', () => {
    expect(modelFamily('RedHatAI/Qwen3.8-27B-INT4')).toBe('Qwen3.8-27B')
  })

  it('strips granularity qualifiers that trail a format tag', () => {
    expect(modelFamily('nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block'))
      .toBe('NVIDIA-Nemotron-3-Ultra-550B-A55B')
    expect(modelFamily('org/Model-8B-INT4-groupwise')).toBe('Model-8B')
    expect(modelFamily('org/Model-8B-W8A8-sym')).toBe('Model-8B')
  })

  it('leaves a qualifier alone when the checkpoint names no format', () => {
    expect(modelFamily('org/Transformer-Block')).toBe('Transformer-Block')
    expect(modelFamily('org/Model-8B-dynamic')).toBe('Model-8B-dynamic')
  })
})

describe('type facets', () => {
  it('marks mixture-of-experts models from the spec', () => {
    expect(modelTypes('Qwen/Qwen3-Next-80B', spec({ num_experts: 512 }))).toContain('moe')
    expect(modelTypes('Qwen/Qwen3-32B', spec({ num_experts: 1 }))).not.toContain('moe')
  })

  it('infers mixture-of-experts from active-parameter and expert notation', () => {
    expect(modelTypes('Qwen/Qwen3-235B-A22B')).toContain('moe')
    expect(modelTypes('meta-llama/Llama-4-Scout-17B-16E-Instruct')).toContain('moe')
    expect(modelTypes('Qwen/Qwen3-32B')).not.toContain('moe')
  })

  it('marks vision models from the architecture and from the checkpoint name', () => {
    expect(modelTypes('x/y', spec({ architecture: 'Gemma3ForConditionalGeneration' }))).toContain('vision')
    expect(modelTypes('Qwen/Qwen3-VL-8B')).toContain('vision')
    expect(modelTypes('llava-hf/llava-1.5-7b-hf')).toContain('vision')
  })

  it('treats vision and text as mutually exclusive but stacks moe on top', () => {
    expect(modelTypes('Qwen/Qwen3-VL-235B-A22B')).toEqual(['vision', 'moe'])
    expect(modelTypes('Qwen/Qwen3-32B')).toEqual(['text'])
  })
})

describe('active parameters', () => {
  it('reads the active-parameter count from MoE checkpoint names', () => {
    expect(modelActiveParameterBillions('Qwen/Qwen3-235B-A22B')).toBe(22)
    expect(modelActiveParameterBillions('RedHatAI/gemma-4-26B-A4B-it')).toBe(4)
    expect(modelActiveParameterBillions('Qwen/Qwen3-32B')).toBeNull()
  })
})

describe('modelFacets', () => {
  it('combines id-derived and spec-derived facts', () => {
    const facets = modelFacets('RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic', {
      spec: spec({ num_experts: 128, context_length: 131_072 }),
      testedModels: ['RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic'],
      catalogModels: ['RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic'],
    })

    expect(facets.vendor).toBe('RedHatAI')
    expect(facets.family).toBe('gemma-4-26B-A4B-it')
    expect(facets.paramsB).toBe(26)
    expect(facets.activeParamsB).toBe(4)
    expect(facets.quantization).toBe('FP8')
    expect(facets.types).toEqual(['text', 'moe'])
    expect(facets.contextLength).toBe(131_072)
    expect(facets.isTested).toBe(true)
    expect(facets.inCatalog).toBe(true)
    expect(facets.isHuggingFace).toBe(false)
  })

  it('degrades to nulls when no spec is available', () => {
    const facets = modelFacets('some-org/Mystery-Model')
    expect(facets.vendor).toBe('some-org')
    expect(facets.paramsB).toBeNull()
    expect(facets.quantization).toBeNull()
    expect(facets.contextLength).toBeNull()
    expect(facets.types).toEqual(['text'])
  })

  it('handles a bare checkpoint name with no org prefix', () => {
    expect(modelVendor('Qwen3-32B')).toBe('')
    expect(modelFacets('Qwen3-32B').family).toBe('Qwen3-32B')
  })
})

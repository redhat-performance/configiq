import { describe, expect, it } from 'vitest'
import {
  isModelListedAsTested,
  modelParameterBillions,
  modelSizeLabel,
  modelTierLabel,
} from './model-catalogue'

describe('hybrid model catalogue labels', () => {
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

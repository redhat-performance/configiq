const MODEL_PARAMETER_OVERRIDES: Array<[RegExp, number]> = [
  [/qwen3\.8-2\.4t/i, 2_400],
  [/llama-4-maverick/i, 401.6],
  [/llama-4-scout/i, 108.6],
  [/minimax-m2\.5/i, 228.7],
  [/minimax-m2\.7/i, 228.7],
  [/minimax-m3/i, 427],
  [/mimo-v2-flash/i, 309.8],
  [/^deepseek-r1(?:-\d{4})?(?:-(?:base|chat|instruct|bf16|fp8))?$/i, 684.5],
  [/^deepseek-v3(?:\.1|\.2)?(?:-\d{4})?(?:-(?:base|chat|instruct|bf16|fp8))?$/i, 685.4],
  [/deepseek-v4-flash/i, 290.9],
  [/deepseek-v4-pro/i, 1_598.8],
  [/kimi-k3/i, 2_779.9],
  [/kimi-k2(?:\.[5-7])?/i, 1_026.9],
  [/glm-5(?:\.[1-3])?/i, 753.9],
  [/step-3\.7-flash/i, 201.4],
]

export function isModelListedAsTested(modelId: string, testedModels: string[]): boolean {
  const normalizedModelId = modelId.trim().toLowerCase()
  return testedModels.some(candidate => candidate.trim().toLowerCase() === normalizedModelId)
}

export function modelParameterBillions(modelId: string): number | null {
  const checkpoint = modelId.split('/').pop() ?? modelId
  const override = MODEL_PARAMETER_OVERRIDES.find(([pattern]) => pattern.test(checkpoint))
  if (override) return override[1]

  const match = checkpoint.match(/(?:^|[-_])(\d+(?:\.\d+)?)([bm])(?:[-_]|$)/i)
  if (!match) return null
  const value = Number(match[1])
  if (!Number.isFinite(value)) return null
  return match[2].toLowerCase() === 'm' ? value / 1_000 : value
}

export function modelSizeLabel(modelId: string): string {
  const billions = modelParameterBillions(modelId)
  if (billions === null) return 'Parameters unavailable'
  if (billions >= 1_000) {
    return `${Number((billions / 1_000).toFixed(2))}T parameters`
  }
  return `${Number(billions.toFixed(1))}B parameters`
}

export function modelTierLabel(modelId: string): 'Small model' | 'Medium model' | 'Large model' {
  const billions = modelParameterBillions(modelId)
  if (billions !== null && billions <= 12) return 'Small model'
  if (billions !== null && billions <= 50) return 'Medium model'
  return 'Large model'
}

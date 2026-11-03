/**
 * Canonical quantization vocabulary shared by classifier features.
 *
 * Serving runtimes describe quantization with different spellings: the sizing
 * request uses kernel modes such as `int4_wo`, while measured runtime arguments
 * carry values such as `awq` or `fp8_e4m3`. Map both to a single category so a
 * verdict always describes the configuration that was measured or estimated.
 */
export type QuantizationField = 'weight_quantization' | 'kv_quantization'

const UNKNOWN = new Set(['', 'none', 'auto', 'unknown', 'nan', 'null'])

const WEIGHT_ALIASES: Record<string, string> = {
  fp8: 'fp8',
  fp8_e4m3: 'fp8',
  fp8_e5m2: 'fp8',
  fp8_dynamic: 'fp8',
  w8a8_fp8: 'fp8',
  int4: 'int4',
  int4_wo: 'int4',
  w4a16: 'int4',
  awq: 'int4',
  awq_marlin: 'int4',
  awq_marlin_24: 'int4',
  gptq: 'int4',
  gptq_marlin: 'int4',
  int8: 'int8',
  int8_wo: 'int8',
  w8a8: 'int8',
  w8a8_int8: 'int8',
  mxfp4: 'mxfp4',
  w4a16_mxfp4: 'mxfp4',
  w4a16_mxfp4_cutlass: 'mxfp4',
  w4a8_mxfp4_mxfp8: 'mxfp4',
  nvfp4: 'nvfp4',
  modelopt_fp4: 'nvfp4',
  w4a8_nvfp4: 'nvfp4',
}

const KV_ALIASES: Record<string, string> = {
  fp8: 'fp8',
  fp8_e4m3: 'fp8',
  fp8_e5m2: 'fp8',
  fp8_dynamic: 'fp8',
  int8: 'int8',
  nvfp4: 'nvfp4',
}

const ALIASES: Record<QuantizationField, Record<string, string>> = {
  weight_quantization: WEIGHT_ALIASES,
  kv_quantization: KV_ALIASES,
}

export function canonicalQuantization(field: QuantizationField, value: string | null | undefined): string {
  if (value == null) return 'unknown'
  const normalized = value.trim().toLowerCase()
  if (!normalized || UNKNOWN.has(normalized)) return 'unknown'
  return ALIASES[field][normalized] ?? normalized
}

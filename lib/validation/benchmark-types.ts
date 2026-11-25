export const BENCHMARK_FILTER_KEYS = [
  'accelerator',
  'model',
  'backend',
  'backendVersion',
  'isl',
  'osl',
  'tp',
] as const

export type BenchmarkFilterKey = typeof BENCHMARK_FILTER_KEYS[number]
export type BenchmarkFilters = Record<BenchmarkFilterKey, string>
export type DatasetRow = Record<string, unknown>

export interface DatasetConfiguration extends Record<string, string | null> {
  measurement_source: string | null
  benchmark_version: string | null
  backend: string | null
  backend_version: string | null
  precision: string | null
  weight_quantization: string | null
  kv_quantization: string | null
  prefix_caching: string | null
  prefix_tokens: string | null
  prefix_count: string | null
  max_seq_len: string | null
  max_num_seqs: string | null
  max_num_batched_tokens: string | null
  pp: string | null
  dp: string | null
  ep: string | null
  cp: string | null
  moe_tp_size: string | null
  moe_ep_size: string | null
  moe_num_experts: string | null
  moe_top_k: string | null
  enable_expert_parallel: string | null
  runtime_args: string | null
  env_vars: string | null
  guidellm_version: string | null
  vllm_rocm_use_aiter: string | null
  vllm_rocm_use_aiter_moe: string | null
  vllm_rocm_use_aiter_moe_situv2: string | null
  gpu_memory_utilization: string | null
  chunked_prefill_enabled: string | null
  spec_decoding: string | null
  request_type: string | null
  turns: string | null
  tp_prefill: string | null
  tp_decode: string | null
  inferencex_topology: string | null
  inferencex_curve: string | null
  inferencex_offload: string | null
}

export interface DatasetSource {
  id: string
  url: string
  revision: string | null
  lastModified: string | null
  loadedAt: string
  rowCount: number
}

export interface ConfigurationOption {
  key: string
  label: string
  values: DatasetConfiguration
  rows: number
}

export interface BenchmarkPoint {
  concurrency: number
  ttftSeconds: number | null
  tpotSeconds: number | null
  outputTokensPerSecond: number | null
  outputTokensPerSecondPerUser: number | null
  requestLatencySeconds: number | null
  sampleCount: number
}

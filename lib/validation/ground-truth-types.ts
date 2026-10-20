export const GROUND_TRUTH_FILTER_KEYS = [
  'accelerator',
  'model',
  'version',
  'isl',
  'osl',
  'tp',
] as const

export type GroundTruthFilterKey = typeof GROUND_TRUTH_FILTER_KEYS[number]
export type GroundTruthFilters = Record<GroundTruthFilterKey, string>
export type DatasetRow = Record<string, unknown>

export interface DatasetConfiguration extends Record<string, string | null> {
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
  gpu_memory_utilization: string | null
  chunked_prefill_enabled: string | null
  spec_decoding: string | null
  request_type: string | null
  turns: string | null
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

export interface GroundTruthPoint {
  concurrency: number
  ttftSeconds: number | null
  tpotSeconds: number | null
  outputTokensPerSecond: number | null
  outputTokensPerSecondPerUser: number | null
  requestLatencySeconds: number | null
  sampleCount: number
}

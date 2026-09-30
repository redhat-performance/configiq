// GPU Specification Types
// Used by /app/api/gpus and related API response formatters
// GPU data is now fetched from AISimulators /systems endpoint, not stored locally

/**
 * GPU specification type combining AISimulators system data with optional live pricing.
 * Used for type-safe GPU catalog responses.
 */
export interface GpuSpec {
  // Core fields from AISimulators /systems?include=specs
  id: string
  name: string
  display_name: string
  sizer_system_id: string | null
  vendor: 'nvidia' | 'amd'
  vram_gb: number
  hardware_cost_usd: number
  memory_bandwidth_tbps: number
  tokens_per_dollar: number
  tflops_bf16: number
  tflops_fp8: number | null
  mfu_prefill: number  // Model FLOPs Utilization for prefill (compute-bound)
  mfu_decode: number   // Memory bandwidth utilization for decode
  tdp_watts: number
  architecture: 'ampere' | 'hopper' | 'blackwell' | 'ada' | 'cdna2' | 'cdna3' | 'cdna4' | 'ada-lovelace' | 'sm_0' | 'sm_103'
  nvlink_bandwidth_gbps?: number
  color: string  // Hex color for visualization

  // Legacy field names (backward compatibility - computed at runtime)
  vramGb: number
  memoryBandwidthGbps: number
  bandwidthTbps: number
  tflops: number
  pricePerHour: number
  hardwareCostPerGpu: number
  powerWatts: number
  cloudAvailabilityPct: number
  tpuAvailabilityPct: number
}

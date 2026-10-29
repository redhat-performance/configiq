export interface WorkloadPreset {
  key: string
  label: string
  isl: number
  osl: number
}

export const DEFAULT_WORKLOAD: WorkloadPreset = {
  key: 'default',
  label: 'Default',
  isl: 2048,
  osl: 128
}

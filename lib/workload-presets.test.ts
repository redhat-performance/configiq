import { describe, expect, it } from 'vitest'
import appConfig from '../public/config.json'
import {
  DEFAULT_WORKLOAD,
  matchWorkloadPreset,
  matchWorkloadPresetByTokens,
  workloadPresetOptions,
  type WorkloadPreset,
} from './workload-presets'

const presets: readonly WorkloadPreset[] = [
  { key: 'chat', label: 'Interactive chat', isl: 4000, osl: 300, prefix: 3950 },
  { key: 'rag', label: 'RAG / Search', isl: 16384, osl: 512, prefix: 2000 },
]

describe('workload presets', () => {
  it('starts with the system prompt every request shares', () => {
    expect(DEFAULT_WORKLOAD.prefix).toBe(200)
    expect(workloadPresetOptions(presets)[0]).toBe(DEFAULT_WORKLOAD)
    expect(matchWorkloadPreset(workloadPresetOptions(presets), 2048, 128, 200)).toBe(DEFAULT_WORKLOAD)
    expect(matchWorkloadPreset(workloadPresetOptions(presets), 2048, 128, 0)).toBeNull()
  })

  it('matches a preset only when lengths and prefix agree', () => {
    expect(matchWorkloadPreset(presets, 4000, 300, 3950)).toBe(presets[0])
    expect(matchWorkloadPreset(presets, 4000, 300, 0)).toBeNull()
    expect(matchWorkloadPreset(presets, 4000, 301, 3950)).toBeNull()
  })

  it('treats a preset without a stated prefix as cold', () => {
    const legacy: WorkloadPreset = { key: 'legacy', label: 'Legacy', isl: 1000, osl: 100 }
    expect(matchWorkloadPreset([legacy], 1000, 100, 0)).toBe(legacy)
    expect(matchWorkloadPreset([legacy], 1000, 100, 10)).toBeNull()
  })

  it('still matches the combined token budget for a page with no prefix control', () => {
    expect(matchWorkloadPresetByTokens(presets, 4300)).toBe(presets[0])
    expect(matchWorkloadPresetByTokens(presets, 16896)).toBe(presets[1])
  })

  it('gives every shipped preset a shared-prefix assumption', () => {
    // #236 asks for a sane prefix on every preset. A preset that leaves the
    // field out is sized as a cold request, so a new preset must state one
    // rather than inherit that silence. The values themselves are product
    // config and change deliberately; only the assumption is an invariant.
    for (const preset of [DEFAULT_WORKLOAD, ...appConfig.workloadPresets]) {
      expect(preset.prefix).toBeDefined()
      expect(preset.prefix ?? 0).toBeGreaterThanOrEqual(0)
      expect(preset.prefix ?? 0).toBeLessThanOrEqual(preset.isl)
    }
  })
})

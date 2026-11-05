import { describe, expect, it } from 'vitest'
import type { HybridWorkload } from './calc'
import {
  analyzeRequestMix,
  DEFAULT_HYBRID_WORKLOAD,
  matchRequestShapePreset,
  proposeRequestMixAlignment,
  REQUEST_MIX_ALIGNMENT_TOLERANCE,
} from './workload-basis'
import { DEFAULT_WORKLOAD, type WorkloadPreset } from '@/lib/workload-presets'

// The preset list is app configuration (DEFAULT_WORKLOAD plus /config.json);
// match against a representative local list.
const presets: readonly WorkloadPreset[] = [
  DEFAULT_WORKLOAD,
  { key: 'rag', label: 'RAG / Search', isl: 6_144, osl: 256, ttft: 800, tpot: 40, concurrency: 8, prefix: 1_024 },
  { key: 'agentic', label: 'Agentic / Coding', isl: 12_288, osl: 1_024, ttft: 1_500, tpot: 25, concurrency: 4, prefix: 2_048 },
]

function workload(overrides: Partial<HybridWorkload> = {}): HybridWorkload {
  return {
    monthlyInputTokens: 2_000_000_000,
    monthlyOutputTokens: 500_000_000,
    averageInputTokens: 2_048,
    averageOutputTokens: 128,
    activeHoursPerMonth: 730,
    peakToAverage: 1,
    ...overrides,
  }
}

describe('analyzeRequestMix', () => {
  it('uses the shared Default request shape without changing that preset', () => {
    const mix = analyzeRequestMix(DEFAULT_HYBRID_WORKLOAD)
    expect(DEFAULT_HYBRID_WORKLOAD.averageInputTokens).toBe(DEFAULT_WORKLOAD.isl)
    expect(DEFAULT_HYBRID_WORKLOAD.averageOutputTokens).toBe(DEFAULT_WORKLOAD.osl)
    expect(matchRequestShapePreset({
      averageInputTokens: DEFAULT_HYBRID_WORKLOAD.averageInputTokens,
      averageOutputTokens: DEFAULT_HYBRID_WORKLOAD.averageOutputTokens,
      targetTtftMs: DEFAULT_HYBRID_WORKLOAD.targetTtftMs,
      targetTpotMs: DEFAULT_HYBRID_WORKLOAD.targetTpotMs,
      targetConcurrency: DEFAULT_HYBRID_WORKLOAD.targetConcurrency,
      prefixTokens: DEFAULT_HYBRID_WORKLOAD.prefixTokens,
    }, [DEFAULT_WORKLOAD])).toBe(DEFAULT_WORKLOAD)
    expect(mix.aligned).toBe(true)
    expect(mix.monthlyInputTokens + mix.monthlyOutputTokens).toBe(2_550_000_000)
    expect(mix.inputImpliedRequests).toBe(1_171_875)
    expect(mix.outputImpliedRequests).toBe(1_171_875)
    expect(mix.sizingRequests).toBe(1_171_875)
    expect(DEFAULT_WORKLOAD.isl).toBe(2048)
    expect(DEFAULT_WORKLOAD.osl).toBe(128)
  })
  it('derives both implied request counts and the conservative sizing count', () => {
    const analysis = analyzeRequestMix(workload())
    expect(analysis.monthlyInputTokens).toBe(2_000_000_000)
    expect(analysis.monthlyOutputTokens).toBe(500_000_000)
    expect(analysis.inputImpliedRequests).toBeCloseTo(976_562.5, 4)
    expect(analysis.outputImpliedRequests).toBeCloseTo(3_906_250, 4)
    expect(analysis.sizingRequests).toBeCloseTo(3_906_250, 4)
    expect(analysis.twoSided).toBe(true)
    expect(analysis.aligned).toBe(false)
    expect(analysis.relativeMismatch).toBeGreaterThan(0.7)
  })

  it('flags mixes within tolerance as aligned', () => {
    const analysis = analyzeRequestMix(workload({ averageInputTokens: 512 }))
    expect(analysis.inputImpliedRequests).toBeCloseTo(3_906_250, 4)
    expect(analysis.relativeMismatch ?? 1).toBeLessThanOrEqual(REQUEST_MIX_ALIGNMENT_TOLERANCE)
    expect(analysis.aligned).toBe(true)
  })

  it('treats the exact 0.5% boundary as aligned and just beyond it as misaligned', () => {
    // 1,000 input-implied requests vs 995 output-implied: |1000 - 995| / 1000
    // is exactly the tolerance.
    const boundary = analyzeRequestMix(workload({
      monthlyInputTokens: 1_000_000,
      monthlyOutputTokens: 995_000,
      averageInputTokens: 1_000,
      averageOutputTokens: 1_000,
    }))
    expect(boundary.relativeMismatch).toBe(REQUEST_MIX_ALIGNMENT_TOLERANCE)
    expect(boundary.aligned).toBe(true)

    const beyond = analyzeRequestMix(workload({
      monthlyInputTokens: 1_000_000,
      monthlyOutputTokens: 994_000,
      averageInputTokens: 1_000,
      averageOutputTokens: 1_000,
    }))
    expect(beyond.relativeMismatch ?? 0).toBeGreaterThan(REQUEST_MIX_ALIGNMENT_TOLERANCE)
    expect(beyond.aligned).toBe(false)
  })

  it('follows the input-implied request count when input requests dominate', () => {
    // 1,000 input-implied requests vs 500 output-implied: the input side
    // drives sizing, the mirror of the default output-heavy workload.
    const analysis = analyzeRequestMix(workload({
      monthlyInputTokens: 2_000_000,
      monthlyOutputTokens: 500_000,
      averageInputTokens: 2_000,
      averageOutputTokens: 1_000,
    }))
    expect(analysis.inputImpliedRequests).toBe(1_000)
    expect(analysis.outputImpliedRequests).toBe(500)
    expect(analysis.sizingRequests).toBe(1_000)
    expect(analysis.aligned).toBe(false)
  })

  it('treats one-sided mixes as not two-sided and follows the positive side', () => {
    const analysis = analyzeRequestMix(workload({ monthlyOutputTokens: 0 }))
    expect(analysis.twoSided).toBe(false)
    expect(analysis.outputImpliedRequests).toBeNull()
    expect(analysis.sizingRequests).toBeCloseTo(976_562.5, 4)
    expect(analysis.relativeMismatch).toBeNull()
    expect(analysis.aligned).toBe(false)
  })

  it('clamps invalid inputs like the engine does', () => {
    const analysis = analyzeRequestMix(workload({
      monthlyInputTokens: Number.NaN,
      monthlyOutputTokens: -1,
      averageInputTokens: 0,
      averageOutputTokens: Number.POSITIVE_INFINITY,
    }))
    expect(analysis.monthlyInputTokens).toBe(0)
    expect(analysis.monthlyOutputTokens).toBe(0)
    expect(analysis.twoSided).toBe(false)
    expect(analysis.inputImpliedRequests).toBeNull()
    expect(analysis.outputImpliedRequests).toBeNull()
    expect(analysis.sizingRequests).toBeNull()
  })
})

describe('proposeRequestMixAlignment', () => {
  it('proposes the output-implied request count, changing only the input length', () => {
    const alignment = proposeRequestMixAlignment(workload(), 0)
    expect(alignment.proposedInputTokens).toBe(512)
    expect(alignment.reason).toBeNull()
    const revised = analyzeRequestMix(workload({ averageInputTokens: alignment.proposedInputTokens! }))
    expect(revised.aligned).toBe(true)
  })

  it('aligns an input-heavy mix by raising the input length to the output-implied count', () => {
    // 1,000 input-implied vs 500 output-implied: the aligned input length is
    // 2,000,000 / 500 = 4,000 tokens (= B × I / O), so both sides imply 500.
    const alignment = proposeRequestMixAlignment(workload({
      monthlyInputTokens: 2_000_000,
      monthlyOutputTokens: 500_000,
      averageInputTokens: 2_000,
      averageOutputTokens: 1_000,
    }), 0)
    expect(alignment.proposedInputTokens).toBe(4_000)
    expect(alignment.reason).toBeNull()
    const revised = analyzeRequestMix(workload({
      monthlyInputTokens: 2_000_000,
      monthlyOutputTokens: 500_000,
      averageInputTokens: alignment.proposedInputTokens!,
      averageOutputTokens: 1_000,
    }))
    expect(revised.relativeMismatch).toBe(0)
    expect(revised.aligned).toBe(true)
  })

  it('rejects proposals beyond the safe-integer range', () => {
    // B × I / O = 1,000 × 9e18 / 1 = 9e21: finite, but not a safe integer.
    const alignment = proposeRequestMixAlignment(workload({
      monthlyInputTokens: 9e18,
      monthlyOutputTokens: 1,
      averageInputTokens: 1,
      averageOutputTokens: 1_000,
    }), 0)
    expect(alignment.proposedInputTokens).toBeNull()
    expect(alignment.reason).toContain('not a safe integer')
  })

  it('returns no proposal when the mix is already aligned', () => {
    const alignment = proposeRequestMixAlignment(workload({ averageInputTokens: 512 }), 0)
    expect(alignment.proposedInputTokens).toBeNull()
    expect(alignment.reason).toBeNull()
  })

  it('returns no proposal for one-sided mixes', () => {
    const alignment = proposeRequestMixAlignment(workload({ monthlyOutputTokens: 0 }), 0)
    expect(alignment.proposedInputTokens).toBeNull()
    expect(alignment.reason).toContain('positive monthly input and output volumes')
  })

  it('refuses proposals shorter than the cached prefix', () => {
    const alignment = proposeRequestMixAlignment(workload(), 600)
    expect(alignment.proposedInputTokens).toBeNull()
    expect(alignment.reason).toContain('cached prefix')
  })

  it('rejects non-finite or sub-token proposals', () => {
    const tiny = proposeRequestMixAlignment(workload({ monthlyInputTokens: 10, averageOutputTokens: 1 }), 0)
    // raw = 1 * 10 / 500,000,000 = 0.00000002 → rounds to 0, below one token.
    expect(tiny.proposedInputTokens).toBeNull()
    expect(tiny.reason).toContain('below one token')

    const infinite = proposeRequestMixAlignment(workload({
      monthlyInputTokens: Number.POSITIVE_INFINITY,
    }), 0)
    // Non-finite monthly totals are clamped to zero exactly like the engine,
    // so the mix is one-sided rather than producing an unrepresentable proposal.
    expect(infinite.proposedInputTokens).toBeNull()
    expect(infinite.reason).toContain('positive monthly input and output volumes')
  })

  it('reports rounding failure when rounding cannot reach tolerance', () => {
    // raw = B × I / O = 1 × 1 / 3 = 0.333… → rounds to 0, below one token,
    // and larger mixes with tiny output can never round to an aligned length.
    const alignment = proposeRequestMixAlignment(workload({
      monthlyInputTokens: 1,
      monthlyOutputTokens: 3,
      averageInputTokens: 3,
      averageOutputTokens: 1,
    }), 0)
    expect(alignment.proposedInputTokens).toBeNull()
    expect(alignment.reason).toContain('below one token')
  })
})

describe('matchRequestShapePreset', () => {
  it('matches the default preset exactly on all six fields', () => {
    const preset = matchRequestShapePreset({
      averageInputTokens: DEFAULT_WORKLOAD.isl,
      averageOutputTokens: DEFAULT_WORKLOAD.osl,
      targetTtftMs: DEFAULT_WORKLOAD.ttft,
      targetTpotMs: DEFAULT_WORKLOAD.tpot,
      targetConcurrency: DEFAULT_WORKLOAD.concurrency,
      prefixTokens: DEFAULT_WORKLOAD.prefix,
    }, presets)
    expect(preset?.key).toBe(DEFAULT_WORKLOAD.key)
  })

  it('returns null when any of the six preset-controlled fields differs', () => {
    const exact = {
      averageInputTokens: DEFAULT_WORKLOAD.isl,
      averageOutputTokens: DEFAULT_WORKLOAD.osl,
      targetTtftMs: DEFAULT_WORKLOAD.ttft,
      targetTpotMs: DEFAULT_WORKLOAD.tpot,
      targetConcurrency: DEFAULT_WORKLOAD.concurrency,
      prefixTokens: DEFAULT_WORKLOAD.prefix,
    }
    const singleFieldEdits: Array<[keyof typeof exact, number]> = [
      ['averageInputTokens', DEFAULT_WORKLOAD.isl + 1],
      ['averageOutputTokens', DEFAULT_WORKLOAD.osl + 1],
      ['targetTtftMs', DEFAULT_WORKLOAD.ttft + 1],
      ['targetTpotMs', DEFAULT_WORKLOAD.tpot + 1],
      ['targetConcurrency', DEFAULT_WORKLOAD.concurrency + 1],
      ['prefixTokens', DEFAULT_WORKLOAD.prefix + 1],
    ]
    for (const [field, value] of singleFieldEdits) {
      expect(matchRequestShapePreset({ ...exact, [field]: value }, presets), field).toBeNull()
    }
    // Every other preset is matched by its own exact shape, and a single-field
    // edit away from any of them is also Custom (no preset shares the other
    // five fields with another, so the +1 can never complete a different
    // preset's shape).
    for (const preset of presets) {
      const shape = {
        averageInputTokens: preset.isl,
        averageOutputTokens: preset.osl,
        targetTtftMs: preset.ttft,
        targetTpotMs: preset.tpot,
        targetConcurrency: preset.concurrency,
        prefixTokens: preset.prefix,
      }
      expect(matchRequestShapePreset(shape, presets)?.key, preset.key).toBe(preset.key)
      for (const [field] of singleFieldEdits) {
        expect(matchRequestShapePreset(
          { ...shape, [field]: shape[field] + 1 },
          presets,
        ), `${preset.key} ${field}`).toBeNull()
      }
    }
  })
})

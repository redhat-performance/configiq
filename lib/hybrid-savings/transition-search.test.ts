import { describe, expect, it } from 'vitest'
import { costOnCurve, firstWinningVolume, TransitionSearchLimitError, type CostCurve } from './transition-search'

function exhaustiveFirst(targets: CostCurve[], competitors: CostCurve[], maximum: number): number | null {
  for (let volume = 1; volume <= maximum; volume += 1) {
    if (targets.some(target => competitors.every(competitor =>
      costOnCurve(target, volume) <= costOnCurve(competitor, volume),
    ))) return volume
  }
  return null
}

describe('exact cost transitions', () => {
  it('agrees with exhaustive search across stepped, linear and scale-to-zero costs', () => {
    let seed = 21
    const random = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0
      return seed / 2 ** 32
    }
    const curve = (): CostCurve => {
      const fixed = Math.floor(random() * 60)
      const rate = Math.floor(random() * 40) / 100
      const kind = Math.floor(random() * 3)
      if (kind === 0) return { kind: 'linear', fixed, rate }
      if (kind === 1) return {
        kind: 'step', fixed, increment: 1 + Math.floor(random() * 30), capacity: 2 + random() * 75,
      }
      return {
        kind: 'scale-to-zero', fixed, rate, firstKnot: 5 + random() * 100,
        parallelism: 1 + Math.floor(random() * 8),
      }
    }

    for (let scenario = 0; scenario < 200; scenario += 1) {
      const targets = [curve(), curve()]
      const competitors = [curve(), curve(), curve()]
      expect(firstWinningVolume(targets, competitors, 2_000)).toBe(
        exhaustiveFirst(targets, competitors, 2_000),
      )
    }
  })

  it('finds a brief win between later capacity steps', () => {
    const target: CostCurve = { kind: 'step', fixed: 0, increment: 20, capacity: 10 }
    const competitor: CostCurve = { kind: 'linear', fixed: 0, rate: 2.1 }
    expect(firstWinningVolume([target], [competitor], 1_000_000_000_000)).toBe(10)
    expect(costOnCurve(target, 11)).toBeGreaterThan(costOnCurve(competitor, 11))
  })

  it('reports an unverified transition instead of silently skipping a search', () => {
    const target: CostCurve = { kind: 'step', fixed: 10, increment: 3, capacity: 7 }
    const competitor: CostCurve = { kind: 'linear', fixed: 0, rate: 0.5 }
    expect(() => firstWinningVolume([target], [competitor], 1_000, { remaining: 0 }))
      .toThrow(TransitionSearchLimitError)
  })
})

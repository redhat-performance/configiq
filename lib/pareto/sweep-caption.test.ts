import { describe, it, expect } from 'vitest'
import { describeSweepCoverage } from './sweep-caption'

const feasible = { status: 'feasible' as const }
const infeasible = { status: 'infeasible' as const }
const timedOut = { status: 'timed_out' as const }

describe('describeSweepCoverage', () => {
  it('calls an enumerated sweep exhaustive only when the grid finished', () => {
    expect(
      describeSweepCoverage(
        { algorithm: 'enumerate', evaluated: 14, budgetExhausted: false },
        [feasible, feasible, infeasible, timedOut],
      ),
    ).toBe('14 evaluated · 2 feasible · 1 timed out · 1 infeasible — exhaustive over the enumerated grid.')
  })

  it('reports budget exhaustion instead of claiming an exhaustive enumeration', () => {
    expect(
      describeSweepCoverage(
        { algorithm: 'enumerate', evaluated: 6, budgetExhausted: true },
        [feasible],
      ),
    ).toBe('6 evaluated · 1 feasible — the search time budget stopped the enumeration before every grid point ran.')
  })

  it('does not call a sampled sweep exhaustive', () => {
    expect(
      describeSweepCoverage(
        { algorithm: 'random', evaluated: 20, budgetExhausted: false },
        [feasible],
      ),
    ).toBe('20 sampled · 1 feasible — not exhaustive.')
  })

  it('reports a wall-clock budget stop for a sampled sweep', () => {
    expect(
      describeSweepCoverage(
        { algorithm: 'random', evaluated: 20, budgetExhausted: true },
        [feasible],
      ),
    ).toBe('20 sampled · 1 feasible — not exhaustive: the search time budget was reached.')
  })

  it('falls back to the ledger length when the sweep did not report a count', () => {
    expect(describeSweepCoverage(null, [feasible, feasible])).toBe(
      '2 sampled · 2 feasible — not exhaustive.',
    )
  })
})

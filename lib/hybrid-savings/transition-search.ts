/** Cost shapes used only to locate transitions. Displayed costs still come from calc.ts. */
export type CostCurve =
  | { kind: 'linear'; fixed: number; rate: number }
  | { kind: 'step'; fixed: number; increment: number; capacity: number }
  | { kind: 'scale-to-zero'; fixed: number; rate: number; firstKnot: number; parallelism: number }

type Line = { fixed: number; rate: number }

export class TransitionSearchLimitError extends Error {
  constructor() { super('The transition search exceeded its safe work limit') }
}

export interface SearchBudget { remaining: number }

function segment(curve: CostCurve, volume: number): number {
  if (curve.kind !== 'scale-to-zero') return 0
  if (volume <= curve.firstKnot) return 0
  if (volume <= curve.firstKnot * curve.parallelism) return 1
  return 2
}

function lineAt(curve: CostCurve, volume: number): Line {
  if (curve.kind === 'linear') return { fixed: curve.fixed, rate: curve.rate }
  if (curve.kind === 'step') {
    return { fixed: curve.fixed + curve.increment * Math.ceil(volume / curve.capacity), rate: 0 }
  }
  switch (segment(curve, volume)) {
    case 0: return { fixed: curve.fixed, rate: curve.rate }
    case 1: return { fixed: curve.fixed + curve.rate * curve.firstKnot, rate: 0 }
    default: return { fixed: curve.fixed, rate: curve.rate / curve.parallelism }
  }
}

export function costOnCurve(curve: CostCurve, volume: number): number {
  const line = lineAt(curve, volume)
  return line.fixed + line.rate * volume
}

function affineOver(curve: CostCurve, first: number, last: number): Line | null {
  if (curve.kind === 'step' && Math.ceil(first / curve.capacity) !== Math.ceil(last / curve.capacity)) return null
  if (curve.kind === 'scale-to-zero' && segment(curve, first) !== segment(curve, last)) return null
  return lineAt(curve, first)
}

function lowerCost(curve: CostCurve, volume: number): number {
  return curve.kind === 'step'
    ? curve.fixed + curve.increment * volume / curve.capacity
    : costOnCurve(curve, volume)
}

function upperCost(curve: CostCurve, volume: number): number {
  return curve.kind === 'step'
    ? curve.fixed + curve.increment * (volume / curve.capacity + 1)
    : costOnCurve(curve, volume)
}

function integerKnots(curve: CostCurve, first: number, last: number): number[] {
  if (curve.kind !== 'scale-to-zero') return []
  return [curve.firstKnot, curve.firstKnot * curve.parallelism]
    .flatMap(knot => [Math.floor(knot), Math.ceil(knot)])
    .filter(knot => knot > first && knot < last)
}

/** A strict lower bound on target minus competitor over an integer interval. */
function cannotBeat(
  target: CostCurve,
  competitor: CostCurve,
  first: number,
  last: number,
): boolean {
  if (
    target.kind === 'step' && competitor.kind === 'step' &&
    target.capacity <= competitor.capacity && target.increment >= competitor.increment &&
    target.fixed + target.increment > competitor.fixed + competitor.increment
  ) return true

  const checkpoints = [first, last, ...integerKnots(target, first, last), ...integerKnots(competitor, first, last)]
  const differences = checkpoints.map(volume => lowerCost(target, volume) - upperCost(competitor, volume))
  const tolerance = Math.max(
    1,
    ...checkpoints.map(volume => Math.max(lowerCost(target, volume), upperCost(competitor, volume))),
  ) * 1e-12
  return Math.min(...differences) > tolerance
}

function firstAffineWin(target: Line, competitors: Line[], first: number, last: number): number | null {
  let lower = first
  let upper = last
  for (const competitor of competitors) {
    const slope = target.rate - competitor.rate
    const wins = (volume: number) =>
      target.fixed + target.rate * volume <= competitor.fixed + competitor.rate * volume
    if (slope === 0) {
      if (!wins(first)) return null
    } else if (slope < 0) {
      if (!wins(last)) return null
      let start = first
      let end = last
      while (start < end) {
        const middle = Math.floor((start + end) / 2)
        if (wins(middle)) end = middle
        else start = middle + 1
      }
      lower = Math.max(lower, start)
    } else {
      if (!wins(first)) return null
      let start = first
      let end = last
      while (start < end) {
        const middle = Math.ceil((start + end) / 2)
        if (wins(middle)) start = middle
        else end = middle - 1
      }
      upper = Math.min(upper, start)
    }
  }
  return lower <= upper ? lower : null
}

/** Find the first whole-token volume where any target is no dearer than every competitor. */
export function firstWinningVolume(
  targets: CostCurve[],
  competitors: CostCurve[],
  maximum: number,
  budget?: SearchBudget,
): number | null {
  if (targets.length === 0) return null
  let firstWinner: number | null = null
  for (const target of targets) {
    const stack: Array<[number, number]> = [[1, firstWinner === null ? maximum : firstWinner - 1]]
    while (stack.length > 0) {
      if (budget && --budget.remaining < 0) throw new TransitionSearchLimitError()
      const [first, last] = stack.pop()!
      if (first > last || (firstWinner !== null && first >= firstWinner)) continue
      if (competitors.some(competitor => cannotBeat(target, competitor, first, last))) continue

      const targetLine = affineOver(target, first, last)
      const competitorLines = competitors.map(competitor => affineOver(competitor, first, last))
      if (targetLine && competitorLines.every((line): line is Line => line !== null)) {
        const winner = firstAffineWin(targetLine, competitorLines, first, last)
        if (winner !== null) {
          firstWinner = winner
          break
        }
        continue
      }

      if (first === last) {
        if (competitors.every(competitor => costOnCurve(target, first) <= costOnCurve(competitor, first))) {
          firstWinner = first
          break
        }
        continue
      }
      const middle = Math.floor((first + last) / 2)
      stack.push([middle + 1, last], [first, middle])
    }
  }
  return firstWinner
}

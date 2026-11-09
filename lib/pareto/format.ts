/**
 * Render an AISimulate candidate identifier with a three-digit sequence.
 *
 * The SDK numbers candidates with six digits (`candidate-000001`). A Pareto
 * sweep never approaches 1,000 candidates, so the display form drops the
 * surplus zero padding. Non-AISimulate identifiers (for example BLIS
 * `agg-tp1-c1`) pass through unchanged, and numbers above 999 keep all of
 * their digits rather than being truncated.
 */
const CANDIDATE_ID_PATTERN = /^candidate-0*(\d+)$/

export function formatCandidateId(id: string): string {
  const match = CANDIDATE_ID_PATTERN.exec(id)
  if (!match) return id
  return `candidate-${String(Number(match[1])).padStart(3, '0')}`
}

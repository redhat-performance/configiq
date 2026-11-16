/**
 * Fire-and-forget tour telemetry.
 *
 * The browser has no metrics provider, so events go to a same-origin route
 * that increments the server counters. Failures are ignored: help must work
 * when telemetry does not.
 */
export function recordTourEvent(tourId: string, phase: 'started' | 'completed'): void {
  if (typeof window === 'undefined') return;

  try {
    void fetch('/api/help/tour', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: tourId, phase }),
      keepalive: true,
    }).catch(() => undefined);
  } catch {
    // Ignore network failures.
  }
}

/**
 * Counts a search without recording the question. Only whether the index had
 * an answer is sent, so a search term never leaves the page.
 */
export function recordSearchEvent(matched: number): void {
  if (typeof window === 'undefined') return;

  try {
    void fetch('/api/help/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ matched: Math.max(0, Math.trunc(matched)) }),
      keepalive: true,
    }).catch(() => undefined);
  } catch {
    // Ignore network failures.
  }
}

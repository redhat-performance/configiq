/**
 * Debug panel data model.
 *
 * A page owns the run it executed and hands the panel a list of runs; the
 * panel only renders. Every artifact is a pane — the request that was sent,
 * the response that came back, the command and files that reproduce the run
 * outside ConfigIQ — so a run can gain artifacts without the panel learning a
 * new shape.
 */

/** A request or response body as the browser captured it. */
export type DebugPayload =
  | Record<string, unknown>
  | Record<string, unknown>[]
  | null

export type DebugLanguage = 'json' | 'yaml' | 'shell' | 'text'

export interface DebugPane {
  /** Stable within its run; names the tab and scopes the copy action. */
  id: string
  label: string
  language: DebugLanguage
  /** Exact text a reader would paste elsewhere. Never re-serialized. */
  contents: string
  /**
   * Optional sub-tab this artifact belongs to: a serving pool, a candidate, a
   * search window. Panes that share a group render behind one group tab.
   */
  group?: string
}

export interface DebugRun {
  id: string
  label: string
  /** Short provenance marker, such as the simulator that produced the run. */
  badge?: string
  panes: DebugPane[]
  /** Lines shown above the artifacts, such as why an artifact is absent. */
  notes?: string[]
}

/** A request/response pair captured on the page, before replay data exists. */
export interface DebugExchange {
  request: DebugPayload
  response: DebugPayload
  /** Run id; keep it stable across re-renders of the same exchange. */
  id?: string
  label?: string
  badge?: string
}

function serialize(value: unknown): string {
  return JSON.stringify(value, null, 2)
}

/**
 * Describe a captured request/response pair as one run with two panes.
 *
 * This carries exactly what the page already holds, so the panel can change
 * shape without changing the data under it. A missing half stays visible as a
 * pane: "no response" is information, and hiding the tab would lose it.
 */
export function exchangeRuns(exchange: DebugExchange): DebugRun[] {
  const { request, response } = exchange
  if (request == null && response == null) return []

  return [{
    id: exchange.id ?? 'exchange',
    label: exchange.label ?? 'Run',
    badge: exchange.badge,
    panes: [
      {
        id: 'request',
        label: 'Request',
        language: 'json',
        contents: request == null ? '(no request)' : serialize(request),
      },
      {
        id: 'response',
        label: 'Response',
        language: 'json',
        contents: response == null ? '(no response)' : serialize(response),
      },
    ],
  }]
}

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
   * search window. Panes that share a group render behind one group tab;
   * ungrouped panes stay visible whichever group is selected.
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
  /**
   * Replay record(s) to show, when the page holds them separately from the
   * response. When omitted, a record carried on the response payload is used.
   * `null` means the page knows there is none, and the payload is not re-read.
   */
  debug?: DebugRecord | DebugRecord[] | null
}

// ─── Wire shape ──────────────────────────────────────────────────────────────
//
// The service returns what a request executed when the caller asks for
// `include=debug`. These types mirror that payload and are the only place the
// panel learns its shape: an artifact is a command plus the files it reads.

/** One input file of a recorded run, relative to the bundle root. */
export interface DebugFile {
  path: string
  contents: string
}

/** One runnable unit: a command plus the input files it reads. */
export interface DebugInvocation {
  id: string
  label: string
  /** Shell command, already quoted, reading its inputs from the bundle. */
  command: string
  /**
   * How the simulator ran: `subprocess` reproduces the exact process,
   * `in_process` the equivalent command for work the service called directly.
   * Kept open so a new execution kind does not require a client release.
   */
  execution: string
  group: string | null
  /** Set when this invocation reproduces one Pareto candidate. */
  candidate_id: string | null
  notes: string[]
  files: DebugFile[]
}

/** What one request executed, returned inline when `include=debug` was asked. */
export interface DebugRecord {
  endpoint: string
  simulator: string
  execution: string
  created_at: number
  /** Releases and data revisions the run was made against. */
  versions: Record<string, string | null>
  /** What the run chose, in one line, when it had a choice to make. */
  selection: string | null
  notes: string[]
  invocations: DebugInvocation[]
  /** `run.sh`: every non-candidate invocation in execution order. */
  script: string | null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function optionalString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function parseFiles(value: unknown): DebugFile[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item): DebugFile[] => {
    if (!isRecord(item)) return []
    return typeof item.path === 'string' && typeof item.contents === 'string'
      ? [{ path: item.path, contents: item.contents }]
      : []
  })
}

function parseInvocation(value: unknown): DebugInvocation | null {
  if (!isRecord(value)) return null
  const { id, label, command, execution } = value
  if (
    typeof id !== 'string' ||
    typeof label !== 'string' ||
    typeof command !== 'string' ||
    typeof execution !== 'string'
  ) {
    return null
  }
  return {
    id,
    label,
    command,
    execution,
    group: optionalString(value.group),
    candidate_id: optionalString(value.candidate_id),
    notes: Array.isArray(value.notes)
      ? value.notes.filter((note): note is string => typeof note === 'string')
      : [],
    files: parseFiles(value.files),
  }
}

/**
 * Read a replay record from an API payload, or null when it is absent or the
 * shape is unknown.
 *
 * Recording is best-effort on the service side, so an absent record is normal.
 * A record that does not describe runnable work is treated the same way rather
 * than rendered as an empty command.
 */
export function parseDebugRecord(value: unknown): DebugRecord | null {
  if (!isRecord(value)) return null
  const { endpoint, simulator, execution } = value
  if (
    typeof endpoint !== 'string' ||
    typeof simulator !== 'string' ||
    typeof execution !== 'string'
  ) {
    return null
  }

  const versions: Record<string, string | null> = {}
  if (isRecord(value.versions)) {
    for (const [name, version] of Object.entries(value.versions)) {
      if (typeof version === 'string' || version === null) versions[name] = version
    }
  }

  return {
    endpoint,
    simulator,
    execution,
    created_at: typeof value.created_at === 'number' ? value.created_at : 0,
    versions,
    selection: optionalString(value.selection),
    notes: Array.isArray(value.notes)
      ? value.notes.filter((note): note is string => typeof note === 'string')
      : [],
    invocations: Array.isArray(value.invocations)
      ? value.invocations
          .map(parseInvocation)
          .filter((invocation): invocation is DebugInvocation => invocation !== null)
      : [],
    script: optionalString(value.script),
  }
}

// ─── Panel mapping ───────────────────────────────────────────────────────────

function languageForPath(path: string): DebugLanguage {
  const lower = path.toLowerCase()
  if (lower.endsWith('.yaml') || lower.endsWith('.yml')) return 'yaml'
  if (lower.endsWith('.json')) return 'json'
  if (lower.endsWith('.sh')) return 'shell'
  return 'text'
}

function groupNames(invocations: DebugInvocation[]): string[] {
  return Array.from(new Set(
    invocations
      .map(invocation => invocation.group)
      .filter((group): group is string => typeof group === 'string' && group !== ''),
  ))
}

/**
 * One pane describing the pins and the choice the run was made with. With
 * several records (a page fanning out to more than one request) each record is
 * an entry rather than being flattened into a misleading single summary.
 */
function versionsPane(records: DebugRecord[]): DebugPane {
  const describe = (record: DebugRecord) => ({
    endpoint: record.endpoint,
    simulator: record.simulator,
    execution: record.execution,
    selection: record.selection,
    versions: record.versions,
  })
  return {
    id: 'versions',
    label: 'Versions',
    language: 'json',
    contents: JSON.stringify(
      records.length === 1 ? describe(records[0]) : records.map(describe),
      null,
      2,
    ),
  }
}

/**
 * Describe recorded replay data as one run of artifacts.
 *
 * Every command travels with the files it reads, so a copied command and its
 * copied inputs reproduce the run outside ConfigIQ. Per-candidate commands are
 * left out: a sweep records one per candidate, and a candidate is reproduced
 * from its own row rather than by listing hundreds of commands here.
 */
export function replayRuns(
  debug: DebugRecord | DebugRecord[] | null | undefined,
): DebugRun[] {
  if (debug == null) return []
  const records = Array.isArray(debug) ? debug : [debug]
  if (records.length === 0) return []

  const multiple = records.length > 1
  const panes: DebugPane[] = [versionsPane(records)]
  const notes: string[] = []
  let omittedCandidates = 0

  records.forEach((record, recordIndex) => {
    record.notes.forEach(note => {
      notes.push(multiple ? `${record.endpoint}: ${note}` : note)
    })

    if (record.script) {
      const groups = groupNames(record.invocations)
      const suffix = multiple
        ? ` · ${groups.length > 0 ? groups.join(' + ') : recordIndex + 1}`
        : ''
      panes.push({
        id: `script-${recordIndex}`,
        label: `run.sh${suffix}`,
        language: 'shell',
        contents: record.script,
      })
    }

    for (const invocation of record.invocations) {
      if (invocation.candidate_id != null) {
        omittedCandidates += 1
        continue
      }
      const prefix = multiple ? `${recordIndex}-` : ''
      const group = invocation.group ?? undefined
      invocation.notes.forEach(note => notes.push(`${invocation.label}: ${note}`))
      panes.push({
        id: `${prefix}${invocation.id}`,
        label: invocation.label,
        language: 'shell',
        contents: invocation.command,
        ...(group === undefined ? {} : { group }),
      })
      for (const file of invocation.files) {
        panes.push({
          id: `${prefix}${invocation.id}/${file.path}`,
          label: file.path,
          language: languageForPath(file.path),
          contents: file.contents,
          ...(group === undefined ? {} : { group }),
        })
      }
    }
  })

  if (omittedCandidates > 0) {
    notes.push(`${omittedCandidates} per-candidate commands were left out of this list.`)
  }

  return [{
    id: 'replay',
    label: 'Replay',
    badge: Array.from(new Set(records.map(record => record.simulator))).join(' + '),
    panes,
    notes,
  }]
}

/** Read every replay record carried on a captured response payload. */
function debugRecordsFromPayload(payload: DebugPayload | undefined): DebugRecord[] {
  if (payload == null) return []
  const items = Array.isArray(payload) ? payload : [payload]
  return items.flatMap((item): DebugRecord[] => {
    if (!isRecord(item)) return []
    const record = parseDebugRecord(item.debug)
    return record === null ? [] : [record]
  })
}

function serialize(value: unknown): string {
  return JSON.stringify(value, null, 2)
}

/**
 * Describe a captured request/response pair as one run with two panes, plus a
 * replay run when the response carries recorded artifacts.
 *
 * This carries exactly what the page already holds, so the panel can change
 * shape without changing the data under it. A missing half stays visible as a
 * pane: "no response" is information, and hiding the tab would lose it.
 */
export function exchangeRuns(exchange: DebugExchange): DebugRun[] {
  const { request, response } = exchange
  if (request == null && response == null) return []

  const runs: DebugRun[] = [{
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

  const debug = exchange.debug === undefined
    ? debugRecordsFromPayload(response)
    : exchange.debug
  return [...runs, ...replayRuns(debug)]
}

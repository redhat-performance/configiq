import { z } from 'zod'

import {
  COMPARISON_SCHEMA_VERSION,
  COMPARISON_STORAGE_KEY,
  MAX_COMPARISON_BYTES,
  MAX_COMPARISON_ENTRIES,
  emptyComparisonDocument,
  type ComparisonDocument,
  type ComparisonEntry,
} from './types'

const nullableNumber = z.number().finite().nullable()
const positiveNumber = z.number().finite().positive()
const positiveInt = z.number().int().positive()

const phaseSchema = z.object({
  tp: positiveInt,
  pp: positiveInt,
  workers: positiveInt,
})

const entrySchema = z.object({
  id: z.string().min(1),
  savedAt: z.string().min(1),
  name: z.string().min(1),
  link: z.string().min(1),
  model: z.string().min(1),
  system: z.string().min(1),
  systemLabel: z.string().min(1),
  simulator: z.string().min(1),
  engineVersion: z.string().nullable(),
  backend: z.string().nullable(),
  backendVersion: z.string().nullable(),
  mode: z.enum(['agg', 'disagg']),
  workload: z.object({
    isl: positiveInt,
    osl: positiveInt,
    concurrency: positiveInt,
    weightPrecision: z.string().min(1),
    kvPrecision: z.string().min(1),
  }),
  topology: z.object({
    tp: positiveInt.nullable(),
    pp: positiveInt.nullable(),
    replicas: positiveInt.nullable(),
    prefill: phaseSchema.nullable(),
    decode: phaseSchema.nullable(),
  }),
  figures: z.object({
    gpus: positiveNumber,
    ttftMs: nullableNumber,
    tpotMs: nullableNumber,
    requestLatencyMs: nullableNumber,
    tokensPerSecond: nullableNumber,
    tokensPerSecondPerGpu: nullableNumber,
    tokensPerSecondPerUser: nullableNumber,
    weightMemoryGb: nullableNumber,
    kvCachePerRequestMb: nullableNumber,
    kvCacheUsedGb: nullableNumber,
    totalVramGb: nullableNumber,
    cloudMonthlyUsd: nullableNumber,
    cloudFiveYearUsd: nullableNumber,
    selfHostedMonthlyUsd: nullableNumber,
    selfHostedFiveYearUsd: nullableNumber,
  }),
})

const documentSchema = z.object({
  schemaVersion: z.literal(COMPARISON_SCHEMA_VERSION),
  entries: z.array(entrySchema),
})

export type ParseResult =
  | { ok: true; document: ComparisonDocument }
  | { ok: false; problem: string }

export interface ComparisonLoad {
  document: ComparisonDocument
  /** Set when stored data could not be read; the empty set is returned. */
  problem: string | null
}

export type ComparisonWrite =
  | { ok: true; document: ComparisonDocument }
  | { ok: false; problem: string }

function bytes(value: string): number {
  return new TextEncoder().encode(value).length
}

function formatLimit(value: number): string {
  return value >= 1_000_000 ? `${value / 1_000_000} MB` : `${Math.round(value / 1000)} KB`
}

function issueMessage(error: z.ZodError): string {
  const issue = error.issues[0]
  if (!issue) return 'the document did not match the comparison schema'
  const path = issue.path.join('.')
  return path ? `${path}: ${issue.message}` : issue.message
}

/**
 * Validate a document as a whole, including the bounds an entry may occupy.
 *
 * Import and storage both go through this, so a file cannot introduce an entry
 * the browser would refuse to save, and a partially written document cannot be
 * read as a comparison. Nothing is written unless the whole document passes.
 */
export function parseComparisonDocument(value: unknown): ParseResult {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return { ok: false, problem: 'The file does not contain a comparison set.' }
  }

  const version = (value as { schemaVersion?: unknown }).schemaVersion
  if (version !== COMPARISON_SCHEMA_VERSION) {
    return {
      ok: false,
      problem: typeof version === 'number'
        ? `Unsupported comparison version ${version}; this build reads version ${COMPARISON_SCHEMA_VERSION}.`
        : `The comparison set does not declare schema version ${COMPARISON_SCHEMA_VERSION}.`,
    }
  }

  const parsed = documentSchema.safeParse(value)
  if (!parsed.success) {
    return { ok: false, problem: `The comparison set is not valid: ${issueMessage(parsed.error)}` }
  }

  if (parsed.data.entries.length > MAX_COMPARISON_ENTRIES) {
    return {
      ok: false,
      problem: `A comparison holds at most ${MAX_COMPARISON_ENTRIES} results; this one has ${parsed.data.entries.length}.`,
    }
  }

  // An identifier names exactly one saved result; two entries sharing one would
  // be removed or refreshed together.
  const ids = new Set<string>()
  for (const entry of parsed.data.entries) {
    if (ids.has(entry.id)) {
      return {
        ok: false,
        problem: `Two results share the identifier ${entry.id}; each saved result needs its own.`,
      }
    }
    ids.add(entry.id)
  }

  const serialized = JSON.stringify(parsed.data)
  if (bytes(serialized) > MAX_COMPARISON_BYTES) {
    return {
      ok: false,
      problem: `The comparison set is larger than ${formatLimit(MAX_COMPARISON_BYTES)}.`,
    }
  }

  return { ok: true, document: parsed.data }
}

function storage(): Storage | null {
  if (typeof window === 'undefined') return null
  try {
    return window.localStorage
  } catch {
    return null
  }
}

/** Read the saved set. Corrupt data yields an empty set and a reason. */
export function loadComparison(): ComparisonLoad {
  const store = storage()
  if (store == null) return { document: emptyComparisonDocument(), problem: null }

  let raw: string | null
  try {
    raw = store.getItem(COMPARISON_STORAGE_KEY)
  } catch {
    return {
      document: emptyComparisonDocument(),
      problem: 'This browser would not let the comparison be read.',
    }
  }
  if (raw == null) return { document: emptyComparisonDocument(), problem: null }

  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return {
      document: emptyComparisonDocument(),
      problem: 'The saved comparison is not valid JSON. Import a valid file to replace it.',
    }
  }

  const result = parseComparisonDocument(parsed)
  return result.ok
    ? { document: result.document, problem: null }
    : { document: emptyComparisonDocument(), problem: `The saved comparison could not be read: ${result.problem}` }
}

/** Replace the saved set. Refuses a set over the entry or byte bound. */
export function writeComparison(document: ComparisonDocument): ComparisonWrite {
  const checked = parseComparisonDocument(document)
  if (!checked.ok) return { ok: false, problem: checked.problem }

  const store = storage()
  if (store == null) {
    return { ok: false, problem: 'This browser does not provide local storage, so the result was not saved.' }
  }

  try {
    store.setItem(COMPARISON_STORAGE_KEY, JSON.stringify(checked.document))
  } catch {
    return {
      ok: false,
      problem: 'The comparison could not be saved; the browser may be out of storage for this site.',
    }
  }
  return { ok: true, document: checked.document }
}

/**
 * Identity of a snapshot: the view it came from, the simulator and engine that
 * produced it, and the figures themselves. Saving the same result again moves
 * it to the front instead of adding a duplicate; a different figure is a
 * different result and stays a second entry.
 */
export function snapshotKey(entry: ComparisonEntry): string {
  return JSON.stringify([entry.link, entry.simulator, entry.engineVersion, entry.figures])
}

/** Add a snapshot, or refresh the existing one with the same identity. */
export function upsertComparisonEntry(entry: ComparisonEntry): ComparisonWrite {
  const current = loadComparison()
  if (current.problem != null) {
    // Writing would replace a set this build could not read. Replacing it stays
    // an explicit import or clear.
    return { ok: false, problem: `${current.problem} This result was not saved.` }
  }
  const key = snapshotKey(entry)
  const existing = current.document.entries.find(candidate => snapshotKey(candidate) === key)
  const rest = current.document.entries.filter(candidate => candidate.id !== existing?.id)
  const next: ComparisonDocument = {
    ...current.document,
    entries: [{ ...entry, id: existing?.id ?? entry.id }, ...rest],
  }
  return writeComparison(next)
}

export function removeComparisonEntry(id: string): ComparisonWrite {
  const current = loadComparison()
  if (current.problem != null) {
    return { ok: false, problem: `${current.problem} Nothing was removed.` }
  }
  return writeComparison({
    ...current.document,
    entries: current.document.entries.filter(entry => entry.id !== id),
  })
}

export function clearComparison(): ComparisonWrite {
  const store = storage()
  if (store == null) {
    return { ok: false, problem: 'This browser does not provide local storage.' }
  }
  try {
    store.removeItem(COMPARISON_STORAGE_KEY)
  } catch {
    return { ok: false, problem: 'The comparison could not be cleared.' }
  }
  return { ok: true, document: emptyComparisonDocument() }
}

/** Read an exported JSON document and replace the saved set with it. */
export function importComparison(raw: string): ComparisonWrite {
  let value: unknown
  try {
    value = JSON.parse(raw)
  } catch {
    return { ok: false, problem: 'The selected file is not valid JSON.' }
  }
  const parsed = parseComparisonDocument(value)
  if (!parsed.ok) return { ok: false, problem: parsed.problem }
  return writeComparison(parsed.document)
}

export function exportComparisonJson(document: ComparisonDocument): string {
  return `${JSON.stringify(document, null, 2)}\n`
}

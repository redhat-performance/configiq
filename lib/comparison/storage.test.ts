// @vitest-environment happy-dom
import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  clearComparison,
  exportComparisonJson,
  importComparison,
  loadComparison,
  parseComparisonDocument,
  removeComparisonEntry,
  snapshotKey,
  upsertComparisonEntry,
  writeComparison,
} from './storage'
import {
  COMPARISON_SCHEMA_VERSION,
  COMPARISON_STORAGE_KEY,
  MAX_COMPARISON_BYTES,
  MAX_COMPARISON_ENTRIES,
  emptyComparisonDocument,
  type ComparisonDocument,
  type ComparisonEntry,
} from './types'

function entry(overrides: Partial<ComparisonEntry> = {}): ComparisonEntry {
  return {
    id: 'entry-1',
    savedAt: '2026-11-21T00:00:00.000Z',
    name: 'Llama · H200',
    link: '/predict?model=llama&system=h200',
    model: 'meta-llama/Llama-3.1-70B',
    system: 'h200',
    systemLabel: 'H200',
    simulator: 'aisimulate',
    engineVersion: '0.14.0',
    backend: 'vllm',
    backendVersion: '0.9.0',
    mode: 'agg',
    workload: {
      isl: 2048,
      osl: 128,
      concurrency: 32,
      weightPrecision: 'FP8',
      kvPrecision: 'FP16',
    },
    topology: { tp: 8, pp: 1, replicas: 1, prefill: null, decode: null },
    figures: {
      gpus: 8,
      ttftMs: 180,
      tpotMs: 22.5,
      requestLatencyMs: 3060,
      tokensPerSecond: 1600,
      tokensPerSecondPerGpu: 200,
      tokensPerSecondPerUser: 50,
      weightMemoryGb: 140,
      kvCachePerRequestMb: 2500,
      kvCacheUsedGb: 80,
      totalVramGb: 640,
      cloudMonthlyUsd: 14_600,
      cloudFiveYearUsd: 876_000,
      selfHostedMonthlyUsd: 4_000,
      selfHostedFiveYearUsd: 240_000,
    },
    ...overrides,
  }
}

function document(entries: ComparisonEntry[]): ComparisonDocument {
  return { schemaVersion: COMPARISON_SCHEMA_VERSION, entries }
}

function memoryStorage(): Storage {
  const values = new Map<string, string>()
  return {
    get length() {
      return values.size
    },
    clear: () => values.clear(),
    getItem: key => values.get(key) ?? null,
    key: index => Array.from(values.keys())[index] ?? null,
    removeItem: key => {
      values.delete(key)
    },
    setItem: (key, value) => {
      values.set(key, value)
    },
  }
}

beforeEach(() => {
  vi.stubGlobal('localStorage', memoryStorage())
})

describe('parseComparisonDocument', () => {
  it('accepts a document this build wrote', () => {
    const parsed = parseComparisonDocument(document([entry()]))

    expect(parsed.ok).toBe(true)
    if (parsed.ok) expect(parsed.document.entries).toHaveLength(1)
  })

  it('names the version it cannot read instead of accepting a guess', () => {
    const parsed = parseComparisonDocument({ schemaVersion: 2, entries: [] })

    expect(parsed).toEqual({
      ok: false,
      problem: 'Unsupported comparison version 2; this build reads version 1.',
    })
  })

  it('rejects a value that is not a comparison set', () => {
    expect(parseComparisonDocument([]).ok).toBe(false)
    expect(parseComparisonDocument(null).ok).toBe(false)
    expect(parseComparisonDocument({ entries: [] }).ok).toBe(false)
  })

  it('rejects an entry with a missing figure', () => {
    const broken = entry() as unknown as Record<string, unknown>
    delete (broken.figures as Record<string, unknown>).tokensPerSecond

    const parsed = parseComparisonDocument(document([broken as unknown as ComparisonEntry]))

    expect(parsed.ok).toBe(false)
    if (!parsed.ok) expect(parsed.problem).toContain('figures.tokensPerSecond')
  })

  it('refuses a set over the entry count', () => {
    const entries = Array.from({ length: MAX_COMPARISON_ENTRIES + 1 }, (_, index) =>
      entry({ id: `entry-${index}`, link: `/predict?system=h200&n=${index}` }),
    )

    const parsed = parseComparisonDocument(document(entries))

    expect(parsed.ok).toBe(false)
    if (!parsed.ok) expect(parsed.problem).toContain(`at most ${MAX_COMPARISON_ENTRIES}`)
  })

  it('refuses a set over the byte limit', () => {
    const parsed = parseComparisonDocument(
      document([entry({ name: 'x'.repeat(MAX_COMPARISON_BYTES + 1) })]),
    )

    expect(parsed.ok).toBe(false)
    if (!parsed.ok) expect(parsed.problem).toContain('larger than')
  })
})

describe('loadComparison', () => {
  it('returns the empty set when nothing was saved', () => {
    expect(loadComparison()).toEqual({ document: emptyComparisonDocument(), problem: null })
  })

  it('returns a saved set', () => {
    window.localStorage.setItem(COMPARISON_STORAGE_KEY, JSON.stringify(document([entry()])))

    expect(loadComparison().document.entries).toHaveLength(1)
  })

  it('returns the empty set and a reason when the stored data is corrupt', () => {
    window.localStorage.setItem(COMPARISON_STORAGE_KEY, '{not json')

    const loaded = loadComparison()

    expect(loaded.document.entries).toEqual([])
    expect(loaded.problem).toContain('not valid JSON')
  })

  it('returns the empty set and a reason when the stored document is invalid', () => {
    window.localStorage.setItem(COMPARISON_STORAGE_KEY, JSON.stringify({ schemaVersion: 7, entries: [] }))

    const loaded = loadComparison()

    expect(loaded.document.entries).toEqual([])
    expect(loaded.problem).toContain('Unsupported comparison version 7')
  })
})

describe('writeComparison', () => {
  it('writes the document to the browser-local key', () => {
    const written = writeComparison(document([entry()]))

    expect(written.ok).toBe(true)
    expect(JSON.parse(window.localStorage.getItem(COMPARISON_STORAGE_KEY) ?? '{}').entries).toHaveLength(1)
  })

  it('refuses a document over the entry bound and leaves the saved set alone', () => {
    writeComparison(document([entry()]))
    const entries = Array.from({ length: MAX_COMPARISON_ENTRIES + 1 }, (_, index) =>
      entry({ id: `entry-${index}`, link: `/predict?n=${index}` }),
    )

    const written = writeComparison(document(entries))

    expect(written.ok).toBe(false)
    expect(loadComparison().document.entries).toHaveLength(1)
  })
})

describe('upsertComparisonEntry', () => {
  it('adds a new snapshot to the front', () => {
    upsertComparisonEntry(entry({ id: 'a' }))
    upsertComparisonEntry(entry({ id: 'b', link: '/predict?n=2' }))

    expect(loadComparison().document.entries.map(saved => saved.id)).toEqual(['b', 'a'])
  })

  it('moves an identical snapshot instead of duplicating it', () => {
    upsertComparisonEntry(entry({ id: 'a' }))
    upsertComparisonEntry(entry({ id: 'b' }))

    const saved = loadComparison().document.entries
    expect(saved).toHaveLength(1)
    expect(saved[0].id).toBe('a')
  })

  it('keeps a snapshot with a different figure as its own entry', () => {
    upsertComparisonEntry(entry({ id: 'a' }))
    upsertComparisonEntry(entry({
      id: 'b',
      figures: { ...entry().figures, tokensPerSecond: 1700 },
    }))

    expect(loadComparison().document.entries).toHaveLength(2)
  })

  it('refuses to save over a stored set it could not read', () => {
    window.localStorage.setItem(COMPARISON_STORAGE_KEY, '{not json')

    const written = upsertComparisonEntry(entry({ id: 'a' }))

    expect(written.ok).toBe(false)
    if (!written.ok) expect(written.problem).toContain('not valid JSON')
    expect(window.localStorage.getItem(COMPARISON_STORAGE_KEY)).toBe('{not json')
  })

  it('keeps a snapshot from a different simulator as its own entry', () => {
    upsertComparisonEntry(entry({ id: 'a' }))
    upsertComparisonEntry(entry({ id: 'b', simulator: 'blis' }))

    expect(loadComparison().document.entries).toHaveLength(2)
  })
})

describe('snapshotKey', () => {
  it('changes when the link, simulator, engine, or figures change', () => {
    const base = snapshotKey(entry())

    expect(snapshotKey(entry({ link: '/predict?n=2' }))).not.toBe(base)
    expect(snapshotKey(entry({ simulator: 'blis' }))).not.toBe(base)
    expect(snapshotKey(entry({ engineVersion: '0.15.0' }))).not.toBe(base)
    expect(snapshotKey(entry({ figures: { ...entry().figures, ttftMs: 1 } }))).not.toBe(base)
  })
})

describe('removeComparisonEntry and clearComparison', () => {
  it('removes one entry', () => {
    writeComparison(document([entry({ id: 'a' }), entry({ id: 'b', link: '/predict?n=2' })]))

    removeComparisonEntry('a')

    expect(loadComparison().document.entries.map(saved => saved.id)).toEqual(['b'])
  })

  it('refuses to remove from a stored set it could not read', () => {
    window.localStorage.setItem(COMPARISON_STORAGE_KEY, '{not json')

    const written = removeComparisonEntry('a')

    expect(written.ok).toBe(false)
    expect(window.localStorage.getItem(COMPARISON_STORAGE_KEY)).toBe('{not json')
  })

  it('clears every entry', () => {
    writeComparison(document([entry()]))

    clearComparison()

    expect(loadComparison().document.entries).toEqual([])
    expect(window.localStorage.getItem(COMPARISON_STORAGE_KEY)).toBeNull()
  })
})

describe('importComparison', () => {
  it('refuses text that is not JSON', () => {
    const result = importComparison('not json')

    expect(result).toEqual({ ok: false, problem: 'The selected file is not valid JSON.' })
  })

  it('refuses an invalid document and leaves the saved set alone', () => {
    writeComparison(document([entry()]))

    const result = importComparison(JSON.stringify({ schemaVersion: 2, entries: [] }))

    expect(result.ok).toBe(false)
    expect(loadComparison().document.entries).toHaveLength(1)
  })

  it('refuses a document whose entries share an identifier', () => {
    writeComparison(document([entry({ id: 'a' })]))
    const incoming = document([
      entry({ id: 'same', link: '/predict?n=1' }),
      entry({ id: 'same', link: '/predict?n=2' }),
    ])

    const result = importComparison(JSON.stringify(incoming))

    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.problem).toContain('Two results share the identifier same')
    expect(loadComparison().document.entries.map(saved => saved.id)).toEqual(['a'])
  })

  it('replaces the saved set with a valid document', () => {
    writeComparison(document([entry()]))
    const incoming = document([entry({ id: 'imported', link: '/predict?n=9' })])

    const result = importComparison(JSON.stringify(incoming))

    expect(result.ok).toBe(true)
    expect(loadComparison().document.entries.map(saved => saved.id)).toEqual(['imported'])
  })
})

describe('exportComparisonJson', () => {
  it('round-trips through import', () => {
    const original = document([entry()])

    const exported = exportComparisonJson(original)
    const result = importComparison(exported)

    expect(result.ok).toBe(true)
    if (result.ok) expect(result.document).toEqual(original)
  })
})

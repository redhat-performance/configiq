import { describe, expect, it } from 'vitest'
import {
  exchangeRuns,
  parseDebugRecord,
  replayRuns,
  type DebugRecord,
} from './types'

function record(overrides: Partial<DebugRecord> = {}): DebugRecord {
  return {
    endpoint: '/predict',
    simulator: 'aisim',
    execution: 'in_process',
    created_at: 1_730_000_000,
    versions: { aisim: '0.14.0', blis: null },
    selection: null,
    notes: [],
    invocations: [
      {
        id: 'prediction',
        label: 'Prediction',
        command: 'aisimulate predict -c prediction.yaml --format json',
        execution: 'in_process',
        group: null,
        candidate_id: null,
        notes: [],
        files: [
          { path: 'prediction.yaml', contents: 'engine: {}\n' },
          { path: 'model/config.json', contents: '{"hidden_size":8}\n' },
        ],
      },
    ],
    script: '#!/usr/bin/env bash\naisimulate predict -c prediction.yaml\n',
    ...overrides,
  }
}

describe('parseDebugRecord', () => {
  it('reads a recorded run and drops its unusable pieces', () => {
    const parsed = parseDebugRecord({
      endpoint: '/predict',
      simulator: 'aisim',
      execution: 'in_process',
      created_at: 1_730_000_000,
      versions: { aisim: '0.14.0', unknown: 7 },
      selection: '8 GPUs, agg',
      notes: ['a note', 4],
      invocations: [
        { id: 'one', label: 'One', command: 'run one', execution: 'subprocess' },
        { id: 'missing-command', label: 'Broken' },
      ],
      script: '#!/usr/bin/env bash\n',
    })

    expect(parsed).not.toBeNull()
    expect(parsed?.versions).toEqual({ aisim: '0.14.0' })
    expect(parsed?.notes).toEqual(['a note'])
    expect(parsed?.invocations.map(invocation => invocation.id)).toEqual(['one'])
    expect(parsed?.invocations[0].files).toEqual([])
  })

  it('treats a missing or unrecognizable record as no record', () => {
    expect(parseDebugRecord(null)).toBeNull()
    expect(parseDebugRecord({ endpoint: '/predict' })).toBeNull()
  })
})

describe('replayRuns', () => {
  it('returns nothing when the request recorded nothing', () => {
    expect(replayRuns(null)).toEqual([])
    expect(replayRuns(undefined)).toEqual([])
    expect(replayRuns([])).toEqual([])
  })

  it('turns the script, pins, command, and input files into panes', () => {
    const [run] = replayRuns(record())

    expect(run.id).toBe('replay')
    expect(run.badge).toBe('aisim')
    expect(run.panes.map(pane => pane.label)).toEqual([
      'Versions',
      'run.sh',
      'Prediction',
      'prediction.yaml',
      'model/config.json',
    ])
    // Input files are rendered with the syntax their extension names.
    expect(run.panes.find(pane => pane.label === 'prediction.yaml')?.language).toBe('yaml')
    expect(run.panes.find(pane => pane.label === 'model/config.json')?.language).toBe('json')
    expect(run.panes.find(pane => pane.label === 'run.sh')?.language).toBe('shell')
    // The version pane is machine-readable rather than a formatted sentence.
    expect(JSON.parse(run.panes[0].contents)).toMatchObject({
      endpoint: '/predict',
      versions: { aisim: '0.14.0' },
    })
  })

  it('keeps grouped artifacts behind their group and ungrouped ones always visible', () => {
    const [run] = replayRuns(record({
      invocations: [
        {
          id: 'prefill',
          label: 'Prefill memory',
          command: 'python prefill.py',
          execution: 'in_process',
          group: 'Prefill',
          candidate_id: null,
          notes: [],
          files: [{ path: 'prefill.py', contents: 'print(1)\n' }],
        },
        {
          id: 'decode',
          label: 'Decode memory',
          command: 'python decode.py',
          execution: 'in_process',
          group: 'Decode',
          candidate_id: null,
          notes: [],
          files: [{ path: 'decode.py', contents: 'print(2)\n' }],
        },
      ],
    }))

    expect(run.panes.find(pane => pane.label === 'Prefill memory')?.group).toBe('Prefill')
    expect(run.panes.find(pane => pane.label === 'Decode memory')?.group).toBe('Decode')
    expect(run.panes.find(pane => pane.label === 'run.sh')?.group).toBeUndefined()
  })

  it('leaves per-candidate commands out and says how many', () => {
    const [run] = replayRuns(record({
      endpoint: '/recommend',
      invocations: [
        {
          id: 'window',
          label: 'GPU window 1-32',
          command: 'aisimulate recommend -c window.yaml',
          execution: 'in_process',
          group: 'Search windows',
          candidate_id: null,
          notes: [],
          files: [],
        },
        {
          id: 'candidate-1',
          label: 'Candidate c1',
          command: 'aisimulate predict -c c1.yaml',
          execution: 'in_process',
          group: 'Candidates',
          candidate_id: 'c1',
          notes: [],
          files: [],
        },
        {
          id: 'candidate-2',
          label: 'Candidate c2',
          command: 'aisimulate predict -c c2.yaml',
          execution: 'in_process',
          group: 'Candidates',
          candidate_id: 'c2',
          notes: [],
          files: [],
        },
      ],
    }))

    expect(run.panes.map(pane => pane.label)).not.toContain('Candidate c1')
    expect(run.notes).toContain('2 per-candidate commands were left out of this list.')
  })

  it('merges several records into one run with qualified pane ids', () => {
    const [run] = replayRuns([
      record({
        endpoint: '/memory',
        invocations: [
          {
            id: 'memory-estimate',
            label: 'Memory estimate',
            command: 'python prefill.py',
            execution: 'in_process',
            group: 'Prefill',
            candidate_id: null,
            notes: [],
            files: [],
          },
        ],
        script: '#!/usr/bin/env bash\npython prefill.py\n',
      }),
      record({
        endpoint: '/memory',
        invocations: [
          {
            id: 'memory-estimate',
            label: 'Memory estimate',
            command: 'python decode.py',
            execution: 'in_process',
            group: 'Decode',
            candidate_id: null,
            notes: [],
            files: [],
          },
        ],
        script: '#!/usr/bin/env bash\npython decode.py\n',
      }),
    ])

    const ids = run.panes.map(pane => pane.id)
    expect(new Set(ids).size).toBe(ids.length)
    // Both scripts survive, labelled by the pool they compute.
    expect(run.panes.map(pane => pane.label)).toEqual([
      'Versions',
      'run.sh · Prefill',
      'Memory estimate',
      'run.sh · Decode',
      'Memory estimate',
    ])
    // The pin pane lists both records rather than pretending there is one.
    expect(Array.isArray(JSON.parse(run.panes[0].contents))).toBe(true)
  })
})

describe('exchangeRuns', () => {
  it('adds a replay run when the response carries a record', () => {
    const runs = exchangeRuns({
      request: { isl: 2048 },
      response: { ttft_ms: 120, debug: record() },
    })

    expect(runs.map(run => run.id)).toEqual(['exchange', 'replay'])
    expect(runs[1].panes.some(pane => pane.contents.includes('aisimulate predict'))).toBe(true)
  })

  it('reads one record from each response of a fanned-out request', () => {
    const runs = exchangeRuns({
      request: [{ pool: 'prefill' }, { pool: 'decode' }],
      response: [{ debug: record() }, { debug: record() }],
    })

    expect(runs).toHaveLength(2)
    expect(runs[1].id).toBe('replay')
    // Two records, so their scripts are told apart by their pool.
    expect(runs[1].panes.filter(pane => pane.label.startsWith('run.sh'))).toHaveLength(2)
  })

  it('does not re-read the payload when the page says there is no record', () => {
    const runs = exchangeRuns({
      request: { isl: 2048 },
      response: { ttft_ms: 120, debug: record() },
      debug: null,
    })

    expect(runs).toHaveLength(1)
  })

  it('hides the whole panel when there is neither half', () => {
    expect(exchangeRuns({ request: null, response: null })).toEqual([])
  })
})

import { describe, expect, it } from 'vitest'

import { invocationBundle, invocationForCandidate } from './replay'
import type { DebugInvocation, DebugRecord } from './types'

function invocation(overrides: Partial<DebugInvocation> = {}): DebugInvocation {
  return {
    id: 'candidate-1',
    label: 'Candidate c1',
    command: 'aisimulate predict -c candidate-1.yaml --format json',
    execution: 'in_process',
    group: null,
    candidate_id: 'candidate-000001',
    notes: ['Equivalent CLI invocation: the service called the API directly.'],
    files: [
      { path: 'candidate-1.yaml', contents: 'engine:\n  model: ./model\n' },
      { path: 'model/config.json', contents: '{"hidden_size":8}\n' },
    ],
    ...overrides,
  }
}

function record(invocations: DebugInvocation[]): DebugRecord {
  return {
    endpoint: '/recommend',
    simulator: 'aisimulate',
    execution: 'in_process',
    created_at: 1_730_000_000,
    versions: {},
    selection: null,
    notes: [],
    invocations,
    script: null,
  }
}

describe('invocationForCandidate', () => {
  it('finds the invocation the sweep tagged with the candidate', () => {
    const found = invocationForCandidate(
      record([invocation(), invocation({ id: 'candidate-2', candidate_id: 'candidate-000002' })]),
      'candidate-000002',
    )

    expect(found?.id).toBe('candidate-2')
  })

  it('returns nothing when the service recorded no command for the candidate', () => {
    expect(invocationForCandidate(record([invocation()]), 'candidate-000009')).toBeNull()
    expect(invocationForCandidate(record([]), 'candidate-000001')).toBeNull()
    expect(invocationForCandidate(null, 'candidate-000001')).toBeNull()
  })

  it('never falls back to the request-level command', () => {
    const requestLevel = invocation({ id: 'sweep', candidate_id: null })

    expect(invocationForCandidate(record([requestLevel]), 'candidate-000001')).toBeNull()
  })
})

describe('invocationBundle', () => {
  it('writes every input file before running the command', () => {
    const bundle = invocationBundle(invocation())

    expect(bundle.startsWith('#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\n')).toBe(true)
    expect(bundle).toContain("# Equivalent CLI invocation: the service called the API directly.")
    expect(bundle).toContain("mkdir -p 'model'")
    expect(bundle).toContain("cat > 'candidate-1.yaml' <<'CONFIGIQ_REPLAY_EOF'")
    expect(bundle).toContain('engine:\n  model: ./model')
    expect(bundle).toContain("cat > 'model/config.json' <<'CONFIGIQ_REPLAY_EOF'")
    // The command runs last, after its inputs exist.
    expect(bundle.trimEnd().endsWith('aisimulate predict -c candidate-1.yaml --format json')).toBe(true)
  })

  it('guards the environment variables the command reads', () => {
    const bundle = invocationBundle(invocation({
      command: '"$BLIS_BINARY" run --catalog "$BLIS_CATALOG" --scenario .',
      notes: [],
      files: [],
    }))

    expect(bundle).toContain(': "${BLIS_BINARY:?set BLIS_BINARY before running this script}"')
    expect(bundle).toContain(': "${BLIS_CATALOG:?set BLIS_CATALOG before running this script}"')
  })

  it('quotes a path containing a single quote', () => {
    const bundle = invocationBundle(invocation({
      notes: [],
      files: [{ path: "it's.yaml", contents: 'a: 1\n' }],
    }))

    expect(bundle).toContain("cat > 'it'\\''s.yaml' <<'CONFIGIQ_REPLAY_EOF'")
  })

  it('keeps a heredoc terminator that appears in the file out of the way', () => {
    const bundle = invocationBundle(invocation({
      notes: [],
      files: [{ path: 'script.sh', contents: 'echo CONFIGIQ_REPLAY_EOF\n' }],
    }))

    expect(bundle).toContain("<<'CONFIGIQ_REPLAY_EOF_'")
    expect(bundle).toContain('\nCONFIGIQ_REPLAY_EOF_\n')
  })
})

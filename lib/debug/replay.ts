import type { DebugInvocation, DebugRecord } from './types'

/**
 * The invocation that reproduces one Pareto candidate.
 *
 * A sweep records one prediction per evaluated candidate and tags it with the
 * candidate's identifier, so a row can copy the command that produced its own
 * figures instead of repeating the whole search. A service that recorded no
 * per-candidate command, or had to omit them to keep the record small, has no
 * match; the row then offers no command rather than a different one.
 */
export function invocationForCandidate(
  record: DebugRecord | null | undefined,
  candidateId: string,
): DebugInvocation | null {
  if (record == null) return null
  return record.invocations.find(
    invocation => invocation.candidate_id === candidateId,
  ) ?? null
}

const ENVIRONMENT_PATTERN = /\$\{([A-Z_][A-Z0-9_]*)|\$([A-Z_][A-Z0-9_]*)/g

/**
 * Names a command reads from the environment, in first-use order.
 *
 * The record's commands are already shell-quoted and never name a temporary
 * path; only runtime locations the caller must supply are left as variables.
 * Collecting them from the command keeps the copied script honest about what
 * it needs without the browser knowing which tool uses which name.
 */
function referencedEnvironment(command: string): string[] {
  const names: string[] = []
  for (const match of command.matchAll(ENVIRONMENT_PATTERN)) {
    const name = match[1] ?? match[2]
    if (!names.includes(name)) names.push(name)
  }
  return names
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`
}

function parentDirectory(path: string): string | null {
  const index = path.lastIndexOf('/')
  return index > 0 ? path.slice(0, index) : null
}

/** Keep the heredoc terminator out of the text it wraps. */
function heredocDelimiter(contents: string): string {
  let delimiter = 'CONFIGIQ_REPLAY_EOF'
  while (contents.includes(delimiter)) delimiter += '_'
  return delimiter
}

/**
 * Render one invocation as a self-contained script.
 *
 * A command alone would not reproduce a run: its inputs are files the service
 * wrote while it executed. The bundle writes each file first, then runs the
 * command from the bundle directory, so one paste on a machine with the
 * simulator installed repeats the work. Notes from the record travel as
 * comments, including whether the command is the exact process or its
 * equivalent.
 */
export function invocationBundle(invocation: DebugInvocation): string {
  const lines: string[] = [
    '#!/usr/bin/env bash',
    'set -euo pipefail',
    'cd "$(dirname "$0")"',
  ]

  if (invocation.notes.length > 0) {
    lines.push('')
    for (const note of invocation.notes) lines.push(`# ${note}`)
  }

  const environment = referencedEnvironment(invocation.command)
  if (environment.length > 0) {
    lines.push('')
    for (const name of environment) {
      lines.push(`: "\${${name}:?set ${name} before running this script}"`)
    }
  }

  for (const file of invocation.files) {
    const directory = parentDirectory(file.path)
    const delimiter = heredocDelimiter(file.contents)
    lines.push('')
    if (directory) lines.push(`mkdir -p ${shellQuote(directory)}`)
    lines.push(`cat > ${shellQuote(file.path)} <<'${delimiter}'`)
    lines.push(file.contents.endsWith('\n') ? file.contents.slice(0, -1) : file.contents)
    lines.push(delimiter)
  }

  lines.push('')
  lines.push(`# ${invocation.label}`)
  lines.push(invocation.command)
  return `${lines.join('\n')}\n`
}

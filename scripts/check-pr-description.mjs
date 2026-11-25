// Check a pull request description against .github/pull_request_template.md.
//
// GitHub renders the template into a new pull request's body but never checks
// that the author kept it. CodeRabbit's built-in description check reads the
// body too, but it reports in its Walkthrough comment, not as a GitHub check
// run — so a missing section leaves the checks list all green with a red row
// buried in a comment. This script backs a named workflow check instead.
//
// The template stays the source of truth for which checklist items exist. Each
// item is matched in the body by its first backticked token, so an author can
// reword the surrounding prose without the check drifting away from the file.

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

export const TEMPLATE_PATH = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '..',
  '.github',
  'pull_request_template.md',
)

const FENCE = /^\s*(`{3,}|~{3,})(.*)$/

/** The opening run of a fenced code block, or null for any other line. */
function fenceRun(line) {
  const match = FENCE.exec(line)
  if (match === null) return null
  return { character: match[1][0], length: match[1].length, rest: match[2] }
}

/**
 * Whether `line` closes `fence`, following CommonMark: the same marker
 * character, at least as many of them, and nothing but whitespace after. A
 * shorter inner fence therefore cannot close a longer outer one.
 */
function closesFence(line, fence) {
  const run = fenceRun(line)
  return (
    run !== null &&
    run.character === fence.character &&
    run.length >= fence.length &&
    run.rest.trim() === ''
  )
}

/**
 * HTML comments are guidance for the author, never content.
 *
 * A single regular-expression replace is not enough: overlapping or repeated
 * comment openers can survive it, so the check would read text the author
 * meant to hide. Scan line by line instead, dropping each comment from its
 * `<!--` through the next `-->`, and drop an unterminated opener to the end of
 * the document.
 *
 * Fenced code blocks are the exception: `sections` already ignores them, so a
 * literal opener in an example must survive. Stripping it would also swallow
 * the closing fence and every section after it.
 */
export function stripComments(markdown) {
  const kept = []
  let fence = null
  let inComment = false

  for (const line of markdown.replace(/\r\n?/g, '\n').split('\n')) {
    if (fence !== null) {
      kept.push(line)
      if (closesFence(line, fence)) fence = null
      continue
    }

    let rest = line
    let output = ''
    while (rest !== '') {
      if (inComment) {
        const end = rest.indexOf('-->')
        if (end === -1) {
          rest = ''
          break
        }
        rest = rest.slice(end + 3)
        inComment = false
        continue
      }
      const start = rest.indexOf('<!--')
      if (start === -1) {
        output += rest
        break
      }
      output += rest.slice(0, start)
      rest = rest.slice(start + 4)
      inComment = true
    }

    if (!inComment) {
      const opener = fenceRun(output)
      if (opener !== null) fence = opener
    }
    kept.push(output)
  }

  return kept.join('\n')
}

const HEADING = /^##\s+(.+?)\s*$/
const ITEM_START = /^-\s+\[([ xX])\]\s+(.*)$/

/**
 * The body of each level-two section, keyed by heading. A repeated heading is
 * kept once, with the first occurrence winning: content under a later
 * duplicate is dropped rather than appended, so a stray second `## Checks`
 * cannot hide a defective first one.
 *
 * Fence delimiters are dropped along with the fenced lines. Keeping a closing
 * fence would make `checklist()` read it as an opening fence and skip every
 * item that follows a fenced example.
 */
export function sections(markdown) {
  const found = new Map()
  let current = null
  let fence = null
  for (const line of markdown.replace(/\r\n/g, '\n').split('\n')) {
    if (fence !== null) {
      if (closesFence(line, fence)) fence = null
      continue
    }
    const opener = fenceRun(line)
    if (opener !== null) {
      fence = opener
      continue
    }
    const heading = line.match(HEADING)
    if (heading) {
      current = found.has(heading[1]) ? null : heading[1]
      if (current !== null) found.set(current, [])
      continue
    }
    if (current !== null) found.get(current).push(line)
  }
  return new Map([...found].map(([name, lines]) => [name, lines.join('\n')]))
}

/**
 * The checklist items in a markdown fragment, joining the indented
 * continuation lines that the template uses to keep its items readable.
 */
export function checklist(markdown) {
  const items = []
  let current = null
  let fence = null
  for (const raw of markdown.replace(/\r\n/g, '\n').split('\n')) {
    if (fence !== null) {
      if (closesFence(raw, fence)) fence = null
      continue
    }
    const opener = fenceRun(raw)
    if (opener !== null) {
      fence = opener
      continue
    }
    const start = raw.match(ITEM_START)
    if (start) {
      current = { ticked: start[1] !== ' ', text: start[2].trim() }
      items.push(current)
      continue
    }
    if (current && /^\s+\S/.test(raw)) {
      current.text = `${current.text} ${raw.trim()}`
      continue
    }
    current = null
  }
  return items
}

/** The token that identifies an item when the author rewords it. */
export function itemKey(text) {
  const code = text.match(/`([^`]+)`/)
  return code ? code[1] : text
}

/**
 * What the author has to change, in the order the template presents it. An
 * empty list means the description follows the template.
 */
export function checkDescription(body, template) {
  const problems = []
  const parsed = sections(stripComments(body ?? ''))

  const changed = parsed.get('What changed')
  if (changed === undefined) {
    problems.push('add a `## What changed` section with a one or two sentence summary')
  } else if (changed.trim() === '') {
    problems.push('`## What changed` has no summary under it')
  }

  const checks = parsed.get('Checks')
  if (checks === undefined) {
    problems.push('add a `## Checks` section from the template')
    return problems
  }

  const ticked = checklist(checks)
  const templateChecks = sections(stripComments(template)).get('Checks') ?? ''
  for (const item of checklist(templateChecks)) {
    const key = itemKey(item.text)
    const match = ticked.find(candidate => candidate.text.includes(key))
    if (!match) {
      problems.push(`\`## Checks\` is missing the item containing \`${key}\``)
    } else if (!match.ticked) {
      problems.push(`\`## Checks\` leaves \`${key}\` unticked`)
    }
  }

  return problems
}

function main() {
  if (process.env.PR_BODY === undefined) {
    console.error('Set PR_BODY to the pull request description.')
    process.exitCode = 2
    return
  }

  const problems = checkDescription(
    process.env.PR_BODY,
    readFileSync(TEMPLATE_PATH, 'utf8'),
  )

  if (problems.length === 0) {
    console.log('The description follows the pull request template.')
    return
  }

  console.error('The description does not follow .github/pull_request_template.md:')
  for (const problem of problems) console.error(`- ${problem}`)
  process.exitCode = 1
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main()
}

import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  TEMPLATE_PATH,
  checkDescription,
  checklist,
  itemKey,
  sections,
  stripComments,
} from './check-pr-description.mjs'

const template = readFileSync(TEMPLATE_PATH, 'utf8')

/** The template with every box ticked and a summary filled in. */
function completed(from = template) {
  return from
    .replace(/^- \[ \]/gm, '- [x]')
    .replace(
      /<!-- One or two sentences a reviewer can hold in their head\. -->/,
      'Make the Workload chips follow a shared view and count the tour steps that exist.',
    )
}

describe('pull request description check', () => {
  it('accepts the filled-in template', () => {
    expect(checkDescription(completed(), template)).toEqual([])
  })

  it('rejects the untouched template, which is how a real description fails', () => {
    // The placeholder summary is a comment, so it is not content, and every
    // box is unticked. This is the description CodeRabbit flagged on #240.
    const problems = checkDescription(template, template)

    expect(problems).toHaveLength(6)
    expect(problems[0]).toBe('`## What changed` has no summary under it')
    expect(problems.slice(1)).toEqual([
      '`## Checks` leaves `npm run type-check` unticked',
      '`## Checks` leaves `defineAccessors` unticked',
      '`## Checks` leaves `lib/deep-links/registry.ts` unticked',
      '`## Checks` leaves `LINK_VERSION` unticked',
      '`## Checks` leaves `Signed-off-by:` unticked',
    ])
  })

  it('rejects a body with no template sections at all', () => {
    expect(checkDescription('Fixed a few things.', template)).toEqual([
      'add a `## What changed` section with a one or two sentence summary',
      'add a `## Checks` section from the template',
    ])
  })

  it('follows the template when the author rewrote the prose', () => {
    // The keys, not the wording, identify the items: the template is a
    // policy, and restating it in your own words still satisfies it.
    const body = [
      '## What changed',
      '',
      'Selected workload chips now follow the values in a shared view.',
      '',
      '## Checks',
      '',
      '- [x] Ran `npm run type-check`, `npm run lint`, and `npm test`',
      '- [x] Shared view inputs bound with `defineAccessors` / `useSharedView`',
      '- [x] No new page, so `lib/deep-links/registry.ts` is unchanged',
      '- [x] Additive only, `LINK_VERSION` untouched',
      '- [x] Commits carry a `Signed-off-by:` trailer',
    ].join('\n')

    expect(checkDescription(body, template)).toEqual([])
  })

  it('reads an item whose key wrapped onto a continuation line', () => {
    const body = [
      '## What changed',
      '',
      'A summary.',
      '',
      '## Checks',
      '',
      '- [x] Shared views — new or changed inputs are bound through the page',
      '      link contract (`defineAccessors` / `useSharedView`)',
      '- [x] `npm run type-check`, `npm run lint`, and the test suite pass',
      '- [x] `lib/deep-links/registry.ts` unchanged',
      '- [x] `LINK_VERSION` unchanged',
      '- [x] `Signed-off-by:` on every commit',
    ].join('\n')

    expect(checkDescription(body, template)).toEqual([])
  })

  it('does not accept a tick outside the Checks section', () => {
    // A ticked box in `Notes for review` says nothing about the checklist, so
    // the item it names is still missing where it counts.
    const body = [
      '## What changed',
      '',
      'A summary.',
      '',
      '## Checks',
      '',
      '- [x] `npm run type-check`, `npm run lint`, and the test suite pass locally',
      '- [x] `defineAccessors` binding',
      '- [x] `lib/deep-links/registry.ts` unchanged',
      '- [x] `Signed-off-by:` on every commit',
      '',
      '## Notes for review',
      '',
      '- [x] `LINK_VERSION` policy reviewed',
    ].join('\n')

    expect(checkDescription(body, template)).toEqual([
      '`## Checks` is missing the item containing `LINK_VERSION`',
    ])
  })

  it('requires a checklist item the template gains later', () => {
    // The check derives its items from the file, so a new policy line is
    // enforced without touching the workflow.
    const grown = template.replace(
      '\n## Notes for review',
      '- [ ] `CHANGELOG.md` updated\n\n## Notes for review',
    )

    expect(checkDescription(completed(grown), grown)).toEqual([])
    expect(checkDescription(completed(), grown)).toEqual([
      '`## Checks` is missing the item containing `CHANGELOG.md`',
    ])
  })

  it('reads a completed checklist that follows a fenced example', () => {
    // The example is documentation, not a claim. A retained closing fence
    // would be read as an opening fence and hide every item below it.
    const body = completed().replace(
      '## Checks\n',
      '## Checks\n\n```text\n- [ ] copy the items below\n```\n',
    )

    expect(checkDescription(body, template)).toEqual([])
  })

  it('does not accept a later duplicate Checks section', () => {
    // The first occurrence is the one the author is held to; a second section
    // appended below it cannot supply the ticks the first one lacks.
    const body = [
      '## What changed',
      '',
      'A summary.',
      '',
      '## Checks',
      '',
      '## Checks',
      '',
      '- [x] `npm run type-check`, `npm run lint`, and the test suite pass locally',
      '- [x] `defineAccessors` binding',
      '- [x] `lib/deep-links/registry.ts` unchanged',
      '- [x] `LINK_VERSION` unchanged',
      '- [x] `Signed-off-by:` on every commit',
    ].join('\n')

    expect(checkDescription(body, template)).toEqual([
      '`## Checks` is missing the item containing `npm run type-check`',
      '`## Checks` is missing the item containing `defineAccessors`',
      '`## Checks` is missing the item containing `lib/deep-links/registry.ts`',
      '`## Checks` is missing the item containing `LINK_VERSION`',
      '`## Checks` is missing the item containing `Signed-off-by:`',
    ])
  })

  it('treats a commented-out heading as absent', () => {
    const body = completed().replace('## What changed', '<!-- ## What changed -->')

    expect(checkDescription(body, template)).toContain(
      'add a `## What changed` section with a one or two sentence summary',
    )
  })

  it('drops an unterminated comment instead of leaving its opener behind', () => {
    // A regex replace removes `<!-- ... -->` pairs but leaves a stray `<!--`
    // when the author never closed the comment. The scanner drops the rest,
    // so no opener survives to hide content from a later reader.
    expect(stripComments('keep<!-- hidden')).toBe('keep')
    expect(stripComments('keep<!-- hidden')).not.toContain('<!--')
  })

  it('removes overlapping comment markers in one pass', () => {
    expect(stripComments('before<!-- hidden -->after')).toBe('beforeafter')
    expect(stripComments('a<!-- <!-- hidden -->b-->c')).toBe('ab-->c')
  })

  it('keeps a comment opener that an author put in a fenced example', () => {
    // `sections` ignores fenced blocks, so the literal opener is harmless.
    // Treating it as a real comment would swallow the closing fence and the
    // checklist that follows.
    const example = '```html\n<!--\n```\n\n'
    const body = completed().replace('## Checks\n', `${example}## Checks\n`)

    expect(stripComments(example)).toContain('<!--')
    expect(checkDescription(body, template)).toEqual([])
  })

  it('does not close a long fence on a shorter nested fence', () => {
    // CommonMark closes a four-backtick fence only with four or more
    // backticks. Reading the inner three-backtick example as a close would
    // drop the unterminated comment and the real `## Checks` that follows.
    const example = ['````markdown', '```html', '<!--', '```', '````', '', ''].join(
      '\n',
    )
    const body = completed().replace('## Checks\n', `${example}## Checks\n`)

    expect(stripComments(example)).toContain('<!--')
    expect(checkDescription(body, template)).toEqual([])
  })

  it('applies the same fence rule in sections and checklist', () => {
    const nested = [
      '````markdown',
      '```html',
      '<!--',
      '```',
      '````',
      '',
      '## Checks',
      '',
      '- [x] visible',
    ].join('\n')

    expect(sections(nested).get('Checks')).toContain('- [x] visible')
    expect(checklist(sections(nested).get('Checks'))).toHaveLength(1)
  })

  it('reads CRLF descriptions', () => {
    expect(checkDescription(completed().replace(/\n/g, '\r\n'), template)).toEqual([])
  })

  it('parses multi-line checklist items from the template', () => {
    const items = checklist(template)

    expect(items).toHaveLength(5)
    expect(items.map(item => itemKey(item.text))).toEqual([
      'npm run type-check',
      'defineAccessors',
      'lib/deep-links/registry.ts',
      'LINK_VERSION',
      'Signed-off-by:',
    ])
    expect(items.every(item => !item.ticked)).toBe(true)
  })

  it('keys the sections it knows about', () => {
    expect([...sections(completed()).keys()]).toEqual([
      'What changed',
      'Checks',
      'Notes for review',
    ])
  })
})

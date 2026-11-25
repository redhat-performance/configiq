import { describe, expect, it } from 'vitest'
import { decodeEntities } from './build-docs.mjs'

describe('docs HTML text extraction', () => {
  it('decodes each entity exactly once', () => {
    expect(decodeEntities('&amp;')).toBe('&')
    expect(decodeEntities('&lt;')).toBe('<')
    expect(decodeEntities('&quot;&#39;&nbsp;&gt;')).toBe('"\' >')
  })

  it('does not unescape text that was itself escaped', () => {
    // `&amp;lt;` is the author's literal `&lt;`, not a tag. Decoding `&amp;`
    // and then `&lt;` in separate passes would turn it into `<`.
    expect(decodeEntities('&amp;lt;')).toBe('&lt;')
    expect(decodeEntities('&amp;amp;')).toBe('&amp;')
  })
})

import { describe, expect, it } from 'vitest'
import { GET } from './route'
import { VERSION } from '@/lib/version'

describe('GET /api/version', () => {
  it('reports the build metadata baked into the image', async () => {
    const response = await GET()
    const body = await response.json()
    expect(body).toEqual({
      version: VERSION.number,
      gitCommit: VERSION.gitCommit,
    })
  })
})

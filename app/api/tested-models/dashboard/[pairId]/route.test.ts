import { afterEach, describe, expect, it, vi } from 'vitest'
import { readFile } from 'node:fs/promises'
import { GET } from './route'

vi.mock('node:fs/promises', () => ({ readFile: vi.fn() }))

const mockReadFile = vi.mocked(readFile)

const context = { params: Promise.resolve({ pairId: 'model__h200' }) }

afterEach(() => {
  vi.resetAllMocks()
  vi.unstubAllEnvs()
})

describe('GET /api/tested-models/dashboard/[pairId]', () => {
  it('serves a schema v2 payload with its throughput curves', async () => {
    const payload = {
      schemaVersion: 2,
      pairId: 'model__h200',
      throughputCurves: [{ isl: 1000, osl: 1000, turns: 1, tp: 1, sweepId: 's', points: [] }],
    }
    mockReadFile.mockResolvedValue(JSON.stringify(payload) as never)

    const response = await GET(new Request('http://localhost'), context)

    expect(response.status).toBe(200)
    expect(await response.json()).toEqual(payload)
    expect(response.headers.get('Cache-Control')).toBe('no-store')
  })

  it('rejects an older payload instead of rendering it as an empty chart', async () => {
    mockReadFile.mockResolvedValue(JSON.stringify({
      schemaVersion: 1,
      throughputLatencyByConcurrency: [{ concurrency: 1, throughput: 100 }],
    }) as never)

    const response = await GET(new Request('http://localhost'), context)

    expect(response.status).toBe(409)
    expect(await response.json()).toMatchObject({ error: expect.stringContaining('not supported') })
  })

  it('rejects a payload that does not declare a schema version', async () => {
    mockReadFile.mockResolvedValue(JSON.stringify({ curves: [] }) as never)

    const response = await GET(new Request('http://localhost'), context)

    expect(response.status).toBe(409)
  })

  it('returns 404 when the pair has no dashboard file', async () => {
    mockReadFile.mockRejectedValue(new Error('missing'))

    const response = await GET(new Request('http://localhost'), context)

    expect(response.status).toBe(404)
  })

  it('rejects a pair id that could escape the dashboard directory', async () => {
    const response = await GET(new Request('http://localhost'), {
      params: Promise.resolve({ pairId: '../registry' }),
    })

    expect(response.status).toBe(400)
    expect(mockReadFile).not.toHaveBeenCalled()
  })
})

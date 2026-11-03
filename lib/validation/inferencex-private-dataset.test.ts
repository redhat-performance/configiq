import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const snapshot = {
  sourceUrl: 'https://inferencex.semianalysis.com/api',
  retrievedAt: '2026-10-29T16:24:02.331Z',
  rows: [{
    benchmark_source: 'inferencex', model: 'openai/gpt-oss-120b',
    isl: 8192, osl: 1024, concurrency: 4,
  }],
}

beforeEach(() => {
  vi.resetModules()
  vi.stubEnv('INFERENCEX_HF_TOKEN', 'test-read-token')
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe('private InferenceX dataset', () => {
  it('fetches server-side with a token and caches the validated snapshot', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(snapshot), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const { getInferenceXDataset } = await import('./ground-truth-dataset')

    const first = await getInferenceXDataset()
    const second = await getInferenceXDataset()

    expect(first.rows).toHaveLength(1)
    expect(second).toBe(first)
    expect(fetchMock).toHaveBeenCalledOnce()
    expect(fetchMock.mock.calls[0][0]).toBe(
      'https://huggingface.co/datasets/redhat-performance/configiq-inferencex-comparison/resolve/main/inferencex-8k1k.json',
    )
    expect(fetchMock.mock.calls[0][1].headers.Authorization).toBe('Bearer test-read-token')
  })

  it('serves the last good snapshot to concurrent callers when a refresh fails', async () => {
    const now = Date.parse('2026-10-29T16:24:02.331Z')
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(now)
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(snapshot), { status: 200 }))
      .mockRejectedValue(new Error('Hugging Face is unreachable'))
    vi.stubGlobal('fetch', fetchMock)
    const { getInferenceXDataset } = await import('./ground-truth-dataset')

    try {
      const first = await getInferenceXDataset()
      nowSpy.mockReturnValue(now + 6 * 60 * 1000)

      const [left, right] = await Promise.all([getInferenceXDataset(), getInferenceXDataset()])

      expect(left).toBe(first)
      expect(right).toBe(first)
      expect(fetchMock).toHaveBeenCalledTimes(2)
    } finally {
      nowSpy.mockRestore()
    }
  })

  it('forces a refresh after the minimum interval when requested', async () => {
    const now = Date.parse('2026-10-29T16:24:02.331Z')
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(now)
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(snapshot), { status: 200 }))
      .mockResolvedValue(new Response(JSON.stringify({ ...snapshot, retrievedAt: '2026-10-29T16:30:00.000Z' }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const { getInferenceXDataset } = await import('./ground-truth-dataset')

    try {
      const first = await getInferenceXDataset()
      nowSpy.mockReturnValue(now + 31_000)
      const refreshed = await getInferenceXDataset(true)

      expect(refreshed).not.toBe(first)
      expect(refreshed.source.retrievedAt).toBe('2026-10-29T16:30:00.000Z')
      expect(fetchMock).toHaveBeenCalledTimes(2)
    } finally {
      nowSpy.mockRestore()
    }
  })

  it('rejects a non-https source URL', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      ...snapshot, sourceUrl: 'javascript:alert(1)',
    }), { status: 200 })))
    const { getInferenceXDataset } = await import('./ground-truth-dataset')

    await expect(getInferenceXDataset()).rejects.toThrow('invalid source URL')
  })

  it('rejects a non-https run URL', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      ...snapshot, rows: [{ ...snapshot.rows[0], source_run_url: 'javascript:alert(1)' }],
    }), { status: 200 })))
    const { getInferenceXDataset } = await import('./ground-truth-dataset')

    await expect(getInferenceXDataset()).rejects.toThrow('invalid or unapproved benchmark rows')
  })

  it('rejects models outside ConfigIQ approved models', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      ...snapshot, rows: [{ ...snapshot.rows[0], model: 'InferenceX/unverified-model' }],
    }), { status: 200 })))
    const { getInferenceXDataset } = await import('./ground-truth-dataset')

    await expect(getInferenceXDataset()).rejects.toThrow('unapproved benchmark rows')
  })
})

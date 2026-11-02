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

  it('rejects models outside ConfigIQ tested models', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      ...snapshot, rows: [{ ...snapshot.rows[0], model: 'InferenceX/unverified-model' }],
    }), { status: 200 })))
    const { getInferenceXDataset } = await import('./ground-truth-dataset')

    await expect(getInferenceXDataset()).rejects.toThrow('untested benchmark rows')
  })
})

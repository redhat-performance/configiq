import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

describe('app config loading', () => {
  beforeEach(() => vi.resetModules())
  afterEach(() => vi.unstubAllGlobals())

  it('ignores malformed tested model data from config.json', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ testedModels: 'Qwen/Qwen3-8B' }),
    }))

    const { loadAppConfig } = await import('./app-config')
    expect((await loadAppConfig()).testedModels).toEqual([])
  })

  it('keeps only string model identifiers', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ testedModels: ['Qwen/Qwen3-8B', null, 42] }),
    }))

    const { loadAppConfig } = await import('./app-config')
    expect((await loadAppConfig()).testedModels).toEqual(['Qwen/Qwen3-8B'])
  })
})

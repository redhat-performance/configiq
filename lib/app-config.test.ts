import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

describe('app config loading', () => {
  beforeEach(() => vi.resetModules())
  afterEach(() => vi.unstubAllGlobals())

  it('does not load tested model data from config.json', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ testedModels: 'Qwen/Qwen3-8B' }),
    }))

    const { loadAppConfig } = await import('./app-config')
    expect('testedModels' in (await loadAppConfig())).toBe(false)
  })
})

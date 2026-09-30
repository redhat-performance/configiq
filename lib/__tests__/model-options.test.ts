import { describe, it, expect, vi, beforeEach } from 'vitest'
import { buildModelItems, needsHfConfig } from '../model-options'
import type { AppConfig } from '../app-config'

// buildModelItems reads testedModels + huggingFaceModels from getAppConfig().
vi.mock('../app-config', () => ({ getAppConfig: vi.fn() }))
import { getAppConfig } from '../app-config'
const mockGetAppConfig = vi.mocked(getAppConfig)

const BASE: AppConfig = {
  defaultBackend: '',
  defaultSystem: '',
  defaultModel: '',
  defaultOpenModel: '',
  defaultFrontierModel: '',
  testedModels: [],
  huggingFaceModels: [],
  suggestedModelNames: [],
  modelRequestUrl: '',
  workloadPresets: [],
}

const CATALOG_MODELS = ['org-a/catalog-model', 'org-a/tested-in-catalog', 'plain-model']

beforeEach(() => {
  mockGetAppConfig.mockReturnValue({
    ...BASE,
    testedModels: ['org-a/tested-in-catalog', 'org-b/tested-only'],
    huggingFaceModels: ['org-c/hf-only', 'org-a/tested-in-catalog'],
  })
})

describe('buildModelItems', () => {
  it('flags a catalog-only model as inCatalog', () => {
    const item = buildModelItems(CATALOG_MODELS).find(i => i.value === 'org-a/catalog-model')!
    expect(item).toMatchObject({ inCatalog: true, isTested: false, isHuggingFace: false })
  })

  it('flags a tested model that is also in the catalog', () => {
    const item = buildModelItems(CATALOG_MODELS).find(i => i.value === 'org-a/tested-in-catalog')!
    // Present in all three sources → all flags set, but appears exactly once.
    expect(item).toMatchObject({ inCatalog: true, isTested: true, isHuggingFace: true })
  })

  it('includes a tested model that is NOT in the catalog', () => {
    const item = buildModelItems(CATALOG_MODELS).find(i => i.value === 'org-b/tested-only')
    expect(item).toBeDefined()
    expect(item).toMatchObject({ inCatalog: false, isTested: true, isHuggingFace: false })
  })

  it('flags a hugging-face-only model', () => {
    const item = buildModelItems(CATALOG_MODELS).find(i => i.value === 'org-c/hf-only')!
    expect(item).toMatchObject({ inCatalog: false, isTested: false, isHuggingFace: true })
  })

  it('dedupes models present in multiple sources', () => {
    const items = buildModelItems(CATALOG_MODELS)
    const dupes = items.filter(i => i.value === 'org-a/tested-in-catalog')
    expect(dupes).toHaveLength(1)
  })

  it('derives group from the org prefix, empty when no slash', () => {
    const items = buildModelItems(CATALOG_MODELS)
    expect(items.find(i => i.value === 'org-a/catalog-model')!.group).toBe('org-a')
    expect(items.find(i => i.value === 'plain-model')!.group).toBe('')
  })

  it('lists catalog models before tested-only and hf-only models', () => {
    const values = buildModelItems(CATALOG_MODELS).map(i => i.value)
    expect(values.indexOf('org-a/catalog-model')).toBeLessThan(values.indexOf('org-b/tested-only'))
    expect(values.indexOf('org-b/tested-only')).toBeLessThan(values.indexOf('org-c/hf-only'))
  })
})

describe('needsHfConfig', () => {
  it('is false for a model in the AISimulators catalog', () => {
    expect(needsHfConfig('org-a/catalog-model', CATALOG_MODELS)).toBe(false)
  })

  it('is true for a model outside the catalog (incl. tested-only)', () => {
    expect(needsHfConfig('org-b/tested-only', CATALOG_MODELS)).toBe(true)
    expect(needsHfConfig('someone/random-model', CATALOG_MODELS)).toBe(true)
  })

  it('is false for an empty model string', () => {
    expect(needsHfConfig('', CATALOG_MODELS)).toBe(false)
  })
})

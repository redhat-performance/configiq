// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/contexts/SettingsContext', () => ({
  simulatorSupportsBackend: () => true,
  useSettings: vi.fn(),
}))
vi.mock('@/lib/hooks/useCatalog', () => ({ useCatalog: vi.fn() }))
vi.mock('@/lib/hooks/useTestedModels', () => ({ useTestedModels: vi.fn() }))
vi.mock('@/lib/app-config', () => ({
  getAppConfig: () => ({ modelRequestUrl: '' }),
}))

import { Settings } from './Settings'
import { useSettings } from '@/contexts/SettingsContext'
import { useCatalog } from '@/lib/hooks/useCatalog'
import { useTestedModels } from '@/lib/hooks/useTestedModels'

let root: ReturnType<typeof createRoot>
let host: HTMLDivElement

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  vi.mocked(useSettings).mockReturnValue({
    hydrated: true,
    defaultModel: '',
    defaultSystem: '',
    hfToken: '',
    backendVersion: '',
    inferenceBackend: 'vllm',
    simulator: 'aisimulate',
    costingsEnabled: false,
    preferredCloudProvider: null,
    pricingSource: 'merged',
    setDefaultModel: vi.fn(),
    setDefaultSystem: vi.fn(),
    setHfToken: vi.fn(),
    setInferenceBackend: vi.fn(),
    setSimulator: vi.fn(),
    setBackendVersion: vi.fn(),
    setCostingsEnabled: vi.fn(),
    setPreferredCloudProvider: vi.fn(),
    setPricingSource: vi.fn(),
    applyViewForSession: vi.fn(),
  })
  vi.mocked(useCatalog).mockReturnValue({
    gpuOptions: [],
    modelOptions: [],
    modelSpecs: new Map(),
    backendOptions: [],
    networkOptions: [],
    simulatorSupport: {},
    simulators: [],
    objectives: [],
    timeoutSeconds: 180,
    isLoading: false,
    error: null,
  })
  vi.mocked(useTestedModels).mockReturnValue({
    pairs: [],
    modelIds: ['org/tested-model'],
    validatedModelIds: ['org/tested-model'],
    measuredModelIds: [],
    approvedModels: ['org/tested-model'],
    modelStatus: () => 'validated',
    gate: null,
    schemaVersion: 2,
    dataset: null,
    isLoading: false,
    isAvailable: true,
  })
  host = document.createElement('div')
  document.body.append(host)
  root = createRoot(host)
})

afterEach(async () => {
  await act(async () => root.unmount())
  host.remove()
  vi.clearAllMocks()
})

async function render() {
  await act(async () => root.render(<Settings />))
}

describe('validated model disclosure', () => {
  it('is a keyboard-operable disclosure that starts closed', async () => {
    await render()

    const toggle = [...host.querySelectorAll('button')]
      .find(button => button.textContent?.includes('Validated models'))!
    expect(toggle).toBeTruthy()
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(toggle.getAttribute('aria-controls')).toBe('validated-models-panel')
    expect(host.querySelector('#validated-models-panel')).toBeNull()

    await act(async () => {
      toggle.click()
      await Promise.resolve()
    })

    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    const panel = host.querySelector('#validated-models-panel')
    expect(panel).not.toBeNull()
    expect(panel!.textContent).toContain('tested-model')
    expect(panel!.querySelector('button')?.textContent).toContain('tested-model')
  })
})

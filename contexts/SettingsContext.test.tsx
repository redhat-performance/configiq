// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/app-config', () => ({
  loadAppConfig: async () => ({
    defaultModel: 'Qwen/Qwen3-32B',
    defaultSystem: 'h200_sxm',
    defaultOpenModel: 'Qwen/Qwen3-32B',
    defaultFrontierModel: 'claude-fable-5',
    defaultBackend: 'vllm',
    testedModels: [],
    huggingFaceModels: [],
    suggestedModelNames: [],
    modelRequestUrl: '',
    workloadPresets: [],
  }),
}))

import { isSimulatorBackendCompatible, SettingsProvider, useSettings } from './SettingsContext'

const VERSION_KEY = 'settings_backend_version_v2'
const DEFAULT_SIMULATOR_VERSION_KEY = `${VERSION_KEY}:aisimulate`
const LEGACY_VERSION_KEY = 'settings_backend_version'
const SIMULATOR_KEY = 'settings_simulator'

let root: ReturnType<typeof createRoot>
let host: HTMLDivElement

function Consumer() {
  const { hydrated, backendVersion, simulator, setBackendVersion, setInferenceBackend, setSimulator } = useSettings()
  return <>
    <output>{JSON.stringify({ hydrated, backendVersion, simulator })}</output>
    <button onClick={() => setBackendVersion('0.24.0')}>Set version</button>
    <button onClick={() => setInferenceBackend('sglang')}>Set backend</button>
    <button onClick={() => setSimulator('blis')}>Set simulator</button>
  </>
}

const state = () => JSON.parse(host.querySelector('output')!.textContent!)

async function render() {
  await act(async () => root.render(<SettingsProvider><Consumer /></SettingsProvider>))
}

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  localStorage.clear()
  host = document.createElement('div')
  document.body.append(host)
  root = createRoot(host)
})

afterEach(async () => {
  await act(async () => root.unmount())
  host.remove()
  vi.restoreAllMocks()
})

describe('Backend version persistence', () => {
  it('restores a version the user explicitly saved', async () => {
    localStorage.setItem(VERSION_KEY, '0.24.0')

    await render()

    expect(state()).toEqual({ hydrated: true, backendVersion: '0.24.0', simulator: 'aisimulate' })
  })

  it('writes nothing when the user never chose a version', async () => {
    await render()

    expect(state().backendVersion).toBe('')
    expect(state().simulator).toBe('aisimulate')
    expect(localStorage.getItem(VERSION_KEY)).toBeNull()
  })

  it('ignores the legacy fleet-wide pin so a stale literal is not resent', async () => {
    // 0.31.0 was the fleet-wide newest version, unqueryable on a100_sxm and b60.
    localStorage.setItem(LEGACY_VERSION_KEY, '0.31.0')

    await render()

    expect(state().backendVersion).toBe('')
    expect(state().simulator).toBe('aisimulate')
  })

  it('clears the persisted version when the backend changes', async () => {
    localStorage.setItem(VERSION_KEY, '0.24.0')
    await render()

    await act(async () => (host.querySelectorAll('button')[1] as HTMLButtonElement).click())

    expect(state().backendVersion).toBe('')
    expect(state().simulator).toBe('aisimulate')
    expect(localStorage.getItem(DEFAULT_SIMULATOR_VERSION_KEY)).toBe('')
  })

  it('persists an explicit user choice', async () => {
    await render()

    await act(async () => (host.querySelector('button') as HTMLButtonElement).click())

    expect(state().backendVersion).toBe('0.24.0')
    expect(state().simulator).toBe('aisimulate')
    expect(localStorage.getItem(DEFAULT_SIMULATOR_VERSION_KEY)).toBe('0.24.0')
  })

  it('restores the persisted global simulator selection', async () => {
    localStorage.setItem(SIMULATOR_KEY, 'blis')
    await render()
    expect(state().simulator).toBe('blis')
  })

  it('normalizes a persisted BLIS selection that conflicts with the inference backend', async () => {
    localStorage.setItem(SIMULATOR_KEY, 'blis')
    localStorage.setItem('settings_inference_backend', 'sglang')

    await render()

    expect(state().simulator).toBe('aisimulate')
    expect(localStorage.getItem(SIMULATOR_KEY)).toBe('aisimulate')
  })

  it('persists a global simulator change', async () => {
    await render()
    await act(async () => (host.querySelectorAll('button')[2] as HTMLButtonElement).click())
    expect(state().simulator).toBe('blis')
    expect(localStorage.getItem(SIMULATOR_KEY)).toBe('blis')
  })
})

describe('Simulator/backend compatibility', () => {
  it('allows all AISimulate backends but only vLLM with BLIS', () => {
    expect(isSimulatorBackendCompatible('aisimulate', 'vllm')).toBe(true)
    expect(isSimulatorBackendCompatible('aisimulate', 'sglang')).toBe(true)
    expect(isSimulatorBackendCompatible('aisimulate', 'tensorrt-llm')).toBe(true)
    expect(isSimulatorBackendCompatible('blis', 'vllm')).toBe(true)
    expect(isSimulatorBackendCompatible('blis', 'sglang')).toBe(false)
    expect(isSimulatorBackendCompatible('blis', 'tensorrt-llm')).toBe(false)
  })
})

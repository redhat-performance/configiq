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
    approvedModels: [],
    huggingFaceModels: [],
    suggestedModelNames: [],
    modelRequestUrl: '',
    workloadPresets: [],
  }),
}))

vi.mock('@/lib/hooks/useCatalog', () => ({
  useCatalog: () => ({
    defaultSimulator: 'aisimulate',
    isLoading: false,
  }),
}))

import { simulatorSupportsBackend, SettingsProvider, useSettings } from './SettingsContext'

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

  it('keeps a persisted simulator instead of silently substituting one', async () => {
    // Compatibility is surfaced by the Settings UI and enforced by the API;
    // the context never rewrites the user's simulator choice.
    localStorage.setItem(SIMULATOR_KEY, 'blis')
    localStorage.setItem('settings_inference_backend', 'sglang')

    await render()

    expect(state().simulator).toBe('blis')
    expect(localStorage.getItem(SIMULATOR_KEY)).toBe('blis')
  })

  it('persists a global simulator change', async () => {
    await render()
    await act(async () => (host.querySelectorAll('button')[2] as HTMLButtonElement).click())
    expect(state().simulator).toBe('blis')
    expect(localStorage.getItem(SIMULATOR_KEY)).toBe('blis')
  })
})

describe('Simulator/backend compatibility', () => {
  it('reads compatibility from the simulator catalog, not a local table', () => {
    const vllmOnly = { backends: [{ id: 'vllm' }] }
    const allBackends = { backends: [{ id: 'vllm' }, { id: 'sglang' }, { id: 'tensorrt-llm' }] }

    expect(simulatorSupportsBackend('aisimulate', 'vllm', allBackends)).toBe(true)
    expect(simulatorSupportsBackend('aisimulate', 'sglang', allBackends)).toBe(true)
    expect(simulatorSupportsBackend('blis', 'vllm', vllmOnly)).toBe(true)
    expect(simulatorSupportsBackend('blis', 'sglang', vllmOnly)).toBe(false)
    expect(simulatorSupportsBackend('blis', 'tensorrt-llm', vllmOnly)).toBe(false)
    // No explicit choice or no catalog entry: do not invent a restriction.
    expect(simulatorSupportsBackend('', 'sglang', undefined)).toBe(true)
    expect(simulatorSupportsBackend('unknown', 'sglang', undefined)).toBe(true)
  })
})

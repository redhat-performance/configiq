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
    simulators: [{ id: 'aisim', label: 'AISim', version: null, default: true }],
    isLoading: false,
  }),
}))

import { simulatorSupportsBackend, SettingsProvider, useSettings } from './SettingsContext'

const VERSION_KEY = 'settings_backend_version_v2'
const DEFAULT_SIMULATOR_VERSION_KEY = `${VERSION_KEY}:aisim`
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

const SYSTEM_KEY = 'settings_default_system'

function SystemConsumer() {
  const { defaultSystem, setDefaultSystem } = useSettings()
  return <>
    <output>{defaultSystem}</output>
    <button onClick={() => setDefaultSystem('a100_sxm')}>Set system</button>
  </>
}

async function renderSystem() {
  await act(async () => root.render(<SettingsProvider><SystemConsumer /></SettingsProvider>))
}

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
  vi.unstubAllGlobals()
})

describe('Backend version persistence', () => {
  it('restores a version the user explicitly saved', async () => {
    localStorage.setItem(VERSION_KEY, '0.24.0')

    await render()

    expect(state()).toEqual({ hydrated: true, backendVersion: '0.24.0', simulator: 'aisim' })
  })

  it('writes nothing when the user never chose a version', async () => {
    await render()

    expect(state().backendVersion).toBe('')
    expect(state().simulator).toBe('aisim')
    expect(localStorage.getItem(VERSION_KEY)).toBeNull()
  })

  it('ignores the legacy fleet-wide pin so a stale literal is not resent', async () => {
    // 0.31.0 was the fleet-wide newest version, unqueryable on a100_sxm and b60.
    localStorage.setItem(LEGACY_VERSION_KEY, '0.31.0')

    await render()

    expect(state().backendVersion).toBe('')
    expect(state().simulator).toBe('aisim')
  })

  it('clears the persisted version when the backend changes', async () => {
    localStorage.setItem(VERSION_KEY, '0.24.0')
    await render()

    await act(async () => (host.querySelectorAll('button')[1] as HTMLButtonElement).click())

    expect(state().backendVersion).toBe('')
    expect(state().simulator).toBe('aisim')
    expect(localStorage.getItem(DEFAULT_SIMULATOR_VERSION_KEY)).toBe('')
  })

  it('persists an explicit user choice', async () => {
    await render()

    await act(async () => (host.querySelector('button') as HTMLButtonElement).click())

    expect(state().backendVersion).toBe('0.24.0')
    expect(state().simulator).toBe('aisim')
    expect(localStorage.getItem(DEFAULT_SIMULATOR_VERSION_KEY)).toBe('0.24.0')
  })

  it('restores the persisted global simulator selection', async () => {
    localStorage.setItem(SIMULATOR_KEY, 'blis')
    await render()
    expect(state().simulator).toBe('blis')
  })

  it('reads a saved simulator id from before the rename as the current one', async () => {
    localStorage.setItem(SIMULATOR_KEY, 'aisimulate')
    await render()
    expect(state().simulator).toBe('aisim')
  })

  it('carries a saved version pin across the simulator rename', async () => {
    // The pin was written under the id the profile had at the time. Losing it
    // would silently replace the user's choice with the simulator default.
    localStorage.setItem(SIMULATOR_KEY, 'aisimulate')
    localStorage.setItem(`${VERSION_KEY}:aisimulate`, '0.24.0')

    await render()

    expect(state().backendVersion).toBe('0.24.0')
    expect(localStorage.getItem(`${VERSION_KEY}:aisim`)).toBe('0.24.0')
    expect(localStorage.getItem(`${VERSION_KEY}:aisimulate`)).toBeNull()
  })

  it('keeps an explicit current pin over the retired one', async () => {
    localStorage.setItem(SIMULATOR_KEY, 'aisimulate')
    localStorage.setItem(`${VERSION_KEY}:aisim`, '')
    localStorage.setItem(`${VERSION_KEY}:aisimulate`, '0.24.0')

    await render()

    expect(state().backendVersion).toBe('')
    expect(localStorage.getItem(`${VERSION_KEY}:aisim`)).toBe('')
  })

  it('migrates the retired pin when the profile never chose a simulator', async () => {
    // The default simulator was renamed too, and its pin was written under
    // the retired id. Without the migration the profile would read no pin at
    // the current key and fall back to the catalog default.
    localStorage.setItem(`${VERSION_KEY}:aisimulate`, '0.24.0')

    await render()

    expect(state().backendVersion).toBe('0.24.0')
    expect(localStorage.getItem(`${VERSION_KEY}:aisim`)).toBe('0.24.0')
    expect(localStorage.getItem(`${VERSION_KEY}:aisimulate`)).toBeNull()
  })

  it('migrates the retired pin even when another simulator is chosen', async () => {
    // The pin belongs to the renamed simulator, not to the resting one.
    localStorage.setItem(SIMULATOR_KEY, 'blis')
    localStorage.setItem(`${VERSION_KEY}:aisimulate`, '0.24.0')
    localStorage.setItem(`${VERSION_KEY}:blis`, '1.3.0')

    await render()

    expect(state().simulator).toBe('blis')
    expect(state().backendVersion).toBe('1.3.0')
    expect(localStorage.getItem(`${VERSION_KEY}:aisim`)).toBe('0.24.0')
    expect(localStorage.getItem(`${VERSION_KEY}:aisimulate`)).toBeNull()
    expect(localStorage.getItem(`${VERSION_KEY}:blis`)).toBe('1.3.0')
  })

  it('still hydrates with the retired pin when storage cannot take the copy', async () => {
    // A full quota makes the copy throw. The provider must still hydrate and
    // use the pin, and the retired key must stay so a later load can retry.
    localStorage.setItem(`${VERSION_KEY}:aisimulate`, '0.24.0')
    const real = window.localStorage
    const failing: Storage = {
      get length() {
        return real.length
      },
      clear: () => real.clear(),
      getItem: key => real.getItem(key),
      key: index => real.key(index),
      removeItem: key => real.removeItem(key),
      setItem: (key, value) => {
        if (key === `${VERSION_KEY}:aisim`) {
          throw new DOMException('The quota has been exceeded.', 'QuotaExceededError')
        }
        real.setItem(key, value)
      },
    }
    vi.stubGlobal('localStorage', failing)

    await render()
    vi.unstubAllGlobals()

    expect(state().hydrated).toBe(true)
    expect(state().backendVersion).toBe('0.24.0')
    expect(localStorage.getItem(`${VERSION_KEY}:aisimulate`)).toBe('0.24.0')
    expect(localStorage.getItem(`${VERSION_KEY}:aisim`)).toBeNull()
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

describe('Settings changed while a shared view is open', () => {
  const BLIS_VERSION_KEY = `${VERSION_KEY}:blis`

  function OverrideConsumer() {
    const { simulator, backendVersion, applyViewForSession, setBackendVersion, setInferenceBackend } = useSettings()
    return <>
      <output>{JSON.stringify({ simulator, backendVersion })}</output>
      <button onClick={() => applyViewForSession({ simulator: 'blis', backend: 'vllm', backendVersion: 'blis-pin' })}>
        Open shared view
      </button>
      <button onClick={() => setBackendVersion('0.24.0')}>Set version</button>
      <button onClick={() => setInferenceBackend('sglang')}>Set backend</button>
    </>
  }

  async function renderOverride() {
    await act(async () => root.render(<SettingsProvider><OverrideConsumer /></SettingsProvider>))
  }

  async function click(label: string) {
    const button = Array.from(host.querySelectorAll('button'))
      .find(candidate => candidate.textContent === label)
    await act(async () => button?.click())
  }

  it('saves a version choice under the simulator the profile rests on', async () => {
    // A shared view borrows another simulator for this session only. Changing
    // the version ends that borrow, so the value has to be stored against the
    // simulator the reader will actually be using.
    await renderOverride()
    await click('Open shared view')
    expect(state()).toMatchObject({ simulator: 'blis', backendVersion: 'blis-pin' })

    await click('Set version')

    expect(state()).toMatchObject({ simulator: 'aisim', backendVersion: '0.24.0' })
    expect(localStorage.getItem(DEFAULT_SIMULATOR_VERSION_KEY)).toBe('0.24.0')
    expect(localStorage.getItem(BLIS_VERSION_KEY)).toBeNull()
  })

  it('clears the resting simulator version when the backend changes', async () => {
    await renderOverride()
    await click('Open shared view')

    await click('Set backend')

    expect(state()).toMatchObject({ simulator: 'aisim', backendVersion: '' })
    expect(localStorage.getItem(DEFAULT_SIMULATOR_VERSION_KEY)).toBe('')
    expect(localStorage.getItem(BLIS_VERSION_KEY)).toBeNull()
  })
})

describe('Default system persistence', () => {
  it('uses the configured default when nothing is stored', async () => {
    await renderSystem()
    expect(host.querySelector('output')!.textContent).toBe('h200_sxm')
  })

  it('restores a saved system', async () => {
    localStorage.setItem(SYSTEM_KEY, 'a100_sxm')
    await renderSystem()
    expect(host.querySelector('output')!.textContent).toBe('a100_sxm')
  })

  it('persists only an explicit system choice', async () => {
    await renderSystem()
    expect(localStorage.getItem(SYSTEM_KEY)).toBeNull()

    await act(async () => (host.querySelector('button') as HTMLButtonElement).click())

    expect(host.querySelector('output')!.textContent).toBe('a100_sxm')
    expect(localStorage.getItem(SYSTEM_KEY)).toBe('a100_sxm')
  })
})

describe('Simulator/backend compatibility', () => {
  it('reads compatibility from the simulator catalog, not a local table', () => {
    const vllmOnly = { backends: [{ id: 'vllm' }] }
    const allBackends = { backends: [{ id: 'vllm' }, { id: 'sglang' }, { id: 'tensorrt-llm' }] }

    expect(simulatorSupportsBackend('aisim', 'vllm', allBackends)).toBe(true)
    expect(simulatorSupportsBackend('aisim', 'sglang', allBackends)).toBe(true)
    expect(simulatorSupportsBackend('blis', 'vllm', vllmOnly)).toBe(true)
    expect(simulatorSupportsBackend('blis', 'sglang', vllmOnly)).toBe(false)
    expect(simulatorSupportsBackend('blis', 'tensorrt-llm', vllmOnly)).toBe(false)
    // No explicit choice or no catalog entry: do not invent a restriction.
    expect(simulatorSupportsBackend('', 'sglang', undefined)).toBe(true)
    expect(simulatorSupportsBackend('unknown', 'sglang', undefined)).toBe(true)
  })
})

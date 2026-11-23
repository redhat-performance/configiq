// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { getConfigurationOptions } from '@/lib/validation/benchmark-dataset'
import BenchmarkComparisonPage from './Compare'

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(window.location.search),
}))

vi.mock('@/contexts/SettingsContext', () => ({
  useSettings: () => ({
    hydrated: true,
    defaultModel: 'org/model',
    defaultSystem: 'H200',
    inferenceBackend: 'vllm',
    backendVersion: '0.31.0',
    simulator: 'aisimulate',
  }),
}))

vi.mock('@/lib/hooks/useCatalog', () => ({
  useCatalog: () => ({
    gpuOptions: [],
    modelOptions: ['org/model'],
    modelSpecs: new Map(),
    simulatorSupport: {},
    simulators: [],
    isLoading: false,
    error: null,
  }),
}))

const [configuration] = getConfigurationOptions([{
  benchmark_source: 'redhat',
  version: 'vLLM-0.31.0',
  backend: 'vllm',
  backend_version: '0.31.0',
  precision: 'FP16',
  request_type: '/v1/chat/completions',
  turns: 1,
}])

const options = {
  accelerator: ['H200'], model: ['org/model'], backend: ['vllm'], backendVersion: ['0.31.0'],
  isl: ['8192'], osl: ['1024'], tp: ['1'],
}

let host: HTMLDivElement
let root: Root

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  window.history.replaceState({}, '', '/validation/compare?v=1&configurationKey=removed')
  host = document.createElement('div')
  document.body.append(host)
  root = createRoot(host)
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const model = new URL(input, window.location.origin).searchParams.get('model') ?? ''
    const payload = {
      status: 'ok',
      source: {
        id: 'redhat', url: 'https://huggingface.co/datasets/example', revision: '1',
        lastModified: null, loadedAt: '2026-11-01T00:00:00Z', rowCount: 1,
      },
      inferencexError: model ? 'Private dataset unavailable.' : null,
      filters: {
        accelerator: model ? 'H200' : '', model, backend: 'vllm', backendVersion: '0.31.0',
        isl: model ? '8192' : '', osl: model ? '1024' : '', tp: model ? '1' : '',
      },
      options,
      configurations: model ? [configuration] : [],
    }
    return new Response(JSON.stringify(payload), { status: 200 })
  }))
})

afterEach(async () => {
  await act(async () => root.unmount())
  host.remove()
  vi.unstubAllGlobals()
  window.history.replaceState({}, '', '/')
})

describe('Compare resolution warnings', () => {
  it('shows a missing linked configuration alongside an InferenceX source error', async () => {
    await act(async () => root.render(<BenchmarkComparisonPage />))

    await vi.waitFor(() => {
      expect(host.textContent).toContain('InferenceX measurements unavailable')
      expect(host.textContent).toContain('Private dataset unavailable.')
      expect(host.textContent).toContain('Linked configuration is no longer available')
    })
  })
})

// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { useCostings } = vi.hoisted(() => ({ useCostings: vi.fn() }))
vi.mock('@/lib/hooks/useCostings', () => ({ useCostings }))
vi.mock('@/contexts/SettingsContext', () => ({
  useSettings: () => ({ pricingSource: 'litellm', preferredCloudProvider: 'aws.us-east-1' }),
}))

import SavedResultsPage from './SavedResults'

let root: ReturnType<typeof createRoot>
let host: HTMLDivElement

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  useCostings.mockReturnValue({
    models: [],
    gpuCloudRates: new Map(),
    gpuHardwareCosts: new Map(),
    health: null,
    modelsUpdatedAt: null,
    modelsStale: false,
    isLoading: false,
    error: null,
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

describe('SavedResultsPage', () => {
  it('reads infrastructure prices even when costings are off elsewhere', async () => {
    await act(async () => root.render(<SavedResultsPage />))

    expect(useCostings).toHaveBeenCalledWith(true, 'litellm', 'systems')
    expect(host.querySelector('h1')?.textContent).toBe('Saved results')
  })
})

// @vitest-environment happy-dom
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import HybridSavings from './HybridSavings'
import HybridSavingsPage from './page'
import { useSettings } from '@/contexts/SettingsContext'
import { useCostings } from '@/lib/hooks/useCostings'
import { readRecommendStream } from '@/lib/api/recommend-stream'
import { fetchModelConfig, type FetchResult } from '@/lib/huggingface/fetch-config'
import type { RecommendResponse } from '@/lib/api/recommend'
import { useCatalog, type GpuOption } from '@/lib/hooks/useCatalog'
import { DEFAULT_WORKLOAD } from '@/lib/workload-presets'
import appConfig from '@/public/config.json'
import { CostAssumptionsProvider, COST_ASSUMPTIONS_STORAGE_KEY } from '@/contexts/CostAssumptionsContext'
import CostAssumptionsEditor from '@/app/sources/CostAssumptionsEditor'

vi.mock('@/contexts/SettingsContext', () => ({ useSettings: vi.fn() }))
vi.mock('@/lib/hooks/useCostings', () => ({ useCostings: vi.fn() }))
vi.mock('@/lib/api/recommend-stream', () => ({ readRecommendStream: vi.fn() }))
vi.mock('@/lib/huggingface/fetch-config', () => ({ fetchModelConfig: vi.fn() }))
vi.mock('@/lib/app-config', () => ({ getAppConfig: () => appConfig }))
vi.mock('next/dynamic', () => ({ default: () => (props: { points: unknown; hostedModelLabel?: string; infrastructureModelLabel?: string }) => <div data-testid="chart">Comparison chart {props.hostedModelLabel} {props.infrastructureModelLabel} {JSON.stringify(props.points)}</div> }))
vi.mock('@/lib/hooks/useCatalog', () => ({ useCatalog: vi.fn() }))

let root: ReturnType<typeof createRoot>
let host: HTMLDivElement
const models = ['first', 'second', 'third', 'fourth'].map((provider, index) => ({
  id: 'Qwen/Qwen3-8B', name: 'Qwen3-8B', provider, tier: 'fast' as const,
  price_per_m_input: 0.1 * (index + 1), price_per_m_output: 0.2 * (index + 1),
  context_window: null, updated_at: null,
}))
const sizing = {
  status: 'completed', recommendation: { gpusPerReplica: 1, replicasNeeded: 1 },
  throughput: { tokensPerSecond: 1000 }, performance: { ttftLatencyMs: 100, tpotMs: 20 },
} as RecommendResponse
beforeEach(async () => {
  localStorage.clear()
  vi.mocked(fetchModelConfig).mockReset()
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  vi.mocked(useCatalog).mockReturnValue({
    gpuOptions: [{ systemId: 'l40s', label: 'NVIDIA L40S', tdpWatts: 350 } as GpuOption],
    modelOptions: ['Qwen/Qwen3-8B', 'unpriced/model'],
    modelSpecs: new Map(), backendOptions: [], timeoutSeconds: 180,
    isLoading: false, error: null,
  })
  vi.mocked(useSettings).mockReturnValue({
    hydrated: true, defaultModel: 'Qwen/Qwen3-8B', testedModels: [],
    hfToken: '', backendVersion: '', inferenceBackend: 'vllm',
    costingsEnabled: true, preferredCloudProvider: null, pricingSource: 'merged',
    setDefaultModel: vi.fn(), setHfToken: vi.fn(), setInferenceBackend: vi.fn(),
    setBackendVersion: vi.fn(), setCostingsEnabled: vi.fn(), setPreferredCloudProvider: vi.fn(),
    setPricingSource: vi.fn(),
  })
  vi.mocked(useCostings).mockReturnValue({
    models, health: null, modelsUpdatedAt: null, modelsStale: false,
    isLoading: false, error: null, gpuCloudRates: new Map(), gpuHardwareCosts: new Map(),
  })
  vi.mocked(readRecommendStream).mockResolvedValue(sizing)
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response()))
  host = document.createElement('div'); document.body.append(host); root = createRoot(host)
})
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.unstubAllGlobals(); vi.useRealTimers() })
async function render() { await act(async () => root.render(<CostAssumptionsProvider><HybridSavings /></CostAssumptionsProvider>)) }
async function click(text: string) {
  const button = [...host.querySelectorAll('button')].find(element => element.textContent === text)!
  expect(button).toBeTruthy()
  await act(async () => button.click())
}
async function select(id: string, value: string) {
  const field = host.querySelector<HTMLSelectElement>('#' + id)!
  await act(async () => { field.value = value; field.dispatchEvent(new Event('change', { bubbles: true })) })
}
async function comboOptions(id: string) {
  await act(async () => host.querySelector<HTMLInputElement>('#' + id)!.click())
  return [...document.querySelectorAll<HTMLElement>(`[id^="${id}-opt-"]`)]
}
async function chooseComboOption(id: string, match: number | string) {
  const options = await comboOptions(id)
  const option = typeof match === 'number' ? options[match] : options.find(item => item.textContent?.includes(match))!
  expect(option).toBeTruthy()
  await act(async () => option.click())
}
async function editNumber(id: string, value: string) {
  const field = host.querySelector<HTMLInputElement>('#' + id)!
  await act(async () => {
    field.focus()
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(field, value)
    field.dispatchEvent(new Event('input', { bubbles: true }))
  })
  await act(async () => field.blur())
}
async function chooseApiModel(label: string) {
  const input = host.querySelector<HTMLInputElement>('#hybrid-api-model')!
  expect(input).toBeTruthy()
  await act(async () => {
    input.focus()
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, label)
    input.dispatchEvent(new Event('input', { bubbles: true }))
  })
  const option = [...document.querySelectorAll<HTMLElement>('[id^="hybrid-api-model-opt-"]')].find(element => element.textContent === label)!
  expect(option).toBeTruthy()
  await act(async () => option.click())
}

const frontierOffer = {
  id: 'openai/gpt-5', name: 'openai/gpt-5', provider: 'OpenAI', tier: 'balanced' as const,
  price_per_m_input: 1.25, price_per_m_output: 10, context_window: 128000,
  updated_at: '2026-10-05T04:00:00Z', source: 'openrouter' as const,
}

describe('Hybrid model configuration readiness', () => {
  const testedModel = 'tested/model'
  const config = { model_type: 'qwen3', hidden_size: 4096 }
  function calculateButton() {
    return [...host.querySelectorAll<HTMLButtonElement>('button')].find(button => button.textContent === 'Calculate')!
  }
  async function chooseModel(id: string) {
    const input = host.querySelector<HTMLInputElement>('#hybrid-model')!
    await act(async () => {
      input.focus()
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, id)
      input.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await act(async () => input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })))
  }
  beforeEach(() => {
    vi.useFakeTimers()
    vi.mocked(useSettings).mockReturnValue({ ...vi.mocked(useSettings)(), testedModels: [testedModel, 'tested/other'] })
    vi.mocked(useCostings).mockReturnValue({
      ...vi.mocked(useCostings)(true),
      models: [...models, ...[testedModel, 'tested/other'].map(id => ({ ...models[0], id, name: id }))],
    })
  })
  it('waits through debounce and fetching, then sends the selected Tested model configuration', async () => {
    let complete!: (result: FetchResult) => void
    vi.mocked(fetchModelConfig).mockReturnValue(new Promise(resolve => { complete = resolve }))
    await render(); await chooseModel(testedModel)
    expect(calculateButton().disabled).toBe(true)
    await act(async () => vi.advanceTimersByTimeAsync(500))
    expect(fetchModelConfig).toHaveBeenLastCalledWith(testedModel, '')
    expect(calculateButton().disabled).toBe(true)
    await act(async () => complete({ success: true, config, source: 'huggingface' }))
    expect(calculateButton().disabled).toBe(false)
    await click('Calculate')
    const request = JSON.parse(vi.mocked(fetch).mock.calls[0][1]!.body as string)
    expect(request.model_path).toBe(testedModel)
    expect(request.model_config).toEqual(config)
  })
  it('keeps calculation disabled when a Tested model configuration lookup fails', async () => {
    vi.mocked(fetchModelConfig).mockResolvedValue({ success: false, error: 'Access denied', source: 'huggingface' })
    await render(); await chooseModel(testedModel)
    await act(async () => vi.advanceTimersByTimeAsync(500))
    expect(host.textContent).toContain('Access denied')
    expect(calculateButton().disabled).toBe(true)
  })
  it('does not reuse a previous model configuration when another Tested model is selected', async () => {
    vi.mocked(fetchModelConfig).mockResolvedValue({ success: true, config, source: 'huggingface' })
    await render(); await chooseModel(testedModel)
    await act(async () => vi.advanceTimersByTimeAsync(500))
    expect(calculateButton().disabled).toBe(false)
    await chooseModel('tested/other')
    expect(calculateButton().disabled).toBe(true)
  })
})

describe('Hybrid page composition', () => {
  it('keeps the default API model label linked to the model selected above', async () => {
    await render(); await click('Calculate')
    expect(host.querySelector<HTMLInputElement>('#hybrid-api-model')!.value).toBe('Qwen/Qwen3-8B')
    const input = host.querySelector<HTMLInputElement>('#hybrid-model')!
    await act(async () => {
      input.focus()
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, 'unpriced/model')
      input.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await act(async () => input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })))
    expect(host.querySelector<HTMLInputElement>('#hybrid-api-model')!.value).toBe('unpriced/model')
  })

  it('uses one API model dropdown and keeps the exact selected offer on the back', async () => {
    vi.mocked(useCostings).mockReturnValue({ ...vi.mocked(useCostings)(true), models: [...models, frontierOffer] })
    await render(); await click('Calculate')
    const front = host.querySelector('[class*="costFront"]')!
    expect(front.querySelectorAll('#hybrid-api-model')).toHaveLength(1)
    expect(front.querySelector('#hybrid-api-model')!.getAttribute('aria-label')).toBe('API model')
    expect(front.querySelector<HTMLInputElement>('#hybrid-api-model')!.value).toBe('Qwen/Qwen3-8B')
    expect(front.textContent).not.toContain('Same as selected model')
    expect(front.textContent).not.toContain('Compare API model')
    expect([...front.querySelectorAll('dt')].map(element => element.textContent)).toEqual(['Pricing source'])
    expect(front.querySelector('label[for="hybrid-hosted-provider"]')!.textContent).toBe('Provider')
    await chooseApiModel('gpt-5')
    await act(async () => host.querySelector<HTMLButtonElement>('button[aria-label="View Hosted API details"]')!.click())
    const back = host.querySelector('[class*="costBack"]')!
    expect(back.getAttribute('aria-hidden')).toBe('false')
    expect(back.textContent).toContain('Offer / checkpointopenai/gpt-5')
    expect(back.textContent).toContain('$8,817')
  })

  it('updates hosted costs, graph and model labels without changing or re-running GPU sizing', async () => {
    vi.mocked(useCostings).mockReturnValue({ ...vi.mocked(useCostings)(true), models: [...models, frontierOffer] })
    await render(); await click('Calculate'); await click('Cost vs workload (break-even analysis)')
    const originalCosts = [...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)
    const originalChart = host.querySelector('[data-testid="chart"]')!.textContent
    const calls = vi.mocked(readRecommendStream).mock.calls.length
    await chooseApiModel('gpt-5')
    const costs = [...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)
    expect(costs[0]).toContain('$8,817') // $2,500 input + $5,000 output + existing $1,317 full-TCO overhead.
    expect(costs.slice(1)).toEqual(originalCosts.slice(1))
    expect(vi.mocked(readRecommendStream).mock.calls.length).toBe(calls)
    expect(host.querySelector('[data-testid="chart"]')!.textContent).not.toBe(originalChart)
    expect(host.querySelector('[data-testid="chart"]')!.textContent).toContain('openai/gpt-5 Qwen/Qwen3-8B')
    expect(host.querySelector('#hybrid-hosted-provider')).toBeNull()
    expect(host.textContent).toContain('OpenRouter catalogue')
    expect(host.textContent).not.toContain('Equal input/output token volumes')
    expect(host.textContent).not.toContain('not equivalent model quality')
    expect(host.querySelector('[class*="costBack"]')!.getAttribute('aria-hidden')).toBe('true')
    await chooseApiModel('Qwen/Qwen3-8B')
    expect([...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)).toEqual(originalCosts)
    expect(host.querySelector('[data-testid="chart"]')!.textContent).toBe(originalChart)
  })

  it('allows a hosted alternative when the open-weight checkpoint has no API price', async () => {
    vi.mocked(useCostings).mockReturnValue({ ...vi.mocked(useCostings)(true), models: [frontierOffer] })
    await render()
    expect([...host.querySelectorAll<HTMLButtonElement>('button')].find(button => button.textContent === 'Calculate')!.disabled).toBe(true)
    await chooseApiModel('gpt-5'); await click('Calculate')
    expect(host.querySelectorAll('[class*="costValue"]')).toHaveLength(3)
    expect(JSON.parse(vi.mocked(fetch).mock.calls[0][1]!.body as string).model).not.toBe('openai/gpt-5')
  })

  it('keeps independent pricing-source offers selectable for the same API model', async () => {
    vi.mocked(useCostings).mockReturnValue({ ...vi.mocked(useCostings)(true), models: [...models, frontierOffer, { ...frontierOffer, source: 'litellm', price_per_m_input: 2 }] })
    await render(); await click('Calculate'); await chooseApiModel('gpt-5')
    const providers = await comboOptions('hybrid-hosted-provider')
    expect(providers).toHaveLength(2)
    expect(providers[0].textContent).toContain('OpenRouter')
    expect(providers[1].textContent).toContain('LiteLLM catalogue')
    const calls = vi.mocked(readRecommendStream).mock.calls.length
    const original = [...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)
    await chooseComboOption('hybrid-hosted-provider', 1)
    const revised = [...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)
    expect(revised[0]).not.toBe(original[0])
    expect(revised.slice(1)).toEqual(original.slice(1))
    expect(vi.mocked(readRecommendStream).mock.calls.length).toBe(calls)
  })

  it('does not silently substitute a different model when a selected live price disappears', async () => {
    vi.mocked(useCostings).mockReturnValue({ ...vi.mocked(useCostings)(true), models: [...models, frontierOffer] })
    await render(); await click('Calculate'); await chooseApiModel('gpt-5')
    vi.mocked(useCostings).mockReturnValue({ ...vi.mocked(useCostings)(true), models })
    await render()
    expect(host.textContent).toContain('Hosted comparison unavailable')
    expect(host.querySelectorAll('[class*="costValue"]')).toHaveLength(0)
    await chooseApiModel('Qwen/Qwen3-8B')
    expect(host.querySelectorAll('[class*="costValue"]')).toHaveLength(3)
  })

  it('preserves a manually selected API model across open-weight model changes', async () => {
    vi.mocked(useCostings).mockReturnValue({ ...vi.mocked(useCostings)(true), models: [...models, frontierOffer] })
    await render(); await click('Calculate'); await chooseApiModel('gpt-5')
    const input = host.querySelector<HTMLInputElement>('#hybrid-model')!
    await act(async () => {
      input.focus()
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, 'unpriced/model')
      input.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await act(async () => input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })))
    expect(host.querySelector<HTMLInputElement>('#hybrid-model')!.value).toBe('unpriced/model')
    expect(host.textContent).not.toContain('Hosted comparison unavailable')
    // The independent API choice still exists; only self-managed sizing is stale.
    expect(host.textContent).toContain('Inputs changed')
  })

  it('flips all three cards independently without changing costs, chart or sizing, and keeps provider selection working', async () => {
    await render(); await click('Calculate'); await click('Cost vs workload (break-even analysis)')
    const chart = host.querySelector('[data-testid="chart"]')!
    const originalChart = chart.textContent
    const costs = [...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)
    const sizingCalls = vi.mocked(readRecommendStream).mock.calls.length
    const titles = ['Hosted API', 'Rented GPU', 'Purchased hardware']
    for (const title of titles) {
      await act(async () => host.querySelector<HTMLButtonElement>(`button[aria-label="View ${title} details"]`)!.click())
    }
    expect([...host.querySelectorAll('[class*="costBack"]')].filter(element => element.getAttribute('aria-hidden') === 'false')).toHaveLength(3)
    expect([...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)).toEqual(costs)
    expect(chart.textContent).toBe(originalChart)
    expect(vi.mocked(readRecommendStream).mock.calls.length).toBe(sizingCalls)
    for (const title of titles) {
      await act(async () => host.querySelector<HTMLButtonElement>(`button[aria-label="Back to ${title} summary"]`)!.click())
    }
    await chooseComboOption('hybrid-hosted-provider', 1)
    expect([...host.querySelectorAll('[class*="costValue"]')][0].textContent).not.toBe(costs[0])
    expect(chart.textContent).not.toBe(originalChart)
    expect(host.querySelector('[class*="costBack"]')?.getAttribute('aria-hidden')).toBe('true')
    await act(async () => host.querySelector<HTMLButtonElement>('button[aria-label="View Hosted API details"]')!.click())
    expect(host.querySelector('[class*="costBack"]')?.textContent).toContain([...host.querySelectorAll('[class*="costValue"]')][0].textContent!.replace(' / month', ''))
  })

  it('keeps the simplified page free of removed notices while retaining controls, details and chart', async () => {
    await render()
    expect(host.textContent).not.toContain('Planning estimate')
    expect(host.querySelector('[aria-label="Planning guidance"]')).toBeNull()
    expect(host.textContent).not.toContain('AISimulators sizing is estimated')
    expect(host.textContent).toContain('Data and assumptions')
    expect(host.textContent).not.toContain('Advanced cost assumptions')
    expect(host.querySelector('a[href="/sources#cost-assumptions"]')!.textContent).toBe('Edit cost assumptions in Sources')
    expect(host.querySelector('#hardware-life')).toBeNull()
    await click('Calculate')
    for (const removedText of [
      'Effective cost uses billed tokens', 'Marginal excludes acquisition',
      'Based on Qwen/Qwen3-8B', 'See at what workload each option',
    ]) expect(host.textContent).not.toContain(removedText)
    expect(host.querySelectorAll('[class*="costValue"]')).toHaveLength(3)
    expect(host.textContent).toContain('Based on Default')
    expect(host.querySelector('#hybrid-cost-view')).not.toBeNull()
    await click('Cost vs workload (break-even analysis)')
    expect(host.textContent).not.toContain('Manual hardware choices apply throughout')
    expect(host.querySelector('[data-testid="chart"]')).not.toBeNull()
    expect(host.querySelector('a[href="/sources#cost-assumptions"]')).not.toBeNull()
  })

  it('uses Sources edits after navigation and updates all costs and chart data without a new sizing run', async () => {
    await act(async () => root.render(<CostAssumptionsProvider><CostAssumptionsEditor /></CostAssumptionsProvider>))
    await click('Shared planning inputs')
    await editNumber('cost-assumption-loadedMonthlyCostPerFte', '10000')
    await render(); await click('Calculate')
    const originalCosts = [...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)
    expect(originalCosts).toHaveLength(3)
    const sharedInputs = () => host.querySelector('[aria-labelledby="hybrid-comparison-basis-heading"]')!
    expect(sharedInputs().textContent).toContain('Staffing rate / FTE$10,000.00 / month')
    await click('Cost vs workload (break-even analysis)')
    const chartBefore = host.querySelector('[data-testid="chart"]')!.textContent
    const sizingCalls = vi.mocked(readRecommendStream).mock.calls.length
    await act(async () => {
      localStorage.setItem(COST_ASSUMPTIONS_STORAGE_KEY, JSON.stringify({ loadedMonthlyCostPerFte: 20000 }))
      window.dispatchEvent(new StorageEvent('storage', { storageArea: localStorage, key: COST_ASSUMPTIONS_STORAGE_KEY }))
    })
    const revisedCosts = [...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)
    expect(sharedInputs().textContent).toContain('Staffing rate / FTE$20,000.00 / month')
    for (let i = 0; i < 3; i++) expect(revisedCosts[i]).not.toBe(originalCosts[i])
    expect(host.querySelector('[data-testid="chart"]')!.textContent).not.toBe(chartBefore)
    expect(vi.mocked(readRecommendStream).mock.calls.length).toBe(sizingCalls)
    await act(async () => root.render(<CostAssumptionsProvider><CostAssumptionsEditor /></CostAssumptionsProvider>))
    await click('Shared planning inputs')
    expect(host.querySelector<HTMLInputElement>('#cost-assumption-loadedMonthlyCostPerFte')!.value).toBe('20000')
  })

  it('keeps the Sources link independent and visible beside the collapsed calculation details', async () => {
    await render()
    const sizingCalls = vi.mocked(readRecommendStream).mock.calls.length
    const toggle = host.querySelector<HTMLButtonElement>('#hybrid-calculation-details-toggle')!
    const content = host.querySelector<HTMLElement>('#hybrid-calculation-details')!
    const link = host.querySelector<HTMLAnchorElement>('a[href="/sources#cost-assumptions"]')!
    expect(toggle.getAttribute('aria-controls')).toBe(content.id)
    expect(content.getAttribute('aria-labelledby')).toBe(toggle.id)
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(content.hidden).toBe(true)
    expect(link.parentElement!.contains(toggle)).toBe(true)
    expect(toggle.contains(link)).toBe(false)
    expect(content.contains(link)).toBe(false)
    expect(link.closest('[hidden]')).toBeNull()
    expect(content.textContent).toContain('Planning capacity90%')
    expect(content.textContent).toContain('Hardware life4 years')
    // Prevent navigation in this DOM test, while exercising the link's event path.
    link.addEventListener('click', event => event.preventDefault())
    await act(async () => link.click())
    expect(content.hidden).toBe(true)
    await click('Data and assumptions')
    expect(content.hidden).toBe(false)
    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    expect(link.closest('[hidden]')).toBeNull()
    await click('Data and assumptions')
    expect(content.hidden).toBe(true)
    expect(vi.mocked(readRecommendStream).mock.calls.length).toBe(sizingCalls)
  })

  it('groups calculation details into readable labelled sections without changing results', async () => {
    await render(); await click('Calculate'); await click('Cost vs workload (break-even analysis)')
    const costs = [...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)
    const chart = host.querySelector('[data-testid="chart"]')!.textContent
    const sizingCalls = vi.mocked(readRecommendStream).mock.calls.length
    await click('Data and assumptions')
    const sections = host.querySelectorAll('#hybrid-calculation-details > section')
    expect([...sections].map(section => section.querySelector('h3')!.textContent)).toEqual([
      'Comparison basis', 'Cost assumptions', 'Data sources',
    ])
    for (const section of sections) {
      expect(section.getAttribute('aria-labelledby')).toBe(section.querySelector('h3')!.id)
      expect(section.querySelectorAll('dt').length).toBeGreaterThan(0)
      expect(section.querySelectorAll('dt').length).toBe(section.querySelectorAll('dd').length)
    }
    expect(sections[2].textContent).toContain('1 concurrent request')
    expect(sections[2].textContent).toContain('2 models · 1 GPU systems')
    expect(sections[2].textContent).toContain('not live quotes')
    expect(sections[1].textContent).toContain('Catalogue allowance — replaces the Sources fallback')
    expect(sections[1].textContent).not.toContain('Complete server price')
    expect(host.querySelectorAll('#hybrid-calculation-details button, #hybrid-calculation-details summary')).toHaveLength(0)
    expect([...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)).toEqual(costs)
    expect(host.querySelector('[data-testid="chart"]')!.textContent).toBe(chart)
    expect(vi.mocked(readRecommendStream).mock.calls.length).toBe(sizingCalls)
  })

  it.each([
    ['2026-10-02T04:16:34.002725+00:00', '2 Oct 2026, 04:16 UTC'],
    ['invalid-date', 'invalid-date'],
    [null, 'Not supplied'],
  ])('formats collection timestamps safely: %s', async (timestamp, expected) => {
    vi.mocked(useCostings).mockReturnValue({
      models, modelsUpdatedAt: timestamp, modelsStale: false, isLoading: false, error: null,
      gpuCloudRates: new Map(), gpuHardwareCosts: new Map(),
      health: { status: 'ok', version: 'test', sources: {
        'cloud.aws': { last_success: timestamp, stale: true, last_error: null },
      } },
    })
    await render(); await click('Data and assumptions')
    const prices = host.querySelector('[aria-labelledby="hybrid-pricing-details-heading"]')!
    expect(prices.textContent).toContain('Stale sourcesAWS')
    expect(prices.textContent).toContain(expected)
    expect(prices.querySelector('a[href="/sources"]')!.textContent).toBe('View source collection times')
    if (timestamp === null) expect(prices.textContent).toContain('Not supplied')
  })

  it('uses the shared grey page-section layout with the heading outside the input card', async () => {
    await act(async () => root.render(<CostAssumptionsProvider><HybridSavingsPage /></CostAssumptionsProvider>))
    const section = host.querySelector<HTMLElement>('.pf-v6-c-page__main-section')!
    expect(section).not.toBeNull()
    expect(section.style.backgroundColor).toBe('#f5f5f5')
    expect(section.style.padding).toBe('0px')
    const heading = section.querySelector('h1')!
    expect(heading.textContent).toBe('Hybrid savings')
    expect(heading.closest('.pf-v6-c-card')).toBeNull()
    expect(section.querySelector('#hybrid-input-tokens')!.closest('.pf-v6-c-card')).not.toBeNull()
    expect(section.querySelector('main')).toBeNull()
  })

  it('integrates the workload disclosure into the input card without submitting', async () => {
    await render()
    const toggle = host.querySelector<HTMLButtonElement>('#hybrid-workload-toggle')!
    const body = host.querySelector<HTMLDivElement>('#hybrid-advanced-workload')!
    const inputCard = host.querySelector('#hybrid-input-tokens')!.closest('.pf-v6-c-card')
    expect(toggle.closest('.pf-v6-c-card')).toBe(inputCard)
    expect(body.closest('.pf-v6-c-card')).toBe(inputCard)
    expect(toggle.type).toBe('button')
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(toggle.getAttribute('aria-controls')).toBe(body.id)
    expect(body.getAttribute('aria-labelledby')).toBe(toggle.id)
    expect(body.hidden).toBe(true)
    expect(toggle.textContent).toContain('1,000 ms TTFT')
    await act(async () => toggle.click())
    expect(body.hidden).toBe(false)
    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    expect(toggle.textContent).not.toContain('1,000 ms TTFT')
    expect(body.querySelectorAll('input')).toHaveLength(8)
    expect(body.querySelectorAll('select')).toHaveLength(1)
    await act(async () => toggle.click())
    expect(body.hidden).toBe(true)
    await click('Adjust')
    expect(body.hidden).toBe(false)
    expect(fetch).not.toHaveBeenCalled()
  })

  it('preserves edits through collapse and sends updated targets to sizing', async () => {
    await render(); await click('Adjust')
    await editNumber('hybrid-average-input', '3000')
    await editNumber('hybrid-average-output', '600')
    await editNumber('hybrid-ttft', '2500')
    await editNumber('hybrid-tpot', '45')
    await editNumber('hybrid-active-hours', '200')
    await editNumber('hybrid-peak', '2')
    await select('hybrid-billing-mode', 'active-window')
    const toggle = host.querySelector<HTMLButtonElement>('#hybrid-workload-toggle')!
    await act(async () => toggle.click())
    expect(toggle.textContent).toContain('2,500 ms TTFT · 45 ms TPOT · 1 concurrent · 0 prefix tokens · 200 hours/month · 2× peak · Active window')
    await click('Adjust')
    expect(host.querySelector<HTMLInputElement>('#hybrid-average-input')!.value).toBe('3000')
    expect(host.querySelector<HTMLSelectElement>('#hybrid-billing-mode')!.value).toBe('active-window')
    const field = host.querySelector<HTMLInputElement>('#hybrid-average-input')!
    const enter = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true })
    await act(async () => { field.focus(); field.dispatchEvent(enter) })
    expect(enter.defaultPrevented).toBe(true)
    expect(fetch).not.toHaveBeenCalled()
    await click('Calculate')
    const payload = JSON.parse(vi.mocked(fetch).mock.calls[0][1]!.body as string)
    expect(payload).toMatchObject({ isl: 3000, osl: 600, ttft: 2500, tpot: 45 })
    expect(host.textContent).toContain('200 hours/month')
    expect(host.textContent).toContain('Active window')
  })

  it('keeps defaults and makes advanced settings secondary', async () => {
    await render()
    expect(host.querySelector('h1')?.textContent).toBe('Hybrid savings')
    expect(host.querySelector<HTMLInputElement>('#hybrid-input-tokens')?.value).toBe('2,000,000,000')
    expect(host.querySelector<HTMLInputElement>('#hybrid-output-tokens')?.value).toBe('500,000,000')
    expect(host.textContent).not.toContain('Select a model from the live catalogue')
    expect(host.querySelector('[aria-expanded="true"]')).toBeNull()
    expect(host.textContent).toContain('730 hours/month')
    expect(host.textContent).toContain('2,048 input / 128 output')
  })

  it('uses Recommend-style workload buttons and opens settings without showing a token-mix notice', async () => {
    await render()
    const group = host.querySelector('#hybrid-profile')!
    const buttons = [...group.querySelectorAll<HTMLButtonElement>('button')]
    expect(buttons.map(button => button.textContent)).toEqual([DEFAULT_WORKLOAD.label, ...appConfig.workloadPresets.map(preset => preset.label)])
    expect(group.getAttribute('role')).toBe('group')
    expect(buttons[0].classList.contains('pf-m-secondary')).toBe(true)
    expect(buttons.slice(1).every(button => button.classList.contains('pf-m-tertiary'))).toBe(true)
    for (const button of buttons) {
      expect(button.type).toBe('button')
      await act(async () => button.click())
      expect(button.getAttribute('aria-pressed')).toBe('true')
      expect(button.classList.contains('pf-m-secondary')).toBe(true)
      expect(buttons.filter(item => item.getAttribute('aria-pressed') === 'true')).toHaveLength(1)
      expect(host.querySelector<HTMLDivElement>('#hybrid-advanced-workload')!.hidden).toBe(false)
      expect(host.textContent).not.toContain('Token mix differs from the average request')
      expect(host.querySelector<HTMLInputElement>('#hybrid-input-tokens')!.value).toBe('2,000,000,000')
      expect(host.querySelector<HTMLInputElement>('#hybrid-output-tokens')!.value).toBe('500,000,000')
      await act(async () => host.querySelector<HTMLButtonElement>('#hybrid-workload-toggle')!.click())
      expect(host.querySelector<HTMLDivElement>('#hybrid-advanced-workload')!.hidden).toBe(true)
    }
    expect(fetch).not.toHaveBeenCalled()
  })

  it.each([DEFAULT_WORKLOAD, ...appConfig.workloadPresets])('applies all shared request settings for $label without changing monthly demand or billing', async preset => {
    await render(); await click('Adjust')
    await editNumber('hybrid-active-hours', '200')
    await editNumber('hybrid-peak', '2')
    await select('hybrid-billing-mode', 'always-on')
    await click(preset.label)
    for (const [id, value] of [
      ['hybrid-average-input', preset.isl], ['hybrid-average-output', preset.osl],
      ['hybrid-ttft', preset.ttft], ['hybrid-tpot', preset.tpot],
      ['hybrid-concurrency', preset.concurrency], ['hybrid-prefix', preset.prefix],
    ] as const) expect(host.querySelector<HTMLInputElement>('#' + id)!.value).toBe(String(value))
    expect(host.querySelector<HTMLInputElement>('#hybrid-active-hours')!.value).toBe('200')
    expect(host.querySelector<HTMLInputElement>('#hybrid-peak')!.value).toBe('2')
    expect(host.querySelector<HTMLSelectElement>('#hybrid-billing-mode')!.value).toBe('always-on')
    expect(host.querySelector<HTMLInputElement>('#hybrid-input-tokens')!.value).toBe('2,000,000,000')
    expect(host.querySelector<HTMLInputElement>('#hybrid-output-tokens')!.value).toBe('500,000,000')
    await click('Calculate')
    expect(JSON.parse(vi.mocked(fetch).mock.calls[0][1]!.body as string)).toMatchObject({
      isl: preset.isl, osl: preset.osl, ttft: preset.ttft, tpot: preset.tpot,
      target_concurrency: preset.concurrency, prefix: preset.prefix,
    })
    await click('Data and assumptions')
    const reference = [...host.querySelectorAll('dt')].find(term => term.textContent === 'Capacity reference')!
    expect(reference.nextElementSibling!.textContent).toBe(`${preset.concurrency.toLocaleString('en-US')} concurrent ${preset.concurrency === 1 ? 'request' : 'requests'}`)
    expect(host.textContent).toContain('Monthly cost breakdown')
  })

  it('shows the actual default and edited sizing concurrency in calculation details', async () => {
    await render(); await click('Data and assumptions')
    const reference = () => [...host.querySelectorAll('dt')]
      .find(term => term.textContent === 'Capacity reference')!.nextElementSibling!.textContent
    expect(reference()).toBe('1 concurrent request')
    await click('Adjust')
    await editNumber('hybrid-concurrency', '64')
    expect(reference()).toBe('64 concurrent requests')
    await click('Calculate')
    expect(JSON.parse(vi.mocked(fetch).mock.calls[0][1]!.body as string)).toMatchObject({ target_concurrency: 64 })
  })

  it('invalidates previous results when concurrency or prefix settings change', async () => {
    await render(); await click('Calculate'); await click('Adjust')
    await editNumber('hybrid-concurrency', '2')
    expect(host.textContent).toContain('Inputs changed')
    await click('Calculate')
    expect(host.textContent).not.toContain('Inputs changed')
    await editNumber('hybrid-prefix', '128')
    expect(host.textContent).toContain('Inputs changed')
    await click('Calculate')
    const payload = JSON.parse(vi.mocked(fetch).mock.calls.at(-1)![1]!.body as string)
    expect(payload).toMatchObject({ target_concurrency: 2, prefix: 128 })
  })

  it('blocks fractional concurrency and a cached prefix longer than the input', async () => {
    await render(); await click('Adjust')
    await editNumber('hybrid-concurrency', '1.5')
    const calculate = [...host.querySelectorAll<HTMLButtonElement>('button')].find(button => button.textContent === 'Calculate')!
    expect(calculate.disabled).toBe(true)
    expect(host.textContent).toContain('Concurrency must be a positive whole number')
    await editNumber('hybrid-concurrency', '1')
    await editNumber('hybrid-prefix', '1024')
    await editNumber('hybrid-average-input', '512')
    expect(calculate.disabled).toBe(true)
    await editNumber('hybrid-prefix', '512')
    expect(calculate.disabled).toBe(false)
    expect(fetch).not.toHaveBeenCalled()
  })

  it('offers all matched providers and genuinely selectable independent hardware', async () => {
    await render(); await click('Calculate')
    const providers = await comboOptions('hybrid-hosted-provider')
    expect(providers).toHaveLength(4)
    expect(providers[0].textContent).toContain('first')
    expect(providers.map(option => option.textContent)).toEqual([
      'first$300/month token usage · cheapest', 'second$600/month token usage',
      'third$900/month token usage', 'fourth$1,200/month token usage',
    ])
    const rented = host.querySelector<HTMLInputElement>('#hybrid-rented-configuration')!
    const owned = host.querySelector<HTMLInputElement>('#hybrid-owned-configuration')!
    expect(rented.value).toContain('Automatic')
    expect(owned.value).toContain('Automatic')
    const rentedItems = await comboOptions('hybrid-rented-configuration')
    const ownedItems = await comboOptions('hybrid-owned-configuration')
    expect(rentedItems.length).toBeGreaterThan(2)
    const rentedAutomaticLabel = rentedItems[0].textContent
    const ownedAutomaticLabel = ownedItems[0].textContent
    expect(rentedAutomaticLabel).toContain('NVIDIA L40S')
    expect(ownedAutomaticLabel).toContain('NVIDIA L40S')
    const originalCosts = [...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)
    await chooseComboOption('hybrid-hosted-provider', 3)
    const hostedCosts = [...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)
    expect(hostedCosts[0]).not.toBe(originalCosts[0])
    expect(hostedCosts.slice(1)).toEqual(originalCosts.slice(1))
    await chooseComboOption('hybrid-rented-configuration', rentedItems.length - 1)
    const rentedCosts = [...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)
    expect(rentedCosts[1]).not.toBe(hostedCosts[1])
    expect(rentedCosts[2]).toBe(hostedCosts[2])
    expect(owned.value).toContain('Automatic')
    expect((await comboOptions('hybrid-rented-configuration'))[0].textContent).toBe(rentedAutomaticLabel)
    await act(async () => rented.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })))
    const pin = rented.value
    await chooseComboOption('hybrid-owned-configuration', ownedItems.length - 1)
    expect(rented.value).toBe(pin)
    expect((await comboOptions('hybrid-owned-configuration'))[0].textContent).toBe(ownedAutomaticLabel)
    await act(async () => owned.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })))
    await select('hybrid-cost-view', 'marginal')
    expect(rented.value).toBe(pin)
    expect([...host.querySelectorAll('[class*="costValue"]')][0].textContent).not.toBe(rentedCosts[0])
    expect(host.textContent).toContain('Cheapest')
  })

  it('updates provider token-usage prices and cheapest order for the entered token mix, independently of the TCO view', async () => {
    vi.mocked(useCostings).mockReturnValue({ ...vi.mocked(useCostings)(true), models: [models[0], { ...models[1], price_per_m_output: 0.01 }] })
    await render(); await click('Calculate')
    let options = await comboOptions('hybrid-hosted-provider')
    expect(options.map(option => option.textContent)).toEqual(['first$300/month token usage · cheapest', 'second$405/month token usage'])
    await select('hybrid-cost-view', 'marginal')
    options = await comboOptions('hybrid-hosted-provider')
    expect(options.map(option => option.textContent)).toEqual(['first$300/month token usage · cheapest', 'second$405/month token usage'])
    await editNumber('hybrid-input-tokens', '1000000')
    await editNumber('hybrid-output-tokens', '20000000')
    await click('Calculate')
    options = await comboOptions('hybrid-hosted-provider')
    expect(options.map(option => option.textContent)).toEqual(['second$0.4/month token usage · cheapest', 'first$4.1/month token usage'])
  })

  it('searches hardware details and supports keyboard selection and automatic reset without flipping or re-sizing', async () => {
    await render(); await click('Calculate'); await click('Cost vs workload (break-even analysis)')
    const input = host.querySelector<HTMLInputElement>('#hybrid-rented-configuration')!
    const originalCosts = [...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)
    const calls = vi.mocked(readRecommendStream).mock.calls.length
    await act(async () => {
      input.focus()
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, 'EC2 g6e.xlarge')
      input.dispatchEvent(new Event('input', { bubbles: true }))
    })
    const options = document.querySelectorAll<HTMLElement>('[id^="hybrid-rented-configuration-opt-"]')
    expect(options).toHaveLength(1)
    expect(options[0].textContent).toContain('EC2 g6e.xlarge')
    expect(host.contains(options[0])).toBe(false) // Portal menu stays outside the animated card.
    await act(async () => input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })))
    expect(input.value).toContain('NVIDIA L40S')
    expect(input.value).not.toContain('Automatic')
    expect(host.querySelectorAll('[class*="costBack"][aria-hidden="false"]')).toHaveLength(0)
    expect(vi.mocked(readRecommendStream).mock.calls.length).toBe(calls)
    const clear = input.closest('[class*="wrapper"]')!.querySelector<HTMLButtonElement>('button[aria-label="Clear selection"]')!
    await act(async () => clear.click())
    await act(async () => input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })))
    expect(input.value).toContain('Automatic')
    expect([...host.querySelectorAll('[class*="costValue"]')].map(element => element.textContent)).toEqual(originalCosts)
  })

  it('shows no fabricated configuration when live sizing fails', async () => {
    vi.mocked(readRecommendStream).mockResolvedValue({ status: 'failed', requestId: 'bad', error: { code: 'NO_FIT', message: 'No fit' } })
    await render(); await click('Calculate')
    expect(host.textContent).toContain('Comparison unavailable')
    expect(host.querySelector('#hybrid-rented-configuration')).toBeNull()
  })

  it('hides stale results, preserves a valid manual choice and resets an ineligible one', async () => {
    await render(); await click('Calculate')
    const rented = host.querySelector<HTMLInputElement>('#hybrid-rented-configuration')!
    await chooseComboOption('hybrid-rented-configuration', 'EC2 g6e.xlarge')
    const pin = rented.value
    await click('RAG / Search')
    expect(host.textContent).toContain('Inputs changed')
    expect(host.querySelector('#hybrid-rented-configuration')).toBeNull()
    await click('Calculate')
    expect(host.querySelector<HTMLInputElement>('#hybrid-rented-configuration')!.value).toBe(pin)
    expect(host.textContent).toContain('EC2 g6e.xlarge')
    vi.mocked(readRecommendStream).mockResolvedValue({
      ...sizing, recommendation: { gpusPerReplica: 8, replicasNeeded: 1 },
    } as RecommendResponse)
    await click('Agentic / Coding'); await click('Calculate')
    expect(host.querySelector<HTMLInputElement>('#hybrid-rented-configuration')!.value).toContain('Automatic')
  })

  it('shows missing rented and purchased coverage instead of inventing prices', async () => {
    vi.mocked(readRecommendStream).mockResolvedValue({
      ...sizing, recommendation: { gpusPerReplica: 1024, replicasNeeded: 1 },
    } as RecommendResponse)
    await render(); await click('Calculate')
    expect(host.textContent).toContain('No sized configuration has an eligible complete cloud offer')
    expect(host.textContent).toContain('No sized configuration has an eligible complete-server purchase price')
    expect(host.querySelector('#hybrid-rented-configuration')).toBeNull()
    expect(host.querySelector('#hybrid-owned-configuration')).toBeNull()
  })

  it('handles disabled costings without rendering an active calculator', async () => {
    vi.mocked(useSettings).mockReturnValue({ ...vi.mocked(useSettings)(), costingsEnabled: false })
    await render()
    expect(host.textContent).toContain('Enable experimental costings')
    expect(host.querySelector('form')).toBeNull()
  })

  it('disables calculation when the feed has no exact hosted price', async () => {
    vi.mocked(useCostings).mockReturnValue({ ...vi.mocked(useCostings)(true), models: [] })
    await render()
    expect(host.textContent).toContain('Hosted comparison unavailable')
    expect([...host.querySelectorAll('button')].find(button => button.textContent === 'Calculate')?.disabled).toBe(true)
  })
})

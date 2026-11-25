// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { PublicRegistryPair, ValidationGateThresholds } from '@/lib/tested-models/registry-types'
import type { ThroughputCurve } from '@/lib/tested-models/throughput-series'
import { TestedModelPair } from './TestedModelPair'

// The page reads its shared view through `useSearchParams`. Tests mount it
// directly, so supply the current search rather than the app router.
const navigation = vi.hoisted(() => ({ search: '' }))

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(navigation.search),
}))

vi.mock('next/link', () => ({
  default: ({
    href,
    children,
    ...rest
  }: {
    href: string
    children?: React.ReactNode
  } & Record<string, unknown>) => <a href={href} {...rest}>{children}</a>,
}))

// Victory needs layout to draw; these stubs expose what the page asked it to
// plot without depending on SVG measurement in happy-dom.
vi.mock('@patternfly/react-charts/victory', () => ({
  Chart: ({ children }: { children?: React.ReactNode }) => <div data-testid="chart">{children}</div>,
  ChartAxis: () => null,
  ChartLine: ({
    data,
    style,
  }: {
    data: Array<{ x: number; y: number; name?: string }>
    style?: { data?: { stroke?: string; opacity?: number; strokeWidth?: number } }
  }) => (
    <div
      data-testid="chart-line"
      data-stroke={style?.data?.stroke}
      data-opacity={String(style?.data?.opacity ?? '')}
      data-stroke-width={String(style?.data?.strokeWidth ?? '')}
      data-points={JSON.stringify(data.map(datum => [datum.x, datum.y]))}
    />
  ),
  ChartScatter: ({ data }: { data: Array<{ x: number; y: number; name?: string }> }) => (
    <div data-testid="chart-scatter" data-points={JSON.stringify(data.map(datum => [datum.x, datum.y, datum.name]))} />
  ),
  ChartTooltip: () => null,
  ChartVoronoiContainer: () => null,
}))

vi.mock('@/lib/hooks/useCatalog', () => ({
  useCatalog: () => ({
    gpuOptions: [
      {
        systemId: 'h200_sxm',
        label: 'NVIDIA H200',
        vendor: 'nvidia',
        vramGb: 141,
        bandwidthTbps: 4.8,
        tflopsBf16: 989,
        architecture: 'hopper',
      },
    ],
    modelSpecs: new Map(),
    isLoading: false,
  }),
}))

vi.mock('../TestedModelsShell', () => ({
  TestedModelsShell: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock('@/components/help/CopyLinkButton', () => ({
  CopyLinkButton: ({ getUrl }: { getUrl: () => string | null }) => (
    <button type="button" data-testid="copy-link" onClick={() => getUrl()}>Copy link</button>
  ),
}))

const pair: PublicRegistryPair = {
  id: 'google_gemma_4_31b_it__h200',
  modelId: 'google/gemma-4-31b-it',
  systemId: 'h200_sxm',
  status: 'validated',
  coverage: {
    records: 233,
    concurrencyLevels: [1, 10, 40],
    concurrencyLevelCount: 3,
    backendVersions: ['vLLM-0.24.0'],
    backends: ['vllm'],
  },
  diagnostics: { trainingStatus: 'trained', reasons: [] },
  metrics: { test: { roc_auc: 0.91, f1: 0.8, precision: 0.82, recall: 0.78 } },
}

const gate: ValidationGateThresholds = {
  minRecords: 100,
  minTestAuc: 0.7,
  minTestF1: 0.3,
  minConcurrencyLevels: 5,
}

function point(concurrency: number, throughput: number) {
  return {
    concurrency,
    measuredConcurrency: concurrency,
    throughput,
    throughputMin: throughput,
    throughputMax: throughput,
    latency: null,
    observations: 1,
  }
}

const singleTurn: ThroughputCurve = {
  isl: 1000,
  osl: 1000,
  turns: 1,
  tp: 1,
  sweepId: 'sweep-a',
  backend: 'vllm',
  backendVersions: ['vLLM-0.24.0'],
  configLabel: 'vLLM-0.24.0',
  points: [point(1, 100), point(10, 900), point(40, 1400)],
}

const curves: ThroughputCurve[] = [
  singleTurn,
  {
    ...singleTurn,
    tp: 2,
    points: [point(1, 200), point(10, 1800), point(40, 2800)],
  },
  {
    ...singleTurn,
    isl: 8000,
    turns: 5,
    sweepId: 'sweep-b',
    points: [point(1, 50), point(10, 300)],
  },
]

const dashboard = {
  modelId: pair.modelId,
  systemId: pair.systemId,
  records: 233,
  labelDistribution: { within: 180, outside: 53 },
  throughputCurves: curves,
  kneePoints: [],
  featureImportance: [{ feature: 'concurrency', importance: 0.4 }],
}

/** A prediction for the single-turn workload, at the measured concurrency 10. */
const predictionSearch = [
  'v=1',
  'isl=1000',
  'osl=1000',
  'turns=1',
  'tp=1',
  'predIsl=1000',
  'predOsl=1000',
  'predConcurrency=10',
  'predTp=1',
  'predThroughput=1200',
  'predStatus=within_validated_range',
  'predConfidence=0.9',
  'predBackend=vllm',
  'predBackendVersion=0.24.0',
  'predServingMode=agg',
].join('&')

let root: ReturnType<typeof createRoot>
let host: HTMLDivElement

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  navigation.search = ''
  host = document.createElement('div')
  document.body.append(host)
  root = createRoot(host)
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    status: 200,
    json: async () => dashboard,
  })))
})

afterEach(async () => {
  await act(async () => root.unmount())
  host.remove()
  vi.unstubAllGlobals()
})

async function waitFor(predicate: () => boolean, label = 'condition') {
  const started = Date.now()
  while (!predicate()) {
    if (Date.now() - started > 2000) throw new Error(`timed out waiting for ${label}`)
    await act(async () => new Promise(resolve => setTimeout(resolve, 5)))
  }
}

async function mount(search = '') {
  navigation.search = search
  window.history.replaceState(
    {},
    '',
    `/tested-models/${pair.id}${search ? `?${search}` : ''}`,
  )
  await act(async () => {
    root.render(<TestedModelPair pair={pair} gate={gate} dataset={null} />)
  })
  await waitFor(() => host.querySelectorAll('[data-testid="chart-line"]').length > 0, 'chart lines')
}

function lines(): Array<{ stroke: string | null; points: Array<[number, number]> }> {
  return [...host.querySelectorAll<HTMLElement>('[data-testid="chart-line"]')].map(element => ({
    stroke: element.getAttribute('data-stroke'),
    points: JSON.parse(element.getAttribute('data-points') ?? '[]') as Array<[number, number]>,
  }))
}

function scatter(): Array<[number, number, string]> | null {
  const element = host.querySelector<HTMLElement>('[data-testid="chart-scatter"]')
  return element ? JSON.parse(element.getAttribute('data-points') ?? '[]') as Array<[number, number, string]> : null
}

function lineStyles(): Array<{ opacity: string; strokeWidth: string }> {
  return [...host.querySelectorAll<HTMLElement>('[data-testid="chart-line"]')].map(element => ({
    opacity: element.getAttribute('data-opacity') ?? '',
    strokeWidth: element.getAttribute('data-stroke-width') ?? '',
  }))
}

describe('TestedModelPair throughput chart', () => {
  it('draws one line per tensor-parallel width for the default workload', async () => {
    await mount()

    const drawn = lines()
    expect(drawn).toHaveLength(2)
    expect(drawn[0].points).toEqual([[1, 100], [10, 900], [40, 1400]])
    expect(drawn[1].points).toEqual([[1, 200], [10, 1800], [40, 2800]])
    expect(new Set(drawn.map(line => line.stroke)).size).toBe(2)

    const text = host.textContent ?? ''
    expect(text).toContain('Throughput by concurrency')
    expect(text).toContain('TP 1')
    expect(text).toContain('TP 2')
    expect(host.querySelector('select')?.textContent).toContain('ISL 1,000 · OSL 1,000 · single-turn')
    expect(scatter()).toBeNull()
  })

  it('plots a shared prediction on its own measured line and explains the gap', async () => {
    await mount(predictionSearch)
    await waitFor(() => scatter() !== null, 'prediction overlay')

    expect(scatter()).toEqual([[10, 1200, 'Your prediction']])

    const text = host.textContent ?? ''
    expect(text).toContain('Prediction vs measurement')
    expect(text).toContain('Your prediction')
    expect(text).toContain('Measured level')
    expect(text).toContain('+33.3%')
    expect(text).toContain('simulator predicts more throughput')
    // The verdict comes from the shared prediction, not from this page.
    expect(text).toContain('within validated range')
    expect(text).toContain('does not re-run the simulator')
  })

  it('keeps the overlay off a different workload and switches on demand', async () => {
    // The link selects a multi-turn workload while carrying a single-turn
    // prediction, so the chart cannot draw the point until the reader asks.
    const search = predictionSearch
      .replace('isl=1000', 'isl=8000')
      .replace('turns=1', 'turns=5')
    await mount(search)

    expect(lines()).toHaveLength(1)
    expect(lines()[0].points).toEqual([[1, 50], [10, 300]])
    expect(scatter()).toBeNull()

    const text = host.textContent ?? ''
    expect(text).toContain('The chart is showing ISL 8,000 · OSL 1,000 · 5-turn.')

    const switchWorkload = [...host.querySelectorAll<HTMLButtonElement>('button')]
      .find(button => button.textContent?.startsWith('Show ISL 1,000 · OSL 1,000 · single-turn'))
    expect(switchWorkload).toBeDefined()

    await act(async () => switchWorkload?.click())
    await waitFor(() => scatter() !== null, 'prediction overlay after switching workload')

    expect(scatter()).toEqual([[10, 1200, 'Your prediction']])
    expect(lines()).toHaveLength(2)
    expect(host.textContent).toContain('+33.3%')
  })

  it('reports an unreadable dashboard schema instead of an empty chart', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 409, json: async () => ({}) })))
    navigation.search = ''
    window.history.replaceState({}, '', `/tested-models/${pair.id}`)

    await act(async () => {
      root.render(<TestedModelPair pair={pair} gate={gate} dataset={null} />)
    })
    await waitFor(
      () => (host.textContent ?? '').includes('schema this build does not support'),
      'unsupported schema notice',
    )

    expect(host.querySelector('[data-testid="chart-line"]')).toBeNull()
    expect(host.textContent).toContain('Rebuild the tested-model bundle')
  })

  it('does not compare a multi-replica cluster total with one deployment', async () => {
    const search = `${predictionSearch}&predReplicas=2`
    await mount(search)

    expect(scatter()).toBeNull()
    const text = host.textContent ?? ''
    expect(text).toContain('Replica basis')
    expect(text).toContain('cluster total across 2 replicas')
    // The difference is not reported across incomparable bases.
    expect(text).toContain('Prediction versus measured')
    const difference = [...host.querySelectorAll<HTMLElement>('strong')]
      .find(element => element.textContent === '—')
    expect(difference).toBeDefined()
    expect(text).not.toContain('Prediction is')
  })

  it('ignores a focused TP that the displayed workload does not measure', async () => {
    await mount('v=1&isl=8000&osl=1000&turns=5&tp=3')

    // The multi-turn workload measured only TP 1, so no line is dimmed.
    expect(lineStyles()).toEqual([{ opacity: '1', strokeWidth: '2.5' }])
  })

  it('offers the prediction workload even when its TP was not measured', async () => {
    const search = predictionSearch
      .replace('predTp=1', 'predTp=3')
      .replace('isl=1000', 'isl=8000')
      .replace('turns=1', 'turns=5')
    await mount(search)

    const text = host.textContent ?? ''
    expect(text).toContain('The chart is showing ISL 8,000 · OSL 1,000 · 5-turn.')
    expect(text).toContain('No comparable measured line')
    expect(
      [...host.querySelectorAll<HTMLButtonElement>('button')]
        .some(button => button.textContent?.startsWith('Show ISL 1,000 · OSL 1,000 · single-turn')),
    ).toBe(true)
  })
})

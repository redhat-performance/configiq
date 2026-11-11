// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { BenchmarkPoint } from '@/lib/validation/benchmark-types'
import ComparisonPreview, { type PredictionSeries } from './ComparisonPreview'

vi.mock('@patternfly/react-charts/victory', () => ({
  Chart: ({ children }: { children?: React.ReactNode }) => <div data-testid="chart">{children}</div>,
  ChartAxis: () => null,
  ChartGroup: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  ChartLine: ({ name }: { name?: string }) => <span data-line={name} />,
  ChartScatter: ({ name }: { name?: string }) => <span data-points={name} />,
  ChartVoronoiContainer: () => null,
}))

const benchmark: BenchmarkPoint[] = [
  {
    concurrency: 8,
    ttftSeconds: 0.1,
    tpotSeconds: 0.01,
    outputTokensPerSecond: 1000,
    outputTokensPerSecondPerUser: 125,
    requestLatencySeconds: 1.2,
    sampleCount: 2,
  },
]

const series: PredictionSeries[] = [{
  simulator: 'blis',
  label: 'BLIS',
  version: 'v0.24.0',
  comparable: true,
  overridden: false,
  points: [{
    concurrency: 8,
    ttftSeconds: 0.2,
    tpotSeconds: 0.02,
    outputTokensPerSecond: 800,
    outputTokensPerSecondPerUser: 100,
    requestLatencySeconds: 1.4,
  }],
}]

let host: HTMLDivElement
let root: Root

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  host = document.createElement('div')
  document.body.append(host)
  root = createRoot(host)
})

afterEach(async () => {
  await act(async () => root.unmount())
  host.remove()
  vi.restoreAllMocks()
})

async function render() {
  await act(async () => {
    root.render(<ComparisonPreview benchmark={benchmark} inferencex={[]} series={series} />)
  })
}

async function click(element: Element) {
  await act(async () => {
    element.dispatchEvent(new MouseEvent('click', { bubbles: true }))
  })
}

describe('ComparisonPreview', () => {
  it('renders one flip tile per metric with benchmark and simulator series', async () => {
    await render()

    const cards = host.querySelectorAll('[data-flip-card]')
    expect(cards).toHaveLength(5)
    expect(host.querySelectorAll('[data-line="red-hat-line"]')).toHaveLength(5)
    expect(host.querySelectorAll('[data-line="blis-line"]')).toHaveLength(5)

    const table = host.querySelector('table')
    expect(table?.textContent).toContain('Red Hat')
    expect(table?.textContent).toContain('BLIS')
    expect(table?.textContent).toContain('+100.0%')
  })

  it('flips a tile on click and through its table button', async () => {
    await render()

    const [first, second] = [...host.querySelectorAll('[data-flip-card]')]
    const firstButton = () => first.querySelector('button[aria-expanded]')
    expect(firstButton()?.getAttribute('aria-expanded')).toBe('false')

    await click(first)
    expect(firstButton()?.getAttribute('aria-expanded')).toBe('true')
    expect(first.querySelector('[class*="flipBack"] button[aria-expanded]')?.textContent).toContain('chart')

    const secondButton = second.querySelector('button[aria-expanded]')
    expect(secondButton).not.toBeNull()
    await click(secondButton as Element)
    expect(second.querySelector('button[aria-expanded]')?.getAttribute('aria-expanded')).toBe('true')
  })

  it('keeps the hidden face out of the accessibility tree', async () => {
    await render()

    const [first] = [...host.querySelectorAll('[data-flip-card]')]
    const faces = first.querySelectorAll('[class*="flipFace"]')
    expect(faces[0].getAttribute('aria-hidden')).toBe('false')
    expect(faces[1].getAttribute('aria-hidden')).toBe('true')

    await click(first)
    expect(faces[0].getAttribute('aria-hidden')).toBe('true')
    expect(faces[1].getAttribute('aria-hidden')).toBe('false')
  })

  it('hides the difference when a prediction is not comparable', async () => {
    await act(async () => {
      root.render(
        <ComparisonPreview
          benchmark={benchmark}
          inferencex={[]}
          series={[{ ...series[0], comparable: false }]}
        />,
      )
    })

    const table = host.querySelector('table')
    expect(table?.textContent).toContain('not comparable')
    expect(table?.textContent).not.toContain('+100.0%')
  })
})

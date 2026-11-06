// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { expect, it, vi } from 'vitest'
import CostComparisonChart from './CostComparisonChart'

vi.mock('@patternfly/react-charts/victory', () => ({
  Chart: ({ children, domain, containerComponent }: { children: React.ReactNode; domain: unknown; containerComponent?: React.ReactNode }) => <div data-domain={JSON.stringify(domain)}>{containerComponent}{children}</div>,
  ChartGroup: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  ChartLine: ({ name, data }: { name?: string; data: unknown }) => <div data-series={name ?? 'cost'}>{JSON.stringify(data)}</div>,
  ChartAxis: ({ label, tickValues }: { label: string; tickValues?: number[] }) => <div data-axis={label} data-ticks={JSON.stringify(tickValues)} />,
  ChartScatter: ({ data }: { data: unknown }) => <div data-marker>{JSON.stringify(data)}</div>,
  ChartTooltip: () => null,
  ChartVoronoiContainer: ({ labels }: { labels: (props: { datum: { x: number; y: number; name: string } }) => string }) => (
    <div data-tooltip>{labels({ datum: { x: 2.5e9, y: 0.05, name: 'hosted' } })}</div>
  ),
}))

it('shows the combined monthly workload, its input/output split and matching tooltip', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  try {
    await act(async () => root.render(<CostComparisonChart
      points={[{ tokens: 0, hosted: 0, rented: 10, owned: 20 }, { tokens: 2.5e9, hosted: 25, rented: 50, owned: 60 }, { tokens: 10e9, hosted: 100, rented: 50, owned: 60 }]}
      currentTokens={2.5e9} outputShare={0.2} transitionsVerified
      hostedModelLabel="openai/gpt-5" infrastructureModelLabel="Qwen/Qwen3-8B"
      rentedBreakEvenTokens={null} ownedBreakEvenTokens={null} rentedLowestCostTokens={null} ownedLowestCostTokens={null}
    />))
    expect(host.textContent).not.toContain('Dashed line:')
    expect(host.textContent).not.toContain('Scale-to-zero rented cost')
    expect(host.textContent).toContain('Your total · 2.5B')
    const summary = host.querySelector('[aria-label="Current monthly workload in tokens"]')!
    expect([...summary.querySelectorAll('dt')].map(term => term.textContent)).toEqual(['Monthly total', 'Input', 'Output'])
    expect([...summary.querySelectorAll('dd')].map(value => value.textContent)).toEqual(['2.5B', '2B', '500M'])
    expect(host.textContent).toContain('Input/output mix stays fixed')
    expect(host.textContent).not.toContain('Full TCO')
    expect(host.textContent).not.toContain('The horizontal axis shows output only')
    expect(host.textContent).toContain('Hosted API: openai/gpt-5 · Rented / purchased: Qwen/Qwen3-8B')
    expect(host.textContent).toContain('model quality and capability may differ')
    expect(host.querySelectorAll('[data-series="cost"]')).toHaveLength(3)
    const currentLine = JSON.parse(host.querySelector('[data-series="current-workload"]')!.textContent!)
    expect(currentLine.map((point: { x: number }) => point.x)).toEqual([2.5e9, 2.5e9])
    expect(host.querySelector('[data-tooltip]')!.textContent).toBe('Hosted API\n2.5B total tokens/month\n2B input + 500M output\n$0.05 / 1M output')
    const plotHeader = host.querySelector('[class*="chartPlotHeader"]')!
    expect([...plotHeader.querySelectorAll('li')].map(item => item.textContent)).toEqual(['Hosted API', 'Rented', 'Purchased'])
    expect(plotHeader.nextElementSibling!.querySelector('[data-domain]')).not.toBeNull()
    await act(async () => host.querySelector<HTMLButtonElement>('button[aria-label="Help for Cost per 1M output tokens"]')!.click())
    expect(document.body.textContent).toContain('including input processing')
    expect(document.body.textContent).toContain('not the provider\'s output-token tariff')
  } finally {
    await act(async () => root.unmount())
    host.remove()
  }
})

it('keeps the range, ticks, current line and crossover markers in total-token coordinates', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  try {
    for (const maximum of [60e9, 12e9, 2e12]) {
      const rentedCrossing = maximum / 2
      const ownedCrossing = maximum * 5 / 6
      await act(async () => root.render(<CostComparisonChart
        points={[
          { tokens: 0, hosted: 0, rented: 50, owned: 100 },
          { tokens: 625e6, hosted: 20, rented: 50, owned: 100 },
          { tokens: 2.5e9, hosted: 25, rented: 50, owned: 100 },
          { tokens: rentedCrossing, hosted: 50, rented: 50, owned: 100 },
          { tokens: ownedCrossing, hosted: 100, rented: 80, owned: 100 },
          { tokens: maximum, hosted: 120, rented: 90, owned: 100 },
        ]}
        currentTokens={2.5e9} outputShare={0.2} transitionsVerified
        rentedBreakEvenTokens={rentedCrossing} ownedBreakEvenTokens={ownedCrossing}
        rentedLowestCostTokens={rentedCrossing} ownedLowestCostTokens={null}
      />))
      expect(JSON.parse(host.querySelector('[data-domain]')!.getAttribute('data-domain')!).x).toEqual([625e6, maximum])
      const ticks = JSON.parse(host.querySelector('[data-axis="Total tokens per month (input + output)"]')!.getAttribute('data-ticks')!)
      expect(ticks[0]).toBe(625e6)
      expect(ticks.at(-1)).toBeCloseTo(maximum, 3)
      expect(host.querySelectorAll('[data-series="cost"]')).toHaveLength(3)
      expect(host.textContent).toContain('Your total · 2.5B')
      expect(host.querySelector('[data-series="current-workload"]')!.textContent).toContain('2500000000')
      const markers = [...host.querySelectorAll('[data-marker]')].map(marker => JSON.parse(marker.textContent!))
      expect(markers.some(marker => marker[0]?.x === rentedCrossing)).toBe(true)
      expect(markers.some(marker => marker[0]?.x === ownedCrossing)).toBe(true)
    }
  } finally {
    await act(async () => root.unmount())
    host.remove()
  }
})

it('uses total demand on x and output-normalized cost on y without a false zero-output point', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  try {
    await act(async () => root.render(<CostComparisonChart
      points={[
        { tokens: 0, hosted: 0, rented: 100, owned: 200 },
        { tokens: 1e9, hosted: 10, rented: 100, owned: 200 },
        { tokens: 4e9, hosted: 40, rented: 100, owned: 200 },
        { tokens: 10e9, hosted: 100, rented: 50, owned: 60 },
      ]}
      currentTokens={4e9} outputShare={0.2} transitionsVerified
      rentedBreakEvenTokens={10e9} ownedBreakEvenTokens={null}
      rentedLowestCostTokens={null} ownedLowestCostTokens={null}
    />))
    expect(host.querySelector('[data-axis="Total tokens per month (input + output)"]')).not.toBeNull()
    expect(host.textContent).toContain('Cost per 1M output tokens (USD)')
    expect(host.textContent).toContain('Rented ≈ 10B total tokens/month')
    expect(host.textContent).toContain('Your total · 4B')
    const domain = JSON.parse(host.querySelector('[data-domain]')!.getAttribute('data-domain')!)
    expect(domain.x).toEqual([1e9, 10e9])
    expect(domain.y[0]).toBe(0)
    // All displayed values fit, including purchased $200/200M = $1/M.
    expect(domain.y[1]).toBeCloseTo(1 * 1.08, 9)
    const lines = [...host.querySelectorAll('[data-series="cost"]')]
    const hosted = JSON.parse(lines[0].textContent!)
    // The zero-output sample is skipped, not drawn as a zero unit cost.
    expect(hosted).toHaveLength(3)
    expect(hosted[0].x).toBe(1e9)
    expect(hosted[0].y).toBeCloseTo(0.05, 9)
    const rented = JSON.parse(lines[1].textContent!)
    expect(rented.find((point: { x: number }) => point.x === 1e9).y).toBeCloseTo(0.5, 9)
    expect(host.textContent).not.toContain('outside the displayed range')
    const markers = [...host.querySelectorAll('[data-marker]')].map(marker => JSON.parse(marker.textContent!))
    expect(markers.some(marker => marker[0]?.x === 10e9)).toBe(true)
    // The rented break-even marker is converted to the unit metric: $50/2B output.
    expect(markers.some(marker => marker[0]?.y === 0.025)).toBe(true)
  } finally {
    await act(async () => root.unmount())
    host.remove()
  }
})

it('keeps a small workload total unchanged when the input/output mix changes', async () => {
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  try {
    for (const [outputShare, input, output, rentedUnitCost] of [
      [0.2, '2M', '500K', 100],
      [1, '0', '2.5M', 20],
    ] as const) {
      await act(async () => root.render(<CostComparisonChart
        points={[
          { tokens: 0, hosted: 0, rented: 50, owned: 60 },
          { tokens: 2.5e6, hosted: 25, rented: 50, owned: 60 },
          { tokens: 10e6, hosted: 100, rented: 50, owned: 60 },
        ]}
        currentTokens={2.5e6} outputShare={outputShare} transitionsVerified
        rentedBreakEvenTokens={null} ownedBreakEvenTokens={null}
        rentedLowestCostTokens={null} ownedLowestCostTokens={null}
      />))
      expect(host.textContent).toContain('Your total · 2.5M')
      const summary = host.querySelector('[aria-label="Current monthly workload in tokens"]')!
      expect([...summary.querySelectorAll('dd')].map(value => value.textContent)).toEqual(['2.5M', input, output])
      const rented = JSON.parse(host.querySelectorAll('[data-series="cost"]')[1].textContent!)
      expect(rented.map((point: { x: number }) => point.x)).toEqual([2.5e6, 10e6])
      expect(rented[0].y).toBeCloseTo(rentedUnitCost, 9)
    }
  } finally {
    await act(async () => root.unmount())
    host.remove()
  }
})

it('does not produce a graph when output has no defined unit cost', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  try {
    await act(async () => root.render(<CostComparisonChart
      points={[
        { tokens: 0, hosted: 0, rented: 100, owned: 200 },
        { tokens: 1e9, hosted: 10, rented: 100, owned: 200 },
        { tokens: 10e9, hosted: 100, rented: 50, owned: 60 },
      ]}
      currentTokens={5e9} outputShare={0} transitionsVerified
      rentedBreakEvenTokens={null} ownedBreakEvenTokens={null}
      rentedLowestCostTokens={null} ownedLowestCostTokens={null}
    />))
    expect(host.querySelector('[data-domain]')).toBeNull()
  } finally {
    await act(async () => root.unmount())
    host.remove()
  }
})

it('fits the full curves in the displayed range and clearly separates output from combined demand', async () => {
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  try {
    await act(async () => root.render(<CostComparisonChart
      points={[
        { tokens: 0, hosted: 0, rented: 5000, owned: 8000 },
        { tokens: 1, hosted: 1000, rented: 5000, owned: 8000 },
        { tokens: 625e6, hosted: 1350, rented: 8000, owned: 8500 },
        { tokens: 2.5e9, hosted: 1467, rented: 12680, owned: 10430 },
        { tokens: 7.5e9, hosted: 1800, rented: 21500, owned: 15000 },
      ]}
      currentTokens={2.5e9} outputShare={0.2} transitionsVerified
      rentedBreakEvenTokens={null} ownedBreakEvenTokens={null}
      rentedLowestCostTokens={null} ownedLowestCostTokens={null}
    />))
    const domain = JSON.parse(host.querySelector('[data-domain]')!.getAttribute('data-domain')!)
    expect(domain.x).toEqual([625e6, 7.5e9])
    expect(domain.y).toEqual([0, 68 * 1.08])
    const allSamples = [...host.querySelectorAll('[data-series="cost"]')].flatMap(line => JSON.parse(line.textContent!))
    expect(allSamples.every((point: { x: number; y: number }) => point.x >= domain.x[0] && point.y <= domain.y[1])).toBe(true)
    const currentMarkers = [...host.querySelectorAll('[data-marker]')]
      .flatMap(marker => JSON.parse(marker.textContent!))
      .filter((point: { x: number }) => point.x === 2.5e9)
    expect(currentMarkers).toHaveLength(3)
    expect(currentMarkers.every((point: { y: number }) => point.y <= domain.y[1])).toBe(true)
    expect(host.querySelector('button[aria-label="Help for Cost per 1M output tokens"]')).not.toBeNull()
    expect(host.textContent).toContain('added hardware')
    const summary = host.querySelector('[aria-label="Current monthly workload in tokens"]')!
    expect([...summary.querySelectorAll('dd')].map(value => value.textContent)).toEqual(['2.5B', '2B', '500M'])
    expect(host.textContent).toContain('Hosted API stays cheaper up to 1T total tokens/month under these assumptions.')
    expect(host.querySelector('[aria-label="Modeled cost milestones"]')).toBeNull()
  } finally {
    await act(async () => root.unmount())
    host.remove()
  }
})

it('distinguishes unavailable prices, unverified searches and crossovers beyond the search horizon', async () => {
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  try {
    for (const verified of [true, false]) {
      await act(async () => root.render(<CostComparisonChart
        points={[
          { tokens: 0, hosted: 0, rented: null, owned: 100 },
          { tokens: 1e9, hosted: 50, rented: null, owned: 200 },
        ]}
        currentTokens={1e9} outputShare={0.2}
        transitionsVerified={verified}
        rentedBreakEvenTokens={null} ownedBreakEvenTokens={null}
        rentedLowestCostTokens={null} ownedLowestCostTokens={null}
      />))
      expect(host.textContent).toContain('Rented Unavailable')
      expect(host.textContent).toContain(verified
        ? 'Purchased Not within 1T total tokens/month'
        : 'Purchased Not verified')
    }
  } finally {
    await act(async () => root.unmount())
    host.remove()
  }
})

// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { expect, it, vi } from 'vitest'
import CostComparisonChart from './CostComparisonChart'

vi.mock('@patternfly/react-charts/victory', () => ({
  Chart: ({ children, domain }: { children: React.ReactNode; domain: unknown }) => <div data-domain={JSON.stringify(domain)}>{children}</div>,
  ChartGroup: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  ChartLine: ({ name, data }: { name?: string; data: unknown }) => <div data-series={name ?? 'cost'}>{JSON.stringify(data)}</div>,
  ChartAxis: ({ label, tickValues }: { label: string; tickValues?: number[] }) => <div data-axis={label} data-ticks={JSON.stringify(tickValues)} />,
  ChartScatter: ({ data }: { data: unknown }) => <div data-marker>{JSON.stringify(data)}</div>,
  ChartTooltip: () => null,
  ChartVoronoiContainer: () => null,
}))

it('removes the chart notes while retaining all cost series and the labelled current-workload line', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  try {
    await act(async () => root.render(<CostComparisonChart
      points={[{ tokens: 0, hosted: 0, rented: 10, owned: 20 }, { tokens: 10e9, hosted: 100, rented: 50, owned: 60 }]}
      currentTokens={2.5e9} costLens="fully-loaded" transitionsVerified
      hostedModelLabel="openai/gpt-5" infrastructureModelLabel="Qwen/Qwen3-8B"
      rentedBreakEvenTokens={null} ownedBreakEvenTokens={null} rentedLowestCostTokens={null} ownedLowestCostTokens={null}
    />))
    expect(host.textContent).not.toContain('Dashed line:')
    expect(host.textContent).not.toContain('Scale-to-zero rented cost')
    expect(host.textContent).toContain('Current · 2.5B')
    expect(host.textContent).toContain('Hosted API: openai/gpt-5 · Rented / purchased: Qwen/Qwen3-8B')
    expect(host.querySelectorAll('[data-series="cost"]')).toHaveLength(3)
    expect(host.querySelector('[data-series="current-workload"]')!.textContent).toContain('2500000000')
  } finally {
    await act(async () => root.unmount())
    host.remove()
  }
})

it('updates the zero-based axis, continuation and crossover markers when the range changes', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  try {
    for (const [maximum, label] of [[60e9, '60B'], [12e9, '12B'], [2e12, '2T']] as const) {
      const rentedCrossing = maximum / 2
      const ownedCrossing = maximum * 5 / 6
      await act(async () => root.render(<CostComparisonChart
        points={[
          { tokens: 0, hosted: 0, rented: 50, owned: 100 },
          { tokens: rentedCrossing, hosted: 50, rented: 50, owned: 100 },
          { tokens: ownedCrossing, hosted: 100, rented: 80, owned: 100 },
          { tokens: maximum, hosted: 120, rented: 90, owned: 100 },
        ]}
        currentTokens={2.5e9} costLens="fully-loaded" transitionsVerified
        rentedBreakEvenTokens={rentedCrossing} ownedBreakEvenTokens={ownedCrossing}
        rentedLowestCostTokens={rentedCrossing} ownedLowestCostTokens={null}
      />))
      expect(host.textContent).toContain(`Cost from 0 to ${label} billed tokens/month`)
      if (maximum > 1e12) expect(host.textContent).toContain('Purchased Not reached by 1T')
      expect(JSON.parse(host.querySelector('[data-domain]')!.getAttribute('data-domain')!).x).toEqual([0, maximum])
      const ticks = JSON.parse(host.querySelector('[data-axis="Billed tokens per month"]')!.getAttribute('data-ticks')!)
      expect(ticks[0]).toBe(0)
      expect(ticks.at(-1)).toBe(maximum)
      expect(host.querySelectorAll('[data-series="cost"]')).toHaveLength(3)
      expect(host.textContent).toContain('Current · 2.5B')
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

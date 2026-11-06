// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { expect, it, vi } from 'vitest'
import FixedDeploymentChart from './FixedDeploymentChart'
import { fixedDeploymentSamples, type FixedDeploymentCapture } from '@/lib/hybrid-savings/fixed-deployment'
import { DEFAULT_COST_ASSUMPTIONS } from '@/lib/costing-assumptions'
import type { CostAssumptions, HostedPrice, HybridWorkload, InfrastructureCandidate } from '@/lib/hybrid-savings/calc'

vi.mock('@patternfly/react-charts/victory', () => ({
  Chart: ({ children, domain, containerComponent }: {
    children: React.ReactNode
    domain: unknown
    containerComponent?: React.ReactNode
  }) => <div data-domain={JSON.stringify(domain)}>{containerComponent}{children}</div>,
  ChartGroup: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  ChartLine: ({ name, data }: { name?: string; data: unknown }) => <div data-series={name ?? 'cost'}>{JSON.stringify(data)}</div>,
  ChartAxis: ({ label, tickValues }: { label: string; tickValues?: number[] }) => <div data-axis={label} data-ticks={JSON.stringify(tickValues)} />,
  ChartScatter: ({ data }: { data: unknown }) => <div data-marker>{JSON.stringify(data)}</div>,
  ChartTooltip: () => null,
  ChartVoronoiContainer: ({ labels }: { labels?: (props: { datum: { x?: number; y?: number; name?: string } }) => string }) => (
    <div data-voronoi>
      {labels ? labels({ datum: { x: 50, y: 200, name: 'hardware' } }) : ''}
    </div>
  ),
}))

// Worked example: one always-on $1/hour instance across a 1,000-hour month is
// a fixed $1,000 over a 10M combined-token capacity, all of it output tokens.
const workload: HybridWorkload = {
  monthlyInputTokens: 0,
  monthlyOutputTokens: 10_000_000,
  averageInputTokens: 1,
  averageOutputTokens: 1_000,
  activeHoursPerMonth: 730,
  peakToAverage: 1,
}
const assumptions: CostAssumptions = {
  ...DEFAULT_COST_ASSUMPTIONS,
  costLens: 'fully-loaded',
  cloudBillingMode: 'always-on',
  hoursPerMonth: 1_000,
  loadedMonthlyCostPerFte: 0,
  rentedDirectInfrastructureMonthly: 0,
  rentedFixedMonthly: 0,
  rentedOperationsFte: 0,
  rentedImplementation: 0,
}
const candidate: InfrastructureCandidate = {
  systemId: 'fixed_gpu', label: 'Fixed GPU', gpusPerReplica: 1, replicasNeeded: 1,
  clusterOutputTokensPerSecond: 1000, cloudRatePerGpuHour: 1, cloudHourlyCostPerInstance: 1,
  cloudGpusPerInstance: 1, cloudProvider: 'test', cloudRateKind: 'on_demand',
  tdpWattsPerGpu: 0,
  purchaseGpusPerServer: 1, purchasePricePerReplica: 100, purchaseInstallationPerReplica: 0,
  purchasePriceIndicative: false, purchasePriceSource: 'Test catalogue',
  purchasePriceSourceUrl: null, purchasePriceSourceDate: null,
}
const capture: FixedDeploymentCapture = {
  path: 'rented', candidateKey: 'example', systemId: 'fixed_gpu', gpusPerReplica: 1,
  wholeUnits: 1, billedGpus: 1, availableReplicaSlots: 1, planningTokenCapacity: 10_000_000,
  basisSignature: 'example',
}
const hostedPrice: HostedPrice = { modelId: 'test/model', label: 'Hosted API', inputPerMillion: 1, outputPerMillion: 2 }

async function renderChart(
  props: { path?: 'rented' | 'owned'; currentCapacityUsedPct?: number },
  assert: (host: HTMLElement) => void | Promise<void>,
) {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  const samples = fixedDeploymentSamples(workload, hostedPrice, assumptions, candidate, capture)
  try {
    await act(async () => root.render(<FixedDeploymentChart
      samples={samples}
      currentCapacityUsedPct={props.currentCapacityUsedPct ?? 50}
      path={props.path ?? 'rented'}
      hardwareLabel="Fixed GPU"
    />))
    await assert(host)
  } finally {
    await act(async () => root.unmount())
    host.remove()
  }
}

it('shows the frozen deployment without automatic-forecast milestones', async () => {
  await renderChart({}, async host => {
    expect(host.textContent).toContain('Cost as hardware usage increases')
    expect(host.querySelector('[aria-label="Fixed hardware"] dd')!.textContent).toBe('Fixed GPU')
    expect(host.textContent).toContain('Higher usage spreads fixed costs')
    expect(host.textContent).not.toContain('Full TCO')
    expect(host.textContent).not.toContain('Cheaper than hosted API')
    expect(host.textContent).not.toContain('Lowest-cost eligible option')
    // The tooltip labels the series after the path, matching the legend.
    expect(host.querySelector('[data-voronoi]')!.textContent).toContain('Rented (fixed)')
    expect(host.querySelector('[data-voronoi]')!.textContent).not.toContain('Purchased hardware')
    expect(host.textContent).toContain('Your usage · 50.0%')
    expect(host.querySelector('[data-axis="Planning capacity used (%)"]')).not.toBeNull()
    expect(host.textContent).toContain('Cost per 1M output tokens (USD)')
    const domain = JSON.parse(host.querySelector('[data-domain]')!.getAttribute('data-domain')!)
    expect(domain.x).toEqual([10, 100])
    // Fit the entire explicitly displayed range, not only selected references.
    expect(domain.y[0]).toBe(0)
    expect(domain.y[1]).toBeCloseTo(1_000 * 1.08, 6)
    const lines = [...host.querySelectorAll('[data-series="cost"]')]
    expect(lines).toHaveLength(2)
    const hardware = JSON.parse(lines[1].textContent!)
    expect(hardware.at(-1).x).toBe(100)
    expect(hardware.at(-1).y).toBeCloseTo(100, 6)
    expect(hardware.find((point: { x: number }) => point.x === 25).y).toBeCloseTo(400, 6)
    expect(hardware.find((point: { x: number }) => point.x === 50).y).toBeCloseTo(200, 6)
    expect(hardware[0].x).toBe(10)
    expect(hardware.every((point: { y: number }) => point.y <= domain.y[1])).toBe(true)
    expect(host.textContent).not.toContain('outside the displayed range')
    const plotHeader = host.querySelector('[class*="chartPlotHeader"]')!
    expect([...plotHeader.querySelectorAll('li')].map(item => item.textContent)).toEqual(['Hosted API', 'Rented (fixed)'])
    expect(plotHeader.nextElementSibling!.querySelector('[data-domain]')).not.toBeNull()
    expect(host.querySelector('button[aria-label="Help for Hardware usage"]')!.closest('p')).toBeNull()
    const markers = [...host.querySelectorAll('[data-marker]')].map(marker => JSON.parse(marker.textContent!))
    expect(markers.some(marker => marker[0]?.x === 50 && Math.abs(marker[0].y - 200) < 1e-6)).toBe(true)
    await act(async () => host.querySelector<HTMLButtonElement>('button[aria-label="Help for Hardware usage"]')!.click())
    expect(document.body.textContent).toContain('not measured GPU utilization or a hardware maximum')
    expect(document.body.textContent).toContain('Savings still depend on prices and rental billing')
  })
})

it('labels the purchased path and includes a current usage below the usual starting point', async () => {
  await renderChart({ path: 'owned', currentCapacityUsedPct: 5 }, host => {
    expect(host.querySelector('[aria-label="Fixed hardware"] dd')!.textContent).toBe('Fixed GPU')
    expect([...host.querySelectorAll('[aria-label="Chart legend"] li')].map(item => item.textContent)).toEqual(['Hosted API', 'Purchased (fixed)'])
    expect(host.querySelector('[data-voronoi]')!.textContent).toContain('Purchased (fixed)')
    expect(host.querySelector('[data-axis="Monthly cost (USD)"]')).toBeNull()
    expect(host.textContent).not.toContain('Higher unit costs near zero output')
    const domain = JSON.parse(host.querySelector('[data-domain]')!.getAttribute('data-domain')!)
    expect(domain.x).toEqual([5, 100])
    expect(host.textContent).toContain('Your usage · 5.0%')
    const samples = [...host.querySelectorAll('[data-series="cost"]')].flatMap(line => JSON.parse(line.textContent!))
    expect(samples.every((point: { x: number; y: number }) => point.x >= domain.x[0] && point.y <= domain.y[1])).toBe(true)
  })
})

import { writeFile } from 'node:fs/promises'
import { expect, test, type Page, type TestInfo } from '@playwright/test'
import appConfig from '../public/config.json'
import type { WorkloadPreset } from '../lib/workload-presets'
import type { ConcurrencyPreset } from '../lib/concurrency-presets'
import { formatCandidateId } from '../lib/pareto/format'
import { formatApiError, parseFrontierBody } from '../lib/pareto/parse'
import { readAisimulateCatalogDetails } from './live-catalog'
import { baselineConcurrencyPreset, matrixConcurrencyPresets, matrixDemandRate, scheduledCoordinates, scheduledMatrix, systemsForModel, workloads } from './test-matrix'

test.describe.configure({ mode: 'parallel' })

interface CompletedRecommendation {
  status: 'completed'
  error?: never
  recommendation: { gpusNeeded: number; gpusPerReplica: number }
  memoryAvailable?: boolean
  memory: { value: number }
  performance: { ttftLatencyMs: number }
  throughput: { tokensPerSecond: number }
  demand?: { achievedRps?: number | null }
  capacity?: { capacitySequences?: number | null; performanceAvailable?: boolean }
}

interface FailedRecommendation {
  status?: 'failed'
  error?: { code?: string; message?: string }
}

type Recommendation = CompletedRecommendation | FailedRecommendation

type RecommendScenario =
  | { action: 'sizing'; goal: 'latency' | 'capacity'; concurrency: ConcurrencyPreset }
  | { action: 'sizing'; goal: 'demand'; concurrency?: never }
  | { action: 'pareto'; goal: 'latency' | 'demand'; concurrency?: never }

const goalLabels = { latency: 'Latency SLO', demand: 'Throughput demand', capacity: 'Concurrency capacity' } as const

const recommendScenarios: RecommendScenario[] = [
  ...matrixConcurrencyPresets.map(concurrency => ({ action: 'sizing', goal: 'latency', concurrency } as const)),
  ...matrixConcurrencyPresets.map(concurrency => ({ action: 'sizing', goal: 'capacity', concurrency } as const)),
  { action: 'sizing', goal: 'demand' },
  { action: 'pareto', goal: 'latency' },
  { action: 'pareto', goal: 'demand' },
]

function recommendCasesForModel(model: string): Array<{ workload: WorkloadPreset; systemId: string; scenario: RecommendScenario }> {
  const systems = systemsForModel(model)
  if (!scheduledMatrix) {
    return workloads.flatMap(workload => systems.flatMap(systemId => recommendScenarios.map(scenario => ({ workload, systemId, scenario }))))
  }
  const baselineScenario = recommendScenarios.find(scenario =>
    scenario.action === 'sizing' && scenario.goal === 'latency' && scenario.concurrency.key === baselineConcurrencyPreset.key,
  )
  if (!baselineScenario) throw new Error('The baseline Recommend sizing scenario is missing')
  const baseline = workloads.flatMap(workload => systems.map(systemId => ({ workload, systemId, scenario: baselineScenario })))
  const extras = recommendScenarios
    .filter(scenario => scenario !== baselineScenario)
    .map((scenario, index) => ({ ...scheduledCoordinates(model, 8 + index), scenario }))
  return [...baseline, ...extras]
}

function recommendErrorTitle(code?: string): string {
  switch (code) {
    case 'AISIM_TIMEOUT': return 'Request timed out'
    case 'AISIM_NO_CONFIGURATION': return 'No valid configuration found'
    case 'AISIM_UNAVAILABLE': return 'Sizing service unavailable'
    case 'AISIM_NOT_CONFIGURED': return 'Service not configured'
    case 'AISIM_INVALID_RESPONSE': return 'Unexpected response'
    case 'SIMULATOR_MISMATCH': return 'Simulator mismatch'
    case 'INVALID_REQUEST': return 'Invalid input'
    case 'NETWORK_ERROR': return 'Connection error'
    default: return 'Something went wrong'
  }
}

async function saveJson(testInfo: TestInfo, name: string, value: unknown) {
  const path = testInfo.outputPath(name)
  await writeFile(path, JSON.stringify(value, null, 2))
  await testInfo.attach(name, { path, contentType: 'application/json' })
}

interface CapturedResponse {
  contentType: string
  body: string
}

async function captureBrowserRecommendation(page: Page) {
  await page.addInitScript(() => {
    const browserWindow = window as typeof window & { __recommendBody?: { contentType: string; body: string } }
    const originalFetch = window.fetch.bind(window)
    window.fetch = async (...args) => {
      const response = await originalFetch(...args)
      const requestUrl = typeof args[0] === 'string' ? args[0] : args[0] instanceof URL ? args[0].href : args[0].url
      if (new URL(requestUrl, location.href).pathname === '/api/recommend') {
        void response.clone().text().then(body => {
          browserWindow.__recommendBody = { contentType: response.headers.get('content-type') ?? '', body }
        })
      }
      return response
    }
  })
}

async function capturedRecommendation(page: Page): Promise<Recommendation> {
  await page.waitForFunction(
    () => Boolean((window as typeof window & { __recommendBody?: CapturedResponse }).__recommendBody),
    undefined,
    { timeout: 240_000 },
  )
  const captured = await page.evaluate(() => (window as typeof window & { __recommendBody: CapturedResponse }).__recommendBody)
  const body = captured.body
  if (captured.contentType.includes('application/json')) return JSON.parse(body) as Recommendation
  const blocks = body.replaceAll('\r\n', '\n').split('\n\n')
  for (const block of blocks.reverse()) {
    const data = block.split('\n').find(line => line.startsWith('data:'))
    if (!data) continue
    const event = JSON.parse(data.slice(5).trim()) as { type: string; response?: Recommendation }
    if (event.type === 'completed' && event.response) return event.response
  }
  throw new Error('Recommendation stream ended without a completed event')
}

async function selectModel(page: Page, inputId: string, model: string) {
  const input = page.locator(`#${inputId}`)
  await expect(input).toBeVisible()
  await input.fill(model)
  await input.press('Enter')
  await expect(input).toHaveValue(model)
}

async function selectSystem(page: Page, inputId: string, name: string) {
  const field = page.locator(`label[for="${inputId}"]`).locator('..')
  await expect(field.getByRole('button')).not.toHaveText('Select GPU system…')
  await field.getByRole('button').click()
  const option = page.getByRole('option').filter({ hasText: name })
  await expect(option).toBeVisible()
  await option.click()
  await expect(field.getByRole('button')).toContainText(name)
}

function displayedTileValue(page: Page, name: RegExp) {
  return page.getByRole('button', { name }).locator('[class*="tileValue"]')
}

async function verifyDeployedPreset(page: Page, model: string, workload: WorkloadPreset, concurrency?: ConcurrencyPreset) {
  const response = await page.request.get('/config.json')
  expect(response.ok(), 'The deployed UI configuration must be available').toBeTruthy()
  const config = await response.json() as {
    approvedModels?: string[]
    workloadPresets?: Array<{ key: string }>
    concurrencyPresets?: Array<{ key: string }>
  }
  expect(config.approvedModels, 'The deployed UI must list this approved model').toContain(model)
  if (concurrency) expect(config.concurrencyPresets?.map(preset => preset.key)).toContain(concurrency.key)
  if (workload.key !== 'default') {
    expect(config.workloadPresets?.map(preset => preset.key), 'The deployed UI must list this workload preset').toContain(workload.key)
  }
}

async function checkRecommend(page: Page, testInfo: TestInfo, model: string, workload: WorkloadPreset, systemId: string, scenario: RecommendScenario) {
  test.setTimeout(300_000)
  await verifyDeployedPreset(page, model, workload, scenario.concurrency)
  const { systems, objectives } = await readAisimulateCatalogDetails(page)
  const system = systems.find(item => item.id === systemId)
  expect(system, `The deployed catalog must contain ${systemId}`).toBeDefined()
  const pageLoadErrors: string[] = []
  page.on('pageerror', error => pageLoadErrors.push(`JavaScript: ${error.message}`))
  page.on('console', message => {
    if (message.type() === 'error') pageLoadErrors.push(`Console: ${message.text()}`)
  })
  page.on('requestfailed', request => {
    pageLoadErrors.push(`Request: ${request.method()} ${request.url()} — ${request.failure()?.errorText ?? 'unknown error'}`)
  })
  page.on('response', response => {
    if (response.status() >= 400) pageLoadErrors.push(`HTTP ${response.status()}: ${response.url()}`)
  })
  if (scenario.action === 'sizing') await captureBrowserRecommendation(page)
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/recommend')
  try {
    await expect(page.getByRole('button', { name: 'Default', exact: true })).toBeVisible({ timeout: 45_000 })
  } catch (error) {
    await saveJson(testInfo, 'page-load-diagnostics.json', {
      url: page.url(),
      errors: pageLoadErrors,
      state: await page.evaluate(() => ({ readyState: document.readyState, stylesheets: document.styleSheets.length, scripts: document.scripts.length })),
    })
    throw error
  }
  const savedVersion = await page.evaluate(() => localStorage.getItem('settings_backend_version_v2:aisimulate'))
  await selectModel(page, 'adv-model', model)
  await selectSystem(page, 'adv-gpu', system!.name)
  const goalButton = page.getByRole('button', { name: goalLabels[scenario.goal], exact: true })
  if (scenario.goal === 'capacity' && !objectives.includes('concurrency_capacity')) {
    await expect(goalButton).toHaveCount(0)
    test.skip(true, 'AISimulators catalog does not offer concurrency-capacity sizing')
  }
  await goalButton.click()
  await page.getByRole('button', { name: workload.label, exact: true }).click()
  if (scenario.concurrency) await page.getByRole('button', { name: scenario.concurrency.label, exact: true }).click()
  const action = page.getByRole('button', { name: scenario.action === 'pareto' ? 'Pareto frontier' : 'Calculate', exact: true })
  if (scenario.action === 'pareto' && !objectives.includes('pareto')) {
    await expect(action).toBeDisabled()
    test.skip(true, 'AISimulators catalog does not offer Pareto sweeps')
  }
  await expect(action).toBeEnabled()
  const responsePromise = page.waitForResponse(
    response => new URL(response.url()).pathname === '/api/recommend',
    { timeout: 240_000 },
  )
  await action.click()
  const response = await responsePromise
  const request = response.request().postDataJSON() as Record<string, unknown>
  await saveJson(testInfo, 'recommend-live-request.json', request)
  expect(request).toMatchObject({
    model_path: model, system: systemId,
    backend: appConfig.defaultBackend,
    simulator: 'aisimulate',
    isl: workload.isl, osl: workload.osl,
    objective: scenario.action === 'pareto' ? 'pareto' : scenario.goal === 'latency' ? 'min_gpus' : scenario.goal === 'capacity' ? 'concurrency_capacity' : 'throughput_demand',
    sla_mode: scenario.goal === 'latency' ? 'enforce' : 'none',
  })
  if (scenario.action === 'pareto') {
    expect(request).not.toHaveProperty('target_concurrency')
    expect(request).not.toHaveProperty('target_request_rate')
    if (scenario.goal === 'demand') {
      expect(request).not.toHaveProperty('ttft')
      expect(request).not.toHaveProperty('tpot')
    } else expect(request).toMatchObject({ ttft: 1000, tpot: 30 })
  } else if (scenario.goal === 'demand') {
    expect(request).toMatchObject({ target_request_rate: matrixDemandRate })
    expect(request).not.toHaveProperty('target_concurrency')
    expect(request).not.toHaveProperty('ttft')
    expect(request).not.toHaveProperty('tpot')
  } else {
    expect(request).toMatchObject({ target_concurrency: scenario.concurrency.concurrency })
    expect(request).not.toHaveProperty('target_request_rate')
    if (scenario.goal === 'latency') expect(request).toMatchObject({ ttft: 1000, tpot: 30 })
    else {
      expect(request).not.toHaveProperty('ttft')
      expect(request).not.toHaveProperty('tpot')
    }
  }
  if (savedVersion) expect(request.backend_version).toBe(savedVersion)
  else expect(request).not.toHaveProperty('backend_version')
  expect(request).not.toHaveProperty('prefix')
  if (scenario.action === 'pareto') {
    const result = await response.json() as unknown
    await saveJson(testInfo, 'recommend-live-response.json', result)
    const frontier = page.getByRole('region', { name: 'Pareto frontier' })
    await expect(frontier).toBeVisible()
    if (!response.ok()) {
      await expect(frontier.getByText(formatApiError(result, `Pareto sweep failed (${response.status()})`), { exact: true })).toBeVisible()
      return
    }
    const parsed = parseFrontierBody(result)
    expect(parsed, 'The current API must return a parseable Pareto response').not.toBeNull()
    expect(parsed!.simulator).toBe('aisimulate')
    const feasible = parsed!.candidates.filter(candidate => candidate.status === 'feasible')
    const failed = parsed!.candidates.filter(candidate => candidate.status !== 'feasible')
    if (parsed!.candidates.length === 0) {
      await expect(frontier.getByText('No candidate configurations were evaluated for this workload and GPU window.')).toBeVisible()
    } else if (feasible.length === 0) {
      await expect(frontier.getByText('No candidate configuration met the workload constraints.', { exact: false })).toBeVisible()
    } else {
      const table = frontier.getByRole('table', { name: /Feasible candidates/ })
      await expect(table.locator('tbody tr')).toHaveCount(feasible.length)
      const first = feasible[0]
      const row = table.locator('tbody tr').filter({ has: page.getByRole('button', { name: formatCandidateId(first.candidate_id) }) })
      await expect(row).toBeVisible()
      await expect(row.locator('td').nth(2)).toHaveText(first.used_gpus == null ? '—' : String(first.used_gpus))
    }
    if (failed.length > 0) {
      await frontier.getByRole('button', { name: `Show failed attempts (${failed.length})` }).click()
      const table = frontier.getByRole('table', { name: /Failed attempts/ })
      await expect(table.locator('tbody tr')).toHaveCount(failed.length)
    }
    return
  }
  const result = await capturedRecommendation(page)
  await saveJson(testInfo, 'recommend-live-response.json', result)
  if (!response.ok() || result.status !== 'completed') {
    await expect(page.getByText(recommendErrorTitle(result.error?.code), { exact: true })).toBeVisible()
    if (result.error?.message && ['AISIM_NO_CONFIGURATION', 'SIMULATOR_MISMATCH'].includes(result.error.code ?? '')) {
      await expect(page.locator('[class*="errorMsg"]')).toHaveText(result.error.message)
    }
    return
  }
  await expect(displayedTileValue(page, /^GPUs required /)).toHaveText(`${result.recommendation.gpusNeeded}× ${system!.name}`)
  const memoryTile = page.getByRole('button', { name: /^Peak mem \/ GPU / })
  if (result.memoryAvailable === false) {
    await expect(memoryTile).toHaveCount(0)
    await expect(page.getByText('This simulator does not provide memory estimates; the memory tile is hidden.')).toBeVisible()
  } else {
    await expect(displayedTileValue(page, /^Peak mem \/ GPU /)).toHaveText(`${Math.round(result.memory.value * 10) / 10}GB`)
  }
  if (scenario.goal === 'latency') {
    await expect(displayedTileValue(page, /^TTFT /)).toHaveText(`${Math.round(result.performance.ttftLatencyMs)}ms`)
  } else if (scenario.goal === 'demand') {
    await expect(displayedTileValue(page, /^Demand /)).toHaveText(`${result.demand?.achievedRps?.toFixed(1) ?? '—'}req/s`)
  } else {
    await expect(displayedTileValue(page, /^Capacity /)).toHaveText(`${result.capacity?.capacitySequences ?? '—'}seq`)
  }
  if (scenario.goal !== 'capacity' || result.capacity?.performanceAvailable) {
    await expect(displayedTileValue(page, /^Throughput [\d,]/)).toHaveText(`${Math.round(result.throughput.tokensPerSecond)}tok/s`)
  } else {
    await expect(page.getByRole('button', { name: /^Throughput [\d,]/ })).toHaveCount(0)
  }
}

for (const model of appConfig.approvedModels) {
  for (const { workload, systemId, scenario } of recommendCasesForModel(model)) {
    const scenarioLabel = `${goalLabels[scenario.goal]}${scenario.action === 'pareto' ? ' Pareto' : ''}${scenario.concurrency ? ` / ${scenario.concurrency.label}` : ''}`
    test(`Recommend sizing / ${model} / ${workload.label} / ${systemId} / ${scenarioLabel} / live`, async ({ page }, testInfo) => {
      await checkRecommend(page, testInfo, model, workload, systemId, scenario)
    })
  }
}

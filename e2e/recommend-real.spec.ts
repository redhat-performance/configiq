import { writeFile } from 'node:fs/promises'
import { expect, test, type Page, type TestInfo } from '@playwright/test'
import appConfig from '../public/config.json'
import type { WorkloadPreset } from '../lib/workload-presets'
import { readAisimulateCatalog } from './live-catalog'
import { matrixConcurrencyPreset, systemsForModel, workloads } from './test-matrix'

test.describe.configure({ mode: 'parallel' })

interface CompletedRecommendation {
  status: 'completed'
  error?: never
  recommendation: { gpusNeeded: number; gpusPerReplica: number }
  memoryAvailable?: boolean
  memory: { value: number }
  performance: { ttftLatencyMs: number }
  throughput: { tokensPerSecond: number }
}

interface FailedRecommendation {
  status?: 'failed'
  error?: { code?: string; message?: string }
}

type Recommendation = CompletedRecommendation | FailedRecommendation

function recommendErrorTitle(code?: string): string {
  switch (code) {
    case 'AISIM_TIMEOUT': return 'Request timed out'
    case 'AISIM_NO_CONFIGURATION': return 'No valid configuration found'
    case 'AISIM_UNAVAILABLE': return 'Sizing service unavailable'
    case 'AISIM_NOT_CONFIGURED': return 'Service not configured'
    case 'AISIM_INVALID_RESPONSE': return 'Unexpected response'
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

async function verifyDeployedPreset(page: Page, model: string, workload: WorkloadPreset) {
  const response = await page.request.get('/config.json')
  expect(response.ok(), 'The deployed UI configuration must be available').toBeTruthy()
  const config = await response.json() as {
    approvedModels?: string[]
    workloadPresets?: Array<{ key: string }>
    concurrencyPresets?: Array<{ key: string }>
  }
  expect(config.approvedModels, 'The deployed UI must list this approved model').toContain(model)
  expect(config.concurrencyPresets?.map(preset => preset.key)).toContain(matrixConcurrencyPreset.key)
  if (workload.key !== 'default') {
    expect(config.workloadPresets?.map(preset => preset.key), 'The deployed UI must list this workload preset').toContain(workload.key)
  }
}

async function checkRecommend(page: Page, testInfo: TestInfo, model: string, workload: WorkloadPreset, systemId: string) {
  test.setTimeout(300_000)
  await verifyDeployedPreset(page, model, workload)
  const systems = await readAisimulateCatalog(page)
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
  await captureBrowserRecommendation(page)
  await page.goto('/recommend')
  try {
    await expect(page.getByRole('button', { name: 'Default', exact: true })).toBeVisible()
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
  await page.getByRole('button', { name: 'Latency SLO', exact: true }).click()
  await page.getByRole('button', { name: workload.label, exact: true }).click()
  await page.getByRole('button', { name: matrixConcurrencyPreset.label, exact: true }).click()
  const calculate = page.getByRole('button', { name: 'Calculate', exact: true })
  await expect(calculate).toBeEnabled()
  const responsePromise = page.waitForResponse(
    response => new URL(response.url()).pathname === '/api/recommend',
    { timeout: 240_000 },
  )
  await calculate.click()
  const response = await responsePromise
  const request = response.request().postDataJSON() as Record<string, unknown>
  const result = await capturedRecommendation(page)
  await saveJson(testInfo, 'recommend-live-request.json', request)
  await saveJson(testInfo, 'recommend-live-response.json', result)
  expect(request).toMatchObject({
    model_path: model, system: systemId,
    backend: appConfig.defaultBackend,
    simulator: 'aisimulate',
    isl: workload.isl, osl: workload.osl,
    ttft: 1000, tpot: 30, target_concurrency: matrixConcurrencyPreset.concurrency,
  })
  if (savedVersion) expect(request.backend_version).toBe(savedVersion)
  else expect(request).not.toHaveProperty('backend_version')
  expect(request).not.toHaveProperty('prefix')
  if (!response.ok() || result.status !== 'completed') {
    await expect(page.getByText(recommendErrorTitle(result.error?.code), { exact: true })).toBeVisible()
    return
  }
  await expect(page.getByRole('button', { name: /^GPUs required / })).toContainText(`${result.recommendation.gpusNeeded}×`)
  const memoryTile = page.getByRole('button', { name: /^Peak mem \/ GPU / })
  if (result.memoryAvailable === false) {
    await expect(memoryTile).toHaveCount(0)
    await expect(page.getByText('This simulator does not provide memory estimates; the memory tile is hidden.')).toBeVisible()
  } else {
    await expect(memoryTile).toContainText(`${Math.round(result.memory.value * 10) / 10}GB`)
  }
  await expect(page.getByRole('button', { name: /^TTFT / })).toContainText(`${Math.round(result.performance.ttftLatencyMs)}ms`)
  await expect(page.getByRole('button', { name: /^Throughput [\d,]/ })).toContainText(`${Math.round(result.throughput.tokensPerSecond)}tok/s`)
}

for (const model of appConfig.approvedModels) {
  for (const workload of workloads) {
    for (const systemId of systemsForModel(model)) {
      test(`Recommend sizing / ${model} / ${workload.label} / ${systemId} / live`, async ({ page }, testInfo) => {
        await checkRecommend(page, testInfo, model, workload, systemId)
      })
    }
  }
}

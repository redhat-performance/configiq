import { writeFile } from 'node:fs/promises'
import { expect, test, type Page, type TestInfo } from '@playwright/test'
import appConfig from '../public/config.json'
import { formatBytes } from '../lib/utils/format'
import type { WorkloadPreset } from '../lib/workload-presets'
import type { ConcurrencyPreset } from '../lib/concurrency-presets'
import { formatApiError, parseFrontierBody } from '../lib/pareto/parse'
import { describeSweepCoverage } from '../lib/pareto/sweep-caption'
import { readAisimulateCatalog, readAisimulateCatalogDetails } from './live-catalog'
import { matrixDemandRate } from './test-matrix'

/**
 * Shared live-UI case runners.
 *
 * Each runner drives one deployed page, captures the request the browser sends,
 * and compares it with the current API response. The scheduled suites only
 * choose which cases to run; the assertions live here so the weekly, sampled,
 * and deployment-sanity suites cannot drift apart.
 */

export interface PredictResponse {
  status?: string
  memory?: number
  ttft?: number
  tpot?: number
  request_latency?: number
  tokens_per_second?: number
  memory_breakdown?: { weights_bytes?: number; kv_cache_bytes?: number } | null
  memory_unavailable_reason?: string | null
  error?: { code?: string; message?: string }
  detail?: string
}

export interface KvCacheResponse {
  status?: string
  kvCache?: { totalBytes?: number; perTokenBytes?: number }
  memoryBreakdown?: { weightsBytes?: number }
  error?: { code?: string; message?: string }
}

export interface CompletedRecommendation {
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

export interface FailedRecommendation {
  status?: 'failed'
  error?: { code?: string; message?: string }
}

export type Recommendation = CompletedRecommendation | FailedRecommendation

export type RecommendScenario =
  | { action: 'sizing'; goal: 'latency' | 'capacity'; concurrency: ConcurrencyPreset }
  | { action: 'sizing'; goal: 'demand'; concurrency?: never }
  | { action: 'pareto'; goal: 'latency' | 'demand'; concurrency?: never }

/**
 * `requireSuccess` turns an API error into a test failure.
 *
 * The scheduled matrices accept an API error when the UI renders the matching
 * error state, because some model/GPU pairings are expected to be unsupported.
 * The deployment sanity suite sets it: a release that cannot produce a result
 * for the one sampled pairing per model is broken, and reporting that as pass
 * would make the suite meaningless.
 */
export interface CaseOptions {
  requireSuccess?: boolean
  /**
   * The one API failure a wholly unavailable model may produce.
   *
   * A sanity case that exempts a model from `requireSuccess` still insists the
   * response is that model's known failure, so an unrelated outage cannot pass
   * as a known gap. Matching a code alone is not enough where the endpoints
   * report different codes; `messageIncludes` pins the credential failure.
   */
  expectFailure?: { code?: string; messageIncludes?: string }
}

function assertKnownFailure(options: CaseOptions, code: string | undefined, message: string | undefined): void {
  const expected = options.expectFailure
  if (!expected) return
  const actual = `code=${code ?? 'none'} message=${message ?? 'none'}`
  if (expected.code && code !== expected.code) {
    throw new Error(`Expected the known ${expected.code} failure, got ${actual}`)
  }
  if (expected.messageIncludes && !(message ?? '').includes(expected.messageIncludes)) {
    throw new Error(`Expected the known failure to mention "${expected.messageIncludes}", got ${actual}`)
  }
}

export const goalLabels = { latency: 'Latency SLO', demand: 'Throughput demand', capacity: 'Concurrency capacity' } as const

function predictErrorTitle(code?: string): string {
  switch (code) {
    case 'OOM': return 'Not enough GPU memory'
    case 'AUTH_REQUIRED': return 'Authentication required'
    case 'MODEL_NOT_FOUND': return 'Model not found'
    case 'SIMULATOR_MISMATCH': return 'Simulator mismatch'
    case 'AISIM_NO_CONFIGURATION': return 'No valid configuration found'
    case 'AISIM_TIMEOUT': return 'Request timed out'
    case 'AISIM_UNAVAILABLE': return 'Sizing service unavailable'
    default: return 'Estimate failed'
  }
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

export async function saveJson(testInfo: TestInfo, name: string, value: unknown) {
  const path = testInfo.outputPath(name)
  await writeFile(path, JSON.stringify(value, null, 2))
  await testInfo.attach(name, { path, contentType: 'application/json' })
}

export async function selectModel(page: Page, inputId: string, model: string) {
  const input = page.locator(`#${inputId}`)
  await expect(input).toBeVisible()
  await expect(input).not.toHaveValue('')
  await input.fill(model)
  await input.press('Enter')
  await expect(input).toHaveValue(model)
}

export async function selectSystem(page: Page, inputId: string, name: string) {
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

/**
 * Check that the deployment exposes the model and presets the case is about to
 * use. This is a capability check, not a version check: the workflow pins the
 * checked-out commit to the deployed build before any case runs.
 */
export async function verifyDeployedPreset(page: Page, model: string, workload: WorkloadPreset, concurrency?: ConcurrencyPreset) {
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

export async function checkPredict(
  page: Page,
  testInfo: TestInfo,
  model: string,
  workload: WorkloadPreset,
  systemId: string,
  concurrency: ConcurrencyPreset,
  parallelism: number,
  options: CaseOptions = {},
) {
  test.setTimeout(300_000)
  await verifyDeployedPreset(page, model, workload, concurrency)
  const systems = await readAisimulateCatalog(page)
  const system = systems.find(item => item.id === systemId)
  expect(system, `The deployed catalog must contain ${systemId}`).toBeDefined()
  await page.addInitScript(() => localStorage.setItem('predict-performance-tour-seen', 'true'))
  await page.goto('/predict')
  const savedVersion = await page.evaluate(() => localStorage.getItem('settings_backend_version_v2:aisim'))
  await selectModel(page, 'predict-performance-model', model)
  await selectSystem(page, 'predict-performance-gpu', system!.name)
  await page.getByRole('button', { name: workload.label, exact: true }).click()
  await page.getByRole('button', { name: concurrency.label, exact: true }).click()
  if (parallelism > 1) {
    await page.getByText('Want to change assumptions?', { exact: true }).click()
    await page.getByRole('spinbutton', { name: 'Tensor parallel size (TP)' }).fill(String(parallelism))
    const expertParallel = page.getByRole('spinbutton', { name: 'MoE expert parallel (EP)' })
    if (await expertParallel.isVisible()) await expertParallel.fill(String(parallelism))
  }
  const calculate = page.getByRole('button', { name: 'Calculate', exact: true })
  await expect(calculate).toBeEnabled()

  const responsePromise = page.waitForResponse(
    response => new URL(response.url()).pathname === '/api/predict',
    { timeout: 240_000 },
  )
  await calculate.click()
  const response = await responsePromise
  const body = response.request().postDataJSON() as Record<string, unknown>
  const result = await response.json() as PredictResponse
  await saveJson(testInfo, 'predict-request.json', body)
  await saveJson(testInfo, 'predict-response.json', result)

  expect(body).toMatchObject({
    model_path: model,
    system: systemId,
    backend: appConfig.defaultBackend,
    simulator: 'aisim',
    tp_size: parallelism,
    isl: workload.isl,
    osl: workload.osl,
    batch_size: concurrency.concurrency,
  })
  if (savedVersion) expect(body.backend_version).toBe(savedVersion)
  else expect(body).not.toHaveProperty('backend_version')
  if (workload.prefix) expect(body).toMatchObject({ prefix: workload.prefix })
  else expect(body).not.toHaveProperty('prefix')

  if (!response.ok() || result.status === 'failed') {
    const errorCode = result.error?.code ?? 'AISIM_NO_CONFIGURATION'
    const errorMessage = result.error?.message ?? result.detail
    if (options.requireSuccess) {
      throw new Error(`Predict did not complete for ${model} on ${systemId}: ${errorCode}${errorMessage ? ` — ${errorMessage}` : ''}`)
    }
    assertKnownFailure(options, errorCode, errorMessage)
    await expect(page.getByText(predictErrorTitle(errorCode), { exact: true })).toBeVisible()
    if (!['OOM', 'AUTH_REQUIRED', 'MODEL_NOT_FOUND', 'AISIM_TIMEOUT'].includes(errorCode)) {
      await expect(page.getByText(errorMessage ?? 'AISimulators prediction failed', { exact: true })).toBeVisible()
    }
    return
  }
  if (options.requireSuccess && (typeof result.ttft !== 'number' || typeof result.tpot !== 'number')) {
    throw new Error(`Predict returned no timing metrics for ${model} on ${systemId}`)
  }

  const ttft = result.ttft ?? 0
  const tpot = result.tpot ?? 0
  const weightTile = page.getByRole('button', { name: /^Weight memory / })
  const kvTile = page.getByRole('button', { name: /^KV cache \/ req / })
  if (typeof result.memory !== 'number' && !result.memory_breakdown) {
    await expect(weightTile).toHaveCount(0)
    await expect(kvTile).toHaveCount(0)
    const memoryWarning = typeof result.memory_unavailable_reason === 'string'
      ? result.memory_unavailable_reason
      : 'The selected simulator did not return a memory breakdown.'
    await expect(page.getByText(memoryWarning)).toBeVisible()
  } else {
    const weightGb = Math.round(result.memory_breakdown
      ? (result.memory_breakdown.weights_bytes ?? 0) / 1024 ** 3
      : (result.memory ?? 0) * 0.5)
    const kvMbPerRequest = Math.round((result.memory_breakdown?.kv_cache_bytes ?? 0) / 1024 ** 3 / concurrency.concurrency * 1000)
    await expect(weightTile).toContainText(`${weightGb}GB`)
    await expect(kvTile).toContainText(`${kvMbPerRequest}MB`)
  }
  await expect(page.locator('[class*="paramGrid"] > div').filter({ hasText: /^TTFT\b/ }).locator(':scope > div').nth(1)).toHaveText(`${ttft.toFixed(1)} ms`)
  await expect(page.locator('[class*="paramGrid"] > div').filter({ hasText: /^TPOT\b/ }).locator(':scope > div').nth(1)).toHaveText(`${tpot.toFixed(1)} ms`)
  const hasTiming = Number.isFinite(ttft) && Number.isFinite(tpot)
  const requestLatency = hasTiming
    ? typeof result.request_latency === 'number'
      ? result.request_latency
      : ttft + tpot * workload.osl
    : 0
  const throughput = typeof result.tokens_per_second === 'number'
    ? result.tokens_per_second
    : hasTiming && workload.osl > 0 && requestLatency > 0 ? workload.osl * 1000 / requestLatency : 0
  await expect(page.locator('[class*="paramGrid"] > div').filter({ hasText: /^Throughput\b/ }).locator(':scope > div').nth(1)).toHaveText(`${throughput.toFixed(1)} tok/s`)
}

export async function checkKvCache(
  page: Page,
  testInfo: TestInfo,
  model: string,
  workload: WorkloadPreset,
  systemId: string,
  concurrency: ConcurrencyPreset,
  parallelism: number,
  options: CaseOptions = {},
) {
  test.setTimeout(300_000)
  await verifyDeployedPreset(page, model, workload, concurrency)
  const systems = await readAisimulateCatalog(page)
  const system = systems.find(item => item.id === systemId)
  expect(system, `The deployed catalog must contain ${systemId}`).toBeDefined()
  await page.goto('/kv-cache')
  const savedVersion = await page.evaluate(() => localStorage.getItem('settings_backend_version_v2:aisim'))
  await selectModel(page, 'kv-model', model)
  await selectSystem(page, 'kv-gpu', system!.name)
  await page.getByRole('button', { name: workload.label, exact: true }).click()
  await page.getByRole('button', { name: concurrency.label, exact: true }).click()
  if (parallelism > 1) {
    await page.getByRole('button', { name: 'Adjust? (edit fields below)', exact: true }).click()
    await page.locator('#kv-tp').fill(String(parallelism))
    await page.locator('#kv-moe-ep').fill(String(parallelism))
  }
  const responsePromise = page.waitForResponse(
    response => new URL(response.url()).pathname === '/api/memory',
    { timeout: 240_000 },
  )
  const calculate = page.getByRole('button', { name: 'Calculate', exact: true })
  await expect(calculate).toBeEnabled()
  await calculate.click()
  const response = await responsePromise
  const body = response.request().postDataJSON() as Record<string, unknown>
  const result = await response.json() as KvCacheResponse
  await saveJson(testInfo, 'memory-request.json', body)
  await saveJson(testInfo, 'memory-response.json', result)

  expect(body).toMatchObject({
    model_path: model,
    system: systemId,
    backend: appConfig.defaultBackend,
    max_num_tokens: workload.isl + workload.osl,
    max_batch_size: concurrency.concurrency,
    tp_size: parallelism,
    ...(parallelism > 1 ? { moe_ep_size: parallelism } : {}),
  })
  if (savedVersion) expect(body.backend_version).toBe(savedVersion)
  else expect(body).not.toHaveProperty('backend_version')
  if (!response.ok() || result.status !== 'completed') {
    const detail = result.error?.message ?? (response.ok() ? 'the response reported no completion' : `HTTP ${response.status()}`)
    if (options.requireSuccess) {
      throw new Error(`KV cache calculation did not complete for ${model} on ${systemId}: ${detail}`)
    }
    assertKnownFailure(options, result.error?.code, result.error?.message)
    if (result.error?.message) {
      await expect(page.getByText(result.error.message, { exact: true })).toBeVisible()
    } else {
      await expect(page.getByRole('heading', { name: /Calculation failed/ })).toBeVisible()
    }
    return
  }
  if (options.requireSuccess) {
    expect(
      typeof result.kvCache?.totalBytes === 'number' || typeof result.kvCache?.perTokenBytes === 'number',
      `KV cache calculation for ${model} on ${systemId} returned no cache metrics`,
    ).toBe(true)
  }
  if (typeof result.kvCache?.totalBytes === 'number') {
    await expect(page.getByRole('button', { name: /^Available for KV cache \/ GPU / }).locator('[class*="tileValue"]')).toHaveText(formatBytes(result.kvCache.totalBytes))
  }
  if (typeof result.kvCache?.perTokenBytes === 'number') {
    await expect(page.getByRole('button', { name: /^Per token / }).locator('[class*="tileValue"]')).toHaveText(formatBytes(result.kvCache.perTokenBytes))
  }
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

export async function checkRecommend(
  page: Page,
  testInfo: TestInfo,
  model: string,
  workload: WorkloadPreset,
  systemId: string,
  scenario: RecommendScenario,
  options: CaseOptions = {},
) {
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
  const savedVersion = await page.evaluate(() => localStorage.getItem('settings_backend_version_v2:aisim'))
  await selectModel(page, 'adv-model', model)
  await selectSystem(page, 'adv-gpu', system!.name)
  const goalButton = page.getByRole('button', { name: goalLabels[scenario.goal], exact: true })
  if (scenario.goal === 'capacity' && !objectives.includes('concurrency_capacity')) {
    await expect(goalButton).toHaveCount(0)
    if (options.requireSuccess) {
      throw new Error(`The deployed catalog does not offer concurrency-capacity sizing for ${model} on ${systemId}, which this suite requires`)
    }
    test.skip(true, 'The deployed catalog does not offer concurrency-capacity sizing')
  }
  await goalButton.click()
  await page.getByRole('button', { name: workload.label, exact: true }).click()
  if (scenario.concurrency) await page.getByRole('button', { name: scenario.concurrency.label, exact: true }).click()
  const action = page.getByRole('button', { name: scenario.action === 'pareto' ? 'Pareto frontier' : 'Calculate', exact: true })
  if (scenario.action === 'pareto' && !objectives.includes('pareto')) {
    await expect(action).toBeDisabled()
    if (options.requireSuccess) {
      throw new Error(`The deployed catalog does not offer Pareto sweeps for ${model} on ${systemId}, which this suite requires`)
    }
    test.skip(true, 'The deployed catalog does not offer Pareto sweeps')
  }
  await expect(action).toBeEnabled()
  // The Pareto action fires the sizing request and the sweep together, so the
  // first `/api/recommend` response on the page may belong to either half.
  // Pick the sweep when that is the action under test.
  const responsePromise = page.waitForResponse(
    response => {
      if (new URL(response.url()).pathname !== '/api/recommend') return false
      if (scenario.action !== 'pareto') return true
      const posted = response.request().postDataJSON() as { objective?: unknown } | null
      return posted?.objective === 'pareto'
    },
    { timeout: 240_000 },
  )
  await action.click()
  const response = await responsePromise
  const request = response.request().postDataJSON() as Record<string, unknown>
  await saveJson(testInfo, 'recommend-live-request.json', request)
  expect(request).toMatchObject({
    model_path: model, system: systemId,
    backend: appConfig.defaultBackend,
    simulator: 'aisim',
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
  if (workload.prefix) expect(request).toMatchObject({ prefix: workload.prefix })
  else expect(request).not.toHaveProperty('prefix')
  if (scenario.action === 'pareto') {
    const result = await response.json() as unknown
    await saveJson(testInfo, 'recommend-live-response.json', result)
    const frontier = page.getByRole('region', { name: 'Pareto frontier' })
    await expect(frontier).toBeVisible()
    if (!response.ok()) {
      if (options.requireSuccess) {
        throw new Error(`Pareto sweep failed for ${model} on ${systemId}: ${formatApiError(result, `HTTP ${response.status()}`)}`)
      }
      const failureCode = typeof result === 'object' && result !== null
        ? (result as { error?: { code?: unknown } }).error?.code
        : undefined
      assertKnownFailure(options, typeof failureCode === 'string' ? failureCode : undefined, formatApiError(result, ''))
      await expect(frontier.getByText(formatApiError(result, `Pareto sweep failed (${response.status()})`), { exact: true })).toBeVisible()
      return
    }
    const hasCandidates = typeof result === 'object' && result !== null && Array.isArray((result as { candidates?: unknown }).candidates)
    const parsed = parseFrontierBody(result)
    expect(parsed, 'The current API must return a parseable Pareto response').not.toBeNull()
    expect(parsed!.simulator).toBe('aisim')
    if (options.requireSuccess && (!hasCandidates || parsed!.candidates.length === 0)) {
      throw new Error(`Pareto sweep for ${model} on ${systemId} returned no evaluated candidates`)
    }
    // The page collates the sizing recommendation into the sweep list, so the
    // feasible table can hold one row the sweep never evaluated. The coverage
    // line is derived from the sweep's own candidates and is the exact check.
    await expect(frontier.getByText(describeSweepCoverage(parsed!.sweep, parsed!.candidates))).toBeVisible()
    const feasible = parsed!.candidates.filter(candidate => candidate.status === 'feasible')
    const failed = parsed!.candidates.filter(candidate => candidate.status !== 'feasible')
    if (feasible.length > 0) {
      const table = frontier.getByRole('table', { name: /Feasible candidates/ })
      expect(await table.locator('tbody tr').count()).toBeGreaterThanOrEqual(feasible.length)
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
    if (options.requireSuccess) {
      const code = result.error?.code ?? `HTTP ${response.status()}`
      const detail = result.error?.message
      throw new Error(`Recommend sizing did not complete for ${model} on ${systemId}: ${code}${detail ? ` — ${detail}` : ''}`)
    }
    assertKnownFailure(options, result.error?.code, result.error?.message)
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

import { writeFile } from 'node:fs/promises'
import { expect, test, type Page, type TestInfo } from '@playwright/test'
import appConfig from '../public/config.json'
import { formatBytes } from '../lib/utils/format'
import { coveredSystems, modelParallelism, workloads } from './test-matrix'

// Each model/workload/GPU combination is independent, allowing CI sharding.
test.describe.configure({ mode: 'parallel' })

interface CatalogResponse {
  models: Array<string | { id: string }>
  backends: Array<{ id: string; default_version: string | null }>
  systems: Array<{ id: string; name: string }>
}

interface PredictResponse {
  status?: string
  memory?: number
  ttft?: number
  tpot?: number
  memory_breakdown?: { weights_bytes?: number; kv_cache_bytes?: number } | null
  error?: { code?: string; message?: string }
}

function predictErrorTitle(code?: string): string {
  switch (code) {
    case 'OOM': return 'Not enough GPU memory'
    case 'AUTH_REQUIRED': return 'Authentication required'
    case 'MODEL_NOT_FOUND': return 'Model not found'
    case 'AISIM_TIMEOUT': return 'Request timed out'
    case 'AISIM_UNAVAILABLE': return 'Sizing service unavailable'
    default: return 'Estimate failed'
  }
}

async function readCatalog(page: Page): Promise<CatalogResponse> {
  const response = await page.request.get('/api/catalog')
  expect(response.ok(), 'The tested-model catalog must be available').toBeTruthy()
  const catalog = await response.json() as CatalogResponse
  expect(catalog.models.length).toBeGreaterThan(0)
  return catalog
}

async function selectModel(page: Page, model: string, inputId: string) {
  const input = page.locator(`#${inputId}`)
  await expect(input).not.toHaveValue('')
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

async function saveJson(testInfo: TestInfo, name: string, value: unknown) {
  const path = testInfo.outputPath(name)
  await writeFile(path, JSON.stringify(value, null, 2))
  await testInfo.attach(name, { path, contentType: 'application/json' })
}

for (const model of appConfig.testedModels) {
  for (const workload of workloads) {
    for (const systemId of coveredSystems) {
      test(`Predict performance / ${model} / ${workload.label} / ${systemId} / live`, async ({ page }, testInfo) => {
        const configResponse = await page.request.get('/config.json')
        expect(configResponse.ok()).toBeTruthy()
        const deployedConfig = await configResponse.json() as { testedModels?: string[]; workloadPresets?: Array<{ key: string }> }
        expect(deployedConfig.testedModels, 'The deployed UI must list this tested model').toContain(model)
        if (workload.key !== 'default') {
          expect(deployedConfig.workloadPresets?.map(preset => preset.key)).toContain(workload.key)
        }

        const catalog = await readCatalog(page)
        const system = catalog.systems.find(item => item.id === systemId)
        expect(system, `The deployed catalog must contain ${systemId}`).toBeDefined()
        await page.addInitScript(() => localStorage.setItem('predict-performance-tour-seen', 'true'))
        await page.goto('/predict')
        const savedVersion = await page.evaluate(() => localStorage.getItem('settings_backend_version_v2'))
        await selectModel(page, model, 'predict-performance-model')
        await selectSystem(page, 'predict-performance-gpu', system!.name)
        await page.getByRole('button', { name: workload.label, exact: true }).click()
        const parallelism = modelParallelism[model] ?? 1
        if (parallelism > 1) {
          await page.getByText('Want to change assumptions?', { exact: true }).click()
          await page.locator('#acc-parallel').click()
          await page.getByRole('spinbutton', { name: 'Tensor parallel size (TP)' }).fill(String(parallelism))
          await page.getByRole('spinbutton', { name: 'MoE expert parallel (EP)' }).fill(String(parallelism))
        }
        const calculate = page.getByRole('button', { name: 'Calculate', exact: true })
        await expect(calculate).toBeEnabled()

        const responsePromise = page.waitForResponse(response => new URL(response.url()).pathname === '/api/predict')
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
          tp_size: parallelism,
          moe_ep_size: parallelism,
          isl: workload.isl,
          osl: workload.osl,
          batch_size: workload.concurrency,
        })
        if (savedVersion) expect(body.backend_version).toBe(savedVersion)
        else expect(body).not.toHaveProperty('backend_version')
        if (workload.prefix > 0) expect(body.prefix).toBe(workload.prefix)
        else expect(body).not.toHaveProperty('prefix')

        if (!response.ok() || result.status === 'failed') {
          await expect(page.getByText(predictErrorTitle(result.error?.code), { exact: true })).toBeVisible()
          return
        }

        const ttft = result.ttft ?? 0
        const tpot = result.tpot ?? 0
        const weightGb = Math.round(result.memory_breakdown
          ? (result.memory_breakdown.weights_bytes ?? 0) / 1024 ** 3
          : (result.memory ?? 0) * 0.5)
        const kvMbPerRequest = Math.round((result.memory_breakdown?.kv_cache_bytes ?? 0) / 1024 ** 3 / workload.concurrency * 1000)
        await expect(page.getByTestId('predict-weight-memory')).toHaveText(`${weightGb}GB`)
        await expect(page.getByTestId('predict-kv-per-request')).toHaveText(`${kvMbPerRequest}MB`)
        await expect(page.getByTestId('predict-ttft')).toHaveText(`${ttft.toFixed(1)} ms`)
        await expect(page.getByTestId('predict-tpot')).toHaveText(`${tpot.toFixed(1)} ms`)
        const latency = ttft + tpot * workload.osl
        const throughput = latency > 0 ? workload.osl * 1000 / latency : 0
        await expect(page.getByTestId('predict-throughput')).toHaveText(`${throughput.toFixed(1)} tok/s`)
      })
    }
  }
}

for (const model of appConfig.testedModels) {
  for (const workload of workloads) {
    for (const systemId of coveredSystems) {
      test(`KV cache / ${model} / ${workload.label} / ${systemId} / live`, async ({ page }, testInfo) => {
        const catalog = await readCatalog(page)
        const system = catalog.systems.find(item => item.id === systemId)
        expect(system, `The deployed catalog must contain ${systemId}`).toBeDefined()
        await page.goto('/kv-cache')
        const savedVersion = await page.evaluate(() => localStorage.getItem('settings_backend_version_v2'))
        await selectModel(page, model, 'kv-model')
        await selectSystem(page, 'kv-gpu', system!.name)
        await page.getByRole('button', { name: /Advanced settings/ }).click()
        // KV cache has no workload preset control. Apply the preset's token budget
        // and concurrency to its editable memory controls for each matrix case.
        await page.locator('#kv-tokens').fill(String(workload.isl + workload.osl))
        await page.locator('#kv-batch').fill(String(workload.concurrency))
        const parallelism = modelParallelism[model] ?? 1
        if (parallelism > 1) {
          await page.locator('#kv-tp').fill(String(parallelism))
          await page.locator('#kv-moe-ep').fill(String(parallelism))
        }
        const responsePromise = page.waitForResponse(response => new URL(response.url()).pathname === '/api/memory')
        const calculate = page.getByRole('button', { name: 'Calculate', exact: true })
        await expect(calculate).toBeEnabled()
        await calculate.click()
        const response = await responsePromise
        const body = response.request().postDataJSON() as Record<string, unknown>
        const result = await response.json() as { status?: string; kvCache?: { totalBytes?: number; perTokenBytes?: number }; memoryBreakdown?: { weightsBytes?: number }; error?: { message?: string } }
        await saveJson(testInfo, 'memory-request.json', body)
        await saveJson(testInfo, 'memory-response.json', result)

        expect(body).toMatchObject({
          model_path: model,
          system: systemId,
          backend: appConfig.defaultBackend,
          max_num_tokens: workload.isl + workload.osl,
          max_batch_size: workload.concurrency,
          tp_size: parallelism,
          ...(parallelism > 1 ? { moe_ep_size: parallelism } : {}),
        })
        if (savedVersion) expect(body.backend_version).toBe(savedVersion)
        else expect(body).not.toHaveProperty('backend_version')
        if (!response.ok() || result.status !== 'completed') {
          if (result.error?.message) {
            await expect(page.getByText(result.error.message, { exact: true })).toBeVisible()
          } else {
            await expect(page.getByRole('heading', { name: /Calculation failed/ })).toBeVisible()
          }
          return
        }
        if (typeof result.kvCache?.totalBytes === 'number') {
          await expect(page.getByTestId('kv-total-value')).toHaveText(formatBytes(result.kvCache.totalBytes))
        }
        if (typeof result.kvCache?.perTokenBytes === 'number') {
          await expect(page.getByTestId('kv-pertoken-value')).toHaveText(formatBytes(result.kvCache.perTokenBytes))
        }
      })
    }
  }
}

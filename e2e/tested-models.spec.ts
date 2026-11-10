import { writeFile } from 'node:fs/promises'
import { expect, test, type Page, type TestInfo } from '@playwright/test'
import appConfig from '../public/config.json'
import { formatBytes } from '../lib/utils/format'
import { readAisimulateCatalog } from './live-catalog'
import { matrixConcurrencyPreset, modelParallelism, systemsForModel, workloads } from './test-matrix'

// Each model/workload/GPU combination is independent, allowing CI sharding.
test.describe.configure({ mode: 'parallel' })

interface PredictResponse {
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

for (const model of appConfig.approvedModels) {
  for (const workload of workloads) {
    for (const systemId of systemsForModel(model)) {
      test(`Predict performance / ${model} / ${workload.label} / ${systemId} / live`, async ({ page }, testInfo) => {
        const configResponse = await page.request.get('/config.json')
        expect(configResponse.ok()).toBeTruthy()
        const deployedConfig = await configResponse.json() as { approvedModels?: string[]; workloadPresets?: Array<{ key: string }>; concurrencyPresets?: Array<{ key: string }> }
        expect(deployedConfig.approvedModels, 'The deployed UI must list this approved model').toContain(model)
        expect(deployedConfig.concurrencyPresets?.map(preset => preset.key)).toContain(matrixConcurrencyPreset.key)
        if (workload.key !== 'default') {
          expect(deployedConfig.workloadPresets?.map(preset => preset.key)).toContain(workload.key)
        }

        const systems = await readAisimulateCatalog(page)
        const system = systems.find(item => item.id === systemId)
        expect(system, `The deployed catalog must contain ${systemId}`).toBeDefined()
        await page.addInitScript(() => localStorage.setItem('predict-performance-tour-seen', 'true'))
        await page.goto('/predict')
        const savedVersion = await page.evaluate(() => localStorage.getItem('settings_backend_version_v2:aisimulate'))
        await selectModel(page, model, 'predict-performance-model')
        await selectSystem(page, 'predict-performance-gpu', system!.name)
        await page.getByRole('button', { name: workload.label, exact: true }).click()
        await page.getByRole('button', { name: matrixConcurrencyPreset.label, exact: true }).click()
        const parallelism = modelParallelism[model] ?? 1
        if (parallelism > 1) {
          await page.getByText('Want to change assumptions?', { exact: true }).click()
          await page.getByRole('spinbutton', { name: 'Tensor parallel size (TP)' }).fill(String(parallelism))
          const expertParallel = page.getByRole('spinbutton', { name: 'MoE expert parallel (EP)' })
          if (await expertParallel.isVisible()) await expertParallel.fill(String(parallelism))
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
          simulator: 'aisimulate',
          tp_size: parallelism,
          isl: workload.isl,
          osl: workload.osl,
          batch_size: matrixConcurrencyPreset.concurrency,
        })
        if (savedVersion) expect(body.backend_version).toBe(savedVersion)
        else expect(body).not.toHaveProperty('backend_version')
        expect(body).not.toHaveProperty('prefix')

        if (!response.ok() || result.status === 'failed') {
          const errorCode = result.error?.code ?? 'AISIM_NO_CONFIGURATION'
          await expect(page.getByText(predictErrorTitle(errorCode), { exact: true })).toBeVisible()
          if (!['OOM', 'AUTH_REQUIRED', 'MODEL_NOT_FOUND', 'AISIM_TIMEOUT'].includes(errorCode)) {
            const errorMessage = result.error?.message ?? result.detail ?? 'AISimulators prediction failed'
            await expect(page.getByText(errorMessage, { exact: true })).toBeVisible()
          }
          return
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
          const kvMbPerRequest = Math.round((result.memory_breakdown?.kv_cache_bytes ?? 0) / 1024 ** 3 / matrixConcurrencyPreset.concurrency * 1000)
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
      })
    }
  }
}

for (const model of appConfig.approvedModels) {
  for (const workload of workloads) {
    for (const systemId of systemsForModel(model)) {
      test(`KV cache / ${model} / ${workload.label} / ${systemId} / live`, async ({ page }, testInfo) => {
        const systems = await readAisimulateCatalog(page)
        const system = systems.find(item => item.id === systemId)
        expect(system, `The deployed catalog must contain ${systemId}`).toBeDefined()
        await page.goto('/kv-cache')
        const savedVersion = await page.evaluate(() => localStorage.getItem('settings_backend_version_v2:aisimulate'))
        await selectModel(page, model, 'kv-model')
        await selectSystem(page, 'kv-gpu', system!.name)
        await page.getByRole('button', { name: workload.label, exact: true }).click()
        await page.getByRole('button', { name: matrixConcurrencyPreset.label, exact: true }).click()
        const parallelism = modelParallelism[model] ?? 1
        if (parallelism > 1) {
          await page.getByRole('button', { name: 'Adjust? (edit fields below)', exact: true }).click()
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
          max_batch_size: matrixConcurrencyPreset.concurrency,
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
          await expect(page.getByRole('button', { name: /^Available for KV cache \/ GPU / }).locator('[class*="tileValue"]')).toHaveText(formatBytes(result.kvCache.totalBytes))
        }
        if (typeof result.kvCache?.perTokenBytes === 'number') {
          await expect(page.getByRole('button', { name: /^Per token / }).locator('[class*="tileValue"]')).toHaveText(formatBytes(result.kvCache.perTokenBytes))
        }
      })
    }
  }
}

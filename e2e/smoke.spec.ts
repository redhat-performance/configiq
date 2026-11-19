import { expect, test, type Page } from '@playwright/test'

// Synthetic catalog values for local smoke tests; they do not come from a live service.
const smokeCatalog = {
  timeoutSeconds: 1,
  simulators: [{
    id: 'aisimulate',
    label: 'AISimulate',
    default: true,
    available: true,
    models: ['smoke/model'],
    systems: [{
      id: 'smoke-gpu',
      name: 'Smoke GPU',
      vendor: 'nvidia',
      architecture: 'Hopper',
      memory_bytes: 85_899_345_920,
      memory_bandwidth_bytes: 3_000_000_000_000,
      bf16_tflops: 1_000,
      gpus_per_node: 8,
    }, {
      id: 'smoke-amd-gpu',
      name: 'AMD MI300X',
      vendor: 'amd',
      architecture: 'cdna3',
      memory_bytes: 206_158_430_208,
      memory_bandwidth_bytes: 5_300_000_000_000,
      bf16_tflops: 1_300,
      gpus_per_node: 8,
    }],
    backends: [{
      id: 'vllm',
      aisimulate_id: 'vllm',
      versions: ['smoke-1'],
      default_version: 'smoke-1',
      systems: { 'smoke-gpu': ['smoke-1'] },
      system_defaults: { 'smoke-gpu': { version: 'smoke-1', memory_fraction: 0.9 } },
    }],
    networks: [],
    objectives: ['min_gpus', 'pareto'],
  }, {
    id: 'blis',
    label: 'BLIS',
    default: false,
    available: true,
    models: ['smoke/model'],
    systems: [{
      id: 'smoke-gpu',
      name: 'Smoke GPU',
      vendor: 'nvidia',
      architecture: 'Hopper',
      memory_bytes: 85_899_345_920,
      memory_bandwidth_bytes: 3_000_000_000_000,
      bf16_tflops: 1_000,
      gpus_per_node: 8,
    }],
    backends: [{
      id: 'vllm',
      versions: ['blis-vllm-1'],
      default_version: 'blis-vllm-1',
      systems: { 'smoke-gpu': ['blis-vllm-1'] },
    }, {
      id: 'sglang',
      versions: ['smoke-sglang-1'],
      default_version: 'smoke-sglang-1',
      systems: { 'smoke-gpu': ['smoke-sglang-1'] },
    }],
    networks: [],
    objectives: ['min_gpus', 'pareto'],
  }],
}

interface SmokeApi {
  posts: Record<string, Record<string, unknown>>
}

async function stubApi(page: Page, responses: Record<string, unknown> = {}): Promise<SmokeApi> {
  const posts: SmokeApi['posts'] = {}

  await page.route('**/api/**', async (route) => {
    const { pathname } = new URL(route.request().url())
    let payload: unknown = responses[pathname] ?? {}

    if (route.request().method() === 'POST') {
      try {
        const body: unknown = route.request().postDataJSON()
        if (body && typeof body === 'object' && !Array.isArray(body)) {
          posts[pathname] = body as Record<string, unknown>
        }
      } catch {
        // Some API requests do not carry JSON bodies; only capture JSON posts.
      }
    }

    if (pathname === '/api/catalog') {
      payload = smokeCatalog
    } else if (pathname === '/api/tested-models') {
      payload = { available: true, pairs: [], approvedModels: [] }
    } else if (pathname === '/api/costings/offers') {
      payload = { schema_version: 1, rented: [], purchased: [], sources: {} }
    } else if (pathname.endsWith('/models')) {
      payload = { models: [], updated_at: null, stale: false }
    } else if (pathname.endsWith('/systems')) {
      payload = { systems: [] }
    } else if (pathname.endsWith('/health')) {
      payload = { sources: {} }
    }

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(payload),
    })
  })

  return { posts }
}

async function selectModel(page: Page, inputId: string): Promise<void> {
  const input = page.locator(`#${inputId}`)
  await expect(input).toBeVisible()
  if (await input.inputValue() === 'smoke/model') return

  const clearSelection = page.getByRole('button', { name: 'Clear selection', exact: true })
  if (await clearSelection.isVisible()) await clearSelection.click()
  await input.fill('smoke/model')
  await input.press('Enter')
  await expect(input).toHaveValue('smoke/model')
}

async function selectSystem(page: Page, inputId: string): Promise<void> {
  const field = page.locator(`label[for="${inputId}"]`).locator('..')
  const button = field.getByRole('button')
  await expect(button).not.toHaveText('Select GPU system…')
  await button.click()
  const option = page.getByRole('option', { name: /Smoke GPU/ })
  await expect(option).toBeVisible()
  await option.click()
  await expect(button).toContainText('Smoke GPU')
}

function inputNearLabel(page: Page, labelText: string) {
  return page.locator('label').filter({ hasText: labelText }).first().locator('..').locator('input')
}

// The mocked sizing result the recommend smoke tests render.
const recommendResponse = {
  requestId: 'smoke-recommend',
  status: 'completed',
  simulator: 'aisimulate',
  mode: 'agg',
  recommendation: {
    gpusNeeded: 2,
    gpusPerReplica: 2,
    totalGpus: 2,
    replicasNeeded: 1,
    tensorParallelSize: 2,
    pipelineParallelSize: 1,
    dataParallelSize: 1,
    contextParallelSize: 1,
    moeTensorParallelSize: null,
    moeExpertParallelSize: null,
    batchSize: 8,
  },
  phases: { prefill: null, decode: null, encode: null },
  performance: {
    ttftLatencyMs: 120,
    tpotMs: 25,
    requestLatencyMs: 3_320,
    concurrency: 1,
  },
  throughput: {
    tokensPerSecond: 38.5,
    tokensPerSecondPerGpu: 19.25,
    tokensPerSecondPerUser: 38.5,
  },
  memory: {
    value: 24,
    unit: 'GB',
    breakdown: { weightsGb: 16, kvCacheGb: 4, overheadGb: 4 },
  },
  memoryAvailable: true,
  metadata: {
    modelPath: 'smoke/model',
    system: 'smoke-gpu',
    inputTokens: 768,
    outputTokens: 96,
    targetTtftMs: 1_000,
    durationMs: 12,
  },
  warnings: [],
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    if (window.location.origin === 'null') return
    if (localStorage.getItem('settings_default_model') === null) localStorage.setItem('settings_default_model', 'smoke/model')
    if (localStorage.getItem('settings_default_system') === null) localStorage.setItem('settings_default_system', 'smoke-gpu')
    if (localStorage.getItem('settings_simulator') === null) localStorage.setItem('settings_simulator', 'aisimulate')
    if (localStorage.getItem('settings_inference_backend') === null) localStorage.setItem('settings_inference_backend', 'vllm')
    if (localStorage.getItem('hf_token') === null) localStorage.setItem('hf_token', 'hf_initial_smoke_token')
  })
})

test('home opens the core tools from the main navigation', async ({ page }) => {
  await stubApi(page)
  await page.goto('/')

  await expect(page.getByRole('heading', { name: 'ConfigIQ', level: 1 })).toBeVisible()

  const navigation = page.getByRole('navigation', { name: 'Main navigation' })
  for (const name of [
    'Recommend sizing', 'KV cache calculator', 'Predict performance', 'GPU explorer',
    'Benchmark comparison', 'Hybrid savings', 'Settings', 'Cost sources', 'Documentation',
  ]) {
    await expect(navigation.getByRole('link', { name, exact: true })).toBeVisible()
  }
})

const modelPages = [
  { path: '/predict', heading: 'Predict performance' },
  { path: '/recommend', heading: 'Recommend sizing' },
  { path: '/kv-cache', heading: 'KV cache calculator' },
] as const

for (const route of modelPages) {
  test(`${route.heading} loads with an editable model control`, async ({ page }) => {
    await stubApi(page)
    await page.goto(route.path)

    await expect(page.getByRole('heading', { name: route.heading, level: 1 })).toBeVisible()

    const modelInput = page.getByRole('combobox', { name: 'Model — Hugging Face ID' })
    await expect(modelInput).toBeEnabled()
    await expect(modelInput).toHaveValue('smoke/model')
    await modelInput.fill('smoke/edited-model')
    await expect(modelInput).toHaveValue('smoke/edited-model')
  })
}

const secondaryPages = [
  { path: '/hybrid-savings', heading: 'Hybrid savings' },
  { path: '/routing', heading: 'Routing economics' },
  { path: '/validation/compare', heading: 'Benchmark comparison' },
  { path: '/sources', heading: 'Cost sources' },
  { path: '/docs', heading: 'ConfigIQ documentation' },
] as const

for (const route of secondaryPages) {
  test(`${route.heading} page loads`, async ({ page }) => {
    await stubApi(page)
    await page.goto(route.path)
    await expect(page.getByRole('heading', { name: route.heading, level: 1 })).toBeVisible()
  })
}

test('GPU explorer loads with an available preset control', async ({ page }) => {
  await stubApi(page)
  await page.goto('/gpu-explorer')

  await expect(page.getByRole('heading', { name: 'GPU explorer', level: 1 })).toBeVisible()

  const performancePreset = page.getByRole('button', { name: 'Performance', exact: true })
  await expect(performancePreset).toBeEnabled()
  await performancePreset.click()
  await expect(performancePreset).toHaveAttribute('aria-pressed', 'true')

  const chart = page.locator('[data-tour="chart"] svg')
  await expect(chart.locator('circle')).toHaveCount(2)
  await page.getByText('AMD', { exact: true }).click()
  await expect(chart.locator('circle')).toHaveCount(1)
  await page.getByText('NVIDIA', { exact: true }).click()
  await expect(chart.locator('circle')).toHaveCount(1)
})

test('Settings loads with an editable token field', async ({ page }) => {
  await stubApi(page)
  await page.goto('/settings')

  await expect(page.getByRole('heading', { name: 'Settings', level: 1 })).toBeVisible()

  const tokenInput = page.getByRole('textbox', { name: 'Hugging Face API token' })
  await expect(tokenInput).toBeEnabled()
  await expect(tokenInput).toHaveValue('hf_initial_smoke_token')
  await tokenInput.fill('hf_smoke_token')
  await expect(tokenInput).toHaveValue('hf_smoke_token')
})

test('Settings enforces and persists simulator and backend selections', async ({ page }) => {
  await stubApi(page)
  await page.goto('/settings')

  const simulator = page.locator('#settings-simulator')
  const backend = page.locator('#settings-backend')
  await expect(simulator).toHaveValue('aisimulate')
  await expect(backend.locator('option[value="sglang"]')).toBeDisabled()

  await simulator.selectOption('blis')
  await expect(simulator).toHaveValue('blis')
  await expect(backend.locator('option[value="sglang"]')).toBeEnabled()
  await backend.selectOption('sglang')

  await page.getByRole('button', { name: 'Backend version' }).click()
  await page.getByRole('option', { name: 'smoke-sglang-1', exact: true }).click()
  await expect(backend).toHaveValue('sglang')
  await expect(page.getByRole('button', { name: 'Backend version' })).toHaveText('smoke-sglang-1')

  await page.reload()
  await expect(simulator).toHaveValue('blis')
  await expect(backend).toHaveValue('sglang')
  await expect(page.getByRole('button', { name: 'Backend version' })).toHaveText('smoke-sglang-1')
})

test('Predict submits selected workload inputs and renders the mocked estimate', async ({ page }) => {
  const api = await stubApi(page, {
    '/api/predict': {
      status: 'success',
      simulator: 'aisimulate',
      memory_breakdown: {
        weights_bytes: 16 * 1_073_741_824,
        kv_cache_bytes: 2 * 1_073_741_824,
      },
      ttft: 640,
      tpot: 24,
      tokens_per_second: 37.8,
      serving_config: {
        tensor_parallel_size: 1,
        max_model_len: 864,
        max_num_seqs: 32,
        gpu_memory_utilization: 0.9,
        enable_chunked_prefill: false,
        enable_prefix_caching: false,
        quantization: 'auto',
      },
    },
  })
  await page.goto('/predict')
  await selectModel(page, 'predict-performance-model')
  await selectSystem(page, 'predict-performance-gpu')

  await page.getByRole('button', { name: /^Adjust\?/ }).click()
  await page.getByLabel('Input sequence length (ISL)').fill('768')
  await page.getByLabel('Output sequence length (OSL)').fill('96')

  const responsePromise = page.waitForResponse(response =>
    new URL(response.url()).pathname === '/api/predict' && response.request().method() === 'POST',
  )
  await page.getByRole('button', { name: 'Calculate', exact: true }).click()
  await responsePromise

  expect(api.posts['/api/predict']).toMatchObject({
    model_path: 'smoke/model',
    system: 'smoke-gpu',
    simulator: 'aisimulate',
    backend: 'vllm',
    isl: 768,
    osl: 96,
  })
  await expect(page.getByTestId('predict-ttft')).toHaveText('640.0 ms')
  await expect(page.getByTestId('predict-tpot')).toHaveText('24.0 ms')
  await expect(page.getByTestId('predict-throughput')).toContainText('37.8 tok/s')
})

test('Recommend submits selected workload inputs and renders the mocked sizing result', async ({ page }) => {
  const api = await stubApi(page, {
    '/api/recommend': recommendResponse,
  })
  await page.goto('/recommend')
  await selectModel(page, 'adv-model')
  await selectSystem(page, 'adv-gpu')
  await page.getByRole('button', { name: 'Latency SLO', exact: true }).click()

  await page.getByRole('button', { name: /^Adjust\?/ }).click()
  await inputNearLabel(page, 'Avg input tokens (ISL)').fill('768')
  await inputNearLabel(page, 'Avg output tokens (OSL)').fill('96')

  const responsePromise = page.waitForResponse(response =>
    new URL(response.url()).pathname === '/api/recommend' && response.request().method() === 'POST',
  )
  await page.getByRole('button', { name: 'Calculate', exact: true }).click()
  await responsePromise

  expect(api.posts['/api/recommend']).toMatchObject({
    model_path: 'smoke/model',
    system: 'smoke-gpu',
    simulator: 'aisimulate',
    backend: 'vllm',
    isl: 768,
    osl: 96,
    objective: 'min_gpus',
  })
  await expect(page.getByTestId('recommend-gpus')).toContainText('2')
  await expect(page.getByTestId('recommend-ttft')).toContainText('120')
  await expect(page.getByTestId('recommend-throughput')).toHaveText('39tok/s')
  await expect(page.getByTestId('recommend-peak-memory')).toContainText('24')
})

test('Recommend submits the deployment view the control shows', async ({ page }) => {
  const api = await stubApi(page, { '/api/recommend': recommendResponse })
  await page.goto('/recommend')
  await selectModel(page, 'adv-model')
  await selectSystem(page, 'adv-gpu')
  await page.getByRole('button', { name: 'Latency SLO', exact: true }).click()
  await page.getByRole('button', { name: /^Adjust\?/ }).click()

  const calculate = async () => {
    const responsePromise = page.waitForResponse(response =>
      new URL(response.url()).pathname === '/api/recommend' && response.request().method() === 'POST',
    )
    await page.getByRole('button', { name: 'Calculate', exact: true }).click()
    await responsePromise
    return api.posts['/api/recommend']
  }

  // Untouched, the page pins no view: it must not narrow the search, so the
  // service picks the deployment. Before a run the summary claims no view.
  await expect(page.getByText(/Deployment view: service default/)).toBeVisible()
  const untouched = await calculate()
  expect(untouched).toMatchObject({ objective: 'min_gpus' })
  expect(untouched).not.toHaveProperty('mode')
  // The mocked service answered with an aggregated recommendation, so the
  // summary reports the deployment that was served.
  await expect(page.getByText(/Deployment view: aggregated/)).toBeVisible()

  await page.getByRole('button', { name: 'Aggregated', exact: true }).click()
  const aggregated = await calculate()
  expect(aggregated).toMatchObject({ mode: 'agg', objective: 'min_gpus' })
  // An aggregated view has no pools to limit and must not claim a topology.
  expect(aggregated).not.toHaveProperty('prefill_max_seq_len')
  expect(aggregated).not.toHaveProperty('decode_max_seq_len')
  await expect(page.getByText(/Deployment view: aggregated/)).toBeVisible()

  await page.getByRole('button', { name: 'Disaggregated', exact: true }).click()
  // Before the run the summary states the view the next request pins.
  await expect(page.getByText(/Deployment view: disaggregated/)).toBeVisible()
  expect((await calculate())).toMatchObject({ mode: 'disagg', objective: 'min_gpus' })
  await expect(page.getByText('Prefill and decode have separate inputs in this view.')).toBeVisible()
  // The mocked service answers with an aggregated recommendation, so the
  // summary must report what was returned rather than what was asked for.
  await expect(page.getByText(/Deployment view: aggregated/)).toBeVisible()
  await expect(page.getByText('The last run returned an aggregated recommendation instead.')).toBeVisible()
})

test('KV cache submits selected capacity inputs and renders the mocked result', async ({ page }) => {
  const api = await stubApi(page, {
    '/api/memory': {
      requestId: 'smoke-kv-cache',
      status: 'completed',
      kvCache: {
        totalBytes: 1_073_741_824,
        perTokenBytes: 262_144,
        totalTokens: 4_096,
      },
      memoryBreakdown: {
        weightsBytes: 16 * 1_073_741_824,
        activationsBytes: 1_073_741_824,
        runtimeOverheadBytes: 2 * 1_073_741_824,
        commOverheadBytes: 0,
      },
      gpuCapacity: {
        totalBytes: 85_899_345_920,
      },
      metadata: {
        simulator: 'aisimulate',
        modelPath: 'smoke/model',
        backend: 'vllm',
        backendVersion: null,
        system: 'smoke-gpu',
        maxNumTokens: 4_096,
        maxBatchSize: 8,
        tpSize: 1,
        ppSize: 1,
        moeTpSize: null,
        moeEpSize: null,
        memoryFractionKind: 'of_total',
        memoryFractionValue: 0.9,
        source: 'smoke-fixture',
        durationMs: 12,
      },
    },
  })
  await page.goto('/kv-cache')
  await selectModel(page, 'kv-model')
  await selectSystem(page, 'kv-gpu')

  await page.getByRole('button', { name: 'Adjust? (edit fields below)', exact: true }).click()
  await page.getByLabel('Max num tokens').fill('4096')
  await page.getByLabel('Max batch size').fill('8')

  const responsePromise = page.waitForResponse(response =>
    new URL(response.url()).pathname === '/api/memory' && response.request().method() === 'POST',
  )
  await page.getByRole('button', { name: 'Calculate', exact: true }).click()
  await responsePromise

  expect(api.posts['/api/memory']).toMatchObject({
    model_path: 'smoke/model',
    system: 'smoke-gpu',
    backend: 'vllm',
    max_num_tokens: 4_096,
    max_batch_size: 8,
  })
  await expect(page.getByTestId('kv-total-value')).toHaveText('1 GiB')
  await expect(page.getByTestId('kv-tokens-value')).toHaveText('4,096')
})

test('Predict shows a service error when the estimate request fails', async ({ page }) => {
  const api = await stubApi(page, {
    '/api/predict': {
      status: 'failed',
      error: {
        code: 'AISIM_UNAVAILABLE',
        message: 'Mock prediction service unavailable',
      },
    },
  })
  await page.goto('/predict')
  await selectModel(page, 'predict-performance-model')
  await selectSystem(page, 'predict-performance-gpu')

  const responsePromise = page.waitForResponse(response =>
    new URL(response.url()).pathname === '/api/predict' && response.request().method() === 'POST',
  )
  await page.getByRole('button', { name: 'Calculate', exact: true }).click()
  await responsePromise

  expect(api.posts['/api/predict']).toMatchObject({ model_path: 'smoke/model', system: 'smoke-gpu' })
  await expect(page.getByText('Sizing service unavailable', { exact: true })).toBeVisible()
  await expect(page.getByText('Mock prediction service unavailable', { exact: true })).toBeVisible()
  await expect(page.getByTestId('predict-ttft')).toHaveCount(0)
})

test('Recommend shows a service error when sizing fails', async ({ page }) => {
  const api = await stubApi(page, {
    '/api/recommend': {
      requestId: 'smoke-recommend-failure',
      status: 'failed',
      error: {
        code: 'AISIM_UNAVAILABLE',
        message: 'Mock recommendation service unavailable',
      },
    },
  })
  await page.goto('/recommend')
  await selectModel(page, 'adv-model')
  await selectSystem(page, 'adv-gpu')
  await page.getByRole('button', { name: 'Latency SLO', exact: true }).click()

  const responsePromise = page.waitForResponse(response =>
    new URL(response.url()).pathname === '/api/recommend' && response.request().method() === 'POST',
  )
  await page.getByRole('button', { name: 'Calculate', exact: true }).click()
  await responsePromise

  expect(api.posts['/api/recommend']).toMatchObject({
    model_path: 'smoke/model',
    system: 'smoke-gpu',
    objective: 'min_gpus',
  })
  await expect(page.getByText('Sizing service unavailable', { exact: true })).toBeVisible()
  await expect(page.getByText(
    'The AISimulators service is temporarily unreachable. This is usually a transient issue.',
    { exact: true },
  )).toBeVisible()
  await expect(page.getByTestId('recommend-gpus')).toHaveCount(0)
})

test('KV cache shows a service error when the calculation fails', async ({ page }) => {
  const api = await stubApi(page, {
    '/api/memory': {
      requestId: 'smoke-kv-cache-failure',
      status: 'failed',
      error: {
        code: 'AISIM_UNAVAILABLE',
        message: 'Mock KV cache service unavailable',
      },
    },
  })
  await page.goto('/kv-cache')
  await selectModel(page, 'kv-model')
  await selectSystem(page, 'kv-gpu')

  const responsePromise = page.waitForResponse(response =>
    new URL(response.url()).pathname === '/api/memory' && response.request().method() === 'POST',
  )
  await page.getByRole('button', { name: 'Calculate', exact: true }).click()
  await responsePromise

  expect(api.posts['/api/memory']).toMatchObject({ model_path: 'smoke/model', system: 'smoke-gpu' })
  await expect(page.getByRole('heading', { name: /Calculation failed/ })).toBeVisible()
  await expect(page.getByText('Mock KV cache service unavailable', { exact: true })).toBeVisible()
  await expect(page.getByTestId('kv-total-value')).toHaveCount(0)
})

test('Predict blocks calculation when input tokens are zero', async ({ page }) => {
  const api = await stubApi(page)
  await page.goto('/predict')
  await selectModel(page, 'predict-performance-model')
  await selectSystem(page, 'predict-performance-gpu')

  await page.getByRole('button', { name: /^Adjust\?/ }).click()
  await page.getByLabel('Input sequence length (ISL)').fill('0')

  await expect(page.getByRole('button', { name: 'Calculate', exact: true })).toBeDisabled()
  expect(api.posts['/api/predict']).toBeUndefined()
})

test('Recommend blocks calculation when input tokens are zero', async ({ page }) => {
  const api = await stubApi(page)
  await page.goto('/recommend')
  await selectModel(page, 'adv-model')
  await selectSystem(page, 'adv-gpu')
  await page.getByRole('button', { name: 'Latency SLO', exact: true }).click()

  await page.getByRole('button', { name: /^Adjust\?/ }).click()
  await inputNearLabel(page, 'Avg input tokens (ISL)').fill('0')

  await expect(page.getByRole('button', { name: 'Calculate', exact: true })).toBeDisabled()
  expect(api.posts['/api/recommend']).toBeUndefined()
})

test('KV cache blocks calculation when batch size is zero', async ({ page }) => {
  const api = await stubApi(page)
  await page.goto('/kv-cache')
  await selectModel(page, 'kv-model')
  await selectSystem(page, 'kv-gpu')

  await page.getByRole('button', { name: 'Adjust? (edit fields below)', exact: true }).click()
  await page.getByLabel('Max batch size').fill('0')

  await expect(page.getByRole('button', { name: 'Calculate', exact: true })).toBeDisabled()
  expect(api.posts['/api/memory']).toBeUndefined()
})

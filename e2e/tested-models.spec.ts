import { test } from '@playwright/test'
import appConfig from '../public/config.json'
import { checkKvCache, checkPredict } from './live-ui'
import { concurrencyCasesForModel, modelParallelism } from './test-matrix'

// Each model/workload/GPU combination is independent, allowing CI sharding.
test.describe.configure({ mode: 'parallel' })

for (const model of appConfig.approvedModels) {
  for (const { workload, systemId, concurrency } of concurrencyCasesForModel(model, 0)) {
    test(`Predict performance / ${model} / ${workload.label} / ${systemId} / ${concurrency.label} / live`, async ({ page }, testInfo) => {
      await checkPredict(page, testInfo, model, workload, systemId, concurrency, modelParallelism[model] ?? 1)
    })
  }
}

for (const model of appConfig.approvedModels) {
  for (const { workload, systemId, concurrency } of concurrencyCasesForModel(model, 4)) {
    test(`KV cache / ${model} / ${workload.label} / ${systemId} / ${concurrency.label} / live`, async ({ page }, testInfo) => {
      await checkKvCache(page, testInfo, model, workload, systemId, concurrency, modelParallelism[model] ?? 1)
    })
  }
}

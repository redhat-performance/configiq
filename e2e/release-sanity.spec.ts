import { test } from '@playwright/test'
import appConfig from '../public/config.json'
import { DEFAULT_WORKLOAD } from '../lib/workload-presets'
import { checkKvCache, checkPredict, checkRecommend, goalLabels, type RecommendScenario } from './live-ui'
import { baselineConcurrencyPreset, modelParallelism, sampledSystemForModel } from './test-matrix'

// Deployment sanity for the release-tracking host. It runs one workload on one
// sampled GPU per approved model across all three tools, plus one case for each
// remaining Recommend control, so a bad image, stale catalog, or broken deploy
// fails quickly without putting the full matrix on the production service.
// Breadth belongs to the dev suites; this suite stays small on purpose.
test.describe.configure({ mode: 'parallel' })

const sanityWorkload = DEFAULT_WORKLOAD

for (const model of appConfig.approvedModels) {
  const systemId = sampledSystemForModel(model)
  const parallelism = modelParallelism[model] ?? 1

  test(`Predict performance / ${model} / ${sanityWorkload.label} / ${systemId} / ${baselineConcurrencyPreset.label} / sanity`, async ({ page }, testInfo) => {
    await checkPredict(page, testInfo, model, sanityWorkload, systemId, baselineConcurrencyPreset, parallelism)
  })

  test(`KV cache / ${model} / ${sanityWorkload.label} / ${systemId} / ${baselineConcurrencyPreset.label} / sanity`, async ({ page }, testInfo) => {
    await checkKvCache(page, testInfo, model, sanityWorkload, systemId, baselineConcurrencyPreset, parallelism)
  })

  test(`Recommend sizing / ${model} / ${sanityWorkload.label} / ${systemId} / ${goalLabels.latency} / sanity`, async ({ page }, testInfo) => {
    await checkRecommend(page, testInfo, model, sanityWorkload, systemId, {
      action: 'sizing',
      goal: 'latency',
      concurrency: baselineConcurrencyPreset,
    })
  })
}

// One case per remaining Recommend control, spread across the first approved
// models so the suite does not grow with the model list.
const extraScenarios: RecommendScenario[] = [
  { action: 'sizing', goal: 'capacity', concurrency: baselineConcurrencyPreset },
  { action: 'sizing', goal: 'demand' },
  { action: 'pareto', goal: 'latency' },
]

for (const [index, scenario] of extraScenarios.entries()) {
  const model = appConfig.approvedModels[index]
  if (!model) continue
  const systemId = sampledSystemForModel(model)
  const scenarioLabel = `${goalLabels[scenario.goal]}${scenario.action === 'pareto' ? ' Pareto' : ''}${scenario.concurrency ? ` / ${scenario.concurrency.label}` : ''}`
  test(`Recommend sizing / ${model} / ${sanityWorkload.label} / ${systemId} / ${scenarioLabel} / sanity`, async ({ page }, testInfo) => {
    await checkRecommend(page, testInfo, model, sanityWorkload, systemId, scenario)
  })
}

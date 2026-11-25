import { expect, test, type Page, type TestInfo } from '@playwright/test'
import appConfig from '../public/config.json'
import { DEFAULT_WORKLOAD } from '../lib/workload-presets'
import { readSimulatorCatalogs } from './live-catalog'
import { checkKvCache, checkPredict, checkRecommend, goalLabels, type CaseOptions, type RecommendScenario } from './live-ui'
import { baselineConcurrencyPreset, modelParallelism, parityModel, paritySystem, sampledSystemForModel } from './test-matrix'

// Deployment sanity for the release-tracking host. It runs one workload on one
// sampled GPU per approved model across all three tools, plus one case for each
// remaining Recommend control, so a bad image, stale catalog, or broken deploy
// fails quickly without putting the full matrix on the production service.
// Breadth belongs to the dev suites; this suite stays small on purpose.
test.describe.configure({ mode: 'parallel' })

const sanityWorkload = DEFAULT_WORKLOAD

// A sanity run exists to prove the deployed release produces results. An API
// error here is a failed release check, not an accepted unsupported pairing.
const requireSuccess: CaseOptions = { requireSuccess: true }

type SanityTool = 'predict' | 'kv' | 'recommend'

// Exceptions to that rule, one reviewable entry at a time. The case still runs
// and still compares the UI with the API response; it accepts that model's one
// known failure instead of demanding a result. Any other error still fails the
// run, so an outage cannot pass as a known gap. Remove an entry once the
// deployment can serve the model, and the case tightens automatically.
const knownUnavailable: Record<string, {
  reason: string
  failures: Record<SanityTool, NonNullable<CaseOptions['expectFailure']>>
}> = {
  // Gated Hugging Face repository: the sizing service cannot download
  // config.json without a Hub token, on every deployment. Predict and sizing
  // report the credential failure; the memory endpoint reports the same model
  // as having no performance data, so its code is the pin there.
  'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block': {
    reason: 'gated model without Hub credentials',
    failures: {
      predict: { code: 'AUTH_REQUIRED', messageIncludes: 'HuggingFace returned HTTP error 401' },
      kv: { code: 'AISIM_UNSUPPORTED' },
      recommend: { code: 'AISIM_UNAVAILABLE', messageIncludes: 'HuggingFace returned HTTP error 401' },
    },
  },
}

function optionsFor(model: string, tool: SanityTool, testInfo: TestInfo): CaseOptions {
  const known = knownUnavailable[model]
  if (!known) return requireSuccess
  testInfo.annotations.push({ type: 'known-unavailable', description: `${model}: ${known.reason}` })
  return { expectFailure: known.failures[tool] }
}

for (const model of appConfig.approvedModels) {
  const systemId = sampledSystemForModel(model)
  const parallelism = modelParallelism[model] ?? 1

  test(`Predict performance / ${model} / ${sanityWorkload.label} / ${systemId} / ${baselineConcurrencyPreset.label} / sanity`, async ({ page }, testInfo) => {
    await checkPredict(page, testInfo, model, sanityWorkload, systemId, baselineConcurrencyPreset, parallelism, optionsFor(model, 'predict', testInfo))
  })

  test(`KV cache / ${model} / ${sanityWorkload.label} / ${systemId} / ${baselineConcurrencyPreset.label} / sanity`, async ({ page }, testInfo) => {
    await checkKvCache(page, testInfo, model, sanityWorkload, systemId, baselineConcurrencyPreset, parallelism, optionsFor(model, 'kv', testInfo))
  })

  test(`Recommend sizing / ${model} / ${sanityWorkload.label} / ${systemId} / ${goalLabels.latency} / sanity`, async ({ page }, testInfo) => {
    await checkRecommend(page, testInfo, model, sanityWorkload, systemId, {
      action: 'sizing',
      goal: 'latency',
      concurrency: baselineConcurrencyPreset,
    }, optionsFor(model, 'recommend', testInfo))
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
    await checkRecommend(page, testInfo, model, sanityWorkload, systemId, scenario, requireSuccess)
  })
}

// The cases above run against the profile a fresh browser starts with: the
// catalog's default simulator. Every other simulator the deployment advertises
// is selectable from Settings and must answer the same pages with its own
// result. Each case names the simulator it selected and checks both the request
// and the response, so an answer from another simulator fails rather than
// passing as a fallback. A simulator the deployment does not advertise cannot
// be selected, so there is nothing to check. The pairing comes from that
// simulator's own catalogue, not the tested-model matrix.
async function advertisedAlternates(page: Page) {
  const alternates = (await readSimulatorCatalogs(page)).filter(entry => !entry.default)
  if (alternates.length === 0) test.skip(true, 'The deployment advertises only its default simulator')
  for (const entry of alternates) {
    expect(entry.available, `${entry.label} is advertised but its capabilities could not be read`).toBe(true)
  }
  return alternates
}

// Each selected simulator drives the page once, and the per-case timeout is
// sized for one run. Give the step loop room for every alternates' case instead
// of letting the second one inherit a budget the first one already spent.
const parityBudget = (count: number) => test.setTimeout(300_000 * Math.max(1, count))

test('Predict performance / every advertised simulator / parity', async ({ page }, testInfo) => {
  const alternates = await advertisedAlternates(page)
  parityBudget(alternates.length)
  for (const entry of alternates) {
    await test.step(entry.label, async () => {
      const model = parityModel(entry.models)
      const parallelism = modelParallelism[model] ?? 1
      await checkPredict(page, testInfo, model, sanityWorkload, paritySystem(entry.systems), baselineConcurrencyPreset, parallelism, {
        requireSuccess: true,
        simulator: entry.id,
      })
    })
  }
})

test('KV cache / every advertised simulator / parity', async ({ page }, testInfo) => {
  const alternates = await advertisedAlternates(page)
  parityBudget(alternates.length)
  for (const entry of alternates) {
    await test.step(entry.label, async () => {
      const model = parityModel(entry.models)
      const parallelism = modelParallelism[model] ?? 1
      await checkKvCache(page, testInfo, model, sanityWorkload, paritySystem(entry.systems), baselineConcurrencyPreset, parallelism, {
        requireSuccess: true,
        simulator: entry.id,
      })
    })
  }
})

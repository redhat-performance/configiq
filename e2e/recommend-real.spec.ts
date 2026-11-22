import { test } from '@playwright/test'
import appConfig from '../public/config.json'
import type { WorkloadPreset } from '../lib/workload-presets'
import { checkRecommend, goalLabels, type RecommendScenario } from './live-ui'
import { baselineConcurrencyPreset, matrixConcurrencyPresets, scheduledCoordinates, scheduledMatrix, systemsForModel, workloads } from './test-matrix'

test.describe.configure({ mode: 'parallel' })

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

for (const model of appConfig.approvedModels) {
  for (const { workload, systemId, scenario } of recommendCasesForModel(model)) {
    const scenarioLabel = `${goalLabels[scenario.goal]}${scenario.action === 'pareto' ? ' Pareto' : ''}${scenario.concurrency ? ` / ${scenario.concurrency.label}` : ''}`
    test(`Recommend sizing / ${model} / ${workload.label} / ${systemId} / ${scenarioLabel} / live`, async ({ page }, testInfo) => {
      await checkRecommend(page, testInfo, model, workload, systemId, scenario)
    })
  }
}

import { describe, expect, it } from 'vitest'
import { modelStatus, parseRegistry, systemStatus, validatedModelIds } from '../tested-models/registry-types'
import { testedModelsFilterHref, testedPairHref } from '../tested-models/links'

const gate = { minRecords: 100, minTestAuc: 0.7, minTestF1: 0.3, minConcurrencyLevels: 5 }
const coverage = { records: 600, concurrencyLevels: [1, 2, 4, 8, 16], concurrencyLevelCount: 5, backendVersions: ['vLLM-0.10.2'] }

const validatedPair = {
  id: 'model__h200',
  modelId: 'org/model',
  systemId: 'h200_sxm',
  status: 'validated',
  coverage,
  dashboardPath: 'dashboard/model__h200.json',
  gate: { passed: true, reasons: [], thresholds: gate },
  classifier: 'xgboost',
  tritonModel: 'model__h200',
  featureSchema: { feature_names: ['tp'], numeric_medians: {} },
  metrics: { test: { roc_auc: 0.88, f1: 0.66 } },
}

const measuredPair = {
  id: 'model__b200',
  modelId: 'org/model',
  systemId: 'b200_sxm',
  status: 'measured_no_envelope',
  coverage: { ...coverage, records: 40, concurrencyLevelCount: 2 },
  dashboardPath: 'dashboard/model__b200.json',
  gate: { passed: false, reasons: ['test_auc_below_minimum'], thresholds: gate },
  diagnostics: { trainingStatus: 'trained', reasons: ['test_auc_below_minimum'] },
}

describe('parseRegistry', () => {
  it('normalizes schema v2 and keeps measured-only pairs separate', () => {
    const registry = parseRegistry({
      schemaVersion: 2,
      generatedAt: '2026-01-01T00:00:00Z',
      gate,
      approvedModels: ['org/model', 'org/approved-only'],
      pairs: [validatedPair, measuredPair],
    })

    expect(registry.schemaVersion).toBe(2)
    expect(registry.gate).toEqual(gate)
    expect(registry.approvedModels).toEqual(['org/model', 'org/approved-only'])
    expect(registry.pairs[0]).toMatchObject({ status: 'validated', legacy: false })
    expect(registry.pairs[1]).toMatchObject({ status: 'measured_no_envelope', legacy: false })
    expect(registry.pairs[1].classifier).toBeUndefined()
  })

  it('downgrades a legacy pair unless its recorded test metrics clear the gate', () => {
    const registry = parseRegistry({
      schemaVersion: 1,
      generatedAt: '2026-01-01T00:00:00Z',
      pairs: [
        { ...validatedPair, legacy: undefined },
        {
          id: 'weak__h200', modelId: 'org/weak', systemId: 'h200_sxm', status: undefined,
          classifier: 'xgboost', tritonModel: 'weak__h200',
          featureSchema: { feature_names: ['tp'], numeric_medians: {} },
          metrics: { test: { roc_auc: 0.27, f1: 0 } },
        },
      ],
    })

    expect(registry.gate).toBeNull()
    expect(registry.pairs[0]).toMatchObject({ status: 'validated', legacy: true })
    expect(registry.pairs[1]).toMatchObject({ status: 'measured_no_envelope', legacy: true })
    expect(registry.pairs[1].gate.reasons).toEqual(['test_auc_below_minimum', 'test_f1_below_minimum'])
  })

  it('rejects unknown schema versions', () => {
    expect(() => parseRegistry({ schemaVersion: 3, pairs: [] })).toThrow('unsupported registry schema')
  })

  it('rejects a validated v2 pair without a feature schema', () => {
    expect(() => parseRegistry({
      schemaVersion: 2,
      gate,
      approvedModels: [],
      pairs: [{ ...validatedPair, featureSchema: undefined }],
    })).toThrow('featureSchema')
  })
})

describe('registry status helpers', () => {
  const registry = parseRegistry({
    schemaVersion: 2,
    gate,
    approvedModels: ['org/model', 'org/approved-only'],
    pairs: [validatedPair, measuredPair],
  })

  it('reports validated, measured, approved-no-data, and untested distinctly', () => {
    expect(modelStatus('org/model', registry)).toBe('validated')
    expect(modelStatus('org/approved-only', registry)).toBe('approved_no_data')
    expect(modelStatus('org/unknown', registry)).toBe('untested')
    expect(validatedModelIds(registry)).toEqual(['org/model'])
  })

  it('derives system status per model', () => {
    expect(systemStatus('h200_sxm', registry.pairs, 'org/model')).toBe('validated')
    expect(systemStatus('b200_sxm', registry.pairs, 'org/model')).toBe('measured_no_envelope')
    expect(systemStatus('mi355x', registry.pairs, 'org/model')).toBe('untested')
  })
})

describe('tested-model links', () => {
  it('builds canonical pair links and filter links', () => {
    expect(testedPairHref('model__h200')).toBe('/tested-models/model__h200')
    expect(testedPairHref('model__h200', 'boundary')).toBe('/tested-models/model__h200#boundary')
    expect(testedModelsFilterHref({ modelId: 'org/model', systemId: 'h200_sxm' }))
      .toBe('/tested-models?model=org%2Fmodel&system=h200_sxm')
  })
})

import { afterEach, describe, expect, it, vi } from 'vitest'
import { readFile } from 'node:fs/promises'
import { GET } from './route'

vi.mock('node:fs/promises', () => ({ readFile: vi.fn() }))

const mockReadFile = vi.mocked(readFile)

afterEach(() => {
  vi.resetAllMocks()
  vi.unstubAllEnvs()
})

describe('GET /api/tested-models', () => {
  it('returns public v2 pair metadata without Triton implementation names', async () => {
    mockReadFile.mockResolvedValue(JSON.stringify({
      schemaVersion: 2,
      generatedAt: '2026-01-01T00:00:00Z',
      dataset: { repo: 'org/ground-truth', revision: 'a'.repeat(40) },
      gate: { minRecords: 100, minTestAuc: 0.7, minTestF1: 0.3, minConcurrencyLevels: 5 },
      approvedModels: ['org/model', 'org/approved-only'],
      pairs: [{
        id: 'model__h200', modelId: 'org/model', systemId: 'h200_sxm',
        status: 'validated', classifier: 'xgboost', tritonModel: 'internal_name',
        dashboardPath: 'dashboard/model__h200.json',
        featureSchema: { feature_names: ['tp'], numeric_medians: {} },
        coverage: { records: 600, concurrencyLevels: [1, 2, 4, 8, 16], concurrencyLevelCount: 5, backendVersions: ['vLLM-0.10.2'] },
        gate: { passed: true, reasons: [], thresholds: { minRecords: 100, minTestAuc: 0.7, minTestF1: 0.3, minConcurrencyLevels: 5 } },
      }],
    }) as never)

    const response = await GET()
    const body = await response.json()

    expect(body.status).toBe('available')
    expect(body.schemaVersion).toBe(2)
    expect(body.dataset).toEqual({ repo: 'org/ground-truth', revision: 'a'.repeat(40) })
    expect(body.approvedModels).toEqual(['org/model', 'org/approved-only'])
    expect(body.pairs[0]).toMatchObject({
      id: 'model__h200',
      modelId: 'org/model',
      systemId: 'h200_sxm',
      status: 'validated',
      dashboardPath: 'dashboard/model__h200.json',
    })
    expect(JSON.stringify(body)).not.toContain('tritonModel')
    expect(JSON.stringify(body)).not.toContain('xgboost')
  })

  it('downgrades a legacy pair without passing test metrics and still hides implementation names', async () => {
    mockReadFile.mockResolvedValue(JSON.stringify({
      schemaVersion: 1,
      generatedAt: '2026-01-01T00:00:00Z',
      pairs: [{
        id: 'model__h200', modelId: 'org/model', systemId: 'h200_sxm',
        classifier: 'xgboost', tritonModel: 'internal_name',
        dashboardPath: 'dashboard/model__h200.json',
      }],
    }) as never)

    const response = await GET()
    const body = await response.json()

    expect(body.status).toBe('available')
    expect(body.schemaVersion).toBe(1)
    expect(body.gate).toBeNull()
    expect(body.pairs[0]).toMatchObject({ id: 'model__h200', status: 'measured_no_envelope', legacy: true })
    expect(JSON.stringify(body)).not.toContain('tritonModel')
  })

  it('returns an unavailable empty response when the registry is absent', async () => {
    mockReadFile.mockRejectedValue(new Error('missing'))
    const response = await GET()
    expect(response.status).toBe(200)
    expect(await response.json()).toMatchObject({ status: 'unavailable', pairs: [], approvedModels: [] })
  })
})

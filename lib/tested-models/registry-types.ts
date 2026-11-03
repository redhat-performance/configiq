/**
 * Tested-model registry contracts shared by the API routes and the browser.
 *
 * The registry is the source of truth for tested status. Schema v2 distinguishes
 * a pair with a validated classifier from a pair that only has measurements.
 * Schema v1 is still accepted during rollout, but a v1 pair is only presented
 * as validated when its recorded test metrics clear the gate; v1 has no
 * coverage data, so it can never claim the full v2 gate.
 */

import type { FeatureSchema } from './feature-vector'

export type RegistryPairStatus = 'validated' | 'measured_no_envelope'

export interface ValidationGateThresholds {
  minRecords: number
  minTestAuc: number
  minTestF1: number
  minConcurrencyLevels: number
}

export interface RegistryCoverage {
  records: number
  concurrencyLevels: number[]
  concurrencyLevelCount: number
  /** Raw version labels, kept for provenance. */
  backendVersions: string[]
  /** Runtime families the pair was measured on, e.g. `vllm`, `sglang`. */
  backends: string[]
}

export interface RegistryGateResult {
  passed: boolean
  reasons: string[]
  thresholds: ValidationGateThresholds | null
}

export interface RegistryDiagnostics {
  trainingStatus: string
  reasons: string[]
  /** Data-shape caveats, such as a single-class validation or test split. */
  warnings?: string[]
  /** Which split picked the hyperparameters: `validation` or `train_fallback`. */
  selectionBasis?: string
}

export interface RegistryPair {
  id: string
  modelId: string
  systemId: string
  status: RegistryPairStatus
  /** Null only for legacy v1 pairs, which carried no coverage summary. */
  coverage: RegistryCoverage | null
  dashboardPath: string | null
  gate: RegistryGateResult
  diagnostics?: RegistryDiagnostics
  /** Present only for validated pairs. */
  classifier?: string
  tritonModel?: string
  metrics?: unknown
  thresholds?: unknown
  featureSchema?: FeatureSchema
  artifactVersion?: string
  legacy: boolean
}

export interface TestedModelsRegistry {
  schemaVersion: 1 | 2
  generatedAt: string | null
  gate: ValidationGateThresholds | null
  approvedModels: string[]
  pairs: RegistryPair[]
}

/**
 * Fields the pair graph page is allowed to send to the browser. Classifier
 * implementation details (Triton model name, feature schema, thresholds) stay
 * server-side, matching the public registry API.
 */
export type PublicRegistryPair = Pick<
  RegistryPair,
  'id' | 'modelId' | 'systemId' | 'status' | 'coverage' | 'diagnostics' | 'metrics'
>

export const DEFAULT_GATE: ValidationGateThresholds = {
  minRecords: 100,
  minTestAuc: 0.7,
  minTestF1: 0.3,
  minConcurrencyLevels: 5,
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function stringOrNull(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null
}

function requireString(record: Record<string, unknown>, key: string, label: string): string {
  const value = stringOrNull(record[key])
  if (value === null) throw new Error(`${label}.${key} must be a non-empty string`)
  return value
}

function parseGate(raw: unknown): ValidationGateThresholds {
  const gate = asRecord(raw)
  if (!gate) throw new Error('registry gate must be an object')
  const result = {
    minRecords: numberOrNull(gate.minRecords),
    minTestAuc: numberOrNull(gate.minTestAuc),
    minTestF1: numberOrNull(gate.minTestF1),
    minConcurrencyLevels: numberOrNull(gate.minConcurrencyLevels),
  }
  if (Object.values(result).some(value => value === null)) {
    throw new Error('registry gate is incomplete')
  }
  return result as ValidationGateThresholds
}

function parseCoverage(raw: unknown, pairId: string): RegistryCoverage {
  const coverage = asRecord(raw)
  if (!coverage) throw new Error(`${pairId}.coverage must be an object`)
  const records = numberOrNull(coverage.records)
  const concurrencyLevelCount = numberOrNull(coverage.concurrencyLevelCount)
  if (records === null || concurrencyLevelCount === null) {
    throw new Error(`${pairId}.coverage is incomplete`)
  }
  const levels = Array.isArray(coverage.concurrencyLevels)
    ? coverage.concurrencyLevels.filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
    : []
  const versions = Array.isArray(coverage.backendVersions)
    ? coverage.backendVersions.filter((value): value is string => typeof value === 'string')
    : []
  // Legacy bundles predate the backend-family list; an absent list means
  // "unknown", so the classify guard stays permissive for them.
  const backends = Array.isArray(coverage.backends)
    ? coverage.backends.filter((value): value is string => typeof value === 'string')
    : []
  return {
    records,
    concurrencyLevels: levels,
    concurrencyLevelCount,
    backendVersions: versions,
    backends,
  }
}

function parseFeatureSchema(raw: unknown, pairId: string): FeatureSchema | undefined {
  const schema = asRecord(raw)
  if (!schema) return undefined
  const names = Array.isArray(schema.feature_names)
    ? schema.feature_names.filter((value): value is string => typeof value === 'string')
    : []
  if (names.length === 0) throw new Error(`${pairId}.featureSchema.feature_names is empty`)
  const medians = asRecord(schema.numeric_medians) ?? {}
  return {
    feature_names: names,
    numeric_medians: Object.fromEntries(
      Object.entries(medians).filter((entry): entry is [string, number] => typeof entry[1] === 'number'),
    ),
  }
}

function parseV2Pair(raw: unknown, index: number): RegistryPair {
  const pair = asRecord(raw)
  if (!pair) throw new Error(`registry pair ${index} must be an object`)
  const id = requireString(pair, 'id', `registry pair ${index}`)
  const status = pair.status
  if (status !== 'validated' && status !== 'measured_no_envelope') {
    throw new Error(`${id}.status is not supported`)
  }
  const gate = asRecord(pair.gate)
  const accepted = gate?.passed
  if (typeof accepted !== 'boolean' || !Array.isArray(gate?.reasons)) {
    throw new Error(`${id}.gate is incomplete`)
  }
  const entry: RegistryPair = {
    id,
    modelId: requireString(pair, 'modelId', id),
    systemId: requireString(pair, 'systemId', id),
    status,
    coverage: parseCoverage(pair.coverage, id),
    dashboardPath: stringOrNull(pair.dashboardPath),
    gate: {
      passed: accepted,
      reasons: gate.reasons.filter((reason): reason is string => typeof reason === 'string'),
      thresholds: gate.thresholds ? parseGate(gate.thresholds) : null,
    },
    legacy: false,
  }
  const diagnostics = asRecord(pair.diagnostics)
  if (diagnostics) {
    entry.diagnostics = {
      trainingStatus: stringOrNull(diagnostics.trainingStatus) ?? 'unknown',
      reasons: Array.isArray(diagnostics.reasons)
        ? diagnostics.reasons.filter((reason): reason is string => typeof reason === 'string')
        : [],
    }
    const warnings = Array.isArray(diagnostics.warnings)
      ? diagnostics.warnings.filter((warning): warning is string => typeof warning === 'string')
      : []
    if (warnings.length > 0) entry.diagnostics.warnings = warnings
    const selectionBasis = stringOrNull(diagnostics.selectionBasis)
    if (selectionBasis !== null) entry.diagnostics.selectionBasis = selectionBasis
  }
  if (status === 'validated') {
    entry.classifier = requireString(pair, 'classifier', id)
    entry.tritonModel = requireString(pair, 'tritonModel', id)
    entry.featureSchema = parseFeatureSchema(pair.featureSchema, id)
    if (!entry.featureSchema) throw new Error(`${id}.featureSchema is required for a validated pair`)
    entry.metrics = pair.metrics
    entry.thresholds = pair.thresholds
    entry.artifactVersion = stringOrNull(pair.artifactVersion) ?? undefined
  }
  return entry
}

function parseV1Pair(raw: unknown, index: number): RegistryPair {
  const pair = asRecord(raw)
  if (!pair) throw new Error(`legacy registry pair ${index} must be an object`)
  const id = requireString(pair, 'id', `legacy registry pair ${index}`)
  const metrics = asRecord(pair.metrics)
  const test = asRecord(metrics?.test)
  const auc = numberOrNull(test?.roc_auc)
  const f1 = numberOrNull(test?.f1)
  const reasons: string[] = []
  if (auc === null) reasons.push('legacy_test_auc_unavailable')
  else if (auc < DEFAULT_GATE.minTestAuc) reasons.push('test_auc_below_minimum')
  if (f1 === null) reasons.push('legacy_test_f1_unavailable')
  else if (f1 < DEFAULT_GATE.minTestF1) reasons.push('test_f1_below_minimum')

  const status: RegistryPairStatus = reasons.length === 0 ? 'validated' : 'measured_no_envelope'
  const entry: RegistryPair = {
    id,
    modelId: requireString(pair, 'modelId', id),
    systemId: requireString(pair, 'systemId', id),
    status,
    coverage: null,
    dashboardPath: stringOrNull(pair.dashboardPath),
    gate: { passed: reasons.length === 0, reasons, thresholds: null },
    legacy: true,
  }
  if (status === 'validated') {
    entry.classifier = requireString(pair, 'classifier', id)
    entry.tritonModel = requireString(pair, 'tritonModel', id)
    entry.featureSchema = parseFeatureSchema(pair.featureSchema, id)
    if (!entry.featureSchema) throw new Error(`${id}.featureSchema is required for a validated pair`)
    entry.metrics = pair.metrics
    entry.thresholds = pair.thresholds
    entry.artifactVersion = stringOrNull(pair.artifactVersion) ?? undefined
  }
  return entry
}

/** Parse and normalize a registry file. Unknown schemas are rejected. */
export function parseRegistry(raw: unknown): TestedModelsRegistry {
  const registry = asRecord(raw)
  if (!registry) throw new Error('registry must be an object')
  const pairs = registry.pairs
  if (!Array.isArray(pairs)) throw new Error('registry.pairs must be an array')
  const schemaVersion = registry.schemaVersion

  if (schemaVersion === 2) {
    const approvedModels = registry.approvedModels
    if (!Array.isArray(approvedModels) || !approvedModels.every(model => typeof model === 'string')) {
      throw new Error('registry.approvedModels must be a string array')
    }
    return {
      schemaVersion: 2,
      generatedAt: stringOrNull(registry.generatedAt),
      gate: parseGate(registry.gate),
      approvedModels,
      pairs: pairs.map(parseV2Pair),
    }
  }

  if (schemaVersion === 1) {
    return {
      schemaVersion: 1,
      generatedAt: stringOrNull(registry.generatedAt),
      gate: null,
      approvedModels: [],
      pairs: pairs.map(parseV1Pair),
    }
  }

  throw new Error(`unsupported registry schema: ${String(schemaVersion)}`)
}

export type TestedModelStatus = 'validated' | 'measured_no_envelope' | 'approved_no_data' | 'untested'

/** Derive a model's status from the registry alone. */
export function modelStatus(modelId: string, registry: Pick<TestedModelsRegistry, 'pairs' | 'approvedModels'>): TestedModelStatus {
  const pairs = registry.pairs.filter(pair => pair.modelId === modelId)
  if (pairs.some(pair => pair.status === 'validated')) return 'validated'
  if (pairs.length > 0) return 'measured_no_envelope'
  if (registry.approvedModels.includes(modelId)) return 'approved_no_data'
  return 'untested'
}

export type TestedSystemStatus = 'validated' | 'measured_no_envelope' | 'untested'

/** Derive a GPU system's status for one model, or across the registry. */
export function systemStatus(systemId: string, pairs: RegistryPair[], modelId?: string): TestedSystemStatus {
  const matching = pairs.filter(pair => pair.systemId === systemId && (modelId == null || pair.modelId === modelId))
  if (matching.some(pair => pair.status === 'validated')) return 'validated'
  if (matching.length > 0) return 'measured_no_envelope'
  return 'untested'
}

export function validatedModelIds(registry: Pick<TestedModelsRegistry, 'pairs'>): string[] {
  return [...new Set(registry.pairs.filter(pair => pair.status === 'validated').map(pair => pair.modelId))]
}

/**
 * Keep every catalog option selectable but list options with tested
 * measurements first. Tested hardware is an annotation, never a filter.
 */
export function testedSystemsFirst<T extends { systemId: string }>(
  options: T[],
  testedSystemIds: ReadonlySet<string>,
): T[] {
  if (testedSystemIds.size === 0) return options
  return [...options].sort(
    (a, b) => Number(testedSystemIds.has(b.systemId)) - Number(testedSystemIds.has(a.systemId)),
  )
}

export function modelNamesForStatus(registry: TestedModelsRegistry, status: TestedModelStatus): string[] {
  if (status === 'approved_no_data') return registry.approvedModels.filter(model => modelStatus(model, registry) === status)
  return [...new Set(registry.pairs.filter(pair => pair.status === status).map(pair => pair.modelId))]
}

import { parquetReadObjects } from 'hyparquet'
import appConfig from '../../public/config.json'
import { backendVersionsMatch, normalizeBackendVersion } from './benchmark-versions'
import {
  BENCHMARK_FILTER_KEYS,
  type ConfigurationOption,
  type DatasetConfiguration,
  type DatasetRow,
  type DatasetSource,
  type BenchmarkFilterKey,
  type BenchmarkFilters,
  type BenchmarkPoint,
} from './benchmark-types'

export {
  backendVersionIdentity,
  backendVersionsMatch,
  normalizeBackendVersion,
  UNREPORTED_VERSION,
} from './benchmark-versions'

const DATASET_ID = 'redhat-performance/configiq-performance-upstream-data'
const DATASET_URL = `https://huggingface.co/datasets/${DATASET_ID}`
const VIEWER_API = 'https://datasets-server.huggingface.co'
const PARQUET_CONCURRENCY = 4
const CACHE_TTL_MS = 5 * 60 * 1000
/** Explicit refreshes are ignored until the cache is at least this old. */
const MIN_REFRESH_INTERVAL_MS = 30 * 1000
const INFERENCEX_DATASET_ID = 'redhat-performance/configiq-inferencex-comparison'
const INFERENCEX_FILE_URL = `https://huggingface.co/datasets/${INFERENCEX_DATASET_ID}/resolve/main/inferencex-8k1k.json`

/** External links are rendered in anchors, so only https URLs are accepted. */
function httpsUrl(value: unknown): string | null {
  if (typeof value !== 'string') return null
  try {
    const url = new URL(value)
    return url.protocol === 'https:' ? url.toString() : null
  } catch {
    return null
  }
}

interface InferenceXCache {
  rows: DatasetRow[]
  source: { url: string; retrievedAt: string; rowCount: number }
  expiresAt: number
}

let cachedInferenceX: InferenceXCache | null = null
let inFlightInferenceX: Promise<InferenceXCache> | null = null

async function loadInferenceXDataset(): Promise<InferenceXCache> {
  const token = process.env.INFERENCEX_HF_TOKEN
  if (!token) throw new Error('Private InferenceX dataset access is not configured')
  const response = await fetch(INFERENCEX_FILE_URL, {
    headers: { Accept: 'application/json', Authorization: `Bearer ${token}` },
    cache: 'no-store',
    signal: AbortSignal.timeout(25_000),
  })
  if (!response.ok) throw new Error(`Private InferenceX dataset returned HTTP ${response.status}`)
  const payload: unknown = await response.json()
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('Private InferenceX dataset has an invalid format')
  }
  const snapshot = payload as Record<string, unknown>
  const rows = snapshot.rows
  const approvedModelIds = appConfig.approvedModels
  if (!Array.isArray(approvedModelIds) || approvedModelIds.length === 0
    || !approvedModelIds.every(model => typeof model === 'string')) {
    throw new Error('ConfigIQ approved model list is missing or invalid')
  }
  const approvedModels = new Set(approvedModelIds)
  if (!Array.isArray(rows) || rows.length === 0 || rows.some(row =>
    !row || typeof row !== 'object' || Array.isArray(row)
    || !approvedModels.has((row as DatasetRow).model as string)
    || (row as DatasetRow).benchmark_source !== 'inferencex'
    || (row as DatasetRow).isl !== 8192 || (row as DatasetRow).osl !== 1024
    || ((row as DatasetRow).source_run_url != null && httpsUrl((row as DatasetRow).source_run_url) === null))) {
    throw new Error('Private InferenceX dataset contains invalid or unapproved benchmark rows')
  }
  const sourceUrl = httpsUrl(snapshot.sourceUrl)
  if (!sourceUrl || typeof snapshot.retrievedAt !== 'string'
    || !Number.isFinite(Date.parse(snapshot.retrievedAt))) {
    throw new Error('Private InferenceX dataset has an invalid source URL or timestamp')
  }
  return {
    rows: rows as DatasetRow[],
    source: { url: sourceUrl, retrievedAt: snapshot.retrievedAt, rowCount: rows.length },
    expiresAt: Date.now() + CACHE_TTL_MS,
  }
}

export async function getInferenceXDataset(refresh = false): Promise<InferenceXCache> {
  const cacheAgeMs = cachedInferenceX ? Date.now() - (cachedInferenceX.expiresAt - CACHE_TTL_MS) : Infinity
  const forceRefresh = refresh && cacheAgeMs > MIN_REFRESH_INTERVAL_MS
  if (!forceRefresh && cachedInferenceX && cachedInferenceX.expiresAt > Date.now()) return cachedInferenceX
  if (inFlightInferenceX) return inFlightInferenceX

  // The fallback lives inside the shared promise so every concurrent caller
  // receives the last successful snapshot, not just the caller that started it.
  const pending = loadInferenceXDataset().then(
    loaded => {
      cachedInferenceX = loaded
      return loaded
    },
    (error: unknown) => {
      // A failed refresh must not turn a working snapshot into an outage.
      if (cachedInferenceX) return cachedInferenceX
      throw error
    },
  )
  inFlightInferenceX = pending
  try {
    return await pending
  } finally {
    if (inFlightInferenceX === pending) inFlightInferenceX = null
  }
}

const CONFIGURATION_FIELDS = [
  'measurement_source',
  'benchmark_version',
  'backend',
  'backend_version',
  'precision',
  'weight_quantization',
  'kv_quantization',
  'prefix_caching',
  'prefix_tokens',
  'prefix_count',
  'max_seq_len',
  'max_num_seqs',
  'gpu_memory_utilization',
  'chunked_prefill_enabled',
  'spec_decoding',
  'request_type',
  'turns',
  'tp_prefill',
  'tp_decode',
  'inferencex_topology',
  'inferencex_curve',
  'inferencex_offload',
] as const

interface ParquetFile {
  url: string
  filename: string
  size: number
}

interface ParquetFileList {
  parquet_files: ParquetFile[]
}

interface HubMetadata {
  sha?: string
  lastModified?: string
}

interface DatasetCache {
  rows: DatasetRow[]
  source: DatasetSource
  expiresAt: number
}

let cachedDataset: DatasetCache | null = null
let inFlightDataset: Promise<DatasetCache> | null = null

function normalizeValue(value: unknown): string | null {
  if (value == null) return null
  if (typeof value === 'string') return value.trim() ? value.trim() : null
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : null
  if (typeof value === 'bigint') return value.toString()
  if (typeof value === 'boolean') return String(value)
  return null
}

function normalizeBackend(value: unknown): string | null {
  return normalizeValue(value)?.toLowerCase() ?? null
}

export function benchmarkFilterValue(row: DatasetRow, key: BenchmarkFilterKey): string | null {
  if (key === 'backend') return normalizeBackend(row.backend)
  if (key === 'backendVersion') return normalizeBackendVersion(row.backend, row.backend_version)
  return normalizeValue(row[key])
}

/** Sorted, de-duplicated values a filter key takes across `rows`. */
export function benchmarkFilterValues(rows: DatasetRow[], key: BenchmarkFilterKey): string[] {
  const values = [...new Set(rows
    .map(row => benchmarkFilterValue(row, key))
    .filter((value): value is string => value !== null))]
  values.sort((left, right) => {
    const leftNumber = Number(left)
    const rightNumber = Number(right)
    if (left !== '' && right !== '' && Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) return leftNumber - rightNumber
    return left.localeCompare(right, undefined, { numeric: true, sensitivity: 'base' })
  })
  return values
}

function configurationForRow(row: DatasetRow): DatasetConfiguration {
  return Object.fromEntries(CONFIGURATION_FIELDS.map(field => {
    if (field === 'measurement_source') return [field, row.benchmark_source === 'inferencex' ? 'InferenceX' : 'Red Hat']
    if (field === 'benchmark_version') return [field, row.benchmark_source === 'inferencex' ? null : normalizeValue(row.version)]
    if (field === 'backend') return [field, normalizeBackend(row.backend)]
    if (field === 'backend_version') return [field, normalizeBackendVersion(row.backend, row.backend_version)]
    return [field, normalizeValue(row[field])]
  })) as DatasetConfiguration
}

function configurationKey(values: DatasetConfiguration): string {
  return JSON.stringify(values)
}

function configurationLabel(values: DatasetConfiguration): string {
  const labels: string[] = []
  if (values.measurement_source) labels.push(values.measurement_source)
  const backend = values.backend?.replaceAll('-', ' ')
  if (backend) labels.push(backend)
  if (values.backend_version && values.backend_version !== values.backend) labels.push(values.backend_version)
  if (values.precision) labels.push(values.precision)
  if (values.benchmark_version) labels.push(`benchmark ${values.benchmark_version}`)
  if (values.weight_quantization) labels.push(`weights ${values.weight_quantization}`)
  if (values.kv_quantization) labels.push(`KV ${values.kv_quantization}`)
  if (values.prefix_caching) labels.push(`prefix caching ${values.prefix_caching}`)
  if (values.prefix_tokens) labels.push(`${values.prefix_tokens} prefix tokens`)
  if (values.prefix_count) labels.push(`${values.prefix_count} prefixes`)
  if (values.max_seq_len) labels.push(`max sequence ${values.max_seq_len}`)
  if (values.max_num_seqs) labels.push(`max sequences ${values.max_num_seqs}`)
  if (values.gpu_memory_utilization) labels.push(`GPU memory ${values.gpu_memory_utilization}`)
  if (values.chunked_prefill_enabled) labels.push(`chunked prefill ${values.chunked_prefill_enabled}`)
  if (values.spec_decoding) labels.push(`speculative decoding ${values.spec_decoding}`)
  if (values.request_type) labels.push(values.request_type)
  if (values.turns) labels.push(`${values.turns} turns`)
  if (values.inferencex_topology) labels.push(values.inferencex_topology)
  if (values.inferencex_offload && values.inferencex_offload !== 'off') labels.push(`offload ${values.inferencex_offload}`)
  if (values.inferencex_curve) labels.push(values.inferencex_curve)
  return labels.length ? labels.join(' · ') : 'Default benchmark configuration'
}

function retryDelay(response: Response | null, attempt: number): number {
  const retryAfter = response?.headers.get('retry-after')
  if (retryAfter) {
    const seconds = Number(retryAfter)
    if (Number.isFinite(seconds)) return Math.min(seconds * 1000, 30_000)
    const date = Date.parse(retryAfter)
    if (Number.isFinite(date)) return Math.min(Math.max(0, date - Date.now()), 30_000)
  }
  return 500 * (2 ** attempt)
}

async function fetchUpstream(url: string, context: string, refresh: boolean): Promise<Response> {
  let lastError: unknown = null
  for (let attempt = 0; attempt < 5; attempt += 1) {
    let response: Response | null = null
    try {
      response = await fetch(url, {
        headers: { Accept: 'application/json' },
        ...(refresh ? { cache: 'no-store' as const } : { next: { revalidate: 300 } }),
        signal: AbortSignal.timeout(25_000),
      })
      if (response.ok) return response
      lastError = new Error(`${context} returned HTTP ${response.status}`)
    } catch (error) {
      lastError = error
    }

    if (response && response.status !== 429 && response.status < 500) {
      throw lastError instanceof Error ? lastError : new Error(`${context} request failed`)
    }
    if (attempt < 4) await new Promise(resolve => setTimeout(resolve, retryDelay(response, attempt)))
  }
  throw lastError instanceof Error ? lastError : new Error(`${context} request failed`)
}

async function fetchJson<T>(url: string, context: string, refresh: boolean): Promise<T> {
  const response = await fetchUpstream(url, context, refresh)
  try {
    return await response.json() as T
  } catch {
    throw new Error(`${context} returned invalid JSON`)
  }
}

async function fetchParquetFile(file: ParquetFile, refresh: boolean): Promise<DatasetRow[]> {
  const fileUrl = new URL(file.url)
  if (fileUrl.protocol !== 'https:' || fileUrl.hostname !== 'huggingface.co') {
    throw new Error(`Hugging Face listed an unsupported Parquet URL for ${file.filename}`)
  }
  const response = await fetchUpstream(fileUrl.toString(), `Hugging Face Parquet file ${file.filename}`, refresh)
  try {
    const parquetBuffer = await response.arrayBuffer()
    const rows = await parquetReadObjects({ file: parquetBuffer })
    return rows as unknown as DatasetRow[]
  } catch (error) {
    const detail = error instanceof Error ? error.message : 'could not be parsed'
    throw new Error(`Hugging Face Parquet file ${file.filename} ${detail}`)
  }
}

async function loadDataset(refresh: boolean): Promise<DatasetCache> {
  const [filesResponse, metadata] = await Promise.all([
    fetchJson<ParquetFileList>(`${VIEWER_API}/parquet?dataset=${encodeURIComponent(DATASET_ID)}`, 'Hugging Face Parquet listing', refresh),
    fetchJson<HubMetadata>(`https://huggingface.co/api/datasets/${DATASET_ID}`, 'Hugging Face dataset metadata', refresh),
  ])
  const files = filesResponse.parquet_files
  if (!Array.isArray(files) || files.length === 0) {
    throw new Error('Hugging Face did not list any Parquet files for this dataset')
  }

  const rows: DatasetRow[] = []
  for (let fileIndex = 0; fileIndex < files.length; fileIndex += PARQUET_CONCURRENCY) {
    const batch = files.slice(fileIndex, fileIndex + PARQUET_CONCURRENCY)
    rows.push(...(await Promise.all(batch.map(file => fetchParquetFile(file, refresh)))).flat())
  }

  if (rows.length === 0) throw new Error('The Hugging Face performance dataset contains no benchmark rows')
  const requiredFields = ['accelerator', 'model', 'version', 'isl', 'osl', 'tp', 'concurrency']
  if (rows.some(row => requiredFields.some(field => !(field in row)))) {
    throw new Error('The Hugging Face dataset is missing comparison fields required by ConfigIQ')
  }

  const source: DatasetSource = {
    id: DATASET_ID,
    url: DATASET_URL,
    revision: metadata.sha ?? null,
    lastModified: metadata.lastModified ?? null,
    loadedAt: new Date().toISOString(),
    rowCount: rows.length,
  }
  return { rows, source, expiresAt: Date.now() + CACHE_TTL_MS }
}

export async function getBenchmarkDataset(refresh = false): Promise<DatasetCache> {
  const cacheAgeMs = cachedDataset ? Date.now() - (cachedDataset.expiresAt - CACHE_TTL_MS) : Infinity
  const forceRefresh = refresh && cacheAgeMs > MIN_REFRESH_INTERVAL_MS
  if (!forceRefresh && cachedDataset && cachedDataset.expiresAt > Date.now()) return cachedDataset
  if (inFlightDataset) return inFlightDataset

  // The fallback lives inside the shared promise so concurrent callers receive
  // it too, rather than only the caller that started the load.
  const pending = loadDataset(forceRefresh).then(
    loaded => {
      cachedDataset = loaded
      return loaded
    },
    (error: unknown) => {
      // A failed reload must not turn a working dataset into an outage.
      if (cachedDataset) return cachedDataset
      throw error
    },
  )
  inFlightDataset = pending
  try {
    return await pending
  } finally {
    if (inFlightDataset === pending) inFlightDataset = null
  }
}

export function parseBenchmarkFilters(searchParams: URLSearchParams): BenchmarkFilters {
  return Object.fromEntries(
    BENCHMARK_FILTER_KEYS.map(key => [key, searchParams.get(key) ?? '']),
  ) as BenchmarkFilters
}

export function filterRows(rows: DatasetRow[], filters: Partial<BenchmarkFilters>): DatasetRow[] {
  return rows.filter(row => BENCHMARK_FILTER_KEYS.every(key => {
    const selected = filters[key]
    if (!selected) return true
    const value = benchmarkFilterValue(row, key)
    // A catalog release pin can satisfy a measured vendor build of the same
    // release, while two specific builds remain distinct.
    return key === 'backendVersion'
      ? backendVersionsMatch(selected, value)
      : value === selected
  }))
}

export function getFilterOptions(rows: DatasetRow[], filters: BenchmarkFilters): Record<BenchmarkFilterKey, string[]> {
  return Object.fromEntries(BENCHMARK_FILTER_KEYS.map(key => {
    const matchingRows = filterRows(rows, { ...filters, [key]: '' })
    return [key, benchmarkFilterValues(matchingRows, key)]
  })) as Record<BenchmarkFilterKey, string[]>
}

export function getConfigurationOptions(rows: DatasetRow[]): ConfigurationOption[] {
  const groups = new Map<string, { values: DatasetConfiguration; rows: number }>()
  for (const row of rows) {
    const values = configurationForRow(row)
    const key = configurationKey(values)
    const existing = groups.get(key)
    if (existing) existing.rows += 1
    else groups.set(key, { values, rows: 1 })
  }
  return [...groups.entries()]
    .map(([key, group]) => ({ key, values: group.values, rows: group.rows, label: configurationLabel(group.values) }))
    .sort((left, right) => left.label.localeCompare(right.label, undefined, { numeric: true, sensitivity: 'base' }))
}

export function rowMatchesConfiguration(row: DatasetRow, values: DatasetConfiguration): boolean {
  return configurationKey(configurationForRow(row)) === configurationKey(values)
}

export function parseConfiguration(value: string): DatasetConfiguration | null {
  try {
    const parsed: unknown = JSON.parse(value)
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null
    const record = parsed as Record<string, unknown>
    if (CONFIGURATION_FIELDS.some(field => record[field] !== null && typeof record[field] !== 'string')) return null
    return Object.fromEntries(CONFIGURATION_FIELDS.map(field => [field, record[field] ?? null])) as DatasetConfiguration
  } catch {
    return null
  }
}

function numericField(row: DatasetRow, key: string): number | null {
  const value = row[key]
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'bigint') {
    const converted = Number(value)
    return Number.isSafeInteger(converted) ? converted : null
  }
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

export function aggregateBenchmarkRows(rows: DatasetRow[]): BenchmarkPoint[] {
  const groups = new Map<number, DatasetRow[]>()
  for (const row of rows) {
    const concurrency = numericField(row, 'concurrency')
    if (concurrency === null || concurrency <= 0) continue
    const group = groups.get(concurrency) ?? []
    group.push(row)
    groups.set(concurrency, group)
  }

  const average = (group: DatasetRow[], field: string, divisor = 1): number | null => {
    const values = group.map(row => numericField(row, field)).filter((value): value is number => value !== null)
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length / divisor : null
  }

  return [...groups.entries()].sort(([left], [right]) => left - right).map(([concurrency, group]) => {
    const outputTokensPerSecond = average(group, 'output_tok/sec')
    return {
      concurrency,
      // Dataset TTFT and TPOT values are milliseconds; request latency is seconds.
      ttftSeconds: average(group, 'ttft_median', 1000),
      tpotSeconds: average(group, 'tpot_median', 1000),
      outputTokensPerSecond,
      outputTokensPerSecondPerUser: outputTokensPerSecond === null ? null : outputTokensPerSecond / concurrency,
      requestLatencySeconds: average(group, 'request_latency_median'),
      sampleCount: group.length,
    }
  })
}

import { NextRequest, NextResponse } from 'next/server'
import {
  aggregateBenchmarkRows,
  filterRows,
  getConfigurationOptions,
  getFilterOptions,
  getBenchmarkDataset,
  getInferenceXDataset,
  normalizeBackendVersion,
  parseBenchmarkFilters,
  parseConfiguration,
  rowMatchesConfiguration,
  UNREPORTED_VERSION,
} from '@/lib/validation/benchmark-dataset'
import { benchmarkOptions, resolveBenchmarkFilters } from '@/lib/validation/benchmark-resolve'
import type { DatasetRow } from '@/lib/validation/benchmark-types'

export const dynamic = 'force-dynamic'

type Mode = 'resolve' | 'filters' | 'data'

function failure(message: string, status: number) {
  return NextResponse.json(
    { status: 'failed', error: message },
    { status, headers: { 'Cache-Control': 'no-store' } },
  )
}

/**
 * The release a set of measured rows reports, normalized for display. Rows in a
 * configuration share one release; report nothing when that is not true so the
 * client does not attribute a mixed set to a single version.
 */
function measuredVersion(rows: DatasetRow[]): string | null {
  const versions = [...new Set(
    rows
      .map(row => normalizeBackendVersion(row.backend, row.backend_version))
      .filter(version => version !== UNREPORTED_VERSION),
  )]
  return versions.length === 1 ? versions[0] : null
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const mode = (searchParams.get('mode') ?? 'filters') as Mode
  if (mode !== 'resolve' && mode !== 'filters' && mode !== 'data') {
    return failure('Unknown benchmark request mode', 400)
  }

  try {
    let benchmark: Awaited<ReturnType<typeof getBenchmarkDataset>> | null = null
    let benchmarkError: string | null = null
    try {
      benchmark = await getBenchmarkDataset(searchParams.get('refresh') === 'true')
    } catch (error) {
      benchmarkError = error instanceof Error ? error.message : 'Hugging Face benchmark data is unavailable'
    }
    let inferencex: Awaited<ReturnType<typeof getInferenceXDataset>> | null = null
    let inferencexError: string | null = null
    try {
      inferencex = await getInferenceXDataset(searchParams.get('refresh') === 'true')
    } catch (error) {
      inferencexError = error instanceof Error ? error.message : 'Private InferenceX dataset is unavailable'
    }
    const inferencexRows = inferencex?.rows ?? []
    if (!benchmark && !inferencex) return failure(benchmarkError ?? inferencexError ?? 'No measured data is available', 502)
    const combinedRows = [...(benchmark?.rows ?? []), ...inferencexRows]
    const requestedFilters = parseBenchmarkFilters(searchParams)

    if (mode === 'resolve' || mode === 'filters') {
      const resolution = resolveBenchmarkFilters(combinedRows, requestedFilters)
      const options = benchmarkOptions(
        combinedRows,
        resolution.filters,
        getFilterOptions(combinedRows, resolution.filters),
      )
      const configurations = resolution.unavailable
        ? []
        : getConfigurationOptions(filterRows(combinedRows, resolution.filters))
      return NextResponse.json(
        {
          status: 'ok',
          source: benchmark?.source ?? null,
          inferencexSource: inferencex?.source ?? null,
          benchmarkError,
          inferencexError,
          filters: resolution.filters,
          adjustments: resolution.adjustments,
          unavailable: resolution.unavailable,
          versionFallback: resolution.versionFallback,
          options,
          configurations,
        },
        { headers: { 'Cache-Control': 'no-store' } },
      )
    }

    const filters = requestedFilters
    if (Object.values(filters).some(value => !value)) {
      return failure('Choose an accelerator, model, serving backend and version, input and output length, and tensor parallel size.', 400)
    }
    const configuration = parseConfiguration(searchParams.get('configuration') ?? '')
    if (!configuration) return failure('Choose a benchmark runtime configuration before comparing.', 400)

    const internalRows = filterRows(benchmark?.rows ?? [], filters)
      .filter(row => rowMatchesConfiguration(row, configuration))
    const selectedInferenceXRows = configuration.measurement_source === 'InferenceX'
      ? filterRows(inferencexRows, filters).filter(row => rowMatchesConfiguration(row, configuration))
      : []
    const inferencexRuns: Array<{ url: string; date: string | null }> = []
    const seenRunUrls = new Set<string>()
    for (const row of selectedInferenceXRows) {
      if (typeof row.source_run_url !== 'string' || !row.source_run_url || seenRunUrls.has(row.source_run_url)) continue
      seenRunUrls.add(row.source_run_url)
      inferencexRuns.push({
        url: row.source_run_url,
        date: typeof row.measurement_date === 'string' ? row.measurement_date : null,
      })
    }
    if (internalRows.length === 0 && selectedInferenceXRows.length === 0) {
      return NextResponse.json(
        { status: 'ok', source: benchmark?.source ?? null, inferencexSource: inferencex?.source ?? null,
          matchedRows: 0, benchmarkPoints: [], inferencexPoints: [], inferencexRunUrl: null, inferencexRuns: [],
          benchmarkVersion: null, inferencexVersion: null },
        { headers: { 'Cache-Control': 'no-store' } },
      )
    }

    return NextResponse.json(
      {
        status: 'ok',
        source: benchmark?.source ?? null,
        inferencexSource: inferencex?.source ?? null,
        benchmarkError,
        inferencexError,
        matchedRows: internalRows.length + selectedInferenceXRows.length,
        benchmarkPoints: aggregateBenchmarkRows(internalRows),
        inferencexPoints: aggregateBenchmarkRows(selectedInferenceXRows),
        benchmarkVersion: measuredVersion(internalRows),
        inferencexVersion: measuredVersion(selectedInferenceXRows),
        inferencexRunUrl: inferencexRuns[0]?.url ?? null,
        inferencexDate: inferencexRuns[0]?.date ?? null,
        inferencexRuns,
      },
      { headers: { 'Cache-Control': 'no-store' } },
    )
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Could not load the Hugging Face benchmark dataset'
    return failure(message, 502)
  }
}

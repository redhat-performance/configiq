import { NextRequest, NextResponse } from 'next/server'
import {
  aggregateGroundTruthRows,
  filterRows,
  getConfigurationOptions,
  getFilterOptions,
  getGroundTruthDataset,
  getInferenceXDataset,
  parseGroundTruthFilters,
  parseConfiguration,
  rowMatchesConfiguration,
} from '@/lib/validation/ground-truth-dataset'

export const dynamic = 'force-dynamic'

function failure(message: string, status: number) {
  return NextResponse.json(
    { status: 'failed', error: message },
    { status, headers: { 'Cache-Control': 'no-store' } },
  )
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const mode = searchParams.get('mode') ?? 'filters'
  if (mode !== 'filters' && mode !== 'data') return failure('Unknown ground truth request mode', 400)

  try {
    let groundTruth: Awaited<ReturnType<typeof getGroundTruthDataset>> | null = null
    let groundTruthError: string | null = null
    try {
      groundTruth = await getGroundTruthDataset(searchParams.get('refresh') === 'true')
    } catch (error) {
      groundTruthError = error instanceof Error ? error.message : 'Hugging Face ground truth is unavailable'
    }
    let inferencex: Awaited<ReturnType<typeof getInferenceXDataset>> | null = null
    let inferencexError: string | null = null
    try {
      inferencex = await getInferenceXDataset()
    } catch (error) {
      inferencexError = error instanceof Error ? error.message : 'Private InferenceX dataset is unavailable'
    }
    const inferencexRows = inferencex?.rows ?? []
    if (!groundTruth && !inferencex) return failure(groundTruthError ?? inferencexError ?? 'No measured data is available', 502)
    const combinedRows = [...(groundTruth?.rows ?? []), ...inferencexRows]
    const filters = parseGroundTruthFilters(searchParams)

    if (mode === 'filters') {
      const matchingRows = filterRows(combinedRows, filters)
      return NextResponse.json(
        {
          status: 'ok',
          source: groundTruth?.source ?? null,
          inferencexSource: inferencex?.source ?? null,
          groundTruthError,
          inferencexError,
          options: getFilterOptions(combinedRows, filters),
          configurations: Object.values(filters).every(Boolean) ? getConfigurationOptions(matchingRows) : [],
        },
        { headers: { 'Cache-Control': 'no-store' } },
      )
    }

    if (Object.values(filters).some(value => !value)) {
      return failure('Choose an accelerator, model, serving backend and version, input and output length, and tensor parallel size.', 400)
    }
    const configuration = parseConfiguration(searchParams.get('configuration') ?? '')
    if (!configuration) return failure('Choose a benchmark runtime configuration before comparing.', 400)

    const internalRows = filterRows(groundTruth?.rows ?? [], filters)
      .filter(row => rowMatchesConfiguration(row, configuration))
    const selectedInferenceXRows = configuration.measurement_source === 'InferenceX'
      ? filterRows(inferencexRows, filters).filter(row => rowMatchesConfiguration(row, configuration))
      : []
    if (internalRows.length === 0 && selectedInferenceXRows.length === 0) {
      return NextResponse.json(
        { status: 'ok', source: groundTruth?.source ?? null, inferencexSource: inferencex?.source ?? null,
          matchedRows: 0, groundTruthPoints: [], inferencexPoints: [], inferencexRunUrl: null },
        { headers: { 'Cache-Control': 'no-store' } },
      )
    }

    return NextResponse.json(
      {
        status: 'ok',
        source: groundTruth?.source ?? null,
        inferencexSource: inferencex?.source ?? null,
        groundTruthError,
        inferencexError,
        matchedRows: internalRows.length + selectedInferenceXRows.length,
        groundTruthPoints: aggregateGroundTruthRows(internalRows),
        inferencexPoints: aggregateGroundTruthRows(selectedInferenceXRows),
        inferencexRunUrl: selectedInferenceXRows[0]?.source_run_url ?? null,
        inferencexDate: selectedInferenceXRows[0]?.measurement_date ?? null,
      },
      { headers: { 'Cache-Control': 'no-store' } },
    )
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Could not load the Hugging Face ground truth dataset'
    return failure(message, 502)
  }
}

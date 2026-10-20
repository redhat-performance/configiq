import { NextRequest, NextResponse } from 'next/server'
import {
  aggregateGroundTruthRows,
  filterRows,
  getConfigurationOptions,
  getFilterOptions,
  getGroundTruthDataset,
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
    const dataset = await getGroundTruthDataset(searchParams.get('refresh') === 'true')
    const filters = parseGroundTruthFilters(searchParams)

    if (mode === 'filters') {
      const matchingRows = filterRows(dataset.rows, filters)
      return NextResponse.json(
        {
          status: 'ok',
          source: dataset.source,
          options: getFilterOptions(dataset.rows, filters),
          configurations: Object.values(filters).every(Boolean) ? getConfigurationOptions(matchingRows) : [],
        },
        { headers: { 'Cache-Control': 'no-store' } },
      )
    }

    if (Object.values(filters).some(value => !value)) {
      return failure('Choose an accelerator, model, version, input and output length, and tensor parallel size.', 400)
    }
    const configuration = parseConfiguration(searchParams.get('configuration') ?? '')
    if (!configuration) return failure('Choose a benchmark runtime configuration before comparing.', 400)

    const selectedRows = filterRows(dataset.rows, filters)
      .filter(row => rowMatchesConfiguration(row, configuration))
    if (selectedRows.length === 0) {
      return NextResponse.json(
        { status: 'ok', source: dataset.source, matchedRows: 0, points: [] },
        { headers: { 'Cache-Control': 'no-store' } },
      )
    }

    return NextResponse.json(
      {
        status: 'ok',
        source: dataset.source,
        matchedRows: selectedRows.length,
        points: aggregateGroundTruthRows(selectedRows),
      },
      { headers: { 'Cache-Control': 'no-store' } },
    )
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Could not load the Hugging Face ground truth dataset'
    return failure(message, 502)
  }
}

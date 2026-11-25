import { readFile } from 'node:fs/promises'
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

/** The pair page reads `throughputCurves`, which schema v2 introduced. */
const SUPPORTED_SCHEMA_VERSION = 2

export async function GET(_request: Request, context: { params: Promise<{ pairId: string }> }) {
  const { pairId } = await context.params
  if (!/^[a-z0-9_]+$/.test(pairId)) {
    return NextResponse.json({ error: 'Invalid pair ID' }, { status: 400 })
  }
  try {
    const path = `${process.env.TESTED_MODELS_DIR ?? '/tested-models'}/dashboard/${pairId}.json`
    const payload: unknown = JSON.parse(await readFile(path, 'utf8'))
    const schemaVersion = payload && typeof payload === 'object'
      ? (payload as { schemaVersion?: unknown }).schemaVersion
      : undefined
    // An older file cannot be rendered as an empty chart: reject it so the page
    // can explain the version skew instead of claiming the pair has no curves.
    if (schemaVersion !== SUPPORTED_SCHEMA_VERSION) {
      return NextResponse.json(
        { error: `Dashboard schema ${String(schemaVersion)} is not supported` },
        { status: 409, headers: { 'Cache-Control': 'no-store' } },
      )
    }
    return NextResponse.json(payload, { headers: { 'Cache-Control': 'no-store' } })
  } catch {
    return NextResponse.json({ error: 'Dashboard data unavailable' }, { status: 404 })
  }
}

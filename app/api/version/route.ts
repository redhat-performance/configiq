import { NextResponse } from 'next/server'
import { VERSION } from '@/lib/version'

export const dynamic = 'force-dynamic'

/**
 * Build identity of the running image.
 *
 * The e2e preflight reads this to pin the checked-out test code to the commit
 * the deployment is actually serving, instead of assuming a host matches a
 * branch or tag.
 */
export async function GET() {
  return NextResponse.json({
    version: VERSION.number,
    gitCommit: VERSION.gitCommit,
    environment: VERSION.environment,
  })
}

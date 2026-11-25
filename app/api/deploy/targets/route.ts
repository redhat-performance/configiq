import { NextRequest } from 'next/server'

import { handleDeployTargets } from '@/lib/api/deploy-proxy'

export async function GET(req: NextRequest) {
  return handleDeployTargets(req)
}

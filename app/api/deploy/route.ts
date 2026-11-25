import { NextRequest } from 'next/server'

import { handleDeploy } from '@/lib/api/deploy-proxy'

export async function POST(req: NextRequest) {
  return handleDeploy(req)
}

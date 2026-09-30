import { NextRequest, NextResponse } from 'next/server'
import { handlePredict } from '@/lib/api/predict-proxy'

export async function POST(req: NextRequest) {
  return handlePredict(req)
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}

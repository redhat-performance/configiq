// GET /api/gpus
// GPU catalog endpoint - fetches from AISimulators /systems and optionally enriches with live pricing

import { NextRequest, NextResponse } from 'next/server'
import { ZodError } from 'zod'
import { GpuCatalogQuerySchema } from '@/lib/api/schemas'
import { ApiErrors } from '@/lib/api/errors'
import { formatGpuCatalogResponse } from '@/lib/api/responses'
import type { GpuSpec } from '@/lib/gpu-math/gpus'
import { recordErrorCode } from '@/lib/otel-metrics'

// AISimulators /systems response schema
interface CatalogSystem {
  id: string
  name: string
  vendor: string
  architecture: string
  memory_bytes: number
  memory_bandwidth_bytes: number
  bf16_tflops: number
  tdp_watts: number
  gpus_per_node: number
}

interface CatalogSystemsResponse {
  systems: CatalogSystem[]
}

export async function GET(req: NextRequest) {
  try {
    // Parse and validate query parameters
    const { searchParams } = new URL(req.url)
    const query: Record<string, string> = {}

    const minMem = searchParams.get('min_memory')
    const maxPrice = searchParams.get('max_price')
    const vendor = searchParams.get('vendor')
    const sort = searchParams.get('sort')

    if (minMem) query.min_memory = minMem
    if (maxPrice) query.max_price = maxPrice
    if (vendor) query.vendor = vendor
    if (sort) query.sort = sort

    const validatedQuery = GpuCatalogQuerySchema.parse(query)

    // Fetch GPU catalog from AISimulators. Require the gateway to be configured;
    // fail loud rather than falling back to the public domain (masks a misconfig
    // and bypasses the per-host internal gateway).
    const gatewayUrl = process.env.AISIMULATORS_GATEWAY_URL
    if (!gatewayUrl) {
      recordErrorCode('/api/gpus', 'AISIM_NOT_CONFIGURED', 503)
      return NextResponse.json(
        { status: 'failed', error: { code: 'AISIM_NOT_CONFIGURED', message: 'AISimulators API URL is not configured' } },
        { status: 503 },
      )
    }
    const gatewayResponse = await fetch(`${gatewayUrl}/systems?include=specs`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      cache: 'no-store',
    })

    if (!gatewayResponse.ok) {
      throw new Error(`AISimulators API error: ${gatewayResponse.status}`)
    }

    const catalogData: CatalogSystemsResponse = await gatewayResponse.json()

    // Transform AISimulators systems to GpuSpec format
    let filteredGpus: GpuSpec[] = catalogData.systems.map((sys, idx) => {
      const vramGb = Math.round(sys.memory_bytes / (1024 ** 3))
      const memoryBandwidthTbps = sys.memory_bandwidth_bytes / (1024 ** 4)

      return {
        // New JSON schema fields
        id: sys.id,
        name: sys.name,
        display_name: sys.name,
        sizer_system_id: sys.id,
        vendor: sys.vendor as 'nvidia' | 'amd',
        vram_gb: vramGb,
        hardware_cost_usd: 0, // TODO: Pricing not yet available from AISimulators
        memory_bandwidth_tbps: memoryBandwidthTbps,
        tokens_per_dollar: 0, // TODO: Pricing not yet available
        tflops_bf16: sys.bf16_tflops,
        tflops_fp8: null,
        mfu_prefill: 0.5, // Default placeholder
        mfu_decode: 0.5, // Default placeholder
        tdp_watts: sys.tdp_watts,
        architecture: sys.architecture as any,
        color: `hsl(${(idx * 137.5) % 360}, 70%, 50%)`, // Generated color

        // Legacy backward compatibility fields
        vramGb: vramGb,
        memoryBandwidthGbps: sys.memory_bandwidth_bytes / (1024 ** 3),
        bandwidthTbps: memoryBandwidthTbps,
        tflops: sys.bf16_tflops,
        pricePerHour: 0, // TODO: Pricing not yet available from AISimulators
        hardwareCostPerGpu: 0, // TODO: Pricing not yet available
        powerWatts: sys.tdp_watts,
        cloudAvailabilityPct: 0, // TODO: Not yet available
        tpuAvailabilityPct: 0, // TODO: Not yet available
      } as GpuSpec
    })


    // Filter by min_memory
    if (validatedQuery.min_memory) {
      filteredGpus = filteredGpus.filter(gpu => gpu.vramGb >= validatedQuery.min_memory!)
    }

    // Filter by max_price
    if (validatedQuery.max_price) {
      filteredGpus = filteredGpus.filter(gpu => gpu.pricePerHour <= validatedQuery.max_price!)
    }

    // Filter by vendor
    if (validatedQuery.vendor) {
      const vendorLower = validatedQuery.vendor.toLowerCase()
      filteredGpus = filteredGpus.filter(gpu =>
        gpu.name.toLowerCase().includes(vendorLower)
      )
    }

    // Sort GPUs
    switch (validatedQuery.sort) {
      case 'memory':
        filteredGpus.sort((a, b) => b.vramGb - a.vramGb)
        break
      case 'price':
        filteredGpus.sort((a, b) => a.pricePerHour - b.pricePerHour)
        break
      case 'performance':
        filteredGpus.sort((a, b) => b.tflops - a.tflops)
        break
    }

    // Return formatted response
    return NextResponse.json(formatGpuCatalogResponse(filteredGpus), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'public, s-maxage=21600, stale-while-revalidate=86400' // 6 hours
      }
    })

  } catch (error) {
    if (error instanceof ZodError) {
      recordErrorCode('/api/gpus', 'validation_error', 400)
      return NextResponse.json(
        ApiErrors.VALIDATION_ERROR(error.issues),
        { status: 400 }
      )
    }

    recordErrorCode('/api/gpus', 'internal_error', 500)
    return NextResponse.json(
      ApiErrors.INTERNAL_ERROR('Failed to fetch GPU catalog'),
      { status: 500 }
    )
  }
}

// CORS preflight
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    }
  })
}

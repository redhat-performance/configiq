import { NextRequest, NextResponse } from 'next/server'
import { recordErrorCode } from '@/lib/otel-metrics'

const HF_BASE = 'https://huggingface.co'
const MODEL_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/

interface HFConfigResponse {
  config:       Record<string, unknown>
  weightBytes:  number | null
  weightSource: 'safetensors_exact' | 'safetensors_overestimate' | 'hf_api_exact' | 'estimated'
  warnings:     string[]
}

interface HFErrorResponse {
  error:   'gated' | 'not_found' | 'network_error' | 'invalid_model_id'
  message: string
}

async function hfFetch(url: string, token?: string): Promise<Response> {
  const headers: Record<string, string> = {
    'Accept': 'application/json',
  }
  if (token) headers['Authorization'] = `Bearer ${token}`
  return fetch(url, { headers, next: { revalidate: 300 } })
}

export async function GET(req: NextRequest): Promise<NextResponse<HFConfigResponse | HFErrorResponse>> {
  const { searchParams } = new URL(req.url)
  const modelId = searchParams.get('model')?.trim()

  if (!modelId || !MODEL_ID_RE.test(modelId)) {
    recordErrorCode('/api/hf-config', 'invalid_model_id', 400)
    return NextResponse.json(
      { error: 'invalid_model_id', message: 'Model ID must be in the format "owner/model-name" using only letters, numbers, ".", "_" or "-".' },
      { status: 400 }
    )
  }

  const [owner, repo] = modelId.split('/')
  const token = req.headers.get('x-hf-token') ?? undefined
  const warnings: string[] = []

  // ── Step 1: fetch config.json ──────────────────────────────────────────────
  const configUrl = `${HF_BASE}/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/raw/main/config.json`
  let configRes: Response

  try {
    configRes = await hfFetch(configUrl, token)
  } catch {
    recordErrorCode('/api/hf-config', 'network_error', 503)
    return NextResponse.json(
      { error: 'network_error', message: 'Could not reach HuggingFace. Check your network connection.' },
      { status: 503 }
    )
  }

  if (configRes.status === 403 || configRes.status === 401) {
    recordErrorCode('/api/hf-config', 'gated', 403)
    return NextResponse.json(
      { error: 'gated', message: 'This model is gated. Provide a HuggingFace access token to continue.' },
      { status: 403 }
    )
  }

  if (configRes.status === 404) {
    recordErrorCode('/api/hf-config', 'not_found', 404)
    return NextResponse.json(
      { error: 'not_found', message: `Model "${modelId}" was not found on HuggingFace.` },
      { status: 404 }
    )
  }

  if (!configRes.ok) {
    recordErrorCode('/api/hf-config', 'network_error', 502)
    return NextResponse.json(
      { error: 'network_error', message: `HuggingFace returned HTTP ${configRes.status} for config.json` },
      { status: 502 }
    )
  }

  let config: Record<string, unknown>
  try {
    config = await configRes.json() as Record<string, unknown>
  } catch {
    recordErrorCode('/api/hf-config', 'network_error', 502)
    return NextResponse.json(
      { error: 'network_error', message: 'config.json from HuggingFace could not be parsed as JSON.' },
      { status: 502 }
    )
  }

  // Detect multimodal — vision models inflate safetensors total_size
  const isMultimodal = !!(
    config.text_config ||
    config.vision_config ||
    (Array.isArray(config.architectures) &&
      (config.architectures as string[]).some(
        (a) => a.includes('Conditional') || a.includes('VL') || a.includes('Vision')
      ))
  )

  // ── Step 2: weight bytes from safetensors.index.json ──────────────────────
  const indexUrl = `${HF_BASE}/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/resolve/main/model.safetensors.index.json`
  let weightBytes: number | null = null
  let weightSource: HFConfigResponse['weightSource'] = 'estimated'

  try {
    const indexRes = await hfFetch(indexUrl, token)
    if (indexRes.ok) {
      const index = await indexRes.json() as Record<string, unknown>
      const metadata = index.metadata as Record<string, unknown> | undefined
      if (typeof metadata?.total_size === 'number') {
        weightBytes = metadata.total_size as number
        weightSource = isMultimodal ? 'safetensors_overestimate' : 'safetensors_exact'
        if (isMultimodal) {
          warnings.push(
            'Weight memory includes vision model weights. ' +
            'Text-only weight cannot be isolated from the safetensors index alone.'
          )
        }
      }
    }
  } catch {
    // fallthrough to HF API
  }

  // ── Step 3: weight bytes from HF API (fallback) ───────────────────────────
  if (weightBytes === null) {
    try {
      const apiUrl = `${HF_BASE}/api/models/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}`
      const apiRes = await hfFetch(apiUrl, token)
      if (apiRes.ok) {
        const meta = await apiRes.json() as Record<string, unknown>
        const safetensors = meta.safetensors as Record<string, unknown> | undefined
        const params = safetensors?.parameters as Record<string, number> | undefined
        if (params) {
          const DTYPE_BYTES: Record<string, number> = {
            F32: 4, F16: 2, BF16: 2, F8_E4M3: 1, F8_E5M2: 1, I8: 1, I4: 0.5,
          }
          weightBytes = Object.entries(params).reduce(
            (sum, [dtype, count]) => sum + count * (DTYPE_BYTES[dtype] ?? 2),
            0
          )
          weightSource = 'hf_api_exact'
        }
      }
    } catch {
      // fallthrough — weightBytes stays null, page will use formula estimate
    }
  }

  if (weightBytes === null) {
    warnings.push(
      'Could not retrieve weight size from HuggingFace. ' +
      'An approximate formula will be used — confidence is lower for quantized or MoE models.'
    )
  }

  return NextResponse.json({ config, weightBytes, weightSource, warnings })
}

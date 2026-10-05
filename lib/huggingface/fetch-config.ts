/**
 * HuggingFace API Integration
 * Fetches model config.json from HuggingFace Hub
 */

export interface HFModelConfig {
  // Common fields across all models
  model_type?: string
  architectures?: string[]
  hidden_size?: number
  num_hidden_layers?: number
  num_attention_heads?: number
  num_key_value_heads?: number
  intermediate_size?: number
  vocab_size?: number
  max_position_embeddings?: number

  // MLA-specific (DeepSeek)
  kv_lora_rank?: number
  qk_rope_head_dim?: number
  qk_nope_head_dim?: number
  v_head_dim?: number

  // CLA-specific (Hunyuan)
  use_cla?: boolean
  cla_share_factor?: number

  // Sliding window (Mistral, Qwen)
  sliding_window?: number | null
  use_sliding_window?: boolean

  // MoE-specific
  num_local_experts?: number
  num_experts_per_tok?: number

  // SSM-specific (Mamba, Jamba)
  state_size?: number
  conv_kernel?: number
  expand?: number

  // Other common fields
  rope_theta?: number
  rms_norm_eps?: number
  tie_word_embeddings?: boolean
  torch_dtype?: string

  // Catch-all for unknown fields
  [key: string]: any
}

export interface FetchResult {
  success: boolean
  config?: HFModelConfig
  error?: string
  source: 'cache' | 'huggingface' | 'catalog'
}

// In-memory cache to avoid repeated fetches
const configCache = new Map<string, HFModelConfig>()

/**
 * Fetch model config.json from HuggingFace Hub
 *
 * @param modelId - HuggingFace model ID (e.g., "meta-llama/Llama-3.1-70B-Instruct")
 * @param hfToken - Optional HuggingFace API token for gated models
 * @returns Model configuration object
 */
export async function fetchModelConfig(
  modelId: string,
  hfToken?: string
): Promise<FetchResult> {
  // Check cache first
  if (configCache.has(modelId)) {
    console.log('📦 Using cached config for', modelId)
    return {
      success: true,
      config: configCache.get(modelId)!,
      source: 'cache'
    }
  }

  // Construct HuggingFace API URL
  const url = `https://huggingface.co/${modelId}/resolve/main/config.json`

  console.log('🔄 Fetching config from HuggingFace:', url)

  try {
    const headers: HeadersInit = {
      'Accept': 'application/json'
    }

    // Add authorization header if token provided
    if (hfToken && hfToken.trim()) {
      headers['Authorization'] = `Bearer ${hfToken}`
      console.log('✅ Authorization header added to request')
    } else {
      console.log('⚠️ No token - requesting as public (gated models will fail)')
    }


    const response = await fetch(url, {
      method: 'GET',
      headers,
      // Add timeout
      signal: AbortSignal.timeout(10000) // 10 second timeout
    })

    console.log('📥 Response status:', response.status, response.statusText)
    console.log('📥 Response headers:', {
      'content-type': response.headers.get('content-type'),
      'x-repo-commit': response.headers.get('x-repo-commit'),
      'x-error-code': response.headers.get('x-error-code'),
      'x-error-message': response.headers.get('x-error-message')
    })

    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        console.error('❌ 401/403 Unauthorized - Token is missing or invalid')
        console.error('   Token was provided:', hfToken ? 'Yes' : 'No')
        console.error('   Token starts with hf_:', hfToken ? hfToken.startsWith('hf_') : 'N/A')
        return {
          success: false,
          error: 'Model is gated or private. Add HuggingFace token above for accurate results.',
          source: 'huggingface'
        }
      } else if (response.status === 404) {
        // Check if this is a GGUF repo (common mistake)
        if (modelId.toLowerCase().includes('gguf') || modelId.toLowerCase().includes('-ggml')) {
          return {
            success: false,
            error: `This appears to be a GGUF/quantized model repository. GGUF repos don't have config.json. Please use the original base model instead (e.g., "google/gemma-2-12b-it" instead of "unsloth/gemma-4-12b-it-GGUF").`,
            source: 'huggingface'
          }
        }
        return {
          success: false,
          error: `Model not found on HuggingFace. Check model name spelling.`,
          source: 'huggingface'
        }
      } else {
        return {
          success: false,
          error: `HuggingFace API error (${response.status}). Using estimated config.`,
          source: 'huggingface'
        }
      }
    }

    // Parse JSON, handling invalid literals (Infinity, -Infinity, NaN)
    // Some HF configs use JavaScript literals instead of valid JSON
    const text = await response.text()
    const sanitized = text
      .replace(/:\s*Infinity\b/g, ': 1e308')        // Infinity → very large number
      .replace(/:\s*-Infinity\b/g, ': -1e308')      // -Infinity → very small number
      .replace(/:\s*NaN\b/g, ': null')              // NaN → null
      .replace(/,\s*Infinity\b/g, ', 1e308')        // Array: [0.0, Infinity]
      .replace(/,\s*-Infinity\b/g, ', -1e308')      // Array: [-Infinity, 0.0]
      .replace(/,\s*NaN\b/g, ', null')              // Array: [NaN, 1.0]

    const config: HFModelConfig = JSON.parse(sanitized)

    // Validate that it's actually a model config
    // Accept if has model_type/architectures OR SSM-specific fields
    const hasStandardFields = config.model_type || config.architectures
    const hasSsmFields = config.ssm_cfg != null || config.d_model != null

    if (!hasStandardFields && !hasSsmFields) {
      return {
        success: false,
        error: 'Invalid config.json - missing model_type or architectures field',
        source: 'huggingface'
      }
    }

    // Cache the result
    configCache.set(modelId, config)

    console.log('✅ Successfully fetched config for', modelId)
    console.log('   Architecture:', config.architectures?.[0] || config.model_type)
    console.log('   Layers:', config.num_hidden_layers)
    console.log('   Hidden size:', config.hidden_size)

    return {
      success: true,
      config,
      source: 'huggingface'
    }

  } catch (error) {
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        return {
          success: false,
          error: 'Request timeout - HuggingFace took too long to respond. Try again.',
          source: 'huggingface'
        }
      }
      return {
        success: false,
        error: `Network error: ${error.message}`,
        source: 'huggingface'
      }
    }
    return {
      success: false,
      error: 'Unknown error occurred while fetching config',
      source: 'huggingface'
    }
  }
}

/**
 * Extract parameter count from config
 * Estimates based on architecture if not explicitly provided
 */
export function extractParamCount(config: HFModelConfig): number {
  // Some configs have explicit parameter count
  if (config.num_parameters) {
    return config.num_parameters
  }

  // Estimate from architecture
  const {
    hidden_size = 4096,
    num_hidden_layers = 32,
    intermediate_size = 11008,
    vocab_size = 32000,
    num_attention_heads = 32
  } = config

  // Rough estimation formula:
  // Params ≈ vocab_size × hidden_size (embeddings)
  //        + num_layers × (4 × hidden_size² + 2 × hidden_size × intermediate_size)
  //        + vocab_size × hidden_size (output)

  const embedding_params = vocab_size * hidden_size
  const attention_params = 4 * hidden_size * hidden_size
  const ffn_params = 2 * hidden_size * intermediate_size
  const layer_params = attention_params + ffn_params
  const total_params = embedding_params + (num_hidden_layers * layer_params) + embedding_params

  return total_params
}

/**
 * Clear the config cache (useful for testing)
 */
export function clearConfigCache() {
  configCache.clear()
  console.log('🗑️ Cleared HuggingFace config cache')
}

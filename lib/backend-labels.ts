/**
 * Display names for inference backends in the UI.
 *
 * The backend ids come from the catalog and are what requests carry. This only
 * gives an id its product spelling, so a page can show "vLLM" while sending
 * `vllm`; it never changes a request value or a capability lookup.
 */
const BACKEND_DISPLAY_LABELS: Record<string, string> = {
  vllm: 'vLLM',
  'tensorrt-llm': 'TensorRT-LLM',
  sglang: 'SGLang',
}

/** The label to show for a backend, falling back to its catalog id. */
export function backendDisplayLabel(id: string): string {
  return BACKEND_DISPLAY_LABELS[id] ?? id
}

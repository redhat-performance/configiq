/**
 * Assistant availability, read per request rather than at module load so a
 * deployment can enable or move the model without a code branch, and so tests
 * can stub the environment.
 *
 * Absent configuration is reported as a reason, never as an empty answer: the
 * help popover hides the question box and says why.
 */
export interface AssistantConfig {
  enabled: boolean;
  model: string | null;
  tritonUrl: string | null;
  reason: string | null;
}

export function assistantConfig(): AssistantConfig {
  const model = process.env.ASSISTANT_MODEL?.trim() || null;
  const enabled = process.env.ASSISTANT_ENABLED === 'true';
  const tritonUrl = process.env.TRITON_INFERENCE_URL?.trim() || null;

  if (!enabled) {
    return {
      enabled: false,
      model: null,
      tritonUrl,
      reason: 'The assistant is not enabled on this deployment.',
    };
  }
  if (!model) {
    return { enabled: false, model: null, tritonUrl, reason: 'No assistant model is configured.' };
  }
  if (!tritonUrl) {
    return { enabled: false, model, tritonUrl, reason: 'The inference service is not configured.' };
  }
  return { enabled: true, model, tritonUrl, reason: null };
}

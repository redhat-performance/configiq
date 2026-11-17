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
  /**
   * Base URL of the OpenAI-compatible server that answers questions. It is the
   * assistant's own vLLM container, which is a different process from the FIL
   * classifier server that serves `/v2`.
   */
  endpoint: string | null;
  reason: string | null;
}

export function assistantConfig(): AssistantConfig {
  const model = process.env.ASSISTANT_MODEL?.trim() || null;
  const enabled = process.env.ASSISTANT_ENABLED === 'true';
  // TRITON_INFERENCE_URL is a compatibility fallback for deployments that
  // predate the split into two inference containers. It is the FIL classifier
  // endpoint, so a deployment that only sets it is misconfigured even though it
  // looks enabled; `deploy.sh` checks the endpoint this module resolves and
  // fails rather than letting the first question discover it.
  const endpoint =
    process.env.ASSISTANT_INFERENCE_URL?.trim() ||
    process.env.TRITON_INFERENCE_URL?.trim() ||
    null;

  if (!enabled) {
    return {
      enabled: false,
      model: null,
      endpoint,
      reason: 'The assistant is not enabled on this deployment.',
    };
  }
  if (!model) {
    return { enabled: false, model: null, endpoint, reason: 'No assistant model is configured.' };
  }
  if (!endpoint) {
    return { enabled: false, model, endpoint, reason: 'The inference service is not configured.' };
  }
  return { enabled: true, model, endpoint, reason: null };
}

const MODEL_PATH = /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/;
const SYSTEM_ID = /^[A-Za-z0-9._-]+$/;

function validParam(value: string | null, pattern: RegExp, maxLength: number) {
  const normalized = value?.trim() ?? '';
  return normalized.length > 0 &&
    normalized.length <= maxLength &&
    pattern.test(normalized)
    ? normalized
    : null;
}

/**
 * Reads the public performance-page handoff contract. Invalid or unexpected
 * values are ignored so a pasted URL cannot force unsupported control state.
 */
export function parsePerformancePrefill(search: string) {
  const params = new URLSearchParams(search);
  return {
    model: validParam(params.get('model'), MODEL_PATH, 200),
    system: validParam(params.get('system'), SYSTEM_ID, 80),
  };
}

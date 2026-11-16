import { metrics } from '@opentelemetry/api';
import { HELP_TOURS } from '@/lib/help/tours';

const meter = metrics.getMeter('configiq-webapp');

const errorResponses = meter.createCounter('configiq.errors', {
  description: 'Structured error responses returned by ConfigIQ',
  unit: '1',
});

const recommendErrors = meter.createCounter('configiq.recommend.errors', {
  description: 'Structured errors returned by recommendation requests',
  unit: '1',
});

const predictErrors = meter.createCounter('configiq.predict.errors', {
  description: 'Structured errors returned by prediction requests',
  unit: '1',
});

const recommendRequests = meter.createCounter('configiq.recommend.requests', {
  description: 'Recommendation requests received by ConfigIQ',
  unit: '1',
});

const predictRequests = meter.createCounter('configiq.predict.requests', {
  description: 'Prediction requests received by ConfigIQ',
  unit: '1',
});

const helpTourStarted = meter.createCounter('configiq.help.tour.started', {
  description: 'Product tours started by the user',
  unit: '1',
});

const helpTourCompleted = meter.createCounter('configiq.help.tour.completed', {
  description: 'Product tours completed or dismissed by the user',
  unit: '1',
});

const ERROR_CATEGORIES = new Set([
  'AISIM_ERROR',
  'AISIM_INVALID_RESPONSE',
  'AISIM_NO_CONFIGURATION',
  'AISIM_NOT_CONFIGURED',
  'AISIM_TIMEOUT',
  'AISIM_UNAVAILABLE',
  'AUTH_REQUIRED',
  'COSTINGS_ERROR',
  'COSTINGS_TIMEOUT',
  'COSTINGS_UNAVAILABLE',
  'gated',
  'INTERNAL_ERROR',
  'internal_error',
  'INVALID_REQUEST',
  'invalid_model_id',
  'MODEL_NOT_FOUND',
  'MOE_PARAMS_REQUIRED',
  'network_error',
  'not_found',
  'OOM',
  'UNHANDLED_ERROR',
  'validation_error',
]);

const MODEL_CATEGORIES = [
  ['gemma', /gemma/i],
  ['nemotron', /nemotron/i],
  ['deepseek', /deepseek/i],
  ['qwen', /qwen/i],
  ['gpt-oss', /gpt[-_ ]?oss/i],
  ['glm', /(?:^|[^a-z])glm(?:[^a-z]|$)/i],
  ['kimi', /kimi/i],
] as const;

function boundedErrorCode(code: string): string {
  return ERROR_CATEGORIES.has(code) ? code : 'unknown';
}

export function modelCategory(model: unknown): string {
  if (typeof model !== 'string') return 'other';
  const category = MODEL_CATEGORIES.find(([, pattern]) => pattern.test(model));
  return category?.[0] ?? 'other';
}

export function recordErrorCode(route: string, code: string, statusCode: number, model?: unknown): void {
  const attributes = {
    'http.route': route,
    'http.response.status_code': statusCode,
    'error.code': boundedErrorCode(code),
  };
  errorResponses.add(1, attributes);

  if (route === '/api/recommend' || route === '/api/predict') {
    const counter = route === '/api/recommend' ? recommendErrors : predictErrors;
    counter.add(1, {
      ...attributes,
      'model.category': modelCategory(model),
    });
  }
}

export function recordModelRequest(route: 'recommend' | 'predict', model: unknown): void {
  const counter = route === 'recommend' ? recommendRequests : predictRequests;
  counter.add(1, { 'model.category': modelCategory(model) });
}

/**
 * Bounded tour id: unknown values are dropped rather than becoming an
 * unbounded metric label.
 */
const HELP_TOUR_IDS = new Set(HELP_TOURS.map((tour) => tour.id));

export function recordHelpTour(tourId: string, phase: 'started' | 'completed'): void {
  if (!HELP_TOUR_IDS.has(tourId)) return;
  const counter = phase === 'started' ? helpTourStarted : helpTourCompleted;
  counter.add(1, { 'help.tour.id': tourId });
}

const helpSearches = meter.createCounter('configiq.help.searches', {
  description: 'Help searches, by whether the index returned a result',
  unit: '1',
});

/**
 * A search is counted by whether it matched, never by its text. An unmatched
 * search is the signal that a page or term needs documentation.
 */
export function recordHelpSearch(matched: number): void {
  helpSearches.add(1, { 'help.search.matched': String(matched > 0) });
}

const assistantAnswers = meter.createCounter('configiq.assistant.answers', {
  description: 'Assistant answers by outcome, including suppressed answers',
  unit: '1',
});

const assistantToolCalls = meter.createCounter('configiq.assistant.tool.calls', {
  description: 'Read-only assistant tool calls by tool and outcome',
  unit: '1',
});

/**
 * Assistants are counted by outcome, never by question text: customer workload
 * shapes can be sensitive and prompts are not logged by default.
 */
export function recordAssistantAnswer(outcome: 'answered' | 'suppressed' | 'error'): void {
  assistantAnswers.add(1, { 'assistant.outcome': outcome });
}

export function recordAssistantToolCall(tool: string, ok: boolean): void {
  // Tool names come from a fixed registry, so the label cannot grow unbounded.
  assistantToolCalls.add(1, { 'assistant.tool': tool, 'assistant.tool.ok': String(ok) });
}

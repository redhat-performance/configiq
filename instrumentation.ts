import type { Instrumentation } from 'next';

export async function register() {
  if (process.env.NEXT_RUNTIME !== 'nodejs') {
    return;
  }

  const { initOtel } = await import('./lib/otel');
  initOtel();
}

export const onRequestError: Instrumentation.onRequestError = async (_error, _request, context) => {
  if (process.env.NEXT_RUNTIME !== 'nodejs') {
    return;
  }

  const { recordErrorCode } = await import('./lib/otel-metrics');
  recordErrorCode(context.routePath, 'UNHANDLED_ERROR', 500);
};

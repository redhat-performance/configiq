import { getPrometheusMetrics } from '@/lib/otel';

export const dynamic = 'force-dynamic';

export async function GET(): Promise<Response> {
  return getPrometheusMetrics();
}

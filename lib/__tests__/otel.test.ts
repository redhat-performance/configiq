import { afterEach, describe, expect, it, vi } from 'vitest';

import { getOtlpEndpoint, getPrometheusMetrics, getTraceEndpoint } from '../otel';

const prometheusExporterKey = '__configiqPrometheusExporter';

describe('getTraceEndpoint', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    delete (globalThis as Record<string, unknown>)[prometheusExporterKey];
  });

  it('returns no endpoint when export is not configured', () => {
    expect(getTraceEndpoint()).toBeUndefined();
  });

  it('appends the trace path without duplicating a trailing slash', () => {
    vi.stubEnv('OTEL_EXPORTER_OTLP_ENDPOINT', 'https://collector.example/');

    expect(getTraceEndpoint()).toBe('https://collector.example/traces');
  });

  it('preserves an OTLP endpoint path prefix', () => {
    vi.stubEnv('OTEL_EXPORTER_OTLP_ENDPOINT', 'https://collector.example/otel/');

    expect(getTraceEndpoint()).toBe('https://collector.example/otel/traces');
  });

  it('prefers the explicit trace endpoint', () => {
    vi.stubEnv('OTEL_EXPORTER_OTLP_ENDPOINT', 'https://collector.example/base');
    vi.stubEnv('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT', 'https://collector.example/custom-traces');

    expect(getTraceEndpoint()).toBe('https://collector.example/custom-traces');
  });

  it('builds the generic metrics endpoint', () => {
    vi.stubEnv('OTEL_EXPORTER_OTLP_ENDPOINT', 'https://collector.example/otel');

    expect(getOtlpEndpoint('OTEL_EXPORTER_OTLP_METRICS_ENDPOINT', 'metrics'))
      .toBe('https://collector.example/otel/metrics');
  });

  it('disables export for an invalid endpoint', () => {
    const warning = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    vi.stubEnv('OTEL_EXPORTER_OTLP_ENDPOINT', 'not-a-url');

    expect(getTraceEndpoint()).toBeUndefined();
    expect(warning).toHaveBeenCalledOnce();
    warning.mockRestore();
  });

  it('serves metrics from the process-global exporter state', async () => {
    const getMetricsRequestHandler = vi.fn((_request, response) => {
      response.setHeader('Content-Type', 'text/plain');
      response.end('# HELP configiq_requests_total test metric\n');
    });
    (globalThis as Record<string, unknown>)[prometheusExporterKey] = {
      getMetricsRequestHandler,
    };

    const response = await getPrometheusMetrics();

    expect(response.status).toBe(200);
    expect(response.headers.get('content-type')).toBe('text/plain');
    expect(await response.text()).toContain('configiq_requests_total');
    expect(getMetricsRequestHandler).toHaveBeenCalledOnce();
  });
});

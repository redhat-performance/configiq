import { NodeSDK } from '@opentelemetry/sdk-node';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http';
import { PrometheusExporter } from '@opentelemetry/exporter-prometheus';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { PeriodicExportingMetricReader, type MetricReader } from '@opentelemetry/sdk-metrics';
import type { IncomingMessage, ServerResponse } from 'node:http';

const serviceName = process.env.OTEL_SERVICE_NAME || 'configiq-webapp';

export function getOtlpEndpoint(
  signalVariable: 'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT' | 'OTEL_EXPORTER_OTLP_METRICS_ENDPOINT',
  signalPath: 'traces' | 'metrics',
): string | undefined {
  const signalEndpoint = process.env[signalVariable];
  if (signalEndpoint) {
    try {
      const parsed = new URL(signalEndpoint);
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        throw new Error(`Unsupported protocol: ${parsed.protocol}`);
      }
      return parsed.toString();
    } catch (error) {
      console.warn(`OpenTelemetry ${signalVariable} is invalid; export disabled.`, error);
      return undefined;
    }
  }

  const endpoint = process.env.OTEL_EXPORTER_OTLP_ENDPOINT;
  if (!endpoint) {
    return undefined;
  }

  try {
    const baseUrl = new URL(endpoint);
    if (baseUrl.protocol !== 'http:' && baseUrl.protocol !== 'https:') {
      throw new Error(`Unsupported protocol: ${baseUrl.protocol}`);
    }
    baseUrl.pathname = `${baseUrl.pathname.replace(/\/+$/, '')}/`;
    return new URL(signalPath, baseUrl).toString();
  } catch (error) {
    console.warn('OpenTelemetry OTLP endpoint is invalid; export disabled.', error);
    return undefined;
  }
}

export function getTraceEndpoint(): string | undefined {
  return getOtlpEndpoint('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT', 'traces');
}

function logEndpoint(endpoint: string): string {
  const parsed = new URL(endpoint);
  return `${parsed.origin}${parsed.pathname}`;
}

type ConfigIqGlobal = typeof globalThis & {
  __configiqPrometheusExporter?: PrometheusExporter;
};

const configIqGlobal = globalThis as ConfigIqGlobal;
let prometheusExporter: PrometheusExporter | undefined;

export function initOtel(): NodeSDK {
  const traceEndpoint = getTraceEndpoint();
  const metricsEndpoint = getOtlpEndpoint('OTEL_EXPORTER_OTLP_METRICS_ENDPOINT', 'metrics');
  prometheusExporter = new PrometheusExporter({ preventServerStart: true });
  // Next can load instrumentation and route handlers in separate server bundles.
  // Keep the exporter in process-global state so /metrics sees the initialized SDK.
  configIqGlobal.__configiqPrometheusExporter = prometheusExporter;
  const metricReaders: MetricReader[] = [prometheusExporter];
  if (metricsEndpoint) {
    metricReaders.push(
      new PeriodicExportingMetricReader({
        exporter: new OTLPMetricExporter({ url: metricsEndpoint }),
      }),
    );
  }
  const sdk = new NodeSDK({
    ...(traceEndpoint
      ? { traceExporter: new OTLPTraceExporter({ url: traceEndpoint }) }
      : { spanProcessors: [] }),
    metricReaders,
    instrumentations: [getNodeAutoInstrumentations()],
    serviceName,
  });

  sdk.start();

  const shutdown = async (signal: NodeJS.Signals) => {
    try {
      await sdk.shutdown();
    } catch (error) {
      console.error('OpenTelemetry shutdown failed.', error);
    } finally {
      // Re-send the signal so Node retains its normal termination behavior.
      process.kill(process.pid, signal);
    }
  };

  process.once('SIGTERM', () => void shutdown('SIGTERM'));
  process.once('SIGINT', () => void shutdown('SIGINT'));

  console.info(
    traceEndpoint
      ? `OpenTelemetry initialized with OTLP endpoint: ${logEndpoint(traceEndpoint)}`
      : 'OpenTelemetry initialized without an OTLP exporter'
  );

  return sdk;
}

export function getPrometheusMetrics(): Promise<Response> {
  const exporter = prometheusExporter ?? configIqGlobal.__configiqPrometheusExporter;
  if (!exporter) {
    return Promise.resolve(new Response('# OpenTelemetry metrics are not initialized\n', { status: 503 }));
  }

  return new Promise(resolve => {
    const headers: Record<string, string> = {};
    let responseStatusCode = 200;
    const response = {
      get statusCode() {
        return responseStatusCode;
      },
      set statusCode(value: number) {
        responseStatusCode = value;
      },
      setHeader(name: string, value: string) {
        headers[name.toLowerCase()] = value;
      },
      end(body?: string) {
        const failed = body?.startsWith('# failed to export metrics:') ?? false;
        resolve(new Response(body ?? '', { status: failed ? 500 : responseStatusCode, headers }));
      },
    } as unknown as ServerResponse;

    exporter.getMetricsRequestHandler({} as IncomingMessage, response);
  });
}

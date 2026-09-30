# SPDX-License-Identifier: Apache-2.0

"""Shared Python utilities for ConfigIQ backend services.

This package holds cross-service machinery so the FastAPI microservices under
`services/` (aisimulators, aicostings, ...) don't each carry their own copy:

- `configiq.systems`       — GPU system ids and display names from the
                             aisimulate SDK (lazy-imported; no hard SDK dep)
- `configiq.otel`          — OpenTelemetry tracing + dual Prometheus/OTLP metrics
- `configiq.metrics`       — generic HTTP + process instruments and middleware
- `configiq.observability` — one-call wiring of the above onto a FastAPI app
- `configiq.mcp`           — expose a FastAPI app as MCP tools

The `otel`/`metrics`/`observability`/`mcp` modules import OpenTelemetry and
fastapi-mcp at module load, so they are only importable when the `otel`/`mcp`
extras are installed. Import them behind a try/except in services that keep
observability optional. `configiq.systems` has no hard dependencies and is
always importable.
"""

__version__ = "0.1.0"

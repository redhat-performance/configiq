# configiq (shared Python library)

Cross-service utilities for the ConfigIQ backend services under `services/`
(aisimulators, aicostings, ...). Consolidates machinery that would otherwise
be copy-pasted into every service.

Import name: `configiq`. Distribution name: `configiq`.

## Modules

| Module | Needs extra | Purpose |
|---|---|---|
| `configiq.systems` | — | GPU system ids (`supported_systems`) and vendor display names (`load_device_names_from_perf_data`) from the aisimulate SDK. Lazy-imports the SDK; degrades to empty results if absent. |
| `configiq.otel` | `otel` | OpenTelemetry tracing + dual Prometheus/OTLP-JSON metric providers. |
| `configiq.metrics` | `otel` | Generic HTTP + process instruments, `record_http_request`, `MetricsMiddleware`, and `get_meter` for domain instruments. |
| `configiq.observability` | `otel` | One call — `enable(app, ...)` — to wire tracing + metrics + middleware, plus `metrics_response(accept)` for a `/metrics` endpoint. |
| `configiq.mcp` | `mcp` | `mount(app, ...)` to expose a FastAPI app as MCP tools at `/mcp`. |

The `otel`/`metrics`/`observability`/`mcp` modules import their dependencies at
module load, so import them behind a `try/except ImportError` in services that
keep observability optional. `configiq.systems` has no hard dependencies and is
always importable.

## Usage in a service

```python
# GPU catalog (always available)
from configiq import systems
names = systems.load_device_names_from_perf_data()   # {id: "NVIDIA H200", ...}
ids = systems.supported_systems()                    # {"h200_sxm", ...}

# Observability (optional)
try:
    from configiq import observability
    observability.enable(app, service_name="aicostings",
                         service_version="0.1.0", meter_name="aicostings.api")
    _OBS = True
except ImportError:
    _OBS = False

# MCP (optional)
try:
    from configiq import mcp as mcp_support
    mcp_support.mount(app, name="aicostings", description="...")
except ImportError:
    pass
```

## Consumed as a path dependency

Services depend on this package via a uv path source, so local edits are picked
up without reinstalling:

```toml
# services/<service>/pyproject.toml
dependencies = ["configiq[otel,mcp]", ...]

[tool.uv.sources]
configiq = { path = "../configiq-py", editable = true }
```

In containers the build context is `services/`, and each service's Containerfile
installs this package before its own:

```dockerfile
COPY configiq-py /src/configiq-py
COPY <service>   /src/<service>
RUN pip install ./configiq-py && pip install ./<service>
```

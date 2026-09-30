# aicostings

GPU and LLM pricing API for AI infrastructure cost modelling.

## Overview

aicostings provides live pricing data for frontier LLM APIs and GPU cloud
compute. It scrapes pricing from multiple providers on a schedule, caches
results in Valkey, and serves them via a REST API.

**Endpoints:**

| Endpoint | Description |
|---|---|
| `GET /models` | Frontier LLM API pricing ($/M tokens, by provider and tier); `?source=merged\|openrouter\|litellm` selects a feed |
| `GET /systems` | GPU pricing (cloud $/hr by provider.region, hardware new/used range) |
| `GET /health` | Service status (last scrape timestamps, staleness flags) |
| `GET /metrics` | Prometheus / OTLP-JSON metrics (requires the `otel` extra) |
| `GET,POST /mcp` | MCP endpoint exposing the API as tools (requires the `mcp` extra) |

The GPU system catalog and vendor display names in `/systems` come from the
aisimulate SDK (via the shared `configiq.systems` module), so aicostings and
the aisimulators API never drift on which GPUs exist or how they're named.

## Quick start

This service depends on the in-tree [`configiq`](../configiq-py) package (a uv
path dependency) and on the aisimulate SDK wheels published by the Red Hat
fork. The SDK's Rust-compiled core is currently **manylinux x86_64 only**, so a
full install (and the test suite) runs on x86-64 Linux or in the container.

```bash
# Install with the shared lib, SDK wheels, and dev + observability extras.
uv sync --extra dev --extra otel --extra mcp

# Start Valkey (requires podman or docker)
podman run -d --name valkey -p 6379:6379 registry.redhat.io/rhel9/valkey-8

# Run the API server
uv run python -m tools.api_service.app

# Run tests
uv run pytest -v
```

The API will be available at `http://localhost:8080`.

### Container

The image bundles the shared `configiq` package, so its build context is the
parent `services/` directory:

```bash
podman build -f services/aicostings/Containerfile -t aicostings services/
# or, with Valkey, via compose from this directory:
podman compose up --build
```

## Data sources

### Frontier model pricing

Two peer pricing feeds plus a curated override layer. Every model in the
response carries a `source` field so callers can trace where a price came from,
and `GET /models?source=` serves a single feed or the merged view:

- **OpenRouter** (`GET https://openrouter.ai/api/v1/models`) — `source="openrouter"`;
  structured pricing for 100+ models.
- **LiteLLM Catalog** (`GET https://api.litellm.ai/model_catalog`) —
  `source="litellm"`; comparable coverage from an independent feed.
- **Curated YAML overrides** (`data/model-overrides.yaml`, optional) —
  `source="override"`; adds models missing from both feeds and corrects
  individual fields.

The default `merged` view is a union keyed by model id. On a duplicate id the
precedence is **overrides > OpenRouter > LiteLLM**. Overrides are applied as
*field-level patches* — an override overlays only the fields it specifies onto
the matching scraped record, so unspecified fields keep tracking the live feeds;
an override for an id neither feed has is a full add.

### Cloud GPU pricing

Credential-free providers (Azure, AWS, Vast.ai) run out of the box. The
remaining providers require a per-provider API key and only run when their
environment variable is set (see [Configuration](#configuration)) — until then
they're skipped, so `/health` only ever reports providers actually being
scraped. Their request/auth is wired but response parsing is **unverified**
against the live APIs; validate with a real key before trusting output.

| Provider | Method | Status |
|---|---|---|
| Azure | Retail Prices API | Implemented (no key) |
| AWS | Price List bulk JSON (on-demand; spot deferred) | Implemented (no key) |
| Vast.ai | Public marketplace API (spot-style median) | Implemented (no key) |
| RunPod | GraphQL API | Key-gated, parsing TODO |
| Lambda Labs | Cloud API | Key-gated, parsing TODO |
| GCP | Cloud Billing Catalog API | Key-gated, parsing TODO |
| Scaleway | Product catalog API | Key-gated, parsing TODO |
| IBM Cloud | Global Catalog API | Key-gated, parsing TODO |

AWS covers on-demand only; spot pricing needs an AWS account
(`DescribeSpotPriceHistory`) and is a deliberate follow-up.

### Hardware purchase costs

- `new_usd`: curated YAML (`data/hardware-costs.yaml`), manually updated
  quarterly. Flagged as indicative in the API response.
- `used_usd`: eBay completed listings (to be implemented).

## Configuration

| Variable | Default | Description |
|---|---|---|
| `VALKEY_URL` | `valkey://localhost:6379/0` | Valkey connection URL |
| `AWS_PRICING_REGIONS` | `us-east-1,us-west-2` | Comma-separated AWS regions to scrape GPU on-demand prices for |
| `RUNPOD_API_KEY` | _(unset)_ | Activates the RunPod scraper (parsing TODO) |
| `LAMBDA_API_KEY` | _(unset)_ | Activates the Lambda Labs scraper (parsing TODO) |
| `GCP_BILLING_API_KEY` | _(unset)_ | Activates the GCP scraper (parsing TODO) |
| `SCALEWAY_SECRET_KEY` | _(unset)_ | Activates the Scaleway scraper (parsing TODO) |
| `IBMCLOUD_API_KEY` | _(unset)_ | Activates the IBM Cloud scraper (parsing TODO) |

## Architecture

- **FastAPI + Pydantic** — REST API
- **APScheduler 3.x** — embedded async scheduler (`AsyncIOScheduler`)
- **aiohttp** — concurrent provider fetches
- **Valkey** — data store with TTL-based staleness
- **configiq** — shared GPU system catalog, OpenTelemetry wiring, and MCP mount
  (observability + MCP are optional extras; the app degrades gracefully without them)

### Observability

With the `otel` extra installed, the app emits OpenTelemetry traces (OTLP HTTP to
`OTEL_EXPORTER_OTLP_ENDPOINT`, default `http://localhost:4318`) and exposes
`/metrics` in either Prometheus text or OTLP-JSON format (content-negotiated via
the `Accept` header). Alongside the standard HTTP/process instruments, aicostings
records domain metrics for scrape health: `costings.scrape.total`,
`costings.scrape.duration`, and `costings.scrape.records`, each labelled by
source.

All pricing data is scraped on schedule and cached in Valkey. API endpoints
read directly from Valkey with no outbound calls at query time.

**Scrape schedule:**

| Interval | Sources |
|---|---|
| 24 hours | Cloud GPU rates, hosted model pricing (OpenRouter + LiteLLM) |
| 7 days | Hardware costs (curated YAML) |

Data is also loaded at startup for any dataset that is missing or already
stale, so a restart self-heals rather than waiting for the next tick.

## API reference

See `docs/api/openapi.yaml` for the full OpenAPI specification.

## Project structure

```text
data/
  gpu-id-mapping.yaml      # provider GPU name → AIC system ID
  hardware-costs.yaml      # GPU purchase prices (curated)
tools/
  api_service/
    app.py                 # FastAPI application + scheduler
    valkey.py              # Valkey client wrapper
    scrapers/
      models.py            # OpenRouter + curated overrides
      cloud_rates.py       # per-provider cloud rate scrapers
      hardware_costs.py    # hardware cost loader
tests/
  unit/                    # unit + integration tests
  fixtures/                # sample API responses for tests
Containerfile
compose.yml
docs/
  api/openapi.yaml
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

# Detailed architecture

This document describes the current ConfigIQ architecture. The frontend is a
Next.js 16 App Router application; GPU sizing and performance calculations are
provided by the AISimulators service rather than a client-side inference engine.

See [architecture.md](./architecture.md) for the short version and
[DESIGN_SYSTEM.md](./DESIGN_SYSTEM.md) for UI conventions.

## System overview

```text
Browser
  |
  v
Next.js App Router
  |-- pages and interactive components
  |-- same-origin /api route handlers
  |     |-- AISimulators proxy and response adapters
  |     |-- aicostings proxies
  |     |-- Hugging Face config lookup
  |     `-- application metrics and health
  |
  `--> AISimulators REST service
       `-- aisimulate SDK: GPU sizing, performance, and memory estimation

Next.js server routes --> aicostings REST service
                          `-- cloud GPU, hardware, and hosted-model pricing
```

In production, deployment is managed outside this repository. The documented
deployment contract is a standalone Next.js container plus the two Python
services, typically behind nginx. See [RELEASE_PROCESS.md](./RELEASE_PROCESS.md).

## Frontend structure

| Area | Location | Responsibility |
|---|---|---|
| App Router pages | `app/*/page.tsx` | Route-level composition |
| App shell | `components/layout/AppShell.tsx` | PatternFly page, masthead, and sidebar |
| Predict workflow | `app/predict/` | Explicit `/api/predict` performance requests and result display |
| Recommend workflow | `app/recommend/` | GPU sizing requests, including streamed progress |
| Simulator setting | `contexts/SettingsContext.tsx`, `app/settings/` | Persisted global simulator selection for Predict and Recommend |
| KV cache calculator | `app/kv-cache/` | `/api/memory` requests and memory breakdown display |
| Catalog hook | `lib/hooks/useCatalog.ts` | Same-origin GPU, model, and backend catalog |
| Costings hook | `lib/hooks/useCostings.ts` | Same-origin pricing and health data |
| Ground truth comparison | `app/validation/compare/` | Hugging Face benchmark filters, AISimulators comparison, and PatternFly charts |
| API adapters | `lib/api/` | Request shapes, response normalization, and error handling |
| Legacy catalogs | `lib/gpu-math/` | Historical static data and types; no new sizing formulas |

React components own presentation and user interaction. They should not contain
GPU sizing formulas or call external service URLs directly from the browser.

## Request flows

### Predict performance

1. `app/predict/Performance.tsx` builds an explicit model, system, workload,
   backend, simulator, and parallelism request. The simulator comes from the
   global Settings selection. BLIS P/D predictions also include the chosen
   network fabric and per-pool settings.
2. `lib/api/estimate-adapter.ts` posts it to `/api/predict?include=config,memory`.
3. The Next.js route delegates to `lib/api/predict-proxy.ts`, which calls the
   AISimulators `/predict` endpoint server-side.
4. The adapter normalizes timing, memory, serving configuration, and warnings
   into the `InferenceConfigResult` shape used by the page.

### Recommend sizing

1. The recommend page sends a validated request to `/api/recommend`.
2. `lib/api/recommend.ts` calls AISimulators `/recommend` and normalizes the
   selected configuration, simulator provenance, topology, phases, throughput,
   and any memory data the selected simulator provides.
3. Streaming clients request `text/event-stream`; AISimulate emits search-window
   progress while `incrementalRecommend` searches and refines GPU bounds. BLIS
   performs one bounded aggregate TP search, plus a bounded P/D grid when a
   fabric is selected, and returns the actual search summary.

### Catalog and pricing

1. `useCatalog` calls `/api/catalog`.
2. The catalog route fetches `/systems`, `/models`, `/backends`, and `/networks`
   from AISimulators and combines them into one browser response. The network
   list supplies fabrics for BLIS P/D predictions; AISimulate does not consume it.
3. When costings are enabled, `useCostings` calls `/api/costings/models`,
   `/api/costings/systems`, and `/api/costings/health`.
4. Those routes proxy the aicostings service; pricing data is not part of the
   AISimulators sizing response.

### Ground truth comparison

1. The comparison page requests the current benchmark rows through the same-origin
   `/api/validation/ground-truth` route.
2. The route reads the public Hugging Face Parquet-file listing and parses its
   files directly, caches the current rows for five minutes, and derives
   dependent filter options and exact measured runtime configurations from them.
3. Selected measurements are grouped by concurrency; ConfigIQ calls its existing
   `/api/predict` proxy for estimates at up to 24 measured concurrency levels,
   with at most three requests in flight. The request carries the simulator
   selected in Settings, and the page refuses unsupported model/system/backend
   combinations instead of falling back to another simulator. MoE expert
   parallelism is set from catalog or Hugging Face expert-count metadata so its
   width matches TP.
4. PatternFly charts show measured and predicted values for TTFT, TPOT,
   throughput, throughput per user, and request latency. The predicted series is
   labeled with the simulator that actually produced the results. Aggregate
   throughput comes from the simulator's cluster-wide rate; per-user throughput
   uses its reported per-user rate when present. An unavailable dataset is shown
   separately from a configuration that simply matches no rows.

### Tested-model envelopes

1. `lib/hooks/useTestedModels.ts` loads `/api/tested-models`, which parses the
   host-mounted `registry.json` through `lib/tested-models/registry-types.ts`.
   Schema v2 exposes each pair's status (`validated` or
   `measured_no_envelope`), coverage, gate result, diagnostics, and the
   `approvedModels` list.
2. Model pickers treat only registry-validated models as tested. Approved models
   and measured-only pairs remain selectable but never claim a validated
   envelope.
3. Before a prediction result is shown, the sizing pages call
   `/api/tested-models/classify`. A pair that fails the gate returns
   `measured_no_envelope` with no verdict; unavailable registry data is distinct
   from "untested".
4. The feedback component and classify response link to the canonical pair graph
   at `/tested-models/<pair-id>`, falling back to
   `/tested-models?model=<modelId>&system=<systemId>`. Legacy `?pair=` links
   redirect to the pair route.

## API route map

| Route | Method | Upstream or role |
|---|---|---|
| `/api/recommend` | POST | AISimulators `/recommend`, with normalized and streaming modes |
| `/api/predict` | POST | AISimulators `/predict` |
| `/api/estimate` | POST | Compatibility alias for predict |
| `/api/memory` | POST | AISimulators `/memory` |
| `/api/catalog` | GET | Combined AISimulators systems, models, backends, and network fabrics |
| `/api/validation/ground-truth` | GET | Hugging Face dataset-backed filters and aggregated benchmark measurements |
| `/api/tested-models` | GET | Host-mounted tested-model registry (schema v1/v2, gate, statuses) |
| `/api/tested-models/classify` | POST | Gate-aware envelope classification; no verdict for non-validated pairs |
| `/api/tested-models/dashboard/[pairId]` | GET | Sanitized measured graph data for one model/system pair |
| `/api/gpus` | GET | AISimulators systems transformed to the legacy GPU shape |
| `/api/hf-config` | GET | Hugging Face model configuration lookup |
| `/api/costings/models` | GET | aicostings hosted-model pricing |
| `/api/costings/systems` | GET | aicostings GPU cloud and hardware pricing |
| `/api/costings/health` | GET | aicostings source health |
| `/api/costings/sources` | GET | aicostings source metadata |
| `/api/health` | GET | Application and gateway health |
| `/api/metrics` | GET | Application metrics |

The canonical external service contracts are maintained by the service
repositories and their OpenAPI specifications under `services/`.

## Service structure

```text
services/
  configiq-py/       Shared Python library: catalog, OpenTelemetry, MCP
  aisimulators/      FastAPI wrapper around the aisimulate SDK
  aicostings/        FastAPI pricing service with scheduled scrapers and Valkey
```

The aisimulators wrapper does not reimplement GPU math. The SDK is installed as
an external wheel in the service container. The aicostings service uses the
shared `configiq` package for GPU system identity and display names.

## Design constraints

- Keep sizing and performance formulas in AISimulators.
- Keep pricing collection and normalization in aicostings.
- Keep browser calls same-origin; resolve gateway URLs on the server.
- Define request and response interfaces in `lib/api/` rather than using `any`.
- Use server components by default and add client components only for browser
  APIs, state, or event handlers.
- Use PatternFly v6 and the Red Hat typography and color tokens.

## Adding a new calculation

1. Add or update the AISimulators API contract and service implementation.
2. Add a typed request/response adapter under `lib/api/`.
3. Add or update the corresponding Next.js route handler if a new proxy is
   required.
4. Update the page and tests without duplicating the formula in React.
5. Update this route map and the relevant service OpenAPI documentation.

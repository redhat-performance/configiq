# AISimulators REST API

A thin FastAPI wrapper over the [aisimulate](https://github.com/redhat-performance/aisimulate)
SDK, exposing GPU recommendation, performance estimation, and memory estimation
as HTTP endpoints. ConfigIQ's `app/api/*` proxy routes call this service.

This directory contains **only** the REST wrapper. The SDK itself is not vendored
here — it is installed as a single wheel published by the Red Hat fork's GitHub
Releases. That one `aisimulate` wheel bundles the Rust-compiled core and exposes
the `aisimulate` and `aisimulate_core` APIs used by this service.

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/recommend` | GPU sizing recommendations |
| POST | `/predict` | Single-point performance prediction |
| POST | `/estimate` | Deprecated alias for `/predict` |
| POST | `/memory` | Memory / KV-cache breakdown |
| GET | `/models` | Supported models |
| GET | `/systems` | Supported GPU systems |

Full spec: [`docs/api/openapi.yaml`](docs/api/openapi.yaml).

## Run with the container (recommended)

The `aisimulate` wheel ships only for `linux/amd64` (manylinux x86_64), so the
container is the portable way to run this service. The build context is
`services/` (not this directory) so the image can pull in the shared
`configiq-py` package; pass the Containerfile with `-f` and build for
`linux/amd64` (emulated on Apple Silicon):

```bash
docker build --platform linux/amd64 \
  -f services/aisimulators/Containerfile -t aisimulators services/
docker run --rm -p 7860:7860 aisimulators
curl http://localhost:7860/systems
```

The image also exposes the simulator CLIs at stable paths:

```text
/usr/local/bin/aisimulate
/usr/local/bin/blis
```

BLIS v0.13.0 is built from source in the multi-stage build (commit
`1813579275e3f782ecc1a83ef508017935121235`) rather than copied from an image
digest. Upstream never published a GHCR image for v0.12.0 or v0.13.0: its own
`Dockerfile` names `golang:1.24-alpine` while `go.mod` requires Go 1.27.2, so the
build failed at the tag. Building here keeps the Go base under our control
(`BLIS_GO_VERSION`). The image bundles the three inputs a BLIS run needs, all at
pinned commits:

| Path | Contents | Pin |
|---|---|---|
| `/opt/blis/catalog` | declared facts: model graphs, chips, fabrics, workloads | blis-catalog `0.3.0`, `676326ec6c708e91ec4981bd6b03922a7fe60512` |
| `/opt/blis/registry` | fitted coefficient sets a scenario names | blis-registry `v0.2.0`, `7f6662d7188dc3dc873a37623e1f7bbfe1d6081b` |
| `/opt/blis/scenarios` | the upstream example scenarios for the same BLIS release | BLIS `v0.13.0`, `1813579275e3f782ecc1a83ef508017935121235` |

The build then runs [`tools/export_blis_catalog.py`](tools/export_blis_catalog.py)
to reconcile the wheel's packaged AISimulate model configs with that catalog:

- **The upstream catalog always wins a shared name.** AISimulate can only add
  models the catalog does not already define, so a reviewed BLIS graph is never
  silently replaced.
- **An added model must earn its place.** The exporter runs blis-catalog's own
  `scripts/derive_graph.py` for every added model and keeps it only if a
  `graph.yaml` comes out. A model BLIS cannot price is removed and recorded.
- **Nothing is dropped silently.** Unsupported architectures, name collisions and
  malformed configs are recorded in `/opt/blis/support.json` with a stable reason
  code, so the API can explain an unavailable model instead of returning a generic
  error.

`/opt/blis/support.json` is the API contract for simulator selection. It records,
per catalog model, whether it came from upstream or AISimulate, whether it has a
graph, its quantization, how an AISimulate config relates to the upstream entry of
the same name (`identical`, `different_quantization`, `different`, with the
differing paths), and any `equivalent_catalog_name`. It also lists the skipped
AISimulate models with reasons, plus the bundled hardware, fabrics and scenarios.

Because the upstream catalog wins, BLIS prices the catalog's reviewed config even
when the wheel's copy differs. For example, upstream `glm-5` is the FP8 variant
while AISimulate's `zai-org/GLM-5` is the unquantized base; the manifest marks that
shared entry `different_quantization` and records the FP8 AISimulate model
separately as `glm-5-fp8` with `equivalent_catalog_name: glm-5`.

## Simulator API behavior

`POST /predict` and `POST /recommend` accept `simulator: "aisimulate" | "blis"`;
the default remains `aisimulate`. A BLIS request is gated by `support.json` and
uses the manifest's catalog model and hardware identities. The service generates
a temporary Scenario + Deployment for the requested configuration and passes it
to `blis run` with the image-pinned catalog, registry, and scenario resources.
Unsupported models, systems, or settings return an explicit client error. A BLIS
failure is never retried through AISimulate.

Simulator provenance includes the requested and resolved model, hardware,
engine version, and any model-configuration relation reported by the selected
simulator. Non-identical model configs are surfaced in the response so clients
can label the comparison. BLIS predictions
do not include KV-cache memory breakdowns; the single-backend `/memory` endpoint
continues to use AISimulate and reports `simulator: "aisimulate"`.

BLIS `/predict` supports both colocated and prefill/decode-disaggregated vLLM
scenarios. P/D requests require a `fabric` from `GET /networks`, the same GPU
system for both pools, and `attention_dp_size=1`. The adapter generates separate
role pools, a NIXL transfer connector, and a router configuration that
disaggregates every request. BLIS `/recommend` always searches aggregate TP
candidates. It defaults to `roce-200g` when the `fabric` field is omitted and
also searches bounded P/D TP and worker-count combinations, with one node per
pool and DP=PP=1. Send `fabric: null` to request aggregate-only candidates; the
response then warns that P/D candidates were not searched.

BLIS recommendation requires a concurrency workload. Its candidate ceiling is the same
`AISIMULATORS_MAX_CANDIDATE_GPUS` service setting used by AISimulate, further
limited by one node for aggregate-only search or two nodes (one per P/D pool)
when a fabric is active. A total search budget bounds the individual simulations. The service
caps that budget at 60 seconds; the web
gateway may lower it further to leave room before its request deadline. The
response reports the configured, hardware, and effective ceilings plus the
actual candidate and evaluated GPU counts.

The image sets `BLIS_CATALOG`, `BLIS_REGISTRY`, `BLIS_SCENARIOS` and
`BLIS_SUPPORT`. BLIS itself only reads `BLIS_CATALOG` from the environment; the
registry and scenario directory have no environment fallback, so callers must pass
`--registry` and `--scenarios` explicitly as well:

```bash
blis run --catalog "$BLIS_CATALOG" --registry "$BLIS_REGISTRY" \
  --scenarios "$BLIS_SCENARIOS" \
  --scenario deepseek-v4.1-flash-h100-tp8-dspark5.yaml \
  --rate 10 --num-requests 100
```

To move to a new BLIS release, bump `BLIS_COMMIT`, `BLIS_CATALOG_COMMIT`,
`BLIS_REGISTRY_COMMIT` and (if `go.mod` requires it) `BLIS_GO_VERSION` together;
see
[docs/RELEASE_PROCESS.md](../../docs/RELEASE_PROCESS.md#bumping-blis).


The `aisimulate` SDK wheel is pinned in `pyproject.toml` by exact fork-release
download URL (the `aisimulate @ …` entry), so the image always installs that
exact artifact — no build args. To move to a newer SDK build, update that URL and
regenerate `uv.lock`; see
[docs/RELEASE_PROCESS.md](../../docs/RELEASE_PROCESS.md#bumping-the-simulation-sdk).

## Local development

The `aisimulate` wheel is amd64-only, so this works on a `linux/amd64` host:

```bash
cd services/aisimulators
uv venv && uv pip install -e . --group dev
uv run pytest          # tests mock the SDK; no perf DB required
uv run ruff check .
uvicorn tools.api_service.app:app --reload --port 7860
```

On other platforms (e.g. macOS/arm) the `aisimulate` wheel is not available; use
the container instead.

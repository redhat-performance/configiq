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

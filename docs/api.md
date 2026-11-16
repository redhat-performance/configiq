# AISimulators REST API reference

The REST API is served by AISimulators, the sizing service behind ConfigIQ. It
provides GPU recommendation and performance estimation for LLM inference.
Interactive docs are at `/docs` on the service; the OpenAPI spec lives in
[`services/aisimulators/docs/api/openapi.yaml`](../services/aisimulators/docs/api/openapi.yaml).

Every sizing response reports the `simulator` that produced it. Optional
sections such as memory breakdowns are present only when that simulator supports
them; absent means unavailable, never zero.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/recommend` | Find an optimal GPU configuration for a workload |
| POST | `/predict` | Single-point performance estimate for a given parallelism configuration |
| POST | `/estimate` | Deprecated alias for `/predict` |
| POST | `/memory` | Estimate the GPU memory breakdown for a model configuration |
| GET | `/models` | List supported model architectures |
| GET | `/systems` | List supported GPU systems |
| GET | `/backends` | List supported serving backends and engine versions |
| GET | `/networks` | List network fabrics reported by the serving simulator |
| GET | `/simulators` | List simulators this deployment can run, and the default |
| GET | `/metrics` | Service metrics via content negotiation |

Which tool to use for which ask is covered in
[user scenarios](user-scenarios.md#which-tool-to-use-when).

## Request defaults

Defaults belong to the API, not to the webapp. Only send the values you
actually chose; when a field is omitted the service resolves it.

`POST /recommend`:

| Field | Default | Notes |
|---|---|---|
| `isl` | `4000` | Input sequence length |
| `osl` | `1000` | Output sequence length |
| `top_n` | `5` | Number of configurations returned. `concurrency_capacity` returns `8` by default, or `6` when no disaggregated option exists, unless `top_n` is set explicitly |
| `objective` | `min_gpus` | Also `throughput_demand`, `concurrency_capacity`, `pareto` |
| `sla_mode` | `enforce` | `none` is required by the demand and capacity objectives |
| `ttft` / `tpot` | none | Required by `sla_mode='enforce'` unless an end-to-end latency bound is set |

`objective='min_gpus'` with `sla_mode='enforce'` requires exactly one of
`target_concurrency` or `target_request_rate`. The demand and capacity
objectives have their own requirements, described in the OpenAPI spec.

`POST /predict` requires a model, a GPU system, and explicit parallelism. It
does not search: it evaluates the configuration you give it.

`POST /memory` requires `model_path` and `system`. It is an AISimulate
capability; a simulator that does not model memory rejects the request rather
than returning an empty breakdown.

Webapp workload presets are not API defaults. Preset names and values come from
the deployment's `config.json`, so a preset is a convenience for filling the
form, not a statement about the API contract.

## Examples

### Recommend a configuration

```bash
curl -X POST https://aisimulators.dev/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "meta-llama/Llama-3.1-70B-Instruct",
    "system": "h100_sxm",
    "target_concurrency": 32,
    "ttft": 2000,
    "tpot": 30,
    "isl": 4000,
    "osl": 1000
  }'
```

### Predict performance

```bash
curl -X POST https://aisimulators.dev/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "meta-llama/Llama-3.1-70B-Instruct",
    "system": "h100_sxm",
    "tp_size": 8,
    "pp_size": 1
  }'
```

`/predict` returns TTFT, TPOT, throughput, and a memory breakdown when the
simulator provides one. `/estimate` is the deprecated alias; use `/predict` in
new work.

### Estimate memory

```bash
curl -X POST https://aisimulators.dev/memory \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "meta-llama/Llama-3.1-70B-Instruct",
    "system": "h100_sxm"
  }'
```

## Run the service locally

```bash
cd services/aisimulators
uv venv && uv pip install -e . --group dev
uv run uvicorn tools.api_service.app:app --reload --port 7860
```

The API is then at `http://localhost:7860` with interactive docs at
`http://localhost:7860/docs`. The `otel` and `mcp` extras are optional:

```bash
uv pip install -e '.[otel]'
uv pip install -e '.[mcp]'
```

## Sizing multiple workloads

One call per workload. There is no single mixed-workload endpoint. See
[user scenarios](user-scenarios.md#sizing-multiple-workloads).

## MCP

The sizing and pricing APIs are available as MCP tools. The ConfigIQ MCP
gateway presents them with short, stable tool names, and the sizing service can
also mount MCP directly with its own generated names. See
[MCP server](mcp-server.md) for the two shapes.

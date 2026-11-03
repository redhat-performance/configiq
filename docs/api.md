# AISimulate REST API reference

The REST API is served by AISimulate, the sizing engine behind ConfigIQ. It
provides GPU recommendation and performance estimation for LLM inference.
Interactive docs: https://aisimulators.dev/docs (OpenAPI spec:
[`services/aisimulators/docs/api/openapi.yaml`](../services/aisimulators/docs/api/openapi.yaml)).

Cost and pricing endpoints are not documented here yet. Pricing
documentation is deferred to a separate change.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/recommend` | Find an optimal GPU configuration for a workload |
| POST | `/predict` | Single point performance estimate for a given parallelism configuration |
| POST | `/estimate` | Deprecated alias for `/predict` |
| POST | `/memory` | Estimate the GPU memory breakdown for a model configuration |
| GET | `/models` | List supported model architectures |
| GET | `/systems` | List supported GPU systems |
| GET | `/backends` | List supported serving backends and engine versions |
| GET | `/networks` | List supported network fabrics |
| GET | `/metrics` | Service metrics via content negotiation |

Which tool to use for which ask is covered in
[user scenarios](user-scenarios.md#which-tool-to-use-when).

## Examples

### Recommend a configuration

```bash
curl -X POST https://aisimulators.dev/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "meta-llama/Llama-3.1-70B",
    "system": "h100_sxm",
    "target_concurrency": 32,
    "isl": 4000,
    "osl": 1000
  }'
```

### Predict performance

`POST /predict` takes a model plus an explicit parallelism configuration and
returns the expected TTFT, TPOT, and throughput. `/estimate` is the
deprecated alias. Use `/predict` in new work.

### Estimate memory

`POST /memory` takes a model configuration and returns the memory breakdown
including the KV cache.

## Run the service locally

```bash
cd services/aisimulators
pip install -e ".[service]"
python -m uvicorn tools.api_service.app:app --reload --host 0.0.0.0 --port 8000
```

The API is then at `http://localhost:8000` with interactive docs at
`http://localhost:8000/docs`.

## Sizing multiple workloads

One call per workload. There is no single mixed workload endpoint. See
[user scenarios](user-scenarios.md#sizing-multiple-workloads).

## MCP

The same API is available as MCP tools through the ConfigIQ MCP gateway.
See [MCP server](mcp-server.md).

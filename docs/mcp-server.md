# ConfigIQ MCP server

ConfigIQ exposes one stable MCP server, presented by the MCP gateway
([`services/mcp-gateway`](../services/mcp-gateway)). It proxies the inference
sizing and pricing APIs while keeping MCP session state in one process,
separate from the horizontally scaled sizing service.

## Install

### Use the deployed instance (no install)

The gateway is already running at `https://aisimulators.dev/mcp`. Point any MCP
client at it and go straight to the example below.

### Run the gateway locally

```bash
git clone https://github.com/redhat-performance/configiq.git
cd configiq/services/mcp-gateway
uv sync
uv run uvicorn api_service.app:app --host 0.0.0.0 --port 8090
```

The MCP server is then at `http://localhost:8090/mcp` (SSE) and
`http://localhost:8090/mcp/http` (Streamable HTTP).

### Expose the REST API as an MCP server directly

The sizing service can serve MCP directly without the gateway. Install the
`mcp` extra, then run the API as usual:

```bash
cd services/aisimulators
uv venv && uv pip install -e '.[mcp]'
uv run uvicorn tools.api_service.app:app --host 0.0.0.0 --port 7860
```

The MCP server is then at `http://localhost:7860/mcp` and
`http://localhost:7860/mcp/http`. Tool names there are FastAPI's generated
operation IDs (for example `post_recommend_recommend_post`), not the gateway's
short names. The shared `configiq` package provides the mount helper; the
service calls it after its last route is declared, because the tool list is
snapshotted when the server is built.

## Connect

Point any MCP client at the gateway URL:

| Client | URL |
|---|---|
| Claude Desktop | `https://aisimulators.dev/mcp` or your local URL |
| MCP Inspector | `npx @modelcontextprotocol/inspector <url>` |

Claude Desktop reads `claude_desktop_config.json` from:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "configiq": {
      "url": "https://aisimulators.dev/mcp"
    }
  }
}
```

For a local gateway, use `"url": "http://localhost:8090/mcp"` instead. For a
client that only speaks Streamable HTTP, use the `/mcp/http` URL.

## Use it: an example

Ask in natural language:

> Recommend a GPU configuration for Llama 3.1 70B Instruct on an H100 for 32
> concurrent users, 4,000 token inputs, 1,000 token outputs, TTFT under 2
> seconds and TPOT under 30 ms.

The client calls the `recommend` tool with:

```json
{
  "model_path": "meta-llama/Llama-3.1-70B-Instruct",
  "system": "h100_sxm",
  "target_concurrency": 32,
  "ttft": 2000,
  "tpot": 30,
  "isl": 4000,
  "osl": 1000
}
```

`model_path` and `system` are required. With `objective='min_gpus'` (the
default) and `sla_mode='enforce'` (the default), set exactly one of
`target_concurrency` or `target_request_rate`, plus a latency bound. `isl`,
`osl`, and `top_n` have API defaults; `ttft` and `tpot` do not, because a
missing latency target is not the same as a generous one. The result lists up
to `top_n` configurations that meet the targets, each with GPU count, TP/PP, and
replicas.

A KV cache ask works the same way:

> Estimate the KV cache memory for Llama 3.1 70B Instruct on h100_sxm.

calls the `memory` tool with:

```json
{
  "model_path": "meta-llama/Llama-3.1-70B-Instruct",
  "system": "h100_sxm"
}
```

`system` is required: the same model has a different memory footprint on each
GPU system, so the service will not guess one.

Sizing requests run one workload per call. See
[user scenarios](user-scenarios.md#sizing-multiple-workloads) for guidance on
sizing multiple workloads.

| Tool | Purpose |
|---|---|
| `recommend` | Find an optimal GPU configuration for a workload |
| `predict` | Predict performance for a given configuration |
| `estimate` | Deprecated alias for `predict` |
| `memory` | Estimate KV cache memory for a model configuration |
| `backends` | List supported serving backends and engine versions |
| `models` | List supported model architectures |
| `systems` | List supported GPU systems |
| `pricing_sources` | List available hosted-model pricing feeds |
| `pricing_models` | List hosted model prices from a pricing feed |
| `pricing_systems` | List GPU system prices from a pricing feed |

The gateway does not currently expose `GET /networks`, even though the sizing
API does. Ask the API directly if you need the fabric list.

## Tools

Inference sizing tools proxy to the AISimulators REST API. Tool names are the
gateway operation IDs. Full request parameters are in the
[REST API reference](api.md). The MCP tools take the same bodies.

Pricing tools read the aicostings service. Pricing is cost data, not sizing
data: it never changes a GPU count.

## Deployment notes

- Legacy `aisimulators.dev` and `aicostings.dev` MCP paths are routed to this
  gateway by the deployment repository during migration. Their REST endpoints
  remain owned by the original services.
- The gateway container listens on port 8090. The sizing service container
  listens on port 7860.

## Related

- [REST API](api.md): endpoint reference and examples
- [User scenarios](user-scenarios.md): which tool to use for which ask

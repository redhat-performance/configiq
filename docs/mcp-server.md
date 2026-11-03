# ConfigIQ MCP server

ConfigIQ exposes one stable MCP server, presented by the MCP gateway
([`services/mcp-gateway`](../services/mcp-gateway)). It proxies the
inference sizing APIs while keeping MCP session state in one process,
separate from the horizontally scaled AISimulators REST service.

Cost and pricing MCP tools exist on the gateway but are not documented here
yet. Pricing documentation is deferred to a separate change.

## Install

### Use the deployed instance (no install)

The gateway is already running at `https://aisimulators.dev/mcp`. Point any
MCP client at it and go straight to the example below.

### Run the gateway locally

```bash
git clone https://github.com/redhat-performance/configiq.git
cd configiq/services/mcp-gateway
uv sync
uv run uvicorn api_service.app:app --host 0.0.0.0 --port 8090
```

The MCP server is then at `http://localhost:8090/mcp` (SSE) and
`http://localhost:8090/mcp/http` (Streamable HTTP).

### Expose the AISimulators API as an MCP server

The FastAPI service can serve MCP directly without the gateway. Install the
`mcp` extra, then run the API as usual:

```bash
cd services/aisimulators
pip install -e ".[mcp]"
python -m uvicorn tools.api_service.app:app --host 0.0.0.0 --port 8000
```

The MCP server is then at `http://localhost:8000/mcp` and
`http://localhost:8000/mcp/http`. Tool names are the FastAPI operation IDs
(for example `post_recommend_recommend_post`) rather than the gateway short
names.

## Connect

Point any MCP client at the gateway URL:

| Client | URL |
|---|---|
| Claude Desktop | `https://aisimulators.dev/mcp` or your local URL |
| MCP Inspector | `npx @modelcontextprotocol/inspector <url>` |

Claude Desktop config (`~/Library/Application Support/Claude/config.json` on
macOS, `~/.config/claude/config.json` elsewhere):

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

> Recommend a GPU configuration for Llama 3.1 70B on an H100 for 32
> concurrent users, 4,000 token inputs, 1,000 token outputs, TTFT under
> 2 seconds and TPOT under 30 ms.

The client calls the `recommend` tool with:

```json
{
  "model_path": "meta-llama/Llama-3.1-70B",
  "system": "h100_sxm",
  "target_concurrency": 32,
  "isl": 4000,
  "osl": 1000,
  "ttft": 2000,
  "tpot": 30
}
```

`model_path` and `system` are required. Set exactly one of
`target_concurrency` or `target_request_rate`. Defaults if omitted: `isl`
4000, `osl` 1000, `ttft` 2000 ms, `tpot` 30 ms, `top_n` 5. The result lists
up to `top_n` configurations that meet the targets, each with GPU count,
TP/PP, and replicas.

A KV cache ask works the same way:

> Estimate the KV cache memory for Llama 3.1 70B on h100_sxm.

calls the `memory` tool with:

```json
{
  "model_path": "meta-llama/Llama-3.1-70B",
  "system": "h100_sxm"
}
```

and returns the memory breakdown including the KV cache.

Sizing requests run one workload per call. See
[user scenarios](user-scenarios.md#sizing-multiple-workloads) for guidance on
sizing multiple workloads.

## Tools

| Tool | Purpose |
|---|---|
| `recommend` | Find an optimal GPU configuration for a workload |
| `predict` | Predict performance for a given configuration |
| `estimate` | Deprecated alias for `predict` |
| `memory` | Estimate KV cache memory for a model configuration |
| `backends` | List supported serving backends and engine versions |
| `models` | List supported model architectures |
| `systems` | List supported GPU systems |

Tool names are the gateway operation IDs. Full request parameters are in the
[REST API reference](api.md). The MCP tools take the same bodies.

## Deployment notes

- Legacy `aisimulators.dev` and `aicostings.dev` MCP paths are routed to
  this gateway by the deployment repository during migration. Their REST
  endpoints remain owned by the original services.
- The gateway runs on port 8090 in its
  [Containerfile](../services/mcp-gateway/Containerfile). The AISimulators
  service defaults to 8000.
- `configiq-py` mounts an MCP server exposing the app endpoints as tools.

## Related

- [REST API](api.md): endpoint reference and examples
- [User scenarios](user-scenarios.md): which tool to use for which ask

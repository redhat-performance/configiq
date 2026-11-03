# ConfigIQ MCP server

ConfigIQ exposes one stable MCP server, presented by the MCP gateway
([`services/mcp-gateway`](../services/mcp-gateway)). It proxies the inference
sizing APIs while keeping MCP session state in one process, separate from the
horizontally-scaled AISimulators REST service.

Cost and pricing MCP tools exist on the gateway but are not documented here
yet; pricing documentation is deferred to a separate change.

## Connect

Point any MCP client at the gateway:

- **SSE:** `GET /mcp`
- **Streamable HTTP:** `GET,POST,DELETE /mcp/http` — the transport the MCP
  spec recommends; use it with clients that do not speak SSE

`/health` and `/metrics` are readiness and observability endpoints, not MCP
tools.

### Claude Desktop

`~/.config/claude/config.json` (or
`~/Library/Application Support/Claude/config.json` on macOS):

```json
{
  "mcpServers": {
    "configiq": {
      "url": "https://aisimulators.dev/mcp"
    }
  }
}
```

### MCP Inspector

```bash
npx @modelcontextprotocol/inspector https://aisimulators.dev/mcp
```

For a streamable-HTTP client, use `https://aisimulators.dev/mcp/http`.

## Tools

Inference sizing tools (proxy to the AISimulators REST API):

| Tool | Purpose |
|---|---|
| `recommend` | Find an optimal GPU configuration for a workload |
| `predict` | Predict performance for a given configuration |
| `estimate` | Deprecated alias for `predict` |
| `memory` | Estimate KV cache memory for a model configuration |
| `backends` | List supported serving backends and engine versions |
| `models` | List supported model architectures |
| `systems` | List supported GPU systems |

Tool names are the gateway's operation IDs. Sizing requests run one workload
row per call — see [user scenarios](user-scenarios.md#sizing-in-bands-batch-sizing)
for band sizing guidance.

## Deployment notes

- Legacy `aisimulators.dev` and `aicostings.dev` MCP paths are routed to this
  gateway by the deployment repository during migration. Their REST endpoints
  remain owned by the original services.
- The AISimulators FastAPI service can also expose itself as an MCP server
  directly when `fastapi-mcp` is installed (`pip install '.[mcp]'`); tool
  names are then the FastAPI operation IDs (e.g.
  `post_recommend_recommend_post`).
- `configiq-py` mounts an MCP server exposing the app's endpoints as tools.

## Running locally

```bash
cd services/mcp-gateway
uv sync
uv run uvicorn api_service.app:app --host 0.0.0.0 --port 8090
```

Then point a client at `http://localhost:8090/mcp`.

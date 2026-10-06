# ConfigIQ MCP gateway

The MCP gateway presents one stable MCP server for ConfigIQ. It proxies the
inference sizing and GPU pricing APIs while keeping MCP session state in one
process, separate from the horizontally-scaled AISimulators REST service.

## Endpoints

- `GET /mcp` — MCP over SSE
- `GET,POST,DELETE /mcp/http` — MCP over Streamable HTTP
- `GET /health` — readiness check (not an MCP tool)

## Tools

- `recommend`, `predict`, `estimate`, `memory`
- `backends`, `models`, `systems`
- `pricing_sources`, `pricing_models`, `pricing_systems`

The old `aisimulators.dev` and `aicostings.dev` MCP paths are routed to this
gateway by the deployment repository during migration. Their REST endpoints
remain owned by their original services.

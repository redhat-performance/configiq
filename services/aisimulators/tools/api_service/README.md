# AISimulators REST API

FastAPI service providing GPU recommendation and performance estimation for LLM inference.

## Quick Start

```bash
# Install with service extras
pip install -e ".[service]"

# Run the API server
python -m uvicorn tools.api_service.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

## Core Endpoints

### POST `/recommend`
Find optimal GPU configuration for a workload.

**Example:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "meta-llama/Llama-3.1-70B",
    "system": "h100_sxm",
    "target_concurrency": 32,
    "isl": 4000,
    "osl": 1000
  }'
```

### POST `/predict`
Predict performance for a specific GPU configuration.

### POST `/estimate` (deprecated)
Compatibility alias for `/predict`.

### POST `/memory`
Estimate KV cache memory usage.

### GET `/models`
List supported model architectures.

### GET `/systems`
List supported GPU systems.

See the full OpenAPI spec at `docs/api/openapi.yaml` or `/docs` when running.

## Model Context Protocol (MCP)

The API can be exposed as an **MCP server**, making it available as tools for Claude and other MCP clients.

### MCP Endpoint

When `fastapi-mcp` is installed, the server automatically exposes:
- **SSE endpoint:** `http://localhost:8000/mcp`

### Available MCP Tools

`fastapi-mcp` automatically exposes all REST API endpoints as MCP tools:

1. **POST `/recommend`** - Find optimal GPU count and parallelism for serving an LLM
2. **POST `/predict`** - Predict throughput/latency for a specific GPU configuration
3. **POST `/estimate`** - Deprecated compatibility alias for `/predict`
4. **POST `/memory`** - Estimate KV cache memory usage
5. **GET `/models`** - Get all supported model architectures
6. **GET `/systems`** - Get all supported GPU systems

Tools are automatically generated from the OpenAPI schema with full parameter validation.

### Connecting an MCP Client

**Claude Desktop** (`~/.config/claude/config.json` or `~/Library/Application Support/Claude/config.json` on macOS):

```json
{
  "mcpServers": {
    "aisimulators": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

**Testing with MCP Inspector:**

```bash
npx @modelcontextprotocol/inspector http://localhost:8000/mcp
```

The MCP protocol handles tool discovery automatically - clients will see all available tools with their schemas and descriptions.

## OpenTelemetry Instrumentation

When OpenTelemetry dependencies are installed, the API automatically instruments:
- **Tracing**: FastAPI request/response traces, HTTP requests (via `requests` library), urllib3 connections
- **Metrics**: HTTP request duration, request counts, exposed via `/metrics` endpoint

### Configuration

Set the OTLP endpoint via environment variable:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
python -m uvicorn tools.api_service.app:app
```

Default: `http://localhost:4318` (standard OTLP HTTP endpoint)

### Running with Jaeger

```bash
# Start Jaeger all-in-one
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest

# Run the API with OTel enabled
python -m uvicorn tools.api_service.app:app

# View traces at http://localhost:16686
```

Traces include:
- Request path, method, status code
- Response time breakdowns
- Upstream API calls
- Database queries

## Metrics Endpoint

**GET `/metrics`** - Exposes metrics in multiple formats via content negotiation:

```bash
curl http://localhost:8000/metrics
```

Returns Prometheus/OpenMetrics text format.

### Available Metrics

**HTTP Request Metrics** (labeled by `http_request_method`, `url_path`, `http_response_status_code`):
- **`http_server_requests_total`** - Total requests counter
- **`http_server_request_body_size_bytes_total`** - Request body bytes received
- **`http_server_response_body_size_bytes_total`** - Response body bytes sent
- **`http_server_request_duration_seconds`** - Request latency histogram

**Process Resource Metrics**:
- **`process_cpu_seconds_total`** - Cumulative CPU seconds consumed (user + system time)
- **`process_memory_usage_bytes`** - Current memory usage in bytes

## Development

### Running Tests

```bash
# Test the API service (includes OTel and MCP tests)
pytest tests/unit/tools/test_api_service.py -v

# Test just OpenTelemetry integration
pytest tests/unit/tools/test_api_service.py::TestOpenTelemetry -v

# Test just MCP integration
pytest tests/unit/tools/test_api_service.py::TestMCPServer -v
```

### Optional Dependencies

Both OpenTelemetry and MCP features are **optional**:
- The service runs without them (logs a warning on startup)
- Tests that require them are automatically skipped if not installed
- Install with: `pip install -e ".[service]"`

### Adding New MCP Tools

To expose a new API endpoint via MCP, add it in `app.py`:

```python
@mcp.tool()
async def your_new_tool(param: str) -> dict:
    """Tool description for MCP clients."""
    # Call your API endpoint handler
    return your_endpoint_handler(param)
```

The `@mcp.tool()` decorator:
- Auto-generates the JSON schema from type hints
- Uses the docstring for tool descriptions
- Makes it discoverable via MCP protocol

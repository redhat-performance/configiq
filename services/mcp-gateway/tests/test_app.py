# SPDX-License-Identifier: Apache-2.0

import json

import httpx
from fastapi.testclient import TestClient

from api_service.app import create_app


def _transport(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/recommend":
        return httpx.Response(200, json={"configs": [{"total_gpus_needed": 2}]})
    if request.url.path == "/models":
        return httpx.Response(200, json={"models": ["Qwen/Qwen3-32B"]})
    if request.url.path == "/pricing/models":
        return httpx.Response(200, json={"models": [{"id": "example"}]})
    if request.url.path == "/sources":
        return httpx.Response(200, json={"sources": ["openrouter"]})
    return httpx.Response(200, json={})


def test_health_is_not_an_mcp_tool():
    app = create_app(transport=httpx.MockTransport(_transport))

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}

    names = {tool.name for tool in app.state.mcp_server.tools}
    assert names == {
        "recommend",
        "predict",
        "estimate",
        "memory",
        "backends",
        "models",
        "systems",
        "pricing_sources",
        "pricing_models",
        "pricing_systems",
    }
    assert "health" not in names


def test_proxy_preserves_json_response():
    app = create_app(transport=httpx.MockTransport(_transport))

    with TestClient(app) as client:
        response = client.post(
            "/recommend",
            json={"model_path": "Qwen/Qwen3-32B", "system": "h100_sxm"},
        )

    assert response.status_code == 200
    assert response.json() == {"configs": [{"total_gpus_needed": 2}]}


def test_mcp_http_lists_curated_tools():
    app = create_app(transport=httpx.MockTransport(_transport))

    with TestClient(app) as client:
        response = client.post(
            "/mcp/http",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
        )
        assert response.status_code == 200
        session = response.headers["mcp-session-id"]
        client.post(
            "/mcp/http",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "MCP-Session-ID": session,
            },
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        listed = client.post(
            "/mcp/http",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "MCP-Session-ID": session,
            },
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )

    assert listed.status_code == 200
    data_line = next(line for line in listed.text.splitlines() if line.startswith("data: "))
    payload = json.loads(data_line[6:])
    assert {tool["name"] for tool in payload["result"]["tools"]} == {
        "recommend",
        "predict",
        "estimate",
        "memory",
        "backends",
        "models",
        "systems",
        "pricing_sources",
        "pricing_models",
        "pricing_systems",
    }

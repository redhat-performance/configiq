# SPDX-License-Identifier: Apache-2.0

"""MCP wiring: every endpoint is a tool, served over SSE and streamable HTTP.

No Valkey needed: only the server's tool list and routes are inspected.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest
from configiq import mcp as mcp_support
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from tools.api_service import app as app_module

EXPECTED_MCP_TOOLS = {
    "get_sources_sources_get",
    "get_models_models_get",
    "get_systems_systems_get",
    "get_health_health_get",
}


@pytest.fixture(autouse=True)
def _need_mcp():
    if not app_module._MCP:
        pytest.skip("fastapi-mcp not installed")


def test_mcp_exposes_every_api_route():
    """Mounted after the last route: fastapi-mcp snapshots routes when built,
    so mounting earlier exposed no tools at all."""
    tools = {t.name for t in app_module._MCP_SERVER.tools}
    missing = EXPECTED_MCP_TOOLS - tools
    assert not missing, f"not exposed as MCP tools: {sorted(missing)}"


def test_mcp_serves_sse_and_streamable_http():
    routes = {
        (getattr(r, "path", ""), m)
        for r in app_module.app.routes
        for m in (getattr(r, "methods", None) or ())
    }
    assert ("/mcp", "GET") in routes
    assert ("/mcp/messages/", "POST") in routes
    assert any(getattr(r, "path", "") == "/mcp/http" for r in app_module.app.routes)


def test_mcp_preserves_existing_lifespan_state():
    @asynccontextmanager
    async def lifespan(_app):
        yield {"marker": "preserved"}

    test_app = FastAPI(lifespan=lifespan)

    @test_app.get("/state", operation_id="get_state")
    def get_state(request: Request):
        return {"marker": getattr(request.state, "marker", None)}

    mcp_support.mount(test_app, name="state-test", description="state test")

    with TestClient(test_app, follow_redirects=False) as client:
        response = client.get("/state")

    assert response.json() == {"marker": "preserved"}


async def test_streamable_http_forwards_get_stream_headers():
    test_app = FastAPI()

    @test_app.get("/hello", operation_id="get_hello")
    def get_hello():
        return {"hello": "world"}

    mcp_support.mount(test_app, name="stream-test", description="stream test")

    base_scope = {
        "type": "http",
        "path": "/mcp/http",
        "raw_path": b"/mcp/http",
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "http_version": "1.1",
        "server": ("test", 80),
        "client": ("test", 1),
        "extensions": {},
    }
    initialize_scope = {
        **base_scope,
        "method": "POST",
        "headers": [
            (b"accept", b"application/json, text/event-stream"),
            (b"content-type", b"application/json"),
        ],
    }
    initialize_body = (
        b'{"jsonrpc":"2.0","id":1,"method":"initialize",'
        b'"params":{"protocolVersion":"2025-03-26","capabilities":{},'
        b'"clientInfo":{"name":"test","version":"0"}}}'
    )

    async with test_app.router.lifespan_context(test_app):
        initialize_messages = await _invoke_asgi(test_app, initialize_scope, initialize_body)
        start = initialize_messages[0]
        session_id = next(value for key, value in start["headers"] if key == b"mcp-session-id")

        get_scope = {
            **base_scope,
            "method": "GET",
            "headers": [(b"accept", b"text/event-stream"), (b"mcp-session-id", session_id)],
        }
        first, get_task, disconnect = await _invoke_asgi(
            test_app, get_scope, stop_after_start=True
        )
        assert first["type"] == "http.response.start"
        assert any(
            key == b"content-type" and b"text/event-stream" in value
            for key, value in first["headers"]
        )
        disconnect.set()
        await asyncio.wait_for(get_task, timeout=2)


async def _invoke_asgi(app, scope, body=b"", stop_after_start=False):
    messages = asyncio.Queue()
    first_request = True
    disconnect = asyncio.Event()

    async def receive():
        nonlocal first_request
        if first_request:
            first_request = False
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        await messages.put(message)

    task = asyncio.create_task(app(scope, receive, send))
    first = await asyncio.wait_for(messages.get(), timeout=2)
    if stop_after_start:
        return first, task, disconnect

    collected = [first]
    while not (
        collected[-1]["type"] == "http.response.body"
        and not collected[-1].get("more_body", False)
    ):
        collected.append(await asyncio.wait_for(messages.get(), timeout=2))
    await task
    return collected

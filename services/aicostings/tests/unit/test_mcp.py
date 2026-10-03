# SPDX-License-Identifier: Apache-2.0

"""MCP wiring: every endpoint is a tool, served over SSE and streamable HTTP.

No Valkey needed: only the server's tool list and routes are inspected.
"""

import pytest

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
    assert ("/mcp/http", "POST") in routes

# SPDX-License-Identifier: Apache-2.0

"""Expose a FastAPI app as MCP tools.

Imports fastapi-mcp at module load, so this module is only importable when the
`mcp` extra is installed. Wire it behind a try/except so MCP stays optional,
and call `mount` AFTER the last route is declared::

    try:
        from configiq import mcp as mcp_support
        _MCP = True
    except ImportError:
        _MCP = False

    ...  # every @app.get / @app.post

    if _MCP:
        _MCP_SERVER = mcp_support.mount(app, name="aicostings", description="...")

fastapi-mcp turns the app's OpenAPI routes into tools once, when the server is
constructed: a route declared after `mount` is not a tool.
"""

import logging

from fastapi_mcp import FastApiMCP

logger = logging.getLogger(__name__)

SSE_PATH = "/mcp"
HTTP_PATH = "/mcp/http"


def mount(app, *, name: str, description: str) -> FastApiMCP:
    """Mount an MCP server that exposes the app's endpoints as MCP tools.

    Serves two transports from the same server:

    - SSE at ``/mcp`` (unchanged; what existing clients and the MCP Inspector
      connect to);
    - Streamable HTTP at ``/mcp/http`` (the transport the MCP spec has
      recommended since 2025-03-26, and the one clients that do not speak SSE
      use).

    Returns the FastApiMCP instance; its ``tools`` list is what clients see.
    """
    server = FastApiMCP(app, name=name, description=description)
    # `mount()` is deprecated; call the transport-specific mounts directly. Their
    # defaults (/sse, /mcp) differ from our paths, so pass them explicitly.
    server.mount_sse(mount_path=SSE_PATH)
    server.mount_http(mount_path=HTTP_PATH)
    logger.info(
        "MCP server initialized - %d tools; SSE at %s, streamable HTTP at %s",
        len(server.tools), SSE_PATH, HTTP_PATH,
    )
    return server

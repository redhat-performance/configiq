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
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.routing import BaseRoute, Match
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)

SSE_PATH = "/mcp"
HTTP_PATH = "/mcp/http"
SESSION_IDLE_TIMEOUT = 30 * 60


class _StreamableHTTPApp:
    """Forward the SDK's ASGI messages without buffering streaming responses."""

    def __init__(self):
        self._manager: StreamableHTTPSessionManager | None = None

    def set_manager(self, manager: StreamableHTTPSessionManager | None) -> None:
        self._manager = manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self._manager is None:
            raise RuntimeError("MCP HTTP transport is not running")
        await self._manager.handle_request(scope, receive, send)


class _StreamableHTTPRoute(BaseRoute):
    """Route an ASGI transport without FastAPI's response buffering."""

    def __init__(self, path: str, endpoint: _StreamableHTTPApp):
        self.path = path
        self.endpoint = endpoint

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        if scope["type"] == "http" and scope["path"] == self.path:
            return Match.FULL, {"endpoint": self.endpoint, "path_params": {}, "route": self}
        return Match.NONE, {}

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.endpoint(scope, receive, send)


def mount(app: FastAPI, *, name: str, description: str) -> FastApiMCP:
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

    # fastapi-mcp 0.4.0 buffers the SDK's ASGI messages into a Response. That
    # breaks the long-lived GET stream required by Streamable HTTP. Mount the
    # SDK transport directly so headers and events are forwarded as sent.
    http_app = _StreamableHTTPApp()
    app.router.routes.append(_StreamableHTTPRoute(HTTP_PATH, http_app))
    app.router.routes.append(_StreamableHTTPRoute(f"{HTTP_PATH}/", http_app))

    # Enter the session manager alongside the service's existing lifespan. The
    # manager must be running before the first HTTP request and must be closed
    # before the service releases its other resources.
    existing_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan_with_mcp(service_app: FastAPI):
        manager = StreamableHTTPSessionManager(
            app=server.server,
            stateless=False,
            session_idle_timeout=SESSION_IDLE_TIMEOUT,
        )
        async with existing_lifespan(service_app) as state:
            http_app.set_manager(manager)
            try:
                async with manager.run():
                    yield state
            finally:
                http_app.set_manager(None)

    app.router.lifespan_context = lifespan_with_mcp
    logger.info(
        "MCP server initialized - %d tools; SSE at %s, streamable HTTP at %s",
        len(server.tools), SSE_PATH, HTTP_PATH,
    )
    return server

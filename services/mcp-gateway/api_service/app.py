# SPDX-License-Identifier: Apache-2.0

"""Unified MCP gateway for ConfigIQ inference and pricing APIs."""

import os
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from configiq import mcp as mcp_support
from configiq import observability
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_AISIMULATORS_URL = "http://host.containers.internal"
DEFAULT_AICOSTINGS_URL = "http://host.containers.internal:8080"


class InferenceRequest(BaseModel):
    """Common inference request fields; unknown service fields are forwarded."""

    model_config = ConfigDict(extra="allow")

    model_path: str = Field(description="HuggingFace model path or SDK model key.")
    system: str = Field(description="GPU system identifier.")
    backend: str = "vllm"
    backend_version: str | None = None
    include: str | None = None
    target_request_rate: float | None = None
    target_concurrency: float | None = None
    isl: int = 4000
    osl: int = 1000
    max_seq_len: int | None = None
    prefill_max_seq_len: int | None = None
    decode_max_seq_len: int | None = None
    ttft: float = 2000.0
    tpot: float = 30.0
    request_latency: float | None = None
    prefix: int = 0
    max_num_seqs: int | None = None
    tp_size: int = 1
    pp_size: int = 1
    batch_size: int = 128
    database_mode: str = "HYBRID"
    top_n: int = 5
    mode: Literal["agg", "disagg"] = "agg"


class MemoryRequest(BaseModel):
    """Memory request fields; unknown service fields are forwarded."""

    model_config = ConfigDict(extra="allow")

    model_path: str = Field(description="HuggingFace model path or SDK model key.")
    system: str = Field(description="GPU system identifier.")
    backend: str = "vllm"
    backend_version: str | None = None
    max_num_tokens: int = 8192
    max_batch_size: int = 128
    memory_fraction_kind: str = "of_total"
    memory_fraction_value: float = 1.0
    tp_size: int = 1
    pp_size: int = 1
    attention_dp_size: int = 1


class GatewayClients:
    def __init__(self, client: httpx.AsyncClient, aisimulators_url: str, aicostings_url: str):
        self.client = client
        self.aisimulators_url = aisimulators_url.rstrip("/")
        self.aicostings_url = aicostings_url.rstrip("/")


def create_app(
    *,
    aisimulators_url: str | None = None,
    aicostings_url: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    resolved_aisimulators_url = aisimulators_url or os.getenv(
        "AISIMULATORS_GATEWAY_URL", DEFAULT_AISIMULATORS_URL,
    )
    resolved_aicostings_url = aicostings_url or os.getenv(
        "AICOSTINGS_GATEWAY_URL", DEFAULT_AICOSTINGS_URL,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), transport=transport) as client:
            app.state.gateway_clients = GatewayClients(
                client, resolved_aisimulators_url, resolved_aicostings_url,
            )
            yield

    app = FastAPI(
        title="ConfigIQ MCP gateway",
        description="Unified inference sizing and GPU pricing tools.",
        lifespan=lifespan,
    )
    observability.enable(
        app,
        service_name="configiq-mcp",
        service_version="0.1.0",
        meter_name="configiq.mcp",
    )

    async def proxy(
        request: Request,
        *,
        base_url: str,
        path: str,
        body: BaseModel | None = None,
    ) -> Response:
        clients: GatewayClients = request.app.state.gateway_clients
        params = dict(request.query_params)
        response = await clients.client.request(
            request.method,
            f"{base_url}{path}",
            params=params,
            json=body.model_dump(mode="json") if body is not None else None,
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )

    @app.post("/recommend", operation_id="recommend", tags=["inference"])
    async def recommend(request: Request, body: InferenceRequest) -> Response:
        clients: GatewayClients = request.app.state.gateway_clients
        return await proxy(request, base_url=clients.aisimulators_url, path="/recommend", body=body)

    @app.post("/predict", operation_id="predict", tags=["inference"])
    async def predict(request: Request, body: InferenceRequest) -> Response:
        clients: GatewayClients = request.app.state.gateway_clients
        return await proxy(request, base_url=clients.aisimulators_url, path="/predict", body=body)

    @app.post("/estimate", operation_id="estimate", tags=["inference"])
    async def estimate(request: Request, body: InferenceRequest) -> Response:
        clients: GatewayClients = request.app.state.gateway_clients
        return await proxy(request, base_url=clients.aisimulators_url, path="/estimate", body=body)

    @app.post("/memory", operation_id="memory", tags=["inference"])
    async def memory(request: Request, body: MemoryRequest) -> Response:
        clients: GatewayClients = request.app.state.gateway_clients
        return await proxy(request, base_url=clients.aisimulators_url, path="/memory", body=body)

    @app.get("/backends", operation_id="backends", tags=["catalog"])
    async def backends(request: Request) -> Response:
        clients: GatewayClients = request.app.state.gateway_clients
        return await proxy(request, base_url=clients.aisimulators_url, path="/backends")

    @app.get("/models", operation_id="models", tags=["catalog"])
    async def models(request: Request, include: str | None = Query(default=None)) -> Response:
        clients: GatewayClients = request.app.state.gateway_clients
        return await proxy(request, base_url=clients.aisimulators_url, path="/models")

    @app.get("/systems", operation_id="systems", tags=["catalog"])
    async def systems(request: Request, include: str | None = Query(default=None)) -> Response:
        clients: GatewayClients = request.app.state.gateway_clients
        return await proxy(request, base_url=clients.aisimulators_url, path="/systems")

    @app.get("/pricing/sources", operation_id="pricing_sources", tags=["pricing"])
    async def pricing_sources(request: Request) -> Response:
        clients: GatewayClients = request.app.state.gateway_clients
        return await proxy(request, base_url=clients.aicostings_url, path="/sources")

    @app.get("/pricing/models", operation_id="pricing_models", tags=["pricing"])
    async def pricing_models(request: Request, source: str = Query(default="merged")) -> Response:
        clients: GatewayClients = request.app.state.gateway_clients
        return await proxy(request, base_url=clients.aicostings_url, path="/models")

    @app.get("/pricing/systems", operation_id="pricing_systems", tags=["pricing"])
    async def pricing_systems(request: Request, include: str | None = Query(default=None)) -> Response:
        clients: GatewayClients = request.app.state.gateway_clients
        return await proxy(request, base_url=clients.aicostings_url, path="/systems")

    # Mount after the public tool routes so FastApiMCP snapshots exactly the
    # curated tool surface. The readiness endpoint below is intentionally not a tool.
    app.state.mcp_server = mcp_support.mount(
        app,
        name="configiq",
        description="Unified GPU sizing, inference performance, and infrastructure pricing",
    )

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics(request: Request) -> Response:
        return observability.metrics_response(request.headers.get("accept", "text/plain"))

    return app


app = create_app()

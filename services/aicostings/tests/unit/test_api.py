# SPDX-License-Identifier: Apache-2.0

"""Tests for the FastAPI endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from tools.api_service.app import app, store


@pytest.fixture(autouse=True)
def setup_store():
    store.connect()
    store.flush()
    yield
    store.flush()
    store.close()


@pytest.fixture
def seed_data():
    store.set_models([
        {"id": "anthropic/claude-sonnet-5", "name": "Claude Sonnet 5", "provider": "Anthropic",
         "tier": "balanced", "price_per_m_input": 3.0, "price_per_m_output": 15.0, "context_window": 200000},
        {"id": "openai/gpt-5.6-terra", "name": "GPT-5.6 Terra", "provider": "OpenAI",
         "tier": "balanced", "price_per_m_input": 2.5, "price_per_m_output": 10.0, "context_window": 128000},
    ])
    store.mark_scrape("models.openrouter")
    store.mark_scrape("models.litellm")
    store.set_cloud_rates("h200_sxm", {
        "cloud_rates": {
            "aws.us-east-1": {"on_demand": 4.50, "spot_median": 1.80},
            "gcp.us-central1": {"on_demand": 3.85, "spot_median": None},
        },
        "rates_updated_at": "2026-08-11T06:00:00Z",
        "rates_stale": False,
    })
    store.set_hardware_cost("h200_sxm", {
        "new_usd": 42000, "used_usd": 28000,
        "tdp_watts": 700, "gpus_per_node": 8,
        "hardware_updated_at": "2026-08-01T00:00:00Z",
    })


class TestGetModels:
    @pytest.mark.asyncio
    async def test_returns_models(self, seed_data):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 2
        assert data["stale"] is False
        assert data["updated_at"] is not None

    @pytest.mark.asyncio
    async def test_model_fields(self, seed_data):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/models")
        model = resp.json()["models"][0]
        assert "id" in model
        assert "name" in model
        assert "provider" in model
        assert "tier" in model
        assert "price_per_m_input" in model
        assert "price_per_m_output" in model
        assert "context_window" in model

    @pytest.mark.asyncio
    async def test_empty_when_no_data(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/models")
        data = resp.json()
        assert data["models"] == []
        assert data["stale"] is True

    @pytest.mark.asyncio
    async def test_default_source_is_merged(self, seed_data):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/models")
        assert resp.json()["source"] == "merged"

    @pytest.mark.asyncio
    async def test_source_selects_feed(self):
        store.set_models([
            {"id": "anthropic/claude-x", "name": "Claude X", "provider": "Anthropic",
             "tier": "balanced", "price_per_m_input": 3.0, "price_per_m_output": 15.0,
             "context_window": 200000, "source": "openrouter"},
        ], source="openrouter")
        store.set_models([
            {"id": "meta-llama/llama-9", "name": "llama-9", "provider": "Meta",
             "tier": "balanced", "price_per_m_input": 0.2, "price_per_m_output": 0.3,
             "context_window": 128000, "source": "litellm"},
        ], source="litellm")
        store.mark_scrape("models.openrouter")
        store.mark_scrape("models.litellm")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            or_resp = await client.get("/models?source=openrouter")
            lt_resp = await client.get("/models?source=litellm")

        assert or_resp.json()["source"] == "openrouter"
        assert [m["id"] for m in or_resp.json()["models"]] == ["anthropic/claude-x"]
        assert or_resp.json()["models"][0]["source"] == "openrouter"
        assert [m["id"] for m in lt_resp.json()["models"]] == ["meta-llama/llama-9"]

    @pytest.mark.asyncio
    async def test_invalid_source_returns_400(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/models?source=bogus")
        assert resp.status_code == 400


class TestGetSources:
    @pytest.mark.asyncio
    async def test_lists_feeds_with_metadata(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/sources")
        assert resp.status_code == 200
        sources = resp.json()["sources"]
        ids = [s["id"] for s in sources]
        assert ids[0] == "merged"  # synthetic default listed first
        assert "openrouter" in ids
        assert "litellm" in ids
        for s in sources:
            assert s["label"]
            assert s["description"]
            assert "url" in s  # present for every source (None for merged)
        by_id = {s["id"]: s for s in sources}
        assert by_id["merged"]["url"] is None
        assert by_id["openrouter"]["url"]  # real feeds carry their scrape URL
        assert by_id["litellm"]["url"]

    @pytest.mark.asyncio
    async def test_ids_match_accepted_model_sources(self):
        # Every advertised source id must be accepted by /models, and any other
        # id rejected — /sources is the single source of truth for the selector.
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            ids = [s["id"] for s in (await client.get("/sources")).json()["sources"]]
            for sid in ids:
                assert (await client.get(f"/models?source={sid}")).status_code == 200
            assert (await client.get("/models?source=bogus")).status_code == 400


class TestGetSystems:
    @pytest.mark.asyncio
    async def test_base_returns_id_and_name(self, seed_data):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/systems")
        data = resp.json()
        assert len(data["systems"]) > 0
        sys = data["systems"][0]
        assert "id" in sys
        assert "name" in sys
        assert "cloud_rates" not in sys
        assert "hardware_cost" not in sys

    @pytest.mark.asyncio
    async def test_include_cloud(self, seed_data):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/systems?include=cloud")
        systems = resp.json()["systems"]
        h200 = next((s for s in systems if s["id"] == "h200_sxm"), None)
        assert h200 is not None
        assert "cloud_rates" in h200
        assert h200["cloud_rates"]["aws.us-east-1"]["on_demand"] == 4.50

    @pytest.mark.asyncio
    async def test_include_hardware(self, seed_data):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/systems?include=hardware")
        systems = resp.json()["systems"]
        h200 = next((s for s in systems if s["id"] == "h200_sxm"), None)
        assert h200 is not None
        assert h200["hardware_cost"]["new_usd"] == 42000
        assert h200["hardware_cost"]["used_usd"] == 28000
        assert h200["tdp_watts"] == 700
        assert h200["gpus_per_node"] == 8

    @pytest.mark.asyncio
    async def test_include_both(self, seed_data):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/systems?include=cloud,hardware")
        systems = resp.json()["systems"]
        h200 = next((s for s in systems if s["id"] == "h200_sxm"), None)
        assert h200 is not None
        assert "cloud_rates" in h200
        assert "hardware_cost" in h200

    @pytest.mark.asyncio
    async def test_no_include_omits_details(self, seed_data):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/systems")
        systems = resp.json()["systems"]
        for sys in systems:
            assert "cloud_rates" not in sys
            assert "hardware_cost" not in sys


class TestHealth:
    @pytest.mark.asyncio
    async def test_returns_status(self, seed_data):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert "sources" in data

    @pytest.mark.asyncio
    async def test_sources_have_scrape_info(self, seed_data):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
        sources = resp.json()["sources"]
        assert "models.openrouter" in sources
        assert "models.litellm" in sources
        assert "last_success" in sources["models.openrouter"]
        assert "last_error" in sources["models.openrouter"]
        assert "stale" in sources["models.openrouter"]

    @pytest.mark.asyncio
    async def test_empty_sources_when_no_data(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.json()["status"] == "ok"
        assert resp.json()["sources"] == {}


class TestCORS:
    @pytest.mark.asyncio
    async def test_cors_headers(self, seed_data):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.options("/models", headers={
                "Origin": "https://configiq.dev",
                "Access-Control-Request-Method": "GET",
            })
        assert resp.headers.get("access-control-allow-origin") == "*"

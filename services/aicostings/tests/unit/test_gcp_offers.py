# SPDX-License-Identifier: Apache-2.0

"""Public GCP response regression and credential-safe collection contract."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from tools.api_service.offers import assemble_offers, load_curated_offers
from tools.api_service.scrapers import cloud_rates, offer_collectors


@pytest.fixture
def catalogue(monkeypatch):
    fixture = Path(__file__).parents[1] / "fixtures" / "gcp_a2_catalogue_sample.json"
    data = json.loads(fixture.read_text())
    clock = MagicMock(wraps=datetime)
    clock.now.return_value = datetime.fromisoformat(data["_meta"]["retrieved_at"])
    monkeypatch.setattr(offer_collectors, "datetime", clock)
    monkeypatch.setenv("GCP_BILLING_API_KEY", "test-catalogue-key")
    monkeypatch.setenv("GCP_PRICING_REGIONS", "us-central1")
    return data["skus"]


def catalogue_session(*pages, status=200):
    session = MagicMock()
    contexts = []
    for page in pages:
        response = MagicMock(status=status)
        response.json = AsyncMock(return_value=page)
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=response)
        context.__aexit__ = AsyncMock(return_value=False)
        contexts.append(context)
    session.get.side_effect = contexts
    return session


def test_real_catalogue_uses_shared_a2_rates_and_excludes_nonstandard_skus(catalogue):
    offers = offer_collectors.parse_gcp_a2_prices(catalogue, ["us-central1"])
    # A100 80GB + 12 vCPU + 170 GiB RAM + 375 GiB SSD at the captured public rates.
    one_gpu = 3.92808 + 12 * .031611 + 170 * .004237 + 375 * .08 / 730
    assert [offer.gpu_count for offer in offers] == [1, 2, 4, 8]
    assert [offer.hourly_cost for offer in offers] == pytest.approx([one_gpu * n for n in [1, 2, 4, 8]])
    assert all(offer.provider_region == "gcp.us-central1" and offer.source.kind == "automated" for offer in offers)


@pytest.mark.asyncio
async def test_collection_waits_for_all_pages_and_keeps_key_out_of_urls(catalogue):
    session = catalogue_session({"skus": catalogue[:5], "nextPageToken": "second"}, {"skus": catalogue[5:]})
    result = await offer_collectors.scrape_gcp_offers(session)
    assert len(result["offers"]) == 4
    assert session.get.call_count == 2
    for call in session.get.call_args_list:
        assert call.args == (offer_collectors.GCP_SKUS_URL,)
        assert call.kwargs["headers"] == {"X-Goog-Api-Key": "test-catalogue-key"}
        assert call.kwargs["allow_redirects"] is False
        assert "key" not in call.kwargs["params"]
    assert session.get.call_args_list[1].kwargs["params"]["pageToken"] == "second"
    assert "test-catalogue-key" not in json.dumps(result)


@pytest.mark.asyncio
async def test_collection_rejects_repeated_pagination_tokens(catalogue):
    page = {"skus": catalogue, "nextPageToken": "repeated"}
    with pytest.raises(ValueError, match="Repeated GCP pagination token"):
        await offer_collectors.scrape_gcp_offers(catalogue_session(page, page))


@pytest.mark.asyncio
@pytest.mark.parametrize("page", [{}, [], {"skus": {}}, {"skus": [None]}, {"skus": [], "nextPageToken": 1}])
async def test_collection_rejects_malformed_later_pages_instead_of_publishing_partial_data(catalogue, page):
    session = catalogue_session({"skus": catalogue, "nextPageToken": "second"}, page)
    with pytest.raises(ValueError, match="invalid page"):
        await offer_collectors.scrape_gcp_offers(session)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [aiohttp.ClientConnectionError("test-catalogue-key"), TimeoutError("test-catalogue-key")])
async def test_collection_does_not_expose_credentials_in_transport_errors(catalogue, error):
    session = MagicMock()
    session.get.side_effect = error
    with pytest.raises(RuntimeError, match="^GCP public catalogue request failed$"):
        await offer_collectors.scrape_gcp_offers(session)


@pytest.mark.asyncio
async def test_http_error_is_explicit_without_echoing_response_details(catalogue):
    session = catalogue_session({"error": {"message": "test-catalogue-key"}}, status=403)
    with pytest.raises(RuntimeError, match="^GCP public catalogue returned HTTP 403$"):
        await offer_collectors.scrape_gcp_offers(session)


@pytest.mark.asyncio
async def test_real_response_publishes_complete_offers_and_preserves_last_good_on_failure(catalogue, monkeypatch):
    from tools.api_service import app as api

    snapshots = {}
    errors = {}
    store = MagicMock()
    store.set_offer_snapshot.side_effect = snapshots.__setitem__
    store.set_scrape_error.side_effect = lambda name, message: errors.update({name: {"message": message}})
    store.mark_scrape.side_effect = lambda name: errors.pop(name, None)
    session = catalogue_session({"skus": catalogue})

    async def collect(_session, mapping):
        return await cloud_rates.scrape_gcp(session, mapping)

    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "scrape_coreweave_offers", AsyncMock(side_effect=ValueError("Not part of this test")))
    monkeypatch.setattr(cloud_rates, "active_scrapers", lambda: [("gcp", collect)])
    await api.job_scrape_cloud_rates()
    good = snapshots["gcp"]
    curated = load_curated_offers().model_dump(mode="json")
    assembled = assemble_offers(curated, snapshots, errors)
    assert assembled.sources["gcp"].state == "fresh"
    assert len([offer for offer in assembled.rented if offer.provider == "gcp" and offer.source.kind == "automated"]) == 4
    assert len([offer for offer in assembled.rented if offer.provider == "gcp" and offer.source.kind == "curated"]) == 3
    assert {offer.id: offer for offer in assembled.rented if offer.provider == "aws"} == {
        offer.id: offer for offer in load_curated_offers().rented if offer.provider == "aws"}
    assert store.set_cloud_rates.call_args.args[1]["cloud_rates"]["gcp.us-central1"]["on_demand"] == pytest.approx(
        good["offers"][0]["hourly_cost"])

    session = catalogue_session({"error": {"message": "Not authorized"}}, status=403)
    await api.job_scrape_cloud_rates()
    assert snapshots["gcp"] is good
    assert assemble_offers(curated, snapshots, errors).sources["gcp"].state == "stale"

    session = catalogue_session({"skus": catalogue})
    await api.job_scrape_cloud_rates()
    assert assemble_offers(curated, snapshots, errors).sources["gcp"].state == "fresh"
    assert "cloud.gcp" not in errors and "offers.gcp" not in errors

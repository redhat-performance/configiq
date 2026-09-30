# SPDX-License-Identifier: Apache-2.0

"""Tests for ValkeyStore — requires a running Valkey instance."""

from unittest.mock import Mock, patch

import pytest

from tools.api_service.valkey import ValkeyStore


class TestValkeyConnection:
    def test_connect_configures_timeout_and_health_checks(self):
        client = Mock()
        with patch("tools.api_service.valkey.valkey.from_url", return_value=client) as from_url:
            ValkeyStore("valkey://example:6379/0").connect()

        from_url.assert_called_once_with(
            "valkey://example:6379/0",
            decode_responses=True,
            socket_timeout=10,
            socket_connect_timeout=5,
            health_check_interval=30,
            retry_on_timeout=True,
        )
        client.ping.assert_called_once_with()


@pytest.fixture
def store():
    s = ValkeyStore("valkey://localhost:6379/15")  # use DB 15 for tests
    s.connect()
    s.flush()
    yield s
    s.flush()
    s.close()


class TestModels:
    def test_set_and_get(self, store: ValkeyStore):
        models = [{"id": "test/model-1", "name": "Test Model", "price_per_m_input": 1.0}]
        store.set_models(models)
        result = store.get_models()
        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "test/model-1"

    def test_get_returns_none_when_empty(self, store: ValkeyStore):
        assert store.get_models() is None

    def test_marks_scrape_time(self, store: ValkeyStore):
        # Scrape time is recorded by _mark_scrape (called by the scrape jobs
        # after a successful set_*), not by the setters themselves.
        store._mark_scrape("models")
        times = store.get_scrape_times()
        assert "models" in times


class TestCloudRates:
    def test_set_and_get(self, store: ValkeyStore):
        rates = {"cloud_rates": {"aws.us-east-1": {"on_demand": 4.50, "spot_median": 1.80}}}
        store.set_cloud_rates("h200_sxm", rates)
        result = store.get_cloud_rates("h200_sxm")
        assert result is not None
        assert result["cloud_rates"]["aws.us-east-1"]["on_demand"] == 4.50

    def test_get_all(self, store: ValkeyStore):
        store.set_cloud_rates("h200_sxm", {"rates": "h200"})
        store.set_cloud_rates("h100_sxm", {"rates": "h100"})
        all_rates = store.get_all_cloud_rates()
        assert "h200_sxm" in all_rates
        assert "h100_sxm" in all_rates

    def test_get_returns_none_when_missing(self, store: ValkeyStore):
        assert store.get_cloud_rates("nonexistent") is None


class TestHardwareCosts:
    def test_set_and_get(self, store: ValkeyStore):
        cost = {"new_usd": 42000, "used_usd": 28000}
        store.set_hardware_cost("h200_sxm", cost)
        result = store.get_hardware_cost("h200_sxm")
        assert result is not None
        assert result["new_usd"] == 42000

    def test_get_all(self, store: ValkeyStore):
        store.set_hardware_cost("h200_sxm", {"new_usd": 42000})
        store.set_hardware_cost("h100_sxm", {"new_usd": 32000})
        all_costs = store.get_all_hardware_costs()
        assert len(all_costs) == 2

    def test_get_returns_none_when_missing(self, store: ValkeyStore):
        assert store.get_hardware_cost("nonexistent") is None


class TestGpuIdMapping:
    def test_set_and_get(self, store: ValkeyStore):
        mapping = {"H100 SXM": "h100_sxm", "A100 80GB": "a100_sxm"}
        store.set_gpu_id_mapping(mapping)
        result = store.get_gpu_id_mapping()
        assert result == mapping

    def test_get_returns_none_when_missing(self, store: ValkeyStore):
        assert store.get_gpu_id_mapping() is None


class TestStaleness:
    def test_stale_when_never_scraped(self, store: ValkeyStore):
        assert store.is_stale("models", 3600) is True

    def test_not_stale_after_scrape(self, store: ValkeyStore):
        store._mark_scrape("models")
        assert store.is_stale("models", 3600) is False

    def test_is_empty(self, store: ValkeyStore):
        assert store.is_empty() is True
        store.set_models([])
        assert store.is_empty() is False

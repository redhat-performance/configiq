# SPDX-License-Identifier: Apache-2.0

"""Valkey client wrapper with typed key helpers and TTL management."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import valkey

logger = logging.getLogger(__name__)

VALKEY_URL = os.environ.get("VALKEY_URL", "valkey://localhost:6379/0")
VALKEY_SOCKET_TIMEOUT = 10
VALKEY_SOCKET_CONNECT_TIMEOUT = 5
VALKEY_HEALTH_CHECK_INTERVAL = 30

# TTLs per data category
TTL_MODELS = 25 * 3600          # 25 hours
TTL_CLOUD_RATES = 25 * 3600     # 25 hours
TTL_HARDWARE = 7 * 24 * 3600    # 7 days
TTL_GPU_ID_MAPPING = 30 * 24 * 3600  # 30 days

# Keys
KEY_HEALTH = "health"
KEY_GPU_ID_MAPPING = "gpu-id-mapping"


def _key_models(source: str) -> str:
    return f"models:{source}"


def _key_cloud(system_id: str) -> str:
    return f"systems:cloud:{system_id}"


def _key_hardware(system_id: str) -> str:
    return f"systems:hardware:{system_id}"


def _key_scrape_last(source: str) -> str:
    return f"scrape:last:{source}"


class ValkeyStore:
    """Thin wrapper around valkey-py with domain-specific helpers."""

    def __init__(self, url: str | None = None):
        self._url = url or VALKEY_URL
        self._client: valkey.Valkey | None = None

    def connect(self) -> None:
        self._client = valkey.from_url(
            self._url,
            decode_responses=True,
            socket_timeout=VALKEY_SOCKET_TIMEOUT,
            socket_connect_timeout=VALKEY_SOCKET_CONNECT_TIMEOUT,
            health_check_interval=VALKEY_HEALTH_CHECK_INTERVAL,
            retry_on_timeout=True,
        )
        self._client.ping()
        logger.info("Connected to Valkey at %s", self._url)

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    @property
    def client(self) -> valkey.Valkey:
        if self._client is None:
            raise RuntimeError("ValkeyStore not connected — call connect() first")
        return self._client

    def is_empty(self) -> bool:
        return self.client.dbsize() == 0

    # ── Models ────────────────────────────────────────────────────────────

    def set_models(self, models: list[dict[str, Any]], source: str = "merged") -> None:
        self.client.setex(_key_models(source), TTL_MODELS, json.dumps(models))

    def get_models(self, source: str = "merged") -> list[dict[str, Any]] | None:
        raw = self.client.get(_key_models(source))
        return json.loads(raw) if raw else None

    # ── Cloud rates ───────────────────────────────────────────────────────

    def set_cloud_rates(self, system_id: str, rates: dict[str, Any]) -> None:
        self.client.setex(_key_cloud(system_id), TTL_CLOUD_RATES, json.dumps(rates))

    def get_cloud_rates(self, system_id: str) -> dict[str, Any] | None:
        raw = self.client.get(_key_cloud(system_id))
        return json.loads(raw) if raw else None

    def get_all_cloud_rates(self) -> dict[str, dict[str, Any]]:
        result = {}
        for key in self.client.scan_iter(match="systems:cloud:*"):
            system_id = key.split(":")[-1]
            raw = self.client.get(key)
            if raw:
                result[system_id] = json.loads(raw)
        return result


    # ── Hardware costs ────────────────────────────────────────────────────

    def set_hardware_cost(self, system_id: str, cost: dict[str, Any]) -> None:
        self.client.setex(_key_hardware(system_id), TTL_HARDWARE, json.dumps(cost))

    def get_hardware_cost(self, system_id: str) -> dict[str, Any] | None:
        raw = self.client.get(_key_hardware(system_id))
        return json.loads(raw) if raw else None

    def get_all_hardware_costs(self) -> dict[str, dict[str, Any]]:
        result = {}
        for key in self.client.scan_iter(match="systems:hardware:*"):
            system_id = key.split(":")[-1]
            raw = self.client.get(key)
            if raw:
                result[system_id] = json.loads(raw)
        return result


    # ── GPU ID mapping ────────────────────────────────────────────────────

    def set_gpu_id_mapping(self, mapping: dict[str, str]) -> None:
        self.client.setex(KEY_GPU_ID_MAPPING, TTL_GPU_ID_MAPPING, json.dumps(mapping))

    def get_gpu_id_mapping(self) -> dict[str, str] | None:
        raw = self.client.get(KEY_GPU_ID_MAPPING)
        return json.loads(raw) if raw else None

    # ── Health / scrape tracking ──────────────────────────────────────────

    def set_scrape_error(self, source: str, message: str) -> None:
        key = f"scrape:error:{source}"
        payload = json.dumps({"at": datetime.now(UTC).isoformat(), "message": message})
        self.client.set(key, payload)

    def clear_scrape_error(self, source: str) -> None:
        self.client.delete(f"scrape:error:{source}")

    def get_scrape_errors(self) -> dict[str, dict[str, str]]:
        result = {}
        for key in self.client.scan_iter(match="scrape:error:*"):
            source = key.split(":", 2)[-1]
            raw = self.client.get(key)
            if raw:
                result[source] = json.loads(raw)
        return result

    def _mark_scrape(self, source: str) -> None:
        now = datetime.now(UTC).isoformat()
        self.client.set(_key_scrape_last(source), now)
        self.clear_scrape_error(source)

    def get_scrape_times(self) -> dict[str, str]:
        result = {}
        for key in self.client.scan_iter(match="scrape:last:*"):
            source = key.split(":")[-1]
            val = self.client.get(key)
            if val:
                result[source] = val
        return result

    def is_stale(self, source: str, max_age_seconds: int) -> bool:
        raw = self.client.get(_key_scrape_last(source))
        if not raw:
            return True
        last = datetime.fromisoformat(raw)
        age = (datetime.now(UTC) - last).total_seconds()
        return age > max_age_seconds

    # ── Utilities ─────────────────────────────────────────────────────────

    def flush(self) -> None:
        self.client.flushdb()

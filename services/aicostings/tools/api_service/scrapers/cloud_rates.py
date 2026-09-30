# SPDX-License-Identifier: Apache-2.0

"""Cloud GPU pricing scrapers.

Each provider scraper returns a list of RateRecord dicts:
  { "system_id": str, "provider_region": str, "on_demand": float|None, "spot_median": float|None }

Provider GPU names are mapped to AIC system IDs via the gpu-id-mapping.
"""

from __future__ import annotations

import logging
import os
import statistics
from pathlib import Path
from typing import Any

import aiohttp
import ijson
import yaml

logger = logging.getLogger(__name__)

GPU_ID_MAPPING_PATH = Path(__file__).parent.parent.parent.parent / "data" / "gpu-id-mapping.yaml"

RateRecord = dict[str, Any]


def load_gpu_id_mapping() -> dict[str, str]:
    if not GPU_ID_MAPPING_PATH.exists():
        return {}
    with open(GPU_ID_MAPPING_PATH) as f:
        data = yaml.safe_load(f)
    return data if data else {}


def _map_gpu_name(name: str, mapping: dict[str, str]) -> str | None:
    lower = name.lower()
    for pattern, system_id in mapping.items():
        if pattern.lower() in lower:
            return system_id
    return None


# ── Azure ─────────────────────────────────────────────────────────────────────

AZURE_GPU_SKUS = ["Standard_ND96asr_v4", "Standard_ND96amsr_A100_v4", "Standard_NC80adis_H100_v5"]
AZURE_GPUS_PER_INSTANCE = {
    "Standard_ND96asr_v4": 8,
    "Standard_ND96amsr_A100_v4": 8,
    "Standard_NC80adis_H100_v5": 2,
}
AZURE_API = "https://prices.azure.com/api/retail/prices"


async def scrape_azure(session: aiohttp.ClientSession, mapping: dict[str, str]) -> list[RateRecord]:
    records: list[RateRecord] = []
    failures = 0  # per-SKU errors; a total failure is propagated as a provider error
    for sku in AZURE_GPU_SKUS:
        try:
            params = {
                "$filter": f"armSkuName eq '{sku}' and priceType eq 'Consumption'",
                "$top": "100",
            }
            async with session.get(AZURE_API, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    logger.warning("Azure %s returned %d", sku, resp.status)
                    failures += 1
                    continue
                data = await resp.json()

            for item in data.get("Items", []):
                region = item.get("armRegionName", "")
                price = item.get("retailPrice")
                meter = item.get("meterName", "")
                if not region or price is None:
                    continue

                # Azure returns both Linux and Windows meters for the same SKU;
                # since records key on (system_id, region), a Windows entry would
                # overwrite the Linux price. GPU sizing is Linux-only, so skip
                # Windows variants (they carry a licence premium anyway).
                if "windows" in item.get("productName", "").lower():
                    continue

                system_id = _map_gpu_name(sku, mapping)
                if not system_id:
                    continue

                gpus_per_instance = AZURE_GPUS_PER_INSTANCE.get(sku)
                if not gpus_per_instance:
                    logger.warning("Azure: GPU count is unknown for %s; skipping rate", sku)
                    continue
                meter_lower = meter.lower()
                is_spot = "spot" in meter_lower or "low priority" in meter_lower
                per_gpu_price = float(price) / gpus_per_instance
                records.append({
                    "system_id": system_id,
                    "provider_region": f"azure.{region}",
                    "on_demand": None if is_spot else round(per_gpu_price, 6),
                    "spot_median": round(per_gpu_price, 6) if is_spot else None,
                    "rate_basis": "gpu_hour",
                    "gpus_per_instance": gpus_per_instance,
                })
        except Exception as e:
            logger.error("Azure scrape failed for %s: %s", sku, e)
            failures += 1

    # Tolerate partial failures, but propagate a complete failure so
    # scrape_all_cloud_rates records the provider as errored rather than as a
    # successful empty scrape — matching the raise-on-failure contract used by
    # scrape_aws and scrape_vastai.
    if failures == len(AZURE_GPU_SKUS):
        raise RuntimeError(f"Azure scrape failed for all {failures} SKU(s)")

    logger.info("Azure: %d rate records", len(records))
    return records


# ── AWS (public Price List bulk JSON, on-demand) ──────────────────────────────
#
# Credential-free: the Price List bulk API is public. The per-region EC2 offer
# file is large (hundreds of MB), so we (a) fetch only the regions in
# AWS_PRICING_REGIONS and (b) stream-parse with ijson, keeping only the SKUs for
# GPU instance types we care about. On-demand only — spot needs an AWS account
# (DescribeSpotPriceHistory), which is a deliberate follow-up.

AWS_PRICING_HOST = "https://pricing.us-east-1.amazonaws.com"
AWS_REGION_INDEX = f"{AWS_PRICING_HOST}/offers/v1.0/aws/AmazonEC2/current/region_index.json"
# Default to the two regions where our GPU instance types are most available;
# override with a comma-separated AWS_PRICING_REGIONS.
AWS_PRICING_REGIONS = [
    r.strip() for r in os.environ.get("AWS_PRICING_REGIONS", "us-east-1,us-west-2").split(",") if r.strip()
]
AWS_GPUS_PER_INSTANCE = {
    "p4d.24xlarge": 8,
    "p5.48xlarge": 8,
}


def _aws_gpu_instance_types(mapping: dict[str, str]) -> set[str]:
    """AWS instance types from the GPU id-mapping (e.g. p5.48xlarge)."""
    return {name for name in mapping if "xlarge" in name.lower()}


async def _aws_region_file_urls(session: aiohttp.ClientSession) -> dict[str, str]:
    async with session.get(AWS_REGION_INDEX, timeout=aiohttp.ClientTimeout(total=60)) as resp:
        if resp.status != 200:
            raise RuntimeError(f"AWS region index returned {resp.status}")
        # AWS serves the Price List JSON as application/octet-stream, which
        # aiohttp's .json() rejects by default ("unexpected mimetype"); the
        # body is valid JSON, so skip the content-type guard.
        index = await resp.json(content_type=None)
    regions = index.get("regions", {})
    return {code: AWS_PRICING_HOST + meta["currentVersionUrl"] for code, meta in regions.items()}


async def _aws_scrape_region(
    session: aiohttp.ClientSession, region: str, url: str, want_types: set[str], mapping: dict[str, str],
) -> list[RateRecord]:
    """Stream a region's EC2 offer file and extract on-demand GPU instance prices.

    The offer file lists `products.<sku>.attributes` (products first) then
    `terms.OnDemand.<sku>...pricePerUnit.USD`. We accumulate one product's
    attributes at a time (they arrive contiguously), keep the SKUs that match a
    GPU instance type + standard Linux/Shared/Used/NA terms, then pick up their
    on-demand USD/hr in the same single pass.

    NOTE: the OnDemand term/rate keys are *composite and contain dots* — e.g.
    `terms.OnDemand.<sku>.<sku>.<offerTermCode>.priceDimensions.<sku>.<offerTermCode>.<rateCode>.pricePerUnit.USD`
    — so the USD leaf's dotted prefix has a variable component count and cannot
    be matched by a fixed `len(parts)`/positional check. The SKU token itself
    never contains a dot, so we take it as the third path component and match the
    USD leaf by suffix instead.
    """
    matched_skus: dict[str, str] = {}  # sku -> instance_type
    seen: set[str] = set()             # (system_id, region) already recorded
    records: list[RateRecord] = []
    cur_sku: str | None = None
    cur_attrs: dict[str, str] = {}

    def finalize(sku: str | None, attrs: dict[str, str]) -> None:
        if not sku:
            return
        if (
            attrs.get("instanceType") in want_types
            and attrs.get("tenancy") == "Shared"
            and attrs.get("operatingSystem") == "Linux"
            and attrs.get("capacitystatus") == "Used"
            and attrs.get("preInstalledSw") == "NA"
        ):
            matched_skus[sku] = attrs["instanceType"]

    async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as resp:
        if resp.status != 200:
            raise RuntimeError(f"AWS offer file for {region} returned {resp.status}")
        async for prefix, event, value in ijson.parse_async(resp.content):
            if prefix.startswith("products."):
                parts = prefix.split(".")
                sku = parts[1] if len(parts) > 1 else None
                if sku != cur_sku:
                    finalize(cur_sku, cur_attrs)
                    cur_sku, cur_attrs = sku, {}
                if len(parts) == 4 and parts[2] == "attributes" and event == "string":
                    cur_attrs[parts[3]] = value
            elif prefix.startswith("terms.OnDemand."):
                if cur_sku is not None:
                    finalize(cur_sku, cur_attrs)
                    cur_sku, cur_attrs = None, {}
                # Match the USD leaf by suffix; the SKU is always the 3rd path
                # component (it has no dots) even though later term/rate keys do.
                if event == "string" and prefix.endswith(".pricePerUnit.USD"):
                    sku = prefix.split(".", 3)[2]  # terms . OnDemand . <sku> . …
                    if sku not in matched_skus:
                        continue
                    try:
                        usd = float(value)
                    except (TypeError, ValueError):
                        continue
                    if usd <= 0:
                        continue
                    instance_type = matched_skus[sku]
                    system_id = _map_gpu_name(instance_type, mapping)
                    if not system_id or (system_id, region) in seen:
                        continue
                    gpus_per_instance = AWS_GPUS_PER_INSTANCE.get(instance_type)
                    if not gpus_per_instance:
                        logger.warning("AWS: GPU count is unknown for %s; skipping rate", instance_type)
                        continue
                    seen.add((system_id, region))
                    records.append({
                        "system_id": system_id,
                        "provider_region": f"aws.{region}",
                        "on_demand": round(usd / gpus_per_instance, 6),
                        "spot_median": None,
                        "rate_basis": "gpu_hour",
                        "gpus_per_instance": gpus_per_instance,
                    })
    return records


async def scrape_aws(session: aiohttp.ClientSession, mapping: dict[str, str]) -> list[RateRecord]:
    want_types = _aws_gpu_instance_types(mapping)
    if not want_types:
        logger.info("AWS: no GPU instance types in the id-mapping; nothing to scrape")
        return []
    region_urls = await _aws_region_file_urls(session)
    records: list[RateRecord] = []
    for region in AWS_PRICING_REGIONS:
        url = region_urls.get(region)
        if not url:
            logger.warning("AWS: no offer file URL for region %s", region)
            continue
        region_records = await _aws_scrape_region(session, region, url, want_types, mapping)
        records.extend(region_records)
        logger.info("AWS %s: %d GPU rate records", region, len(region_records))
    # We map real, widely-available instance types (p4d/p5), so a completely
    # empty result means the parse silently matched nothing — almost certainly a
    # Price List schema change. Propagate it as an error (matching Azure's
    # raise-on-total-failure contract) so /health surfaces it instead of
    # recording a misleading "fresh" success with no data.
    if not records:
        raise RuntimeError(
            f"AWS scrape produced no rate records for {len(want_types)} GPU instance "
            f"type(s) across region(s) {AWS_PRICING_REGIONS}; the Price List schema may "
            "have changed"
        )
    logger.info("AWS: %d rate records across %d region(s)", len(records), len(AWS_PRICING_REGIONS))
    return records


# ── Vast.ai (public marketplace API) ──────────────────────────────────────────
#
# Credential-free. Vast is a GPU marketplace, so prices vary per offer; we take
# the median per-GPU $/hr across current rentable offers and record it as a
# spot-style rate. NOTE: the response schema below is best-effort and should be
# validated against the live API.

VASTAI_URL = "https://console.vast.ai/api/v0/bundles/"


async def scrape_vastai(session: aiohttp.ClientSession, mapping: dict[str, str]) -> list[RateRecord]:
    import json as _json

    query = {"rentable": {"eq": True}, "rented": {"eq": False}}
    params = {"q": _json.dumps(query)}
    async with session.get(VASTAI_URL, params=params, timeout=aiohttp.ClientTimeout(total=60)) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Vast.ai returned {resp.status}")
        data = await resp.json()

    # Group per-GPU $/hr by system so we can report a stable median.
    per_system: dict[str, list[float]] = {}
    for offer in data.get("offers", []):
        gpu_name = offer.get("gpu_name")
        num_gpus = offer.get("num_gpus") or 0
        dph_total = offer.get("dph_total")
        if not gpu_name or not num_gpus or dph_total is None:
            continue
        system_id = _map_gpu_name(gpu_name, mapping)
        if not system_id:
            continue
        try:
            per_gpu = float(dph_total) / int(num_gpus)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if per_gpu > 0:
            per_system.setdefault(system_id, []).append(per_gpu)

    records: list[RateRecord] = []
    for system_id, prices in per_system.items():
        records.append({
            "system_id": system_id,
            "provider_region": "vastai.marketplace",
            "on_demand": None,
            "spot_median": round(statistics.median(prices), 4),
            "rate_basis": "gpu_hour",
            "gpus_per_instance": 1,
        })
    logger.info("Vast.ai: %d systems priced from %d offers", len(records), len(data.get("offers", [])))
    return records


# ── Key-gated providers ───────────────────────────────────────────────────────
#
# These are only invoked when their API key is set (see _SCRAPERS / active_scrapers),
# so they don't pollute /health when unconfigured. Their auth is wired via the
# named env var, but the response parsing is UNVERIFIED against the live APIs, so
# each raises NotImplementedError until it has been implemented and validated with
# a real key — that surfaces in /health as an error rather than emitting wrong
# prices.


async def scrape_runpod(session: aiohttp.ClientSession, mapping: dict[str, str]) -> list[RateRecord]:
    """RunPod GPU pricing via the public GraphQL API.

    Auth is wired via RUNPOD_API_KEY; GraphQL response→system parsing is
    UNVERIFIED. Raises until implemented and tested.
    """
    os.environ["RUNPOD_API_KEY"]  # presence already gated; kept explicit
    raise NotImplementedError("RunPod GraphQL response parsing not yet implemented")


async def scrape_lambda(session: aiohttp.ClientSession, mapping: dict[str, str]) -> list[RateRecord]:
    """Lambda Labs on-demand pricing via the Cloud API.

    Auth is wired via LAMBDA_API_KEY; instance-type response→system parsing is
    UNVERIFIED. Raises until implemented and tested.
    """
    os.environ["LAMBDA_API_KEY"]  # presence already gated; kept explicit
    raise NotImplementedError("Lambda Labs response parsing not yet implemented")


async def scrape_gcp(session: aiohttp.ClientSession, mapping: dict[str, str]) -> list[RateRecord]:
    """GCP GPU pricing via the Cloud Billing Catalog API.

    Auth (API key) is wired, but mapping GCP's per-SKU catalog descriptions to
    our GPU system ids is non-trivial and UNVERIFIED. Raises until the SKU→system
    parsing is implemented and tested, so it surfaces in /health as an error
    rather than emitting wrong prices.
    """
    os.environ["GCP_BILLING_API_KEY"]  # presence already gated; kept explicit
    raise NotImplementedError("GCP catalog SKU→system parsing not yet implemented")


async def scrape_scaleway(session: aiohttp.ClientSession, mapping: dict[str, str]) -> list[RateRecord]:
    """Scaleway GPU instance pricing.

    Auth is wired via SCALEWAY_SECRET_KEY; product-catalog→system parsing is
    UNVERIFIED. Raises until implemented and tested.
    """
    os.environ["SCALEWAY_SECRET_KEY"]
    raise NotImplementedError("Scaleway product-catalog parsing not yet implemented")


async def scrape_ibmcloud(session: aiohttp.ClientSession, mapping: dict[str, str]) -> list[RateRecord]:
    """IBM Cloud GPU pricing via the Global Catalog API.

    Auth is wired via IBMCLOUD_API_KEY; catalog→system parsing is UNVERIFIED.
    Raises until implemented and tested.
    """
    os.environ["IBMCLOUD_API_KEY"]
    raise NotImplementedError("IBM Cloud catalog parsing not yet implemented")


# Scraper registry. `env` is the API key required to activate the provider (None
# = credential-free, always active). active_scrapers() filters to the ones whose
# key is present so /health only ever reports providers actually being scraped.
_SCRAPERS: list[tuple[str, Any, str | None]] = [
    ("azure", scrape_azure, None),
    ("aws", scrape_aws, None),
    ("vastai", scrape_vastai, None),
    ("runpod", scrape_runpod, "RUNPOD_API_KEY"),
    ("lambda", scrape_lambda, "LAMBDA_API_KEY"),
    ("gcp", scrape_gcp, "GCP_BILLING_API_KEY"),
    ("scaleway", scrape_scaleway, "SCALEWAY_SECRET_KEY"),
    ("ibmcloud", scrape_ibmcloud, "IBMCLOUD_API_KEY"),
]


def active_scrapers() -> list[tuple[str, Any]]:
    """Credential-free scrapers plus any key-gated ones whose key is set."""
    active: list[tuple[str, Any]] = []
    for name, fn, env in _SCRAPERS:
        if env is None or os.environ.get(env):
            active.append((name, fn))
        else:
            logger.info("Skipping %s scraper: %s not set", name, env)
    return active


# Back-compat alias for callers/tests that referenced the old registry.
ALL_SCRAPERS = [(name, fn) for name, fn, _ in _SCRAPERS]


def _merge_records(records: list[RateRecord]) -> dict[str, dict[str, dict[str, Any]]]:
    """Merge rate records into per-system, per-provider.region structure."""
    systems: dict[str, dict[str, dict[str, Any]]] = {}

    for rec in records:
        sid = rec["system_id"]
        pr = rec["provider_region"]
        if sid not in systems:
            systems[sid] = {}
        if pr not in systems[sid]:
            systems[sid][pr] = {
                "on_demand": None,
                "spot_median": None,
                "rate_basis": rec.get("rate_basis"),
                "gpus_per_instance": rec.get("gpus_per_instance"),
            }

        if rec.get("on_demand") is not None:
            current = systems[sid][pr]["on_demand"]
            systems[sid][pr]["on_demand"] = min(rec["on_demand"], current) if current is not None else rec["on_demand"]
        if rec.get("spot_median") is not None:
            current = systems[sid][pr]["spot_median"]
            systems[sid][pr]["spot_median"] = min(rec["spot_median"], current) if current is not None else rec["spot_median"]
        if systems[sid][pr].get("rate_basis") is None:
            systems[sid][pr]["rate_basis"] = rec.get("rate_basis")
        if systems[sid][pr].get("gpus_per_instance") is None:
            systems[sid][pr]["gpus_per_instance"] = rec.get("gpus_per_instance")

    return systems


ProviderResult = tuple[str, list[RateRecord] | Exception]


async def scrape_all_cloud_rates(
    session: aiohttp.ClientSession,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Exception | None]]:
    """Scrape all cloud providers concurrently.

    Returns:
        merged: per-system, per-provider.region rate data
        errors: mapping of provider name → Exception (or None if successful)
    """
    import asyncio

    mapping = load_gpu_id_mapping()
    scrapers = active_scrapers()
    tasks = [scraper(session, mapping) for _, scraper in scrapers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_records: list[RateRecord] = []
    errors: dict[str, Exception | None] = {}

    for (name, _), result in zip(scrapers, results):
        if isinstance(result, Exception):
            logger.error("Scraper %s failed: %s", name, result)
            errors[name] = result
        else:
            errors[name] = None
            all_records.extend(result)

    merged = _merge_records(all_records)
    logger.info("Cloud rates: %d systems with pricing, %d provider errors",
                len(merged), sum(1 for e in errors.values() if e is not None))
    return merged, errors

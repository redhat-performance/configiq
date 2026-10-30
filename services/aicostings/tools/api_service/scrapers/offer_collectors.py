# SPDX-License-Identifier: Apache-2.0

"""Complete billable offers. No GPU arithmetic or pricing-page browser automation."""

from __future__ import annotations

import os
import re
from collections import Counter
from datetime import UTC, datetime
from html.parser import HTMLParser

import aiohttp

from ..offers import PriceSource, RentalOffer, load_curated_offers
from .cloud_rates import AWS_PRICING_REGIONS, RateRecord

COREWEAVE_URL = "https://www.coreweave.com/pricing"
GCP_SKUS_URL = "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus"

# These aliases describe *different* physical GPUs from the SDK identities used
# by the legacy aggregate feed. Do not inherit those historical approximations.
UNSUPPORTED_VARIANTS = {"p4d.24xlarge", "Standard_ND96asr_v4", "Standard_NC80adis_H100_v5"}


def snapshot(offers: list[RentalOffer], scope: dict) -> dict:
    return {"offers": [offer.model_dump(mode="json") for offer in offers],
            "collected_at": datetime.now(UTC).isoformat(), "scope": scope}


def snapshot_from_rates(provider: str, records: list[RateRecord]) -> dict:
    """The provider is fetched once; this is a second view of the raw records."""
    curated = load_curated_offers().rented
    if any(record.get("collection_partial") for record in records):
        raise ValueError("Incomplete provider collection; retain the last complete snapshot")
    result: dict[tuple, RentalOffer] = {}
    now = datetime.now(UTC)
    managed = set()
    for record in records:
        name = record.get("instance_name")
        if not name or name in UNSUPPORTED_VARIANTS:
            continue
        if record.get("currency") != "USD" or record.get("unit") not in {"Hrs", "1 Hour"}:
            continue
        kind = "spot" if record.get("on_demand") is None else "on_demand"
        prior = next((offer for offer in curated if offer.provider == provider and offer.instance_name == name), None)
        region = record["provider_region"]
        offer = RentalOffer(
            id=f"{provider}:{region}:{name}:{kind}", system_id=record["system_id"], provider=provider,
            provider_region=region, instance_name=name, gpu_count=record["gpus_per_instance"],
            hourly_cost=record["instance_hourly_cost"], rate_kind=kind,
            max_instances_per_replica=prior.max_instances_per_replica if prior else None,
            interconnect=prior.interconnect if prior else None,
            source=PriceSource(kind="automated", label=f"{provider.upper()} public instance price",
                               url=record["source_url"], retrieved_at=now, effective_at=record.get("effective_at")),
        )
        key = (region, name, kind)
        # Exact duplicate meters are not separate offers. Never mix a price
        # with the GPU count from a different SKU as aggregate min() can do.
        if key not in result or offer.hourly_cost < result[key].hourly_cost:
            result[key] = offer
        managed.add(name)
    if not result:
        raise ValueError("No validated complete instance offers")
    # Include previously managed shapes so an instance removed by the provider
    # cannot return from the seed catalogue on the next successful refresh.
    if provider == "aws":
        from .cloud_rates import AWS_GPUS_PER_INSTANCE
        managed.update(set(AWS_GPUS_PER_INSTANCE) - UNSUPPORTED_VARIANTS)
    regions = [f"aws.{region}" for region in AWS_PRICING_REGIONS] if provider == "aws" else sorted({r[0] for r in result})
    if provider == "gcp":
        regions.append("gcp.public-reference")
    return snapshot(list(result.values()), {"instances": sorted(managed), "regions": regions,
                                            "rate_kinds": ["on_demand", "spot"]})


class CoreWeavePricingParser(HTMLParser):
    """Read the published GPU rows, not desktop/mobile price duplicates.

    The provider currently renders pricing in HTML. A schema change produces
    an explicit collector failure instead of guessing at nearby dollar values.
    """

    def __init__(self):
        super().__init__()
        self.depth = 0
        self.region = ""
        self.region_text: list[str] | None = None
        self.row_depth: int | None = None
        self.cell_depth: int | None = None
        self.cell: list[str] = []
        self.cells: list[str] = []
        self.rows: list[tuple[str, list[str]]] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "h5":
            self.region_text = []
        if tag != "div":
            return
        self.depth += 1
        classes = attrs.get("class", "").split()
        if "kubernetes-gpu-pricing" in classes:
            self.row_depth = self.depth
            self.cells = []
        elif self.row_depth is not None and "table-v2-cell" in classes:
            self.cell_depth = self.depth
            self.cell = []

    def handle_data(self, data):
        if self.region_text is not None:
            self.region_text.append(data)
        if self.cell_depth is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "h5" and self.region_text is not None:
            label = " ".join(self.region_text).strip().upper()
            self.region = {"REGION: NORTH AMERICA": "north-america", "REGION: EUROPE": "europe"}.get(label, "")
            self.region_text = None
        if tag != "div":
            return
        if self.cell_depth == self.depth:
            self.cells.append(" ".join(" ".join(self.cell).split()))
            self.cell_depth = None
        if self.row_depth == self.depth:
            self.rows.append((self.region, self.cells))
            self.row_depth = None
        self.depth -= 1


COREWEAVE_SYSTEMS = {
    "NVIDIA HGX B200": "b200_sxm", "NVIDIA HGX B300": "b300_sxm",
    "NVIDIA GB200 NVL72": "gb200", "NVIDIA GB300 NVL72": "gb300",
    "NVIDIA HGX H100": "h100_sxm", "NVIDIA HGX H200": "h200_sxm",
    "NVIDIA A100": "a100_sxm", "NVIDIA L40S": "l40s",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition (High Memory)": "rtx_pro_6000_server",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition (Standard Memory)": "rtx_pro_6000_server",
}


def parse_coreweave_prices(
    html: str, configurations: list[RentalOffer] | None = None, *, complete_page: bool = False,
) -> list[RentalOffer]:
    parser = CoreWeavePricingParser()
    parser.feed(html)
    if complete_page:
        # The public page currently publishes the same product rows in two
        # regional tables. Validate that contract before replacing the provider
        # snapshot. Do not hardcode prices or demand that delisted GPUs remain.
        regions = {region for region, _ in parser.rows}
        if (not html.rstrip().lower().endswith("</html>") or regions != {"north-america", "europe"}
                or "w-pagination-next" in html or any(len(cells) not in {8, 9} for _, cells in parser.rows)):
            raise ValueError("CoreWeave pricing page is incomplete or its table contract changed")
        tables = [Counter((cells[0], cells[1], cells[2]) for region, cells in parser.rows if region == name)
                  for name in ("north-america", "europe")]
        if tables[0] != tables[1]:
            raise ValueError("CoreWeave regional product tables differ; completeness requires review")
    offers = []
    now = datetime.now(UTC)
    for region, cells in parser.rows:
        if not region or len(cells) not in {8, 9}:
            raise ValueError("CoreWeave GPU pricing layout changed")
        name = cells[0]
        system = COREWEAVE_SYSTEMS.get(name)
        if not system:
            continue  # Unknown GPUs need an explicit, reviewed SDK identity.
        count = int(cells[1].split("^")[0])
        # Published A100 pricing must describe the 80GB variant we simulate.
        if system == "a100_sxm" and cells[2] != "80":
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        for column, kind, eligibility in [(6, "on_demand", "standard"), (7, "spot", "standard"),
                                           (8, "on_demand", "inference_platform")]:
            if column >= len(cells) or not re.fullmatch(r"\$[\d,]+(?:\.\d+)?", cells[column]):
                continue  # N/A and Contact sales are never zero-price offers.
            gpu_count = 1 if eligibility == "inference_platform" else count
            prior = next((record for record in configurations or [] if record.provider == "coreweave"
                          and record.system_id == system and record.gpu_count == gpu_count
                          and record.eligibility == eligibility), None)
            offers.append(RentalOffer(
                id=f"coreweave:{region}:{slug}:{gpu_count}:{kind}:{eligibility}",
                system_id=system, provider="coreweave", provider_region=f"coreweave.{region}",
                instance_name=f"CoreWeave {gpu_count}× {name}", gpu_count=gpu_count,
                hourly_cost=float(cells[column].removeprefix("$").replace(",", "")), rate_kind=kind,
                eligibility=eligibility,
                max_instances_per_replica=prior.max_instances_per_replica if prior else None,
                interconnect=prior.interconnect if prior else None,
                source=PriceSource(kind="automated", label="CoreWeave public pricing page",
                                   url=COREWEAVE_URL, retrieved_at=now),
            ))
    if not offers:
        raise ValueError("CoreWeave page contains no validated GPU offers")
    if any(offer.eligibility == "inference_platform" for offer in offers) and "exclusively applicable" not in html:
        raise ValueError("CoreWeave inference eligibility terms changed")
    if len({offer.id for offer in offers}) != len(offers):
        raise ValueError("Duplicate CoreWeave GPU rows")
    return offers


async def scrape_coreweave_offers(session: aiohttp.ClientSession) -> dict:
    async with session.get(COREWEAVE_URL, timeout=aiohttp.ClientTimeout(total=30)) as response:
        if response.status != 200:
            raise RuntimeError(f"CoreWeave pricing returned HTTP {response.status}")
        offers = parse_coreweave_prices(await response.text(), load_curated_offers().rented, complete_page=True)
    return snapshot(offers, {"all": True})


def parse_gcp_a2_prices(skus: list[dict], regions: list[str]) -> list[RentalOffer]:
    """Conservative A2 Ultra composition: GPU + vCPU + RAM + bundled Local SSD.

    Other machine families remain explicitly curated until their component
    mappings are implemented and validated. Ambiguous/tiered prices fail closed.
    Public catalogue prices only: never call billingAccounts or private quotes.
    """
    offers = []
    now = datetime.now(UTC)
    for region in regions:
        components: dict[str, list[tuple[float, str, str | None]]] = {key: [] for key in ("gpu", "cpu", "ram", "ssd")}
        for sku in skus:
            if region not in sku.get("serviceRegions", []) or sku.get("category", {}).get("usageType") != "OnDemand":
                continue
            description = sku.get("description", "").lower()
            component = None
            if "a100" in description and "80gb" in description.replace(" ", ""):
                component = "gpu"
            elif "a2" in description and "ultra" in description:
                if "core" in description or "cpu" in description:
                    component = "cpu"
                elif "ram" in description:
                    component = "ram"
            elif "local ssd" in description and "preemptible" not in description:
                component = "ssd"
            if not component:
                continue
            pricing = sorted(
                (item for item in sku.get("pricingInfo", []) if item.get("effectiveTime")
                 and datetime.fromisoformat(item["effectiveTime"]) <= now),
                key=lambda item: item["effectiveTime"],
            )
            if not pricing:
                continue
            current = pricing[-1]
            expression = current.get("pricingExpression", {})
            tiers = expression.get("tieredRates", [])
            expected_units = {"h"} if component in {"gpu", "cpu"} else {"GiBy.h", "GiBy.mo"}
            if expression.get("usageUnit") not in expected_units or len(tiers) != 1 or tiers[0].get("startUsageAmount", 0) != 0:
                continue
            price = tiers[0].get("unitPrice", {})
            if price.get("currencyCode") != "USD":
                continue
            hourly = int(price.get("units", 0)) + int(price.get("nanos", 0)) / 1_000_000_000
            if expression["usageUnit"] == "GiBy.mo":
                hourly /= 730
            if hourly > 0:
                components[component].append((hourly, sku.get("skuId", ""), current.get("effectiveTime")))
        if any(len(matches) != 1 for matches in components.values()):
            raise ValueError(f"GCP region {region} has incomplete or ambiguous A2 Ultra component prices")
        rates = {key: matches[0][0] for key, matches in components.items()}
        for count in (1, 2, 4, 8):
            name = f"a2-ultragpu-{count}g"
            offers.append(RentalOffer(
                id=f"gcp:{region}:{name}:on_demand", system_id="a100_sxm", provider="gcp",
                provider_region=f"gcp.{region}", instance_name=f"GCP {name}", gpu_count=count,
                hourly_cost=count * (rates["gpu"] + 12 * rates["cpu"] + 170 * rates["ram"] + 375 * rates["ssd"]),
                rate_kind="on_demand", source=PriceSource(
                    kind="automated", label="GCP public catalogue: GPU + vCPU + RAM + bundled Local SSD",
                    url=GCP_SKUS_URL,
                    retrieved_at=now, effective_at=max(match[0][2] or "" for match in components.values()) or None,
                ),
            ))
    if not offers:
        raise ValueError("GCP catalogue has no unambiguous complete A2 Ultra prices; component mapping requires validation")
    return offers


async def scrape_gcp_offers(session: aiohttp.ClientSession) -> dict:
    key = os.environ["GCP_BILLING_API_KEY"]
    skus = []
    token = ""
    seen = set()
    while True:
        params = {"key": key, "currencyCode": "USD", "pageSize": "5000", "pageToken": token}
        try:
            async with session.get(GCP_SKUS_URL, params=params, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status != 200:
                    raise RuntimeError(f"GCP public catalogue returned HTTP {response.status}")
                data = await response.json()
        except (aiohttp.ClientError, TimeoutError):
            # aiohttp exceptions may contain the API-key query string.
            raise RuntimeError("GCP public catalogue request failed") from None
        skus.extend(data.get("skus", []))
        token = data.get("nextPageToken", "")
        if not token:
            break
        if token in seen:
            raise ValueError("Repeated GCP pagination token")
        seen.add(token)
    regions = [region.strip() for region in os.getenv("GCP_PRICING_REGIONS", "us-central1").split(",") if region.strip()]
    offers = parse_gcp_a2_prices(skus, regions)
    return snapshot(offers, {"instances": [f"GCP a2-ultragpu-{count}g" for count in (1, 2, 4, 8)],
                             "regions": [f"gcp.{region}" for region in regions] + ["gcp.public-reference"],
                             "rate_kinds": ["on_demand"]})

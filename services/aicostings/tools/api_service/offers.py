# SPDX-License-Identifier: Apache-2.0

"""Configuration-level pricing, separate from the legacy per-GPU API."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)
CATALOGUE_PATH = Path(__file__).parents[2] / "data" / "infrastructure-offers.yaml"


class PriceSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["automated", "curated"]
    label: str = Field(min_length=1)
    url: str | None = Field(default=None, pattern=r"^https?://")
    price_date: date | None = None
    retrieved_at: datetime | None = None
    effective_at: datetime | None = None
    stale: bool = False


class RentalOffer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    system_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    provider_region: str = Field(min_length=1)
    instance_name: str = Field(min_length=1)
    gpu_count: int = Field(gt=0, strict=True)
    hourly_cost: float = Field(gt=0, allow_inf_nan=False, strict=True)
    currency: Literal["USD"] = "USD"
    rate_basis: Literal["instance_hour"] = "instance_hour"
    rate_kind: Literal["on_demand", "spot", "capacity_block"]
    eligibility: Literal["standard", "inference_platform"] = "standard"
    max_instances_per_replica: int | None = Field(default=None, gt=0, strict=True)
    interconnect: str | None = None
    source: PriceSource


class ServerOffer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    system_id: str = Field(min_length=1)
    gpu_count: int = Field(gt=0, strict=True)
    purchase_price: float = Field(gt=0, allow_inf_nan=False, strict=True)
    installation_cost: float | None = Field(default=None, ge=0, allow_inf_nan=False, strict=True)
    currency: Literal["USD"] = "USD"
    price_basis: Literal["complete_server"] = "complete_server"
    includes: str = Field(min_length=1)
    indicative: bool = True
    source: PriceSource


class CollectionStatus(BaseModel):
    state: Literal["fresh", "stale", "unconfigured", "unavailable", "curated"]
    last_success: datetime | None = None
    message: str | None = None


class OffersResponse(BaseModel):
    schema_version: Literal[1] = 1
    rented: list[RentalOffer] = Field(default_factory=list)
    purchased: list[ServerOffer] = Field(default_factory=list)
    sources: dict[str, CollectionStatus] = Field(default_factory=dict)


def offer_identity(offer: RentalOffer) -> tuple[str, str, str, str, str]:
    return (offer.provider, offer.provider_region, offer.instance_name, offer.rate_kind, offer.eligibility)


def load_curated_offers(path: Path = CATALOGUE_PATH) -> OffersResponse:
    """Validate the entire snapshot before publishing any of it.

    Dates in this file describe the estimate, not the time YAML was reloaded.
    Missing or invalid data must never be replaced by a fabricated zero price.
    """
    with path.open() as stream:
        data = OffersResponse.model_validate(yaml.safe_load(stream))
    ids = [record.id for record in [*data.rented, *data.purchased]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate infrastructure offer IDs")
    if len({offer_identity(record) for record in data.rented}) != len(data.rented):
        raise ValueError("Duplicate rental identities")
    return data


def assemble_offers(curated: dict | None, snapshots: dict[str, dict], errors: dict) -> OffersResponse:
    """Join independent provider snapshots without disguising a failed refresh.

    A successful collector owns its declared scope. Removed offers in that scope
    cannot reappear from the old curated catalogue. Failed collectors retain
    their last-good snapshot, marked stale, for the store's bounded retention.
    """
    result = OffersResponse.model_validate(curated) if curated else OffersResponse()
    result.sources["curated"] = CollectionStatus(
        state="curated" if curated else "unavailable",
        message="Manually maintained, dated estimates; not live quotes." if curated else "Catalogue unavailable.",
    )
    if "offers.curated" in errors:
        result.sources["curated"].message = "Catalogue reload failed; retained estimates keep their original price dates."
    for provider in ("aws", "azure", "coreweave", "gcp"):
        snapshot = snapshots.get(provider)
        if not snapshot:
            failed = f"offers.{provider}" in errors
            result.sources[provider] = CollectionStatus(
                state="unconfigured" if provider == "gcp" and not failed else "unavailable",
                message="Collection failed or the retained snapshot expired; no automatic prices available." if failed
                else "API key required for automatic collection." if provider == "gcp"
                else "No collected offers available; any catalogue estimates are labelled curated.",
            )
            # Once automatic collection has failed, do not present a curated
            # price as if it were a replacement successful live response.
            if f"offers.{provider}" in errors:
                result.rented = [offer for offer in result.rented if offer.provider != provider]
            continue
        try:
            collected_at = datetime.fromisoformat(snapshot["collected_at"])
            records = [RentalOffer.model_validate(record) for record in snapshot["offers"]]
            if any(record.provider != provider for record in records):
                raise ValueError("Provider snapshot contains a different provider")
            age = (datetime.now(UTC) - collected_at).total_seconds()
            stale = age < 0 or age > 25 * 3600 or f"offers.{provider}" in errors
            scope = snapshot["scope"]
            result.rented = [offer for offer in result.rented if not (
                offer.provider == provider
                and (scope.get("all") or (
                    offer.instance_name in scope.get("instances", [])
                    and offer.provider_region in scope.get("regions", [])
                    and offer.rate_kind in scope.get("rate_kinds", ["on_demand"])
                ))
            )]
            for record in records:
                record.source.stale = stale
            result.rented.extend(records)
            result.sources[provider] = CollectionStatus(
                state="stale" if stale else "fresh", last_success=collected_at,
                message="Refresh failed; showing last-good collected prices." if f"offers.{provider}" in errors else None,
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            logger.warning("Invalid %s offer snapshot; excluding provider", provider)
            result.rented = [offer for offer in result.rented if offer.provider != provider]
            result.sources[provider] = CollectionStatus(state="unavailable", message="Invalid pricing snapshot.")
    result.rented.sort(key=lambda offer: offer.id)
    result.purchased.sort(key=lambda offer: offer.id)
    return result

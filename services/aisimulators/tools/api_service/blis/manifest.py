"""Read the image-built BLIS support manifest and resolve request identities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


class BlisManifestError(RuntimeError):
    """The BLIS support manifest is missing or malformed."""


class BlisUnsupportedRequest(ValueError):
    """BLIS cannot represent a model or hardware choice from the request."""


@dataclass(frozen=True)
class ModelResolution:
    catalog_name: str
    origin: str
    aisimulate_model_id: str | None
    config_relation: str | None
    equivalent_catalog_name: str | None


def _hardware_candidates(system: str) -> tuple[str, ...]:
    """Translate system-ID spelling into catalog candidates; support.json decides availability."""
    normalized = system.lower().replace("_", "-")
    candidates = [normalized]
    if normalized.endswith(("-sxm", "-pcie")):
        candidates.append(normalized.rsplit("-", 1)[0])
    # The BLIS catalog distinguishes A100's PCIe and SXM models by memory SKU.
    if normalized == "a100-pcie":
        candidates.insert(0, "a100-80")
    return tuple(dict.fromkeys(candidates))


@dataclass(frozen=True)
class SupportManifest:
    models: dict[str, dict[str, Any]]
    skipped: tuple[dict[str, Any], ...]
    hardware: frozenset[str]
    networks: frozenset[str]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SupportManifest:
        if raw.get("schema_version") != 1:
            raise BlisManifestError("BLIS support manifest has an unsupported schema version")
        models = raw.get("models")
        skipped = raw.get("skipped")
        hardware = raw.get("hardware")
        networks = raw.get("networks")
        if not isinstance(models, dict) or not isinstance(skipped, list):
            raise BlisManifestError("BLIS support manifest is missing model support data")
        if not isinstance(hardware, list) or not all(isinstance(item, str) for item in hardware):
            raise BlisManifestError("BLIS support manifest has invalid hardware entries")
        if not isinstance(networks, list) or not all(isinstance(item, str) for item in networks):
            raise BlisManifestError("BLIS support manifest has invalid network entries")
        return cls(
            models={name: entry for name, entry in models.items() if isinstance(name, str) and isinstance(entry, dict)},
            skipped=tuple(entry for entry in skipped if isinstance(entry, dict)),
            hardware=frozenset(hardware),
            networks=frozenset(networks),
        )

    def resolve_model(self, model_path: str) -> ModelResolution:
        direct = self.models.get(model_path)
        if direct is not None:
            return self._resolution(model_path, direct)

        for catalog_name, entry in self.models.items():
            if entry.get("aisimulate_model_id") == model_path:
                return self._resolution(catalog_name, entry)

        skipped = next(
            (item for item in self.skipped if item.get("aisimulate_model_id") == model_path),
            None,
        )
        if skipped is not None:
            reason = skipped.get("reason") or "unsupported model configuration"
            detail = skipped.get("detail")
            suffix = f": {detail}" if isinstance(detail, str) and detail else ""
            raise BlisUnsupportedRequest(f"BLIS cannot price model {model_path!r} ({reason}){suffix}")
        raise BlisUnsupportedRequest(
            f"BLIS does not support model {model_path!r}; choose a model listed in the BLIS support catalog"
        )

    @staticmethod
    def _resolution(catalog_name: str, entry: dict[str, Any]) -> ModelResolution:
        if entry.get("graph") is not True:
            raise BlisUnsupportedRequest(
                f"BLIS model {catalog_name!r} has no derived model graph and cannot be priced"
            )
        return ModelResolution(
            catalog_name=catalog_name,
            origin=str(entry.get("origin") or "unknown"),
            aisimulate_model_id=(
                entry.get("aisimulate_model_id")
                if isinstance(entry.get("aisimulate_model_id"), str)
                else None
            ),
            config_relation=(
                entry.get("config_relation") if isinstance(entry.get("config_relation"), str) else None
            ),
            equivalent_catalog_name=(
                entry.get("equivalent_catalog_name")
                if isinstance(entry.get("equivalent_catalog_name"), str)
                else None
            ),
        )

    def resolve_hardware(self, system: str) -> str:
        hardware = next((candidate for candidate in _hardware_candidates(system) if candidate in self.hardware), None)
        if hardware is None:
            available = ", ".join(sorted(self.hardware)) or "none"
            raise BlisUnsupportedRequest(
                f"BLIS cannot price ConfigIQ system {system!r}; available BLIS hardware keys: {available}"
            )
        return hardware

    def supports_hardware(self, system: str) -> bool:
        """Whether this system ID resolves to a catalog chip, without raising."""
        return any(candidate in self.hardware for candidate in _hardware_candidates(system))

    def resolve_network(self, network: str) -> str:
        if network not in self.networks:
            available = ", ".join(sorted(self.networks)) or "none"
            raise BlisUnsupportedRequest(
                f"BLIS cannot price network fabric {network!r}; available fabric keys: {available}"
            )
        return network


@lru_cache(maxsize=4)
def _load_manifest(path: str) -> SupportManifest:
    source = Path(path)
    if not source.is_file():
        raise BlisManifestError(f"BLIS support manifest is unavailable at {source}")
    try:
        raw = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BlisManifestError(f"could not read BLIS support manifest at {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BlisManifestError("BLIS support manifest must be a JSON object")
    return SupportManifest.from_dict(raw)


def load_support_manifest(path: str | None = None) -> SupportManifest:
    """Load the immutable image-built manifest, with an injectable test path."""
    return _load_manifest(path or os.environ.get("BLIS_SUPPORT", "/opt/blis/support.json"))

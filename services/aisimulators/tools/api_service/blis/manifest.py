"""Read the pinned BLIS catalog: models, hardware, and network fabrics.

The catalog is the only source of BLIS support facts. There is no
build-time support manifest and no translation to another simulator's model
IDs: an entry exists when blis-catalog ships it, and a request either resolves
to one or is refused.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class BlisManifestError(RuntimeError):
    """The BLIS catalog is missing or malformed."""


class BlisUnsupportedRequest(ValueError):
    """BLIS cannot represent a model, hardware, or fabric choice from the request."""


@dataclass(frozen=True)
class ModelResolution:
    """A request's model identity resolved to a catalog entry."""

    catalog_name: str
    hf_repo: str | None
    context_length: int | None = None
    origin: str = "blis-catalog"


#: ConfigIQ system IDs whose spelling differs from the catalog hardware key.
#: Only exact catalog keys are offered; PCIe variants are deliberately absent
#: because the catalog's chip files describe SXM parts.
_HARDWARE_ALIASES: dict[str, tuple[str, ...]] = {
    "gb200": ("gb200-nvl72",),
    "b200-sxm": ("b200",),
    "b300-sxm": ("b300",),
    "h100-sxm": ("h100",),
    "h200-sxm": ("h200",),
    "a100-sxm": ("a100-sxm", "a100-80"),
}


def _hardware_candidates(system: str) -> tuple[str, ...]:
    """Translate a ConfigIQ system ID into catalog hardware candidates."""
    normalized = system.strip().lower().replace("_", "-")
    explicit = _HARDWARE_ALIASES.get(normalized)
    if explicit is not None:
        return explicit
    candidates = [normalized]
    # `h100_sxm` -> `h100`; never strip `-pcie`, so a PCIe part cannot resolve
    # to the catalog's SXM specs.
    if normalized.endswith("-sxm"):
        candidates.append(normalized.rsplit("-", 1)[0])
    return tuple(dict.fromkeys(candidates))


def _config_value(config: dict[str, Any], *names: str) -> Any:
    """Read a model-config field, checking `text_config` for multimodal configs."""
    containers: list[dict[str, Any]] = [config]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        containers.append(text_config)
    for container in containers:
        for name in names:
            value = container.get(name)
            if value is not None:
                return value
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and int(value) > 0:
        return int(value)
    return None


def model_specs(model_id: str, config: dict[str, Any]) -> dict[str, Any]:
    """Metering metadata for one model, in the shape `/models?include=specs` uses."""
    architectures = config.get("architectures")
    architecture = (
        architectures[0]
        if isinstance(architectures, list) and architectures and isinstance(architectures[0], str)
        else config.get("model_type") if isinstance(config.get("model_type"), str) else None
    )
    return {
        "id": model_id,
        "num_experts": _positive_int(_config_value(config, "num_experts", "num_local_experts")),
        "num_experts_per_tok": _positive_int(_config_value(config, "num_experts_per_tok", "topk")),
        "context_length": _positive_int(_config_value(config, "max_position_embeddings", "context")),
        "num_attn_heads": _positive_int(_config_value(config, "num_attention_heads", "n")),
        "num_kv_heads": _positive_int(_config_value(config, "num_key_value_heads", "n_kv")),
        "architecture": architecture,
    }


@dataclass(frozen=True)
class CatalogModel:
    slug: str
    hf_repo: str | None
    config: dict[str, Any]

    @property
    def model_id(self) -> str:
        """The public identifier: the Hugging Face repo when the catalog states one."""
        return self.hf_repo or self.slug


@dataclass(frozen=True)
class BlisCatalog:
    models: dict[str, CatalogModel]
    hardware: frozenset[str]
    networks: frozenset[str]

    def model_ids(self) -> list[str]:
        """Public model identifiers a picker can offer, sorted."""
        return sorted({model.model_id for model in self.models.values()})

    def resolve_model(self, model_path: str) -> ModelResolution:
        model = self.models.get(model_path)
        if model is None:
            # A catalog slug is accepted as an input alias for its HF repo ID.
            model = next(
                (candidate for candidate in self.models.values() if candidate.model_id == model_path),
                None,
            )
        if model is None:
            available = ", ".join(self.model_ids()) or "none"
            raise BlisUnsupportedRequest(
                f"BLIS does not support model {model_path!r}; choose a model listed in the BLIS catalog: {available}"
            )
        return ModelResolution(
            catalog_name=model.slug,
            hf_repo=model.hf_repo,
            context_length=_positive_int(
                _config_value(model.config, "max_position_embeddings", "context")
            ),
            origin="blis-catalog",
        )

    def specs_for(self, model_path: str) -> dict[str, Any]:
        model = self.models.get(model_path)
        if model is None:
            model = next(
                (candidate for candidate in self.models.values() if candidate.model_id == model_path),
                None,
            )
        if model is None:
            raise BlisUnsupportedRequest(f"BLIS does not support model {model_path!r}")
        return model_specs(model.model_id, model.config)

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


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise BlisManifestError(f"could not read BLIS catalog file {path}: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise BlisManifestError(f"BLIS catalog file {path} must contain a mapping")
    return value


def _read_model(directory: Path) -> CatalogModel:
    metadata = _read_yaml(directory / "model.yaml") if (directory / "model.yaml").is_file() else {}
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    repo = source.get("repo") if isinstance(source.get("repo"), str) and source.get("repo").strip() else None
    config: dict[str, Any] = {}
    config_path = directory / "config.json"
    if config_path.is_file():
        try:
            raw_config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise BlisManifestError(f"could not read BLIS model config {config_path}: {exc}") from exc
        if isinstance(raw_config, dict):
            config = raw_config
    slug = directory.name
    return CatalogModel(slug=slug, hf_repo=repo, config=config)


def _stems(directory: Path) -> frozenset[str]:
    if not directory.is_dir():
        return frozenset()
    return frozenset(
        path.stem for path in directory.iterdir() if path.is_file() and path.suffix in {".yaml", ".yml"}
    )


@lru_cache(maxsize=4)
def _load_catalog(path: str) -> BlisCatalog:
    root = Path(path)
    models_directory = root / "models"
    if not models_directory.is_dir():
        raise BlisManifestError(f"BLIS catalog is unavailable at {root}")
    models: dict[str, CatalogModel] = {}
    for directory in sorted(models_directory.iterdir()):
        if not directory.is_dir():
            continue
        # A model is priceable only when blis-catalog ships its derived graph.
        if not (directory / "graph.yaml").is_file() and not (directory / "graph.yml").is_file():
            continue
        models[directory.name] = _read_model(directory)
    if not models:
        raise BlisManifestError(f"BLIS catalog at {root} contains no models with derived graphs")
    return BlisCatalog(
        models=models,
        hardware=_stems(root / "hardware"),
        networks=_stems(root / "networks"),
    )


def load_catalog(path: str | None = None) -> BlisCatalog:
    """Load the pinned catalog, with an injectable test path."""
    return _load_catalog(path or os.environ.get("BLIS_CATALOG", "/opt/blis/catalog"))

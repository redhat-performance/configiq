"""Read the pinned BLIS catalog: models, hardware, and network fabrics.

The catalog is the only source of BLIS support facts. There is no
build-time support manifest and no translation to another simulator's model
IDs: an entry exists when blis-catalog ships it, and a request either resolves
to one or is refused.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
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
    #: Smallest engine block size every attention kernel of this model accepts,
    #: from the graph's attention-cache sizes. `BlisCatalog.block_size_for`
    #: widens it to the part's sparse-kernel rule.
    block_size: int | None = None
    #: The graph's `global.architecture`, which selects the sparse-kernel rule.
    architecture: str | None = None
    #: Whether the graph's attention is the DSA sparse-MLA family.
    dsa: bool = False
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
    block_size: int | None = None
    architecture: str | None = None
    dsa: bool = False

    @property
    def model_id(self) -> str:
        """The public identifier: the Hugging Face repo when the catalog states one."""
        return self.hf_repo or self.slug


@dataclass(frozen=True)
class BlisCatalog:
    models: dict[str, CatalogModel]
    hardware: frozenset[str]
    networks: frozenset[str]
    #: Major CUDA compute capability each hardware key states, when it states one.
    compute_capabilities: dict[str, int | None] = field(default_factory=dict)

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
            block_size=model.block_size,
            architecture=model.architecture,
            dsa=model.dsa,
            origin="blis-catalog",
        )

    def block_size_for(self, model: ModelResolution, hardware: str) -> int | None:
        """The engine block size a model needs on one catalog hardware part.

        The graph's attention-cache block sizes are a lower bound. BLIS's sparse
        backends also require their own kernel block, which depends on the model
        family and, for DeepSeek-V4.1, the part's compute capability. The
        emitted size must be a multiple of both.
        """
        required = _kernel_block_size(
            model.architecture,
            model.dsa,
            self.compute_capabilities.get(hardware),
        )
        if required is None:
            return model.block_size
        if model.block_size is None:
            return required
        return math.lcm(required, model.block_size)

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


def _attention_block_size(graph: dict[str, Any]) -> int | None:
    """The smallest engine block every attention kernel of the model accepts.

    A derived model graph states a kernel `block_size` on each attention cache
    that has one. A scenario's `engine.block_size` must be a multiple of every
    stated value (DeepSeek-V4's sparse attention and indexer, for example, run
    256-token kernel blocks), so the least common multiple is the smallest block
    that fits every cache. Models whose graph states none have no constraint.

    The walk covers every layer kind the graph defines, including layers BLIS's
    target-stack expansion does not execute (MTP and draft stacks). Overstating
    the multiple is safe: BLIS checks only that the stated size is a multiple of
    its sparse kernel block, while underestimating one would be refused at
    startup.
    """
    sizes: list[int] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("op") == "Attention":
                caches = value.get("caches")
                if isinstance(caches, list):
                    for cache in caches:
                        if isinstance(cache, dict):
                            size = _positive_int(cache.get("block_size"))
                            if size is not None:
                                sizes.append(size)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(graph)
    return math.lcm(*sizes) if sizes else None


def _has_dsa_sparse_mla(graph: dict[str, Any]) -> bool:
    """Whether the graph's attention is the DSA sparse-MLA family.

    BLIS identifies it by a 576-wide sparse-MLA head on a plain attention node
    (no role), which its indexer runs on 64-token kernel blocks.
    """
    found = False

    def walk(value: Any) -> None:
        nonlocal found
        if found:
            return
        if isinstance(value, dict):
            if (
                value.get("op") == "Attention"
                and value.get("kind") == "sparse_mla"
                and not value.get("role")
                and _positive_int(value.get("d_h")) == 576
            ):
                found = True
                return
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(graph)
    return found


def _kernel_block_size(
    architecture: str | None,
    dsa: bool,
    compute_capability: int | None,
) -> int | None:
    """The block multiple BLIS's sparse kernels require beyond the graph caches.

    BLIS's kernel refuses a DeepSeek-V4-family block that is not a multiple of
    its sparse backend's kernel block: V4 runs 256-token blocks on every part,
    while V4.1 runs 64 on SM90 and 128 elsewhere. A DSA sparse-MLA indexer runs
    64-token blocks. Those are kernel facts, not graph cache sizes, so they are
    applied alongside the graph-derived cache LCM. Unknown compute capability is
    treated as non-SM90, where 128 is also a valid multiple of the SM90 rule.

    Mirrors `v4BlockSize` and `dsaBlockSize` in blis-substrate's
    `kernel/sparse_mla.go` at the pinned release; change both together.
    """
    if architecture == "DeepseekV4ForCausalLM":
        return 256
    if architecture == "DeepseekV41ForCausalLM":
        return 64 if compute_capability == 9 else 128
    if dsa:
        return 64
    return None


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
    graph_path = directory / "graph.yaml"
    if not graph_path.is_file():
        graph_path = directory / "graph.yml"
    graph = _read_yaml(graph_path) if graph_path.is_file() else {}
    global_section = graph.get("global") if isinstance(graph.get("global"), dict) else {}
    architecture = (
        global_section.get("architecture")
        if isinstance(global_section.get("architecture"), str)
        else None
    )
    slug = directory.name
    return CatalogModel(
        slug=slug,
        hf_repo=repo,
        config=config,
        block_size=_attention_block_size(graph),
        architecture=architecture,
        dsa=_has_dsa_sparse_mla(graph),
    )


def _stems(directory: Path) -> frozenset[str]:
    if not directory.is_dir():
        return frozenset()
    return frozenset(
        path.stem for path in directory.iterdir() if path.is_file() and path.suffix in {".yaml", ".yml"}
    )


def _read_compute_capabilities(directory: Path) -> dict[str, int | None]:
    """The major CUDA compute capability each catalog chip states."""
    capabilities: dict[str, int | None] = {}
    if not directory.is_dir():
        return capabilities
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix not in {".yaml", ".yml"}:
            continue
        raw = _read_yaml(path).get("ComputeCapability")
        major: int | None = None
        if isinstance(raw, str):
            try:
                major = int(raw.split(".", 1)[0].strip())
            except ValueError:
                major = None
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            major = int(raw)
        capabilities[path.stem] = major
    return capabilities


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
        compute_capabilities=_read_compute_capabilities(root / "hardware"),
    )


def load_catalog(path: str | None = None) -> BlisCatalog:
    """Load the pinned catalog, with an injectable test path."""
    return _load_catalog(path or os.environ.get("BLIS_CATALOG", "/opt/blis/catalog"))

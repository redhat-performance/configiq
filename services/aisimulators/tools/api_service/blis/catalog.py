"""Read the catalog and hardware facts the pinned BLIS build embeds.

The catalog is the only source of BLIS support facts. There is no build-time
support manifest, no copied data tree, and no translation to another simulator's
model IDs: an entry exists when the build's own `blis catalog list --json`
reports it, and a request either resolves to one or is refused.

The build embeds the catalog, so the report is read from the binary rather than
from a directory beside it: the facts the API lists are the facts the same
binary prices, and a deployment cannot pair a binary with a stale copy.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .errors import BlisCatalogError, BlisUnsupportedRequest
from .runtime import resolve_runtime

logger = logging.getLogger(__name__)

#: How long `blis catalog list --json` may take. The catalog is embedded, so this
#: only bounds a wedged binary.
CATALOG_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class ModelResolution:
    """A request's model identity resolved to a catalog entry."""

    catalog_name: str
    hf_repo: str | None
    context_length: int | None = None
    #: The engine block size this model's kernels require on the resolved part,
    #: stamped once the part is known (`BlisCatalog.block_size_for`). None means
    #: no sparse kernel rule applies, so the caller's default stands.
    block_size: int | None = None
    #: The graph's `architecture`, which selects the sparse-kernel rule.
    architecture: str | None = None
    #: Whether the graph's attention is the DSA sparse-MLA family.
    dsa: bool = False
    origin: str = "blis-catalog"


#: ConfigIQ system IDs whose spelling differs from the catalog chip key.
#: Only exact catalog keys are offered; PCIe variants are deliberately absent
#: because the catalog's chip entries describe SXM parts.
_HARDWARE_ALIASES: dict[str, tuple[str, ...]] = {
    "gb200": ("gb200-nvl72",),
    "b200-sxm": ("b200",),
    "b300-sxm": ("b300",),
    "h100-sxm": ("h100",),
    "h200-sxm": ("h200",),
    "a100-sxm": ("a100-sxm", "a100-80"),
    # The catalog spells the part without the "PRO" separator.
    "rtx-pro-6000-server": ("rtx6000pro-server",),
}


def _hardware_candidates(system: str) -> tuple[str, ...]:
    """Translate a ConfigIQ system ID into catalog chip candidates."""
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


def _string(source: dict[str, Any], key: str) -> str | None:
    value = source.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _mapping(source: dict[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key)
    return value if isinstance(value, dict) else {}


def _positive_int(source: dict[str, Any], key: str) -> int | None:
    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if int(value) > 0 else None


def _positive_number(source: dict[str, Any], key: str) -> float | None:
    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if float(value) > 0 else None


@dataclass(frozen=True)
class CatalogModel:
    """One model the build can price, as the build reports it."""

    name: str
    hf_repo: str | None = None
    architecture: str | None = None
    #: The longest context the engine admits when the deployment does not raise
    #: the limit. Only some graphs state one; absent means the catalog does not
    #: report a window, not that the model has none.
    context_window: int | None = None
    experts: int | None = None
    experts_top_k: int | None = None
    attention_kind: str | None = None
    attention_heads: int | None = None
    attention_kv_heads: int | None = None
    attention_head_dim: int | None = None
    #: Names of the draft models this entry's graph lists, if any. A draft model
    #: is an auxiliary graph, not a model a request can serve.
    drafters: tuple[str, ...] = ()

    @property
    def dsa(self) -> bool:
        """Whether the graph's attention is the DSA sparse-MLA family.

        The build reports only the KV-holding attention layer's shape, so a
        sparse-MLA head that is 576 wide is the DSA indexer's signature: a
        512-wide sparse head is a different backend and carries no such rule.
        """
        return self.attention_kind == "sparse_mla" and self.attention_head_dim == 576

    @property
    def model_id(self) -> str:
        """The public identifier: the Hugging Face repo when the catalog states one."""
        return self.hf_repo or self.name

    def specs(self) -> dict[str, Any]:
        """Metering metadata for this model, in the `/models?include=specs` shape."""
        return {
            "id": self.model_id,
            "num_experts": self.experts,
            "num_experts_per_tok": self.experts_top_k,
            # The graph's stated window only. The catalog carries no model config
            # file, so a model whose graph leaves the window open reports null
            # rather than a value derived from something the build does not read.
            "context_length": self.context_window,
            "num_attn_heads": self.attention_heads,
            "num_kv_heads": self.attention_kv_heads,
            "architecture": self.architecture,
        }


@dataclass(frozen=True)
class CatalogChip:
    """One accelerator the build can price, with the facts it states."""

    name: str
    memory_gib: float | None = None
    memory_bandwidth_tbs: float | None = None
    bf16_tflops: float | None = None
    fp8_tflops: float | None = None
    nvfp4_tflops: float | None = None
    intra_node_bw_gbps: float | None = None
    sm_count: int | None = None
    compute_capability: str | None = None
    #: The board power the vendor specifies for the part as shipped, watts per
    #: GPU. None when the entry states none: the catalog leaves a part whose
    #: board power no source states unstated rather than borrowing a sibling
    #: SKU's figure, so a client must show it as unavailable.
    tdp_watts: float | None = None
    #: Packaging, stated only by chips whose entry fixes it.
    gpus_per_node: int | None = None
    gpus_per_rack: int | None = None


@dataclass(frozen=True)
class BlisCatalog:
    """Everything the build reports about what it can price."""

    models: dict[str, CatalogModel]
    chips: dict[str, CatalogChip]
    fabrics: tuple[str, ...]
    #: Catalog names referenced as a draft model by another entry's graph.
    drafters: frozenset[str] = frozenset()

    def model_ids(self) -> list[str]:
        """Public model identifiers a picker can offer, sorted."""
        return sorted({model.model_id for model in self.models.values()})

    def _find_model(self, model_path: str) -> CatalogModel | None:
        model = self.models.get(model_path)
        if model is not None:
            return model
        # A catalog name is accepted as an input alias for its HF repo ID.
        candidates = [
            candidate for candidate in self.models.values() if candidate.model_id == model_path
        ]
        if not candidates:
            return None
        # A draft model can share its served model's repo, so an ID that names
        # both resolves to the served entry: the draft graph is not a model a
        # request can serve.
        served = [candidate for candidate in candidates if candidate.name not in self.drafters]
        return (served or candidates)[0]

    def resolve_model(self, model_path: str) -> ModelResolution:
        model = self._find_model(model_path)
        if model is None:
            available = ", ".join(self.model_ids()) or "none"
            raise BlisUnsupportedRequest(
                f"BLIS does not support model {model_path!r}; choose a model listed in the BLIS catalog: {available}"
            )
        return ModelResolution(
            catalog_name=model.name,
            hf_repo=model.hf_repo,
            context_length=model.context_window,
            architecture=model.architecture,
            dsa=model.dsa,
            origin="blis-catalog",
        )

    def block_size_for(self, model: ModelResolution, hardware: str) -> int | None:
        """The smallest engine block size this model's sparse kernels accept on a part.

        BLIS refuses a sparse-attention block that is not a multiple of the
        backend's kernel block: DeepSeek-V4 needs 256 tokens, DeepSeek-V4.1
        needs 64 on SM90 and 128 elsewhere, and a DSA sparse-MLA indexer needs
        64. Those are kernel facts, not catalog values, so they are applied
        here rather than read from the report. A model with no rule keeps the
        caller's default, which this returns None for.
        """
        return _kernel_block_size(
            model.architecture,
            model.dsa,
            _compute_capability_major(self.chips.get(hardware)),
        )

    def specs_for(self, model_path: str) -> dict[str, Any]:
        model = self._find_model(model_path)
        if model is None:
            raise BlisUnsupportedRequest(f"BLIS does not support model {model_path!r}")
        return model.specs()

    def resolve_hardware(self, system: str) -> str:
        hardware = next(
            (candidate for candidate in _hardware_candidates(system) if candidate in self.chips),
            None,
        )
        if hardware is None:
            available = ", ".join(sorted(self.chips)) or "none"
            raise BlisUnsupportedRequest(
                f"BLIS cannot price ConfigIQ system {system!r}; available BLIS hardware keys: {available}"
            )
        return hardware

    def supports_hardware(self, system: str) -> bool:
        """Whether this system ID resolves to a catalog chip, without raising."""
        return any(candidate in self.chips for candidate in _hardware_candidates(system))

    def chip(self, system: str) -> CatalogChip | None:
        """The catalog chip a system ID resolves to, or None when it resolves to none."""
        hardware = next(
            (candidate for candidate in _hardware_candidates(system) if candidate in self.chips),
            None,
        )
        return self.chips[hardware] if hardware is not None else None

    def resolve_network(self, network: str) -> str:
        if network not in self.fabrics:
            available = ", ".join(self.fabrics) or "none"
            raise BlisUnsupportedRequest(
                f"BLIS cannot price network fabric {network!r}; available fabric keys: {available}"
            )
        return network


def _compute_capability_major(chip: CatalogChip | None) -> int | None:
    """The major CUDA compute capability a chip states ("9.0" -> 9), or None."""
    if chip is None or not chip.compute_capability:
        return None
    try:
        return int(chip.compute_capability.split(".", 1)[0].strip())
    except ValueError:
        return None


def _kernel_block_size(
    architecture: str | None,
    dsa: bool,
    compute_capability: int | None,
) -> int | None:
    """The block multiple BLIS's sparse kernels require beyond the catalog.

    BLIS's kernel refuses a DeepSeek-V4-family block that is not a multiple of
    its sparse backend's kernel block: V4 runs 256-token blocks on every part,
    while V4.1 runs 64 on SM90 and 128 elsewhere. A DSA sparse-MLA indexer runs
    64-token blocks. Unknown compute capability is treated as non-SM90, where
    128 is also a valid multiple of the SM90 rule, so it cannot produce a size
    the engine refuses.

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


def _entries_of(source: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """The mapping entries of an optional list field, or none when it is absent."""
    value = source.get(key)
    return [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []


def _entries(document: dict[str, Any], key: str) -> list[dict[str, Any]]:
    entries = document.get(key)
    if not isinstance(entries, list):
        raise BlisCatalogError(f"BLIS catalog report has no {key} list")
    return [entry for entry in entries if isinstance(entry, dict)]


def parse_model(entry: dict[str, Any]) -> CatalogModel | None:
    """Parse one `models` entry, or None when it states no name."""
    name = _string(entry, "name")
    if not name:
        return None
    source = _mapping(entry, "source")
    attention = _mapping(entry, "attention")
    experts = _mapping(entry, "experts")
    return CatalogModel(
        name=name,
        hf_repo=_string(source, "repo"),
        architecture=_string(entry, "architecture"),
        context_window=_positive_int(entry, "context_window"),
        experts=_positive_int(experts, "count"),
        experts_top_k=_positive_int(experts, "top_k"),
        attention_kind=_string(attention, "kind"),
        attention_heads=_positive_int(attention, "heads"),
        attention_kv_heads=_positive_int(attention, "kv_heads"),
        attention_head_dim=_positive_int(attention, "head_dim"),
        drafters=tuple(
            name
            for drafter in _entries_of(entry, "drafters")
            if (name := _string(drafter, "name"))
        ),
    )


def parse_chip(entry: dict[str, Any]) -> CatalogChip | None:
    """Parse one `chips` entry, or None when it states no name."""
    name = _string(entry, "name")
    if not name:
        return None
    return CatalogChip(
        name=name,
        memory_gib=_positive_number(entry, "memory_gib"),
        memory_bandwidth_tbs=_positive_number(entry, "memory_bandwidth_tbs"),
        bf16_tflops=_positive_number(entry, "bf16_tflops"),
        fp8_tflops=_positive_number(entry, "fp8_tflops"),
        nvfp4_tflops=_positive_number(entry, "nvfp4_tflops"),
        intra_node_bw_gbps=_positive_number(entry, "intra_node_bw_gbps"),
        sm_count=_positive_int(entry, "sm_count"),
        compute_capability=_string(entry, "compute_capability"),
        tdp_watts=_positive_number(entry, "tdp_watts"),
        gpus_per_node=_positive_int(entry, "gpus_per_node"),
        gpus_per_rack=_positive_int(entry, "gpus_per_rack"),
    )


def parse_catalog(document: Any) -> BlisCatalog:
    """Parse a `blis catalog list --json` report into the catalog it describes."""
    if not isinstance(document, dict):
        raise BlisCatalogError("BLIS catalog report must be a JSON object")
    models: dict[str, CatalogModel] = {}
    for entry in _entries(document, "models"):
        model = parse_model(entry)
        if model is not None:
            models[model.name] = model
    if not models:
        raise BlisCatalogError("BLIS catalog reports no models it can price")
    chips: dict[str, CatalogChip] = {}
    for entry in _entries(document, "chips"):
        chip = parse_chip(entry)
        if chip is not None:
            chips[chip.name] = chip
    if not chips:
        raise BlisCatalogError("BLIS catalog reports no accelerator entries")
    fabrics = tuple(sorted(
        name
        for entry in _entries(document, "fabrics")
        if (name := _string(entry, "name"))
    ))
    return BlisCatalog(
        models=models,
        chips=chips,
        fabrics=fabrics,
        drafters=frozenset(
            name for model in models.values() for name in model.drafters
        ),
    )


def read_catalog(binary: str, *, timeout_seconds: float = CATALOG_TIMEOUT_SECONDS) -> BlisCatalog:
    """Ask the pinned build for the catalog it embeds."""
    try:
        completed = subprocess.run(
            [binary, "catalog", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BlisCatalogError(f"could not ask BLIS for its catalog: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-600:]
        raise BlisCatalogError(
            f"BLIS did not report its catalog (exit {completed.returncode}): "
            f"{detail or 'no diagnostics'}"
        )
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise BlisCatalogError("BLIS catalog output was not JSON") from exc
    return parse_catalog(document)


@lru_cache(maxsize=4)
def _read_catalog_cached(binary: str) -> BlisCatalog:
    return read_catalog(binary)


def load_catalog(binary: str | None = None) -> BlisCatalog:
    """The pinned build's catalog, cached per binary path.

    The build cannot change while the service runs, so the report is read once.
    An injectable path keeps unit tests off the deployed image.
    """
    return _read_catalog_cached(binary or resolve_runtime().binary)

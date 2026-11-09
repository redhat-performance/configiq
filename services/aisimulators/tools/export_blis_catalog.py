#!/usr/bin/env python3
"""Augment the BLIS catalog with AISimulate's packaged HF configs and emit the support manifest.

The image ships BLIS's own catalog (``blis-catalog``) and the checked-out AISimulate
wheel. This tool reconciles the two:

* an upstream catalog entry ALWAYS wins — AISimulate can only add names the catalog
  does not already define, so a reviewed BLIS graph is never silently replaced;
* an added model is priced only if BLIS can derive a ``graph.yaml`` for it. The
  derivation is the same ``scripts/derive_graph.py`` the catalog repo runs in its own
  CI, so "addable" means exactly "BLIS can price it";
* every model the tool could not add is recorded with a reason in the manifest rather
  than dropped, so the API can explain why a model is unavailable instead of failing
  with a generic error;
* the manifest (``support.json``) is the API contract: which models BLIS can price,
  where each came from, how an AISimulate config relates to the upstream entry of the
  same name, and which hardware/networks are bundled.

This is catalog *augmentation*, not a sync. BLIS reads only each model's
``graph.yaml`` (derived from the vendor ``config.json``); it never reads AISimulate's
configs itself, so this tool is the only bridge between an AISimulate model ID and a
BLIS-priced one.

Usage::

    export_blis_catalog.py CATALOG_ROOT --manifest /opt/blis/support.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from hashlib import sha256
from importlib import resources
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

MANIFEST_SCHEMA_VERSION = 1
GENERATOR = "services/aisimulators/tools/export_blis_catalog.py"

#: Wall-clock bound on one graph derivation. Deriving a single model measures at ~0.1s on a
#: laptop (cold process and PyYAML import included), so this is ~100x headroom: it exists only
#: to turn a wedged deriver into a recorded skip instead of a hung image build. It is
#: deliberately loose, because a timeout here drops a model from the manifest — the expensive
#: mistake is a model we could have priced, not a build that takes ten seconds longer.
DERIVE_TIMEOUT_SECONDS = 10.0

#: Skip reasons, stable strings the API may branch on.
SKIP_NOT_HF_CONFIG = "not_hf_config"
SKIP_GRAPH_DERIVATION_FAILED = "graph_derivation_failed"
SKIP_GRAPH_DERIVER_MISSING = "graph_deriver_missing"
SKIP_CATALOG_NAME_COLLISION = "catalog_name_collision"

#: How an AISimulate config relates to the upstream catalog entry of the same name.
RELATION_IDENTICAL = "identical"
RELATION_DIFFERENT_QUANTIZATION = "different_quantization"
RELATION_DIFFERENT = "different"


@dataclass(frozen=True)
class SkippedModel:
    aisimulate_model_id: str
    catalog_name: str
    reason: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "aisimulate_model_id": self.aisimulate_model_id,
            "catalog_name": self.catalog_name,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass
class ExportResult:
    added: int = 0
    shared: int = 0
    skipped: list[SkippedModel] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


def model_id_from_filename(path: Path) -> str:
    stem = path.name.removesuffix("_config.json")
    return stem.replace("--", "/", 1)


def safe_name(model_id: str) -> str:
    name = model_id.split("/", 1)[-1]
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")


def is_hf_config(config: object) -> bool:
    if not isinstance(config, dict):
        return False

    def has_layer_evidence(candidate: dict) -> bool:
        blocks = candidate.get("layers_block_type")
        return (
            "num_hidden_layers" in candidate
            or "hidden_size" in candidate
            or (isinstance(blocks, list) and bool(blocks))
        )

    if has_layer_evidence(config):
        return True
    text_config = config.get("text_config")
    return isinstance(text_config, dict) and has_layer_evidence(text_config)


def quantization(config: dict[str, Any]) -> str | None:
    """Return the weight quantization method the config declares, if any.

    Multimodal configs keep it under ``text_config`` — the text model is the only part
    BLIS prices.
    """
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        text_config = config.get("text_config")
        quant = text_config.get("quantization_config") if isinstance(text_config, dict) else None
    if not isinstance(quant, dict):
        return None
    method = quant.get("quant_method")
    fmt = quant.get("fmt")
    if method and fmt:
        return f"{method}:{fmt}"
    return method or quant.get("quantization_status") or "unknown"


def config_digest(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()


def config_differences(left: object, right: object, path: str = "") -> list[str]:
    """List the leaf paths where two configs differ (``missing`` marks an absent key)."""
    differences: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left:
                differences.append(f"{child}: missing in AISimulate")
            elif key not in right:
                differences.append(f"{child}: missing upstream")
            else:
                differences.extend(config_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append(f"{path}[len]: {len(left)} AISimulate vs {len(right)} upstream")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(config_differences(left_item, right_item, f"{path}[{index}]"))
        return differences
    if left != right:
        differences.append(f"{path}: {json.dumps(left, sort_keys=True)} vs {json.dumps(right, sort_keys=True)}")
    return differences


def config_relation(aisimulate_config: dict[str, Any], upstream_config: dict[str, Any]) -> str:
    if aisimulate_config == upstream_config:
        return RELATION_IDENTICAL
    if quantization(aisimulate_config) != quantization(upstream_config):
        return RELATION_DIFFERENT_QUANTIZATION
    return RELATION_DIFFERENT


def _model_yaml(name: str, model_id: str, package_version: str, build: str | None = None) -> str:
    # `revision` names what the config was taken from: the fork's release build when
    # pip recorded it, else just the installed version.
    revision = build or f"aisimulate-{package_version}"
    return (
        f"name: {json.dumps(name)}\n"
        "source:\n"
        "  provider: aisimulate\n"
        f"  repo: {json.dumps(model_id)}\n"
        f"  revision: {json.dumps(revision)}\n"
    )


def installed_wheel() -> dict[str, str | None]:
    """The exact fork-release wheel this image installed, when pip recorded one.

    ``pip`` writes ``direct_url.json`` beside a distribution installed from a URL —
    which is how the Containerfile installs this SDK. Its URL names the release build
    (``deploy-api-v0.13.0+dd7494b9``) the bundled model configs came from, so the
    manifest can say which build a derived catalog entry was derived from.
    """
    try:
        dist = distribution("aisimulate")
    except PackageNotFoundError:
        return {"version": None, "build": None, "url": None}
    url = build = None
    try:
        payload = json.loads(dist.read_text("direct_url.json") or "null")
    except (OSError, ValueError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("url"), str):
        url = payload["url"]
        segments = [unquote(part) for part in urlparse(url).path.split("/") if part]
        # .../releases/download/<tag>/aisimulate-<version>-<abi>.whl
        if len(segments) >= 2:
            build = segments[-2]
    return {"version": dist.version, "build": build, "url": url}


def _catalog_names(catalog_root: Path, subdir: str, suffix: str) -> list[str]:
    directory = catalog_root / subdir
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob(f"*{suffix}"))


def _derive_graph(catalog_root: Path, name: str, derive_script: Path) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [sys.executable, str(derive_script), str(catalog_root), "--model", name],
            capture_output=True,
            text=True,
            check=False,
            timeout=DERIVE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        # Recorded as a skip like any other derivation failure: the export's contract is that
        # nothing abort the build, and the manifest says which model was lost and why.
        return False, f"derive_graph.py timed out after {DERIVE_TIMEOUT_SECONDS:g}s"
    if completed.returncode == 0 and (catalog_root / "models" / name / "graph.yaml").is_file():
        return True, ""
    detail = (completed.stderr or completed.stdout or "").strip().splitlines()
    return False, detail[-1] if detail else f"derive_graph.py exited {completed.returncode}"


def export_catalog(
    catalog_root: Path,
    source_root: Path | None = None,
    package_version: str | None = None,
    derive_script: Path | None = None,
    manifest_path: Path | None = None,
) -> ExportResult:
    """Augment ``catalog_root`` with AISimulate configs and return the support manifest."""
    source_root = source_root or Path(str(resources.files("aisimulate_core") / "model_configs"))
    wheel = installed_wheel()
    # An explicit package_version is the caller's stated provenance, so it wins:
    # only a version derived from the installed wheel records that wheel's build.
    build = wheel["build"] if package_version is None else None
    package_version = package_version or wheel["version"] or version("aisimulate")
    derive_script = derive_script or catalog_root / "scripts" / "derive_graph.py"
    destination_root = catalog_root / "models"
    destination_root.mkdir(parents=True, exist_ok=True)

    # Upstream entries as they stand before this run: the only configs that may win a
    # name collision, and the reference set for equivalence/relation reporting.
    upstream: dict[str, dict[str, Any]] = {}
    for entry in sorted(p for p in destination_root.iterdir() if p.is_dir()):
        config_path = entry / "config.json"
        if config_path.is_file():
            upstream[entry.name] = json.loads(config_path.read_text())

    upstream_by_digest: dict[str, str] = {}
    for name, config in sorted(upstream.items()):
        upstream_by_digest.setdefault(config_digest(config), name)

    models: dict[str, dict[str, Any]] = {}
    for name, config in sorted(upstream.items()):
        models[name] = {
            "origin": "upstream",
            "graph": (destination_root / name / "graph.yaml").is_file(),
            "aisimulate_model_id": None,
            "config_relation": None,
            "equivalent_catalog_name": None,
            "quantization": {"aisimulate": None, "upstream": quantization(config)},
            "differences": [],
        }

    result = ExportResult()
    owners: dict[str, str] = {}

    for source in sorted(source_root.glob("*_config.json")):
        if source.name.endswith("_hf_quant_config.json"):
            continue
        model_id = model_id_from_filename(source)
        name = safe_name(model_id)
        source_config = json.loads(source.read_text())

        previous = owners.get(name)
        if previous is not None and previous != model_id:
            result.skipped.append(
                SkippedModel(
                    aisimulate_model_id=model_id,
                    catalog_name=name,
                    reason=SKIP_CATALOG_NAME_COLLISION,
                    detail=f"{name!r} already taken by {previous!r}",
                )
            )
            continue

        if not is_hf_config(source_config):
            result.skipped.append(
                SkippedModel(
                    aisimulate_model_id=model_id,
                    catalog_name=name,
                    reason=SKIP_NOT_HF_CONFIG,
                    detail="config has no layer or hidden-size evidence",
                )
            )
            continue

        upstream_config = upstream.get(name)
        if upstream_config is not None:
            # Upstream wins unconditionally; report the relationship and let the API
            # decide whether the AISimulate selection is interchangeable.
            entry = models[name]
            relation = config_relation(source_config, upstream_config)
            entry["origin"] = "upstream"
            entry["aisimulate_model_id"] = model_id
            entry["config_relation"] = relation
            entry["quantization"]["aisimulate"] = quantization(source_config)
            if relation != RELATION_IDENTICAL:
                entry["differences"] = config_differences(source_config, upstream_config)
            owners[name] = model_id
            result.shared += 1
            continue

        destination = destination_root / name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / "config.json")
        (destination / "model.yaml").write_text(_model_yaml(name, model_id, package_version, build))

        if not derive_script.is_file():
            shutil.rmtree(destination)
            result.skipped.append(
                SkippedModel(
                    aisimulate_model_id=model_id,
                    catalog_name=name,
                    reason=SKIP_GRAPH_DERIVER_MISSING,
                    detail=f"{derive_script} is not present",
                )
            )
            continue

        derived, detail = _derive_graph(catalog_root, name, derive_script)
        if not derived:
            shutil.rmtree(destination)
            result.skipped.append(
                SkippedModel(
                    aisimulate_model_id=model_id,
                    catalog_name=name,
                    reason=SKIP_GRAPH_DERIVATION_FAILED,
                    detail=detail,
                )
            )
            continue

        equivalent = upstream_by_digest.get(config_digest(source_config))
        owners[name] = model_id
        models[name] = {
            "origin": "aisimulate",
            "graph": True,
            "aisimulate_model_id": model_id,
            "config_relation": None,
            "equivalent_catalog_name": equivalent if equivalent != name else None,
            "quantization": {"aisimulate": quantization(source_config), "upstream": None},
            "differences": [],
        }
        result.added += 1

    if result.added == 0 and result.shared == 0:
        raise RuntimeError(f"no usable AISimulate model configs found in {source_root}")

    result.manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_by": GENERATOR,
        "aisimulate_version": package_version,
        "aisimulate_wheel": wheel,
        "models": dict(sorted(models.items())),
        "skipped": [skipped.as_dict() for skipped in result.skipped],
        "hardware": _catalog_names(catalog_root, "hardware", ".yaml"),
        "networks": _catalog_names(catalog_root, "networks", ".yaml"),
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(result.manifest, indent=2, sort_keys=False) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog_root", type=Path, help="blis-catalog clone root (modified in place)")
    parser.add_argument("--source", type=Path, default=None, help="AISimulate model_configs directory")
    parser.add_argument("--package-version", default=None, help="AISimulate version recorded in model.yaml")
    parser.add_argument("--derive-script", type=Path, default=None, help="path to blis-catalog scripts/derive_graph.py")
    parser.add_argument("--manifest", type=Path, default=None, help="write the support manifest here")
    args = parser.parse_args(argv)

    result = export_catalog(
        catalog_root=args.catalog_root,
        source_root=args.source,
        package_version=args.package_version,
        derive_script=args.derive_script,
        manifest_path=args.manifest,
    )
    print(
        f"BLIS catalog: {result.added} AISimulate model(s) added, "
        f"{result.shared} shared with upstream, {result.skipped_count} skipped"
    )
    for skipped in result.skipped:
        print(f"  skipped {skipped.aisimulate_model_id}: {skipped.reason} ({skipped.detail})", file=sys.stderr)
    if args.manifest is not None:
        print(f"support manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

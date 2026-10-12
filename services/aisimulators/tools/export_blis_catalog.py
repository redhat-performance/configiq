#!/usr/bin/env python3
"""Export AISimulate's packaged HF configs into a BLIS catalog overlay."""

from __future__ import annotations

import json
import re
import shutil
import sys
from importlib import resources
from importlib.metadata import version
from pathlib import Path

UPSTREAM_PREFERRED = {
    # BLIS 0.1.1 carries a curated FP8 GLM-5.3 config while AISimulate packages
    # the unquantized base config under the same short name. Preserve BLIS's
    # explicitly reviewed catalog entry rather than silently replacing it.
    "glm-5.3",
}


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
        return "num_hidden_layers" in candidate or "hidden_size" in candidate or (isinstance(blocks, list) and bool(blocks))

    if has_layer_evidence(config):
        return True
    text_config = config.get("text_config")
    return isinstance(text_config, dict) and has_layer_evidence(text_config)


def export_catalog(
    catalog_root: Path,
    source_root: Path | None = None,
    package_version: str | None = None,
) -> int:
    source_root = source_root or Path(str(resources.files("aisimulate_core") / "model_configs"))
    package_version = package_version or version("aisimulate")
    destination_root = catalog_root / "models"
    destination_root.mkdir(parents=True, exist_ok=True)
    exported = 0
    used_names: dict[str, str] = {}

    for source in sorted(source_root.glob("*_config.json")):
        if source.name.endswith("_hf_quant_config.json"):
            continue
        model_id = model_id_from_filename(source)
        name = safe_name(model_id)
        previous = used_names.get(name)
        if previous is not None and previous != model_id:
            raise RuntimeError(f"BLIS catalog name collision: {previous!r} and {model_id!r} -> {name!r}")
        used_names[name] = model_id
        destination = destination_root / name
        source_config = json.loads(source.read_text())
        if not is_hf_config(source_config):
            raise RuntimeError(f"AISimulate model config is not Hugging Face compatible: {source}")
        upstream_config = destination / "config.json"
        if upstream_config.exists():
            if json.loads(upstream_config.read_text()) != source_config and name not in UPSTREAM_PREFERRED:
                raise RuntimeError(f"conflicting upstream BLIS catalog entry for {model_id!r} at {destination}")
            # Preserve the upstream BLIS config and provenance for shared names.
            exported += 1
            continue
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / "config.json")
        (destination / "model.yaml").write_text(
            f"name: {json.dumps(name)}\nsource:\n  provider: aisimulate\n"
            f"  repo: {json.dumps(model_id)}\n  revision: {json.dumps(f'aisimulate-{package_version}')}\n"
        )
        exported += 1

    if exported == 0:
        raise RuntimeError(f"no AISimulate model configs found in {source_root}")
    return exported


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <blis-catalog-root>")
    count = export_catalog(Path(sys.argv[1]))
    print(f"exported {count} AISimulate model configs into {sys.argv[1]}/models")

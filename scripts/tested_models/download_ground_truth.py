#!/usr/bin/env python3
"""Download a pinned public ground-truth dataset into the local pipeline layout."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd
from dataset_common import file_sha256, validate_public_frame
from huggingface_hub import snapshot_download

PAIR_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="redhat-performance/configiq-performance-data")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tested-models"))
    return parser.parse_args()


def _commit_dataset(
    staging: Path,
    datasets_dir: Path,
    staged_manifest: Path,
    manifest_path: Path,
    replace_file=os.replace,
) -> None:
    """Replace dataset files and manifest together, restoring both on failure."""
    datasets_backup = datasets_dir.with_name(f".{datasets_dir.name}.backup")
    manifest_backup = manifest_path.with_name(f".{manifest_path.name}.backup")
    if datasets_backup.exists() or manifest_backup.exists():
        raise RuntimeError("previous dataset backup exists; inspect it before retrying")

    datasets_backed_up = False
    manifest_backed_up = False
    datasets_installed = False
    manifest_installed = False
    try:
        if datasets_dir.exists():
            datasets_dir.rename(datasets_backup)
            datasets_backed_up = True
        if manifest_path.exists():
            replace_file(manifest_path, manifest_backup)
            manifest_backed_up = True
        staging.rename(datasets_dir)
        datasets_installed = True
        replace_file(staged_manifest, manifest_path)
        manifest_installed = True
    except Exception:
        if datasets_installed and datasets_dir.exists():
            shutil.rmtree(datasets_dir)
        if datasets_backed_up:
            datasets_backup.rename(datasets_dir)
        if manifest_installed and manifest_path.exists():
            manifest_path.unlink()
        if manifest_backed_up:
            replace_file(manifest_backup, manifest_path)
        raise
    else:
        if datasets_backup.exists():
            shutil.rmtree(datasets_backup)
        if manifest_backup.exists():
            manifest_backup.unlink()
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if staged_manifest.exists():
            staged_manifest.unlink()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("--revision must be a full 40-character lowercase Hugging Face commit SHA")
    snapshot = Path(snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=["manifest.json", "pairs/*/ground_truth.parquet"],
        token=False,
    ))
    source_manifest = json.loads((snapshot / "manifest.json").read_text())
    if source_manifest.get("schema_version") != 1:
        raise ValueError("unsupported ground-truth manifest schema")
    source_pairs = source_manifest.get("pairs")
    if not isinstance(source_pairs, list) or not source_pairs:
        raise ValueError("ground-truth manifest must contain at least one pair")
    seen_ids: set[str] = set()
    planned_pairs = []
    for pair in source_pairs:
        if not isinstance(pair, dict):
            raise TypeError("ground-truth manifest pairs must be objects")
        pair_id = pair.get("id")
        if not isinstance(pair_id, str) or not PAIR_ID_PATTERN.fullmatch(pair_id):
            raise ValueError(f"invalid filesystem pair id: {pair_id!r}")
        if pair_id in seen_ids:
            raise ValueError(f"duplicate ground-truth pair id: {pair_id}")
        seen_ids.add(pair_id)
        pair_path = Path(pair["path"])
        expected_path = Path("pairs") / pair_id / "ground_truth.parquet"
        if pair_path != expected_path:
            raise ValueError(f"invalid dataset path: {pair_path}")
        source_path = snapshot / pair_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        expected_sha = pair.get("file_sha256")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ValueError(f"pair {pair_id} is missing a valid file_sha256")
        if file_sha256(source_path) != expected_sha:
            raise ValueError(f"pair {pair_id} parquet checksum mismatch")
        frame = pd.read_parquet(source_path)
        validate_public_frame(frame)
        if pair.get("records") != len(frame) or pair.get("columns") != list(frame.columns):
            raise ValueError(f"pair {pair_id} parquet metadata does not match its manifest")
        planned_pairs.append((pair, source_path))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = args.output_dir / "datasets"
    staging = args.output_dir / "datasets.staging"
    manifest_path = args.output_dir / "dataset-manifest.json"
    staged_manifest = args.output_dir / ".dataset-manifest.json.staging"
    if staging.exists():
        shutil.rmtree(staging)
    if staged_manifest.exists():
        staged_manifest.unlink()
    staging.mkdir(parents=True)
    pairs = []
    for pair, source_path in planned_pairs:
        pair_id = pair["id"]
        destination = staging / f"{pair_id}.parquet"
        shutil.copy2(source_path, destination)
        pairs.append({**pair, "path": str(destination)})
    for pair in pairs:
        pair["path"] = str(datasets_dir / f"{pair['id']}.parquet")
    manifest = {"schema_version": 1, "source": args.repo_id, "revision": args.revision, "pairs": pairs}
    staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _commit_dataset(staging, datasets_dir, staged_manifest, manifest_path)
    print(f"downloaded {len(pairs)} ground-truth pairs from {args.repo_id}@{args.revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Publish sanitized ground-truth parquet pairs to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd
from dataset_common import file_sha256, validate_public_frame
from huggingface_hub import HfApi, hf_hub_download

PAIR_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _dataset_readme(pair_count: int, row_count: int) -> str:
    return f"""---
license: cc-by-4.0
task_categories:
- tabular-classification
pretty_name: ConfigIQ tested-model performance data
---

# ConfigIQ tested-model performance data

Sanitized aggregate GuideLLM measurements used to train and assess ConfigIQ's
model/hardware performance-envelope classifiers.

- Model/hardware pairs: {pair_count}
- Aggregate benchmark rows: {row_count}
- Format: one Parquet file per model/hardware pair
- License: CC BY 4.0

The dataset excludes run IDs, UUIDs, MLflow identifiers, timestamps, raw runtime
arguments, internal endpoints, and other administrative metadata. `manifest.json`
records pair identity, row count, columns, and checksums. This dataset provides
benchmark observations, not performance guarantees, and is supplied without warranty.
"""


def _schema() -> dict:
    return {
        "schema_version": 1,
        "layout": "pairs/<pair-id>/ground_truth.parquet",
        "identity_fields": ["model_id", "accelerator", "pair_id"],
        "configuration_fields": [
            "backend", "backend_version", "tp", "pp", "dp", "ep", "cp", "isl", "osl",
            "concurrency", "max_seq_len", "max_num_seqs", "prefix_caching", "prefix_tokens",
            "prefix_count", "precision", "weight_quantization", "kv_quantization",
        ],
        "measurement_families": ["throughput", "request rate", "TTFT", "TPOT", "ITL", "request latency", "reliability"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/tested-models/dataset-manifest.json"))
    parser.add_argument("--datasets-dir", type=Path, default=Path("data/tested-models/datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tested-models/hf-ground-truth"))
    parser.add_argument("--repo-id", default="redhat-performance/configiq-performance-data")
    parser.add_argument("--revision", default=None, help="Optional branch or tag to update; defaults to the repository default branch.")
    parser.add_argument("--check-only", action="store_true", help="Compare prepared data with Hugging Face without uploading.")
    return parser.parse_args()


def _content_signature(manifest: dict) -> list[tuple[str, int, tuple[str, ...], str | None]]:
    return sorted(
        (
            pair["id"], pair["records"], tuple(pair["columns"]),
            pair.get("data_sha256"),
        )
        for pair in manifest["pairs"]
    )


def prepare(manifest_path: Path, datasets_dir: Path, output_dir: Path) -> dict:
    manifest_path = manifest_path.resolve()
    datasets_dir = datasets_dir.resolve()
    output_dir = output_dir.resolve()
    staging = output_dir.with_name(f"{output_dir.name}.staging")
    if any(_paths_overlap(candidate, source) for candidate in (output_dir, staging) for source in (manifest_path, datasets_dir)):
        raise ValueError("output and staging directories must not overlap input paths")
    source = json.loads(manifest_path.read_text())
    if source.get("schema_version") != 1:
        raise ValueError("unsupported ground-truth manifest schema")
    source_pairs = source.get("pairs")
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
        model_id = pair.get("model_id")
        accelerator = pair.get("accelerator")
        if not isinstance(model_id, str) or not model_id or not isinstance(accelerator, str) or not accelerator:
            raise ValueError(f"pair {pair_id} requires non-empty model_id and accelerator")
        source_path = datasets_dir / f"{pair_id}.parquet"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        frame = pd.read_parquet(source_path)
        validate_public_frame(frame)
        expected_records = pair.get("records")
        if expected_records is not None and expected_records != len(frame):
            raise ValueError(f"pair {pair_id} row count does not match its manifest")
        expected_columns = pair.get("columns")
        if expected_columns and expected_columns != list(frame.columns):
            raise ValueError(f"pair {pair_id} columns do not match its manifest")
        planned_pairs.append((pair, source_path, frame))

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    output_pairs = []
    for pair, source_path, frame in planned_pairs:
        pair_id = pair["id"]
        destination = staging / "pairs" / pair_id / "ground_truth.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        output_pairs.append({
            "id": pair_id,
            "model_id": pair["model_id"],
            "accelerator": pair["accelerator"],
            "records": len(frame),
            "path": f"pairs/{pair_id}/ground_truth.parquet",
            "columns": list(frame.columns),
            "data_sha256": pair.get("sha256") or pair.get("data_sha256"),
            "file_sha256": file_sha256(destination),
        })
    manifest = {
        "schema_version": 1,
        "source": "configiq-tested-models-ground-truth",
        "pairs": output_pairs,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (staging / "schema.json").write_text(json.dumps(_schema(), indent=2, sort_keys=True) + "\n")
    (staging / "README.md").write_text(_dataset_readme(len(output_pairs), sum(pair["records"] for pair in output_pairs)))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.rename(output_dir)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = prepare(args.manifest, args.datasets_dir, args.output_dir)
    api = HfApi()
    parent_commit = api.repo_info(args.repo_id, repo_type="dataset", revision=args.revision, token=False).sha
    remote_manifest_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        filename="manifest.json",
        revision=parent_commit,
        token=False,
    )
    remote_manifest = json.loads(Path(remote_manifest_path).read_text())
    if _content_signature(manifest) == _content_signature(remote_manifest):
        print(f"ground truth is unchanged from {args.repo_id}@{parent_commit}; no upload needed")
        return 0
    if args.check_only:
        print(f"ground truth differs from {args.repo_id}@{parent_commit}; upload was not attempted")
        return 0
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.output_dir),
        revision=args.revision,
        delete_patterns=["pairs/**", "manifest.json", "schema.json"],
        parent_commit=parent_commit,
        commit_message=f"Publish {len(manifest['pairs'])} ground-truth model hardware pairs",
    )
    print(f"published {len(manifest['pairs'])} pairs to {args.repo_id} at revision {commit.oid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

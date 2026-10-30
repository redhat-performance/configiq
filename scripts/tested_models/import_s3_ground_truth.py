#!/usr/bin/env python3
"""Prepare public ground-truth pair files from an aggregated S3 CSV download."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from dataset_common import dataframe_sha256, load_required_models, normalize_rows, pair_id, write_parquet

REQUIRED_COLUMNS = {"model", "accelerator", "version"}
NUMERIC_COLUMNS = {
    "isl", "osl", "tp", "concurrency", "measured concurrency", "measured rps",
    "output_tok/sec", "total_tok/sec", "prompt_token_count_mean", "prompt_token_count_p99",
    "output_token_count_mean", "output_token_count_p99", "ttft_median", "ttft_p95",
    "ttft_p1", "ttft_p999", "ttft_mean", "ttft_p99", "tpot_median", "tpot_p95",
    "tpot_p99", "tpot_p999", "tpot_p1", "itl_median", "itl_p95", "itl_p999",
    "itl_p1", "itl_mean", "request_latency_median", "request_latency_min",
    "request_latency_max", "successful_requests", "errored_requests", "turns",
    "prefix_tokens", "prefix_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Local copy of the S3 aggregate")
    parser.add_argument("--config", type=Path, default=Path("public/config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tested-models/datasets"))
    parser.add_argument("--manifest", type=Path, default=Path("data/tested-models/dataset-manifest.json"))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _convert_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for name in NUMERIC_COLUMNS.intersection(frame.columns):
        source = frame[name].replace({"": None, "nan": None, "NaN": None, "null": None, "None": None})
        converted = pd.to_numeric(source, errors="coerce")
        if (source.notna() & converted.isna()).any():
            raise ValueError(f"non-numeric value in {name}")
        frame[name] = converted
    return frame


def import_csv(csv_path: Path, config_path: Path, output_dir: Path, manifest_path: Path) -> dict:
    approved_models = set(load_required_models(config_path))
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    source_rows = 0
    approved_rows = 0

    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("source CSV has a missing or duplicate header")
        if missing := REQUIRED_COLUMNS - set(reader.fieldnames):
            raise ValueError(f"source CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            if None in row:
                raise ValueError("source CSV contains a row with extra columns")
            source_rows += 1
            model_id = (row["model"] or "").strip()
            if model_id not in approved_models:
                continue
            accelerator = (row["accelerator"] or "").strip()
            if not accelerator:
                raise ValueError(f"approved model has a missing accelerator on CSV row {source_rows + 1}")
            groups[(model_id, accelerator)].append(row)
            approved_rows += 1

    if not groups:
        raise ValueError("source CSV contains no approved model and accelerator pairs")

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for (model_id, accelerator), rows in sorted(groups.items()):
        frame = normalize_rows(rows, model_id, accelerator)
        frame = _convert_numeric_columns(frame)
        # The aggregate's row order can change without changing its measurements.
        row_keys = frame.astype("string").fillna("<null>").agg("\x1f".join, axis=1)
        frame = frame.iloc[row_keys.argsort(kind="stable")].reset_index(drop=True)
        item_id = pair_id(model_id, accelerator)
        destination = output_dir / f"{item_id}.parquet"
        write_parquet(frame, destination)
        pairs.append({
            "id": item_id,
            "model_id": model_id,
            "accelerator": accelerator,
            "path": str(destination),
            "records": len(frame),
            "columns": list(frame.columns),
            "sha256": dataframe_sha256(frame),
        })

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "s3_aggregate",
        "source_sha256": _sha256(csv_path),
        "source_rows": source_rows,
        "approved_rows": approved_rows,
        "pairs": pairs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {approved_rows} approved rows in {len(pairs)} model/accelerator pairs from {source_rows} CSV rows")
    print("All source version labels were retained; no backend version filter was applied")
    return manifest


def main() -> int:
    args = parse_args()
    import_csv(args.csv, args.config, args.output_dir, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

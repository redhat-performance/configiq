#!/usr/bin/env python3
"""Prepare public ground-truth pair files from an aggregated S3 CSV download."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
from dataset_common import (
    dataframe_sha256,
    is_upstream_vllm_version,
    load_required_models,
    normalize_rows,
    pair_id,
    write_parquet,
)

REQUIRED_COLUMNS = {"model", "accelerator", "version"}
# These source fields describe the recorded workload and serving setup. Raw
# runtime arguments, environment, and image tags are used only while selecting
# runs; none of them are copied into the public Parquet files.
SOURCE_CURVE_FIELDS = (
    "model", "accelerator", "version", "prompt toks", "output toks", "TP", "DP",
    "PP", "EP", "CP", "TP_prefill", "TP_decode", "tp_prefill", "tp_decode",
    "moe_tp_size", "moe_ep_size", "enable_expert_parallel",
    "backend", "backend_version", "precision",
    "weight_quantization", "kv_quantization", "max_num_seqs",
    "max_num_batched_tokens", "max_model_len", "max_seq_len",
    "gpu_memory_utilization", "chunked_prefill_enabled",
    "runtime_args", "env_vars", "image_tag", "dataset", "guidellm_version",
    "spec_decoding", "prefix_caching", "turns", "prefix_tokens", "prefix_count",
    "request_type",
)
PUBLIC_CURVE_FIELDS = (
    "model_id", "accelerator", "version", "backend", "backend_version",
    "tp", "pp", "dp", "ep", "cp", "tp_prefill", "tp_decode",
    "moe_tp_size", "moe_ep_size", "enable_expert_parallel", "isl", "osl", "request_type", "turns",
    "prefix_caching", "prefix_tokens", "prefix_count", "spec_decoding",
    "precision", "weight_quantization", "kv_quantization", "max_seq_len",
    "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
    "chunked_prefill_enabled", "moe_num_experts", "moe_top_k",
    "vllm_rocm_use_aiter", "vllm_rocm_use_aiter_moe", "vllm_rocm_use_aiter_moe_situv2",
    "runtime_args", "env_vars", "guidellm_version",
)
NUMERIC_COLUMNS = {
    "isl", "osl", "tp", "pp", "dp", "ep", "cp", "tp_prefill", "tp_decode",
    "moe_tp_size", "moe_ep_size", "concurrency", "measured concurrency", "measured rps",
    "output_tok/sec", "total_tok/sec", "prompt_token_count_mean", "prompt_token_count_p99",
    "output_token_count_mean", "output_token_count_p99", "ttft_median", "ttft_p95",
    "ttft_p1", "ttft_p999", "ttft_mean", "ttft_p99", "tpot_median", "tpot_p95",
    "tpot_p99", "tpot_p999", "tpot_p1", "itl_median", "itl_p95", "itl_p999",
    "itl_p1", "itl_mean", "request_latency_median", "request_latency_min",
    "request_latency_max", "successful_requests", "errored_requests", "turns",
    "prefix_tokens", "prefix_count", "max_num_batched_tokens", "turn_index",
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


def _source_curve_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple((row.get(field) or "").strip() for field in SOURCE_CURVE_FIELDS)


def _start_time(row: dict[str, str]) -> Decimal:
    value = (row.get("guidellm_start_time_ms") or "").strip()
    try:
        timestamp = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("repeated configuration has no valid GuideLLM start time") from error
    if not timestamp.is_finite():
        raise ValueError("repeated configuration has no valid GuideLLM start time")
    return timestamp


def _public_curve_key(row: dict[str, object]) -> tuple[str | None, ...]:
    return tuple(None if pd.isna(row.get(field)) else str(row[field]) for field in PUBLIC_CURVE_FIELDS)


def _public_identity(row: dict[str, object]) -> tuple[tuple[str | None, ...], str]:
    return _public_curve_key(row), str(row.get("concurrency"))


def _newest_public_rows(rows: list[dict[str, str]], frame: pd.DataFrame) -> tuple[list[int], int]:
    """Keep one newest run for every configuration users can distinguish."""
    candidates: dict[tuple[tuple[str | None, ...], str], list[int]] = defaultdict(list)
    for index, normalized in enumerate(frame.to_dict(orient="records")):
        candidates[_public_identity(normalized)].append(index)
    keep = []
    for indices in candidates.values():
        if len(indices) == 1:
            keep.append(indices[0])
            continue
        ranked = sorted(indices, key=lambda index: _start_time(rows[index]))
        if _start_time(rows[ranked[-1]]) == _start_time(rows[ranked[-2]]):
            raise ValueError("repeated public configuration has tied GuideLLM start times")
        keep.append(ranked[-1])
    return sorted(keep), len(rows) - len(keep)


def _run_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("uuid") or "").strip(), (row.get("intended concurrency") or "").strip()


def import_csv(csv_path: Path, config_path: Path, output_dir: Path, manifest_path: Path) -> dict:
    approved_models = set(load_required_models(config_path))
    candidates: dict[tuple[tuple[str, ...], str], list[dict[str, str]]] = defaultdict(list)
    source_rows = 0
    approved_rows = 0
    upstream_rows = 0
    detail_rows_seen = 0
    duplicate_source_rows_dropped = 0
    dropped_versions: Counter[str] = Counter()
    seen_source_rows: set[tuple[tuple[str, str], ...]] = set()
    details_by_run: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

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
            approved_rows += 1
            version = (row["version"] or "").strip()
            if not is_upstream_vllm_version(version):
                # Productized RHAIIS builds and non-release labels must never
                # widen the public upstream-only dataset.
                dropped_versions[version or "<blank>"] += 1
                continue
            upstream_rows += 1

            source_row_key = tuple(sorted(row.items()))
            if source_row_key in seen_source_rows:
                duplicate_source_rows_dropped += 1
                continue
            seen_source_rows.add(source_row_key)
            if (row.get("turn_index") or "").strip():
                detail_rows_seen += 1
                if _run_key(row)[0]:
                    details_by_run[_run_key(row)].append(row)
                continue
            candidates[(_source_curve_key(row), (row.get("intended concurrency") or "").strip())].append(row)

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    superseded_rows_dropped = 0
    for repeated in candidates.values():
        if len(repeated) == 1:
            selected = repeated[0]
        else:
            ranked = sorted(repeated, key=_start_time)
            if _start_time(ranked[-1]) == _start_time(ranked[-2]):
                raise ValueError("repeated configuration has tied GuideLLM start times")
            selected = ranked[-1]
            superseded_rows_dropped += len(repeated) - 1
        groups[(selected["model"].strip(), selected["accelerator"].strip())].append(selected)

    if not groups:
        raise ValueError("source CSV contains no approved model and accelerator pairs")

    ids: dict[str, tuple[str, str]] = {}
    for model_id, accelerator in groups:
        item_id = pair_id(model_id, accelerator)
        previous = ids.get(item_id)
        if previous is not None and previous != (model_id, accelerator):
            raise ValueError(
                f"distinct model/accelerator pairs map to the same ID {item_id}: "
                f"{previous!r} and {(model_id, accelerator)!r}"
            )
        ids[item_id] = (model_id, accelerator)

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    public_duplicate_rows_dropped = 0
    detail_rows_kept = 0
    for (model_id, accelerator), rows in sorted(groups.items()):
        summary_frame = _convert_numeric_columns(normalize_rows(rows, model_id, accelerator))
        keep, dropped = _newest_public_rows(rows, summary_frame)
        public_duplicate_rows_dropped += dropped
        selected_summaries = [rows[index] for index in keep]
        selected_details = [
            detail
            for summary in selected_summaries
            for detail in details_by_run.get(_run_key(summary), [])
            if detail["model"].strip() == model_id and detail["accelerator"].strip() == accelerator
            and _source_curve_key(detail) == _source_curve_key(summary)
        ]
        detail_rows_kept += len(selected_details)
        frame = normalize_rows(selected_summaries + selected_details, model_id, accelerator)
        if "turn_index" not in frame:
            frame["turn_index"] = None
        frame = _convert_numeric_columns(frame)
        if "concurrency" not in frame or frame["concurrency"].isna().any():
            raise ValueError(f"pair {model_id}/{accelerator} requires nonblank concurrency")
        if "turn_index" in frame:
            details = frame["turn_index"].dropna()
            if ((details < 0) | (details % 1 != 0)).any():
                raise ValueError(f"pair {model_id}/{accelerator} has invalid turn indexes")
        identity_fields = [field for field in PUBLIC_CURVE_FIELDS if field in frame]
        row_fields = [*identity_fields, "concurrency", *(["turn_index"] if "turn_index" in frame else [])]
        if frame.duplicated(row_fields).any():
            raise ValueError(f"pair {model_id}/{accelerator} has repeated published settings, concurrency, and turn rows")
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
        "upstream_rows": upstream_rows,
        "detail_rows_seen": detail_rows_seen,
        "detail_rows_kept": detail_rows_kept,
        "detail_rows_dropped": detail_rows_seen - detail_rows_kept,
        "duplicate_source_rows_dropped": duplicate_source_rows_dropped,
        "superseded_rows_dropped": superseded_rows_dropped,
        "public_duplicate_rows_dropped": public_duplicate_rows_dropped,
        "published_rows": sum(pair["records"] for pair in pairs),
        "dropped_version_rows": approved_rows - upstream_rows,
        "dropped_versions": dict(sorted(dropped_versions.items())),
        "pairs": pairs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Prepared {manifest['published_rows']} upstream summary rows in {len(pairs)} model/accelerator pairs "
        f"from {source_rows} CSV rows ({approved_rows} approved rows before the version filter)"
    )
    if dropped_versions:
        dropped_summary = ", ".join(f"{version}={count}" for version, count in sorted(dropped_versions.items()))
        print(f"Dropped {approved_rows - upstream_rows} approved rows with non-upstream version labels: {dropped_summary}")
    return manifest


def main() -> int:
    args = parse_args()
    import_csv(args.csv, args.config, args.output_dir, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

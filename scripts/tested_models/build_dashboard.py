#!/usr/bin/env python3
"""Build sanitized per-pair dashboard payloads from generated artifacts.

Each measured pair gets a payload, even when it has no trained classifier. A
pair that fails the validation gate must stay inspectable: its graphs explain
why there is no verdict instead of the pair disappearing from the UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from build_registry import coverage_for, evaluate_gate, system_id
from xgboost import XGBClassifier

# Columns that identify one measured sweep. Workload (isl/osl/turns), the
# parallelism width (tp), and the x-axis (concurrency) are separate keys, and
# outcomes are deliberately absent. Two rows with the same sweep id were run
# under the same configuration, so their medians are comparable.
_SWEEP_COLUMNS = (
    "backend",
    "backend_version",
    "pp",
    "dp",
    "ep",
    "cp",
    "precision",
    "weight_quantization",
    "kv_quantization",
    "gpu_memory_utilization",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_seq_len",
    "chunked_prefill_enabled",
    "spec_decoding",
    "prefix_caching",
    "prefix_tokens",
    "prefix_count",
    "moe_num_experts",
    "moe_top_k",
)

_SWEEP_LABELS = (
    ("precision", "precision"),
    ("weight_quantization", "weights"),
    ("kv_quantization", "KV cache"),
    ("chunked_prefill_enabled", "chunked prefill"),
)

_UNKNOWN_VALUES = {"", "nan", "none", "unknown"}


def _json_number(value: float) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def _numeric(frame: pd.DataFrame, name: str, default: float = float("nan")) -> pd.Series:
    value = frame.get(name)
    if value is None:
        return pd.Series(float(default), index=frame.index)
    return pd.to_numeric(value, errors="coerce")


def _sweep_string(frame: pd.DataFrame) -> pd.Series:
    """Stable per-row identity for the run configuration, excluding the x-axis."""
    values = []
    for column in _SWEEP_COLUMNS:
        if column in frame:
            series = frame[column].map(lambda value: "" if pd.isna(value) else str(value))
        else:
            series = pd.Series("", index=frame.index)
        values.append(series)
    return pd.concat(values, axis=1).agg("|".join, axis=1)


def _config_label(group: pd.DataFrame) -> str:
    versions = sorted({
        str(value).strip()
        for value in group.get("backend_version", pd.Series(dtype=object)).dropna()
        if str(value).strip().lower() not in _UNKNOWN_VALUES
    })
    parts = [", ".join(versions) if versions else "unknown backend"]
    for column, label in _SWEEP_LABELS:
        if column not in group:
            continue
        details = []
        for value in group[column].dropna().unique():
            if isinstance(value, bool):
                details.append(f"{label} {'on' if value else 'off'}")
                continue
            text = str(value).strip()
            if text.lower() in _UNKNOWN_VALUES:
                continue
            details.append(f"{label} {text}")
        parts.extend(sorted(set(details)))
    return " · ".join(parts)


def _throughput_curves(frame: pd.DataFrame) -> list[dict]:
    """One clean throughput curve per sweep, never mixing workloads or units.

    Grouping by concurrency alone mixes ISL/OSL, tensor-parallel width, turn
    count, backend configuration, and — when multi-turn windows are present —
    per-request rates with benchmark aggregates. Each of those produces a
    different line, so the median over the mixture alternates between them and
    draws a saw-tooth that no run measured.
    """
    work = frame.assign(
        _throughput=_numeric(frame, "output_tok/sec"),
        _latency=_numeric(frame, "request_latency_median"),
        _concurrency=_numeric(frame, "concurrency"),
        _measured=_numeric(frame, "measured concurrency"),
        _turns=_numeric(frame, "turns").fillna(1),
        _isl=_numeric(frame, "isl").fillna(0),
        _osl=_numeric(frame, "osl").fillna(0),
        _tp=_numeric(frame, "tp").fillna(1),
    )
    work["_sweep"] = _sweep_string(work)
    work = work.dropna(subset=["_concurrency", "_throughput"])

    curves = []
    for (isl, osl, turns, tp, sweep), group in work.groupby(
        ["_isl", "_osl", "_turns", "_tp", "_sweep"], dropna=False,
    ):
        points = []
        for concurrency, level in group.groupby("_concurrency"):
            throughput = level["_throughput"].dropna()
            if throughput.empty:
                continue
            latency = level["_latency"].dropna()
            measured = level["_measured"].dropna()
            points.append({
                "concurrency": _json_number(concurrency),
                "measuredConcurrency": _json_number(measured.median()) if not measured.empty else None,
                "throughput": float(throughput.median()),
                "throughputMin": float(throughput.min()),
                "throughputMax": float(throughput.max()),
                "latency": float(latency.median()) if not latency.empty else None,
                "observations": int(throughput.size),
            })
        if not points:
            continue
        points.sort(key=lambda point: point["concurrency"])
        backends = sorted({
            str(value).strip()
            for value in group.get("backend", pd.Series(dtype=object)).dropna()
            if str(value).strip().lower() not in _UNKNOWN_VALUES
        })
        versions = sorted({
            str(value).strip()
            for value in group.get("backend_version", pd.Series(dtype=object)).dropna()
            if str(value).strip().lower() not in _UNKNOWN_VALUES
        })
        curves.append({
            "isl": _json_number(isl),
            "osl": _json_number(osl),
            "turns": int(turns),
            "tp": int(tp),
            "sweepId": hashlib.sha256(sweep.encode()).hexdigest()[:16],
            "backend": backends[0] if backends else None,
            "backendVersions": versions,
            "configLabel": _config_label(group),
            "points": points,
        })

    curves.sort(key=lambda curve: (
        curve["isl"], curve["osl"], curve["turns"], curve["tp"], curve["sweepId"],
    ))
    return curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/tested-models/dataset-manifest.json"))
    parser.add_argument("--labeled-dir", type=Path, default=Path("data/tested-models/stage5/labeled"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/tested-models/stage5/models"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tested-models/stage6/registry/dashboard"))
    return parser.parse_args()


def _labeled_frame(pair_id: str, labeled_dir: Path) -> pd.DataFrame | None:
    frames = [
        pd.read_parquet(labeled_dir / f"{pair_id}.{split}.parquet")
        for split in ("train", "validation", "test")
        if (labeled_dir / f"{pair_id}.{split}.parquet").exists()
    ]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _feature_importance(metadata: dict, model_path: Path) -> list[dict]:
    model = XGBClassifier()
    model.load_model(model_path)
    feature_names = metadata["feature_schema"]["feature_names"]
    return sorted(
        ({"feature": name, "importance": float(value)} for name, value in zip(feature_names, model.feature_importances_)),
        key=lambda item: item["importance"],
        reverse=True,
    )[:10]


def build_dashboard(manifest_path: Path, labeled_dir: Path, model_dir: Path, output_dir: Path) -> int:
    manifest = json.loads(manifest_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    for pair in manifest.get("pairs", []):
        pair_id = pair["id"]
        frame = _labeled_frame(pair_id, labeled_dir)
        if frame is None:
            # The pair has no aggregate labeled rows, so this run cannot
            # describe it. Remove a payload left by an earlier run instead of
            # publishing a dashboard that no current labels support.
            (output_dir / f"{pair_id}.json").unlink(missing_ok=True)
            continue
        # Defense in depth: a labeled frame produced before the measurement-family
        # split may still carry per-turn windows.
        if "measurement_kind" in frame:
            frame = frame.loc[frame["measurement_kind"].astype(str) != "per_turn"]

        metadata_path = model_dir / f"{pair_id}.metadata.json"
        model_path = model_dir / f"{pair_id}.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        coverage = coverage_for(pair_id, labeled_dir, int(pair.get("records", len(frame))))
        validated = evaluate_gate(metadata, coverage)["passed"]

        knee_columns = [column for column in ("isl", "osl", "turns", "concurrency", "tp") if column in frame]
        knees = []
        if "label_knee" in frame:
            for _, row in frame[frame["label_knee"].astype(bool)].iterrows():
                point = {}
                for column in knee_columns:
                    value = row[column]
                    if pd.isna(value):
                        point[column] = 1 if column == "turns" else None
                    else:
                        point[column] = _json_number(value)
                knees.append(point)

        within = int(frame["within_tested_region"].sum()) if "within_tested_region" in frame else 0
        payload = {
            "schemaVersion": 2,
            "generatedAt": datetime.now(UTC).isoformat(),
            "id": pair_id,
            "modelId": pair["model_id"],
            "systemId": system_id(pair["accelerator"]),
            "records": len(frame),
            "hasEnvelope": validated,
            "trainingStatus": metadata.get("status", "not_trained"),
            "labelDistribution": {
                "within": within,
                "outside": len(frame) - within,
            },
            "throughputCurves": _throughput_curves(frame),
            "kneePoints": knees[:100],
            "featureImportance": _feature_importance(metadata, model_path) if validated else [],
        }
        if validated:
            payload["validation"] = metadata.get("validation")
            payload["test"] = metadata.get("test")
        (output_dir / f"{pair_id}.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        generated += 1
    print(f"wrote {generated} dashboard payloads to {output_dir}")
    return generated


def main() -> int:
    args = parse_args()
    build_dashboard(args.manifest, args.labeled_dir, args.model_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

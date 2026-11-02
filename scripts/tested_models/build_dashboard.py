#!/usr/bin/env python3
"""Build sanitized per-pair dashboard payloads from generated artifacts.

Each measured pair gets a payload, even when it has no trained classifier. A
pair that fails the validation gate must stay inspectable: its graphs explain
why there is no verdict instead of the pair disappearing from the UI.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from build_registry import system_id
from xgboost import XGBClassifier


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
            continue

        metadata_path = model_dir / f"{pair_id}.metadata.json"
        model_path = model_dir / f"{pair_id}.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        has_envelope = metadata.get("status") == "trained" and model_path.exists()

        throughput = pd.to_numeric(frame.get("output_tok/sec"), errors="coerce")
        latency = pd.to_numeric(frame.get("request_latency_median"), errors="coerce")
        curves = []
        for concurrency, group in frame.assign(_throughput=throughput, _latency=latency).groupby("concurrency", dropna=True):
            curves.append({
                "concurrency": float(concurrency),
                "throughput": float(group["_throughput"].median()) if group["_throughput"].notna().any() else None,
                "latency": float(group["_latency"].median()) if group["_latency"].notna().any() else None,
                "points": len(group),
            })
        knee_columns = [column for column in ("isl", "osl", "concurrency", "tp") if column in frame]
        knees = []
        if "label_knee" in frame:
            for _, row in frame[frame["label_knee"].astype(bool)].iterrows():
                knees.append({column: row[column] for column in knee_columns})

        within = int(frame["within_tested_region"].sum()) if "within_tested_region" in frame else 0
        payload = {
            "schemaVersion": 1,
            "generatedAt": datetime.now(UTC).isoformat(),
            "id": pair_id,
            "modelId": pair["model_id"],
            "systemId": system_id(pair["accelerator"]),
            "records": len(frame),
            "hasEnvelope": has_envelope,
            "trainingStatus": metadata.get("status", "not_trained"),
            "labelDistribution": {
                "within": within,
                "outside": len(frame) - within,
            },
            "validation": metadata.get("validation"),
            "test": metadata.get("test"),
            "throughputLatencyByConcurrency": sorted(curves, key=lambda item: item["concurrency"]),
            "kneePoints": knees[:100],
            "featureImportance": _feature_importance(metadata, model_path) if has_envelope else [],
        }
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

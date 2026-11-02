#!/usr/bin/env python3
"""Build the public tested-model registry from validated classifier artifacts.

Registry schema v2 separates three concepts that used to be conflated:

* ``validated`` — a trained classifier whose held-out measurements pass the
  agreed quality gate. Only these pairs may emit a classifier verdict.
* ``measured_no_envelope`` — measurements and dashboard graphs exist, but the
  pair has no classifier that passes the gate. It must not emit a verdict.
* ``approvedModels`` — models approved for measurement. Absence of a pair for
  one of these models means "approved, no measurements yet", not "tested".
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dataset_common import load_approved_models

GATE_MIN_RECORDS = 100
GATE_MIN_TEST_AUC = 0.70
GATE_MIN_TEST_F1 = 0.30
GATE_MIN_CONCURRENCY_LEVELS = 5

VALIDATED = "validated"
MEASURED_NO_ENVELOPE = "measured_no_envelope"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/tested-models/dataset-manifest.json"))
    parser.add_argument("--labeled-dir", type=Path, default=Path("data/tested-models/stage5/labeled"))
    parser.add_argument("--config", type=Path, default=Path("public/config.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/tested-models/stage5/models"))
    parser.add_argument("--triton-dir", type=Path, default=Path("data/tested-models/stage5/triton"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tested-models/stage6/registry"))
    parser.add_argument("--dashboard-dir", type=Path, default=Path("data/tested-models/stage6/registry/dashboard"))
    return parser.parse_args()


def system_id(accelerator: str) -> str:
    return {
        "h200": "h200_sxm",
        "b200": "b200_sxm",
        "b300": "b300_sxm",
        "mi300x": "mi300x",
        "mi355x": "mi355x",
    }.get(accelerator.lower(), accelerator.lower())


def gate_thresholds() -> dict:
    return {
        "minRecords": GATE_MIN_RECORDS,
        "minTestAuc": GATE_MIN_TEST_AUC,
        "minTestF1": GATE_MIN_TEST_F1,
        "minConcurrencyLevels": GATE_MIN_CONCURRENCY_LEVELS,
    }


def coverage_for(pair_id: str, labeled_dir: Path | None, fallback_records: int) -> dict:
    """Summarize the measured coverage of one pair from its labeled splits."""
    frames: list[pd.DataFrame] = []
    if labeled_dir is not None:
        for split in ("train", "validation", "test"):
            path = labeled_dir / f"{pair_id}.{split}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
    if not frames:
        return {
            "records": int(fallback_records),
            "concurrencyLevels": [],
            "concurrencyLevelCount": 0,
            "backendVersions": [],
        }

    frame = pd.concat(frames, ignore_index=True)
    concurrency = pd.to_numeric(frame.get("concurrency"), errors="coerce").dropna()
    levels = sorted({float(value) for value in concurrency.unique()})
    versions = frame.get("backend_version", pd.Series(dtype="object"))
    return {
        "records": len(frame),
        "concurrencyLevels": levels,
        "concurrencyLevelCount": len(levels),
        "backendVersions": sorted({str(value) for value in versions.dropna().unique()}),
    }


def evaluate_gate(report_entry: dict, coverage: dict) -> dict:
    """Apply the agreed quality gate and return the reasons for any failure."""
    reasons: list[str] = []
    if report_entry.get("status") != "trained":
        reasons.append("classifier_not_trained")

    test = report_entry.get("test") or {}
    auc = test.get("roc_auc")
    f1 = test.get("f1")

    if coverage["records"] < GATE_MIN_RECORDS:
        reasons.append("records_below_minimum")
    if auc is None or not math.isfinite(auc):
        reasons.append("test_auc_unavailable")
    elif auc < GATE_MIN_TEST_AUC:
        reasons.append("test_auc_below_minimum")
    if f1 is None or not math.isfinite(f1):
        reasons.append("test_f1_unavailable")
    elif f1 < GATE_MIN_TEST_F1:
        reasons.append("test_f1_below_minimum")
    if coverage["concurrencyLevelCount"] < GATE_MIN_CONCURRENCY_LEVELS:
        reasons.append("concurrency_levels_below_minimum")

    return {"passed": not reasons, "reasons": reasons, "thresholds": gate_thresholds()}


def build_registry(
    manifest_path: Path,
    model_dir: Path,
    triton_dir: Path,
    output_dir: Path,
    dashboard_dir: Path,
    labeled_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    pairs = {pair["id"]: pair for pair in manifest.get("pairs", [])}
    report = json.loads((model_dir / "training-report.json").read_text())
    report_pairs = report.get("pairs", {})
    approved_models = load_approved_models(config_path) if config_path is not None else []
    registry_pairs = []
    triton_output = output_dir / "triton"
    if triton_output.exists():
        shutil.rmtree(triton_output)

    unknown_trained = sorted(set(report_pairs) - set(pairs))
    if unknown_trained:
        raise ValueError(f"trained pair is absent from dataset manifest: {unknown_trained[0]}")

    for pair_id, source in sorted(pairs.items()):
        report_entry = report_pairs.get(pair_id, {})
        coverage = coverage_for(pair_id, labeled_dir, int(source.get("records", 0)))
        gate = evaluate_gate(report_entry, coverage)
        dashboard_path = dashboard_dir / f"{pair_id}.json"
        if not dashboard_path.is_file():
            raise FileNotFoundError(dashboard_path)

        entry: dict = {
            "id": pair_id,
            "modelId": source["model_id"],
            "systemId": system_id(source["accelerator"]),
            "status": VALIDATED if gate["passed"] else MEASURED_NO_ENVELOPE,
            "coverage": coverage,
            "dashboardPath": f"dashboard/{pair_id}.json",
            "gate": gate,
        }

        if gate["passed"]:
            model_path = model_dir / f"{pair_id}.json"
            metadata_path = model_dir / f"{pair_id}.metadata.json"
            triton_path = triton_dir / pair_id / "1" / "xgboost.json"
            config_path_artifact = triton_dir / pair_id / "config.pbtxt"
            for path in (model_path, metadata_path, triton_path, config_path_artifact):
                if not path.exists():
                    raise FileNotFoundError(path)
            shutil.copytree(triton_dir / pair_id, triton_output / pair_id)
            entry.update(
                {
                    "classifier": "xgboost",
                    "tritonModel": pair_id,
                    "metrics": {
                        "validation": report_entry.get("validation"),
                        "test": report_entry.get("test"),
                    },
                    "thresholds": report_entry.get("thresholds", {}),
                    "featureSchema": report_entry.get("feature_schema", {}),
                    "artifactVersion": report_entry.get("model_format"),
                }
            )
        else:
            entry["diagnostics"] = {
                "trainingStatus": report_entry.get("status", "not_trained"),
                "reasons": gate["reasons"],
            }

        registry_pairs.append(entry)

    registry = {
        "schemaVersion": 2,
        "generatedAt": datetime.now(UTC).isoformat(),
        "gate": gate_thresholds(),
        "approvedModels": approved_models,
        "pairs": registry_pairs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return registry


def main() -> int:
    args = parse_args()
    registry = build_registry(
        args.manifest,
        args.model_dir,
        args.triton_dir,
        args.output_dir,
        args.dashboard_dir,
        labeled_dir=args.labeled_dir,
        config_path=args.config,
    )
    validated = sum(pair["status"] == VALIDATED for pair in registry["pairs"])
    measured = len(registry["pairs"]) - validated
    print(
        f"wrote {len(registry['pairs'])} tested-model measurements to {args.output_dir / 'registry.json'} "
        f"({validated} validated, {measured} measured without a validated envelope)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

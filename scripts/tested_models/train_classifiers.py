#!/usr/bin/env python3
"""Train leakage-safe XGBoost envelope classifiers and export Triton FIL bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from stage5_common import feature_matrix, read_manifest
from xgboost import XGBClassifier

WITHIN_THRESHOLD = 0.85
UNCERTAIN_THRESHOLD = 0.50


def _metrics(model: XGBClassifier, matrix: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    probabilities = model.predict_proba(matrix)[:, 1]
    predictions = (probabilities >= WITHIN_THRESHOLD).astype(int)
    accuracy = float(accuracy_score(labels, predictions))
    balanced_accuracy = accuracy if labels.nunique() < 2 else float(balanced_accuracy_score(labels, predictions))
    result: dict[str, Any] = {"rows": len(labels), "positive": int(labels.sum()), "negative": int((labels == 0).sum()), "accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "f1": float(f1_score(labels, predictions, zero_division=0)), "precision": float(precision_score(labels, predictions, zero_division=0)), "recall": float(recall_score(labels, predictions, zero_division=0))}
    result["roc_auc"] = float(roc_auc_score(labels, probabilities)) if labels.nunique() == 2 else None
    return result


def _candidates() -> list[dict[str, Any]]:
    return [
        {"max_depth": 2, "learning_rate": 0.08, "n_estimators": 80, "min_child_weight": 2},
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 120, "min_child_weight": 2},
        {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 160, "min_child_weight": 1},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/tested-models/dataset-manifest.json"))
    parser.add_argument("--labeled-dir", type=Path, default=Path("data/tested-models/stage5/labeled"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tested-models/stage5/models"))
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"seed": args.seed, "pairs": {}}
    label_rules = json.loads((args.labeled_dir / "label-rules.json").read_text())
    split_summary_path = args.labeled_dir.parent / "splits" / "split-summary.json"
    split_summary = json.loads(split_summary_path.read_text()) if split_summary_path.exists() else {"pairs": {}}
    for source in read_manifest(args.manifest):
        pair = str(source["pair_id"].iloc[0])
        frames = {split: pd.read_parquet(args.labeled_dir / f"{pair}.{split}.parquet") for split in ("train", "validation", "test")}
        target = "within_tested_region"
        y_train, y_validation, y_test = (frames[split][target].astype(int) for split in ("train", "validation", "test"))
        pair_report: dict[str, Any] = {"split_counts": {split: {"rows": len(frame), "positive": int(frame[target].sum()), "negative": int((frame[target] == 0).sum())} for split, frame in frames.items()}}
        pair_report["warnings"] = []
        if split_summary["pairs"].get(pair, {}).get("status") == "insufficient_groups":
            pair_report["status"] = "skipped_insufficient_groups"
            pair_report["warnings"].append("too_few_configuration_groups_for_holdout_splits")
            report["pairs"][pair] = pair_report
            continue
        if y_validation.nunique() < 2:
            pair_report["warnings"].append("validation_single_class")
        if y_test.nunique() < 2:
            pair_report["warnings"].append("test_single_class")
        if y_train.nunique() < 2:
            pair_report["status"] = "skipped_single_class_train"
            report["pairs"][pair] = pair_report
            continue
        train_matrix, validation_info = feature_matrix(frames["train"], frames["validation"])
        validation_matrix = validation_info["features"]
        _, test_info = feature_matrix(frames["train"], frames["test"])
        test_matrix = test_info["features"]
        best_model = None
        best_score = float("-inf")
        best_params = None
        for params in _candidates():
            model = XGBClassifier(**params, objective="binary:logistic", eval_metric="logloss", random_state=args.seed, n_jobs=1, tree_method="hist")
            model.fit(train_matrix, y_train)
            score = _metrics(model, validation_matrix, y_validation)["balanced_accuracy"] if y_validation.nunique() > 1 else _metrics(model, train_matrix, y_train)["balanced_accuracy"]
            if score > best_score:
                best_model, best_score, best_params = model, score, params
        assert best_model is not None and best_params is not None
        model_path = args.output_dir / f"{pair}.json"
        best_model.save_model(model_path)
        # This is the only test evaluation; test data is not used for selection or fitting.
        pair_report.update({"status": "trained", "selected_params": best_params, "validation_selection_score": best_score, "validation": _metrics(best_model, validation_matrix, y_validation), "test": _metrics(best_model, test_matrix, y_test), "feature_schema": {"feature_names": test_info["feature_names"], "numeric_medians": test_info["numeric_medians"]}, "thresholds": {"within_region_probability": WITHIN_THRESHOLD, "uncertain_probability": UNCERTAIN_THRESHOLD}, "label_rules": label_rules[pair], "model_format": "xgboost-json"})
        (args.output_dir / f"{pair}.metadata.json").write_text(json.dumps(pair_report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (args.output_dir / f"{pair}.config.pbtxt").write_text(_fil_config(pair, len(test_info["feature_names"])))
        report["pairs"][pair] = pair_report
    (args.output_dir / "training-report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _fil_config(pair: str, feature_count: int) -> str:
    return f'''name: "{pair}"
platform: "fil"
max_batch_size: 0
input [{{ name: "input__0" data_type: TYPE_FP32 dims: [ {feature_count} ] }}]
output [{{ name: "output__0" data_type: TYPE_FP32 dims: [ 2 ] }}]
instance_group [{{ kind: KIND_CPU count: 1 }}]
parameters: {{ key: "model_type" value: {{ string_value: "xgboost_json" }} }}
parameters: {{ key: "predict_proba" value: {{ string_value: "true" }} }}
parameters: {{ key: "threshold" value: {{ string_value: "0.5" }} }}
'''


if __name__ == "__main__":
    raise SystemExit(main())

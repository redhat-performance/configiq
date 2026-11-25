#!/usr/bin/env python3
"""Fit conservative, train-only performance-envelope labels for each pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stage5_common import aggregate_measurements, read_manifest


def _column(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    value = frame.get(name)
    return value if value is not None else pd.Series(default, index=frame.index)


def _aggregate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only benchmark-level rows; per-turn windows are a different family."""
    return frame.loc[aggregate_measurements(frame)].copy()


def _throughput(frame: pd.DataFrame) -> pd.Series:
    value = pd.to_numeric(_column(frame, "output_tok/sec"), errors="coerce")
    fallback = pd.to_numeric(_column(frame, "measured rps"), errors="coerce") * pd.to_numeric(_column(frame, "output_token_count_mean", 0), errors="coerce")
    return value.fillna(fallback)


def _latency(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(_column(frame, "request_latency_max", _column(frame, "request_latency_median")), errors="coerce")


def _curvature(train: pd.DataFrame) -> pd.Series:
    """Estimate local second differences only within matched configuration slices."""
    from stage5_common import normalize_features

    features = normalize_features(train)
    # Turn count is part of the workload, not a numeric axis to sweep: a
    # multi-turn aggregate row and a single-turn row must never be ordered
    # together as if only concurrency separated them.
    features["turns"] = pd.to_numeric(_column(train, "turns"), errors="coerce").fillna(1).to_numpy()
    throughput = _throughput(train).fillna(0)
    score = pd.Series(0.0, index=train.index)
    parameters = [name for name in ("tp", "pp", "dp", "ep", "cp", "isl", "osl", "concurrency") if name in features]
    group_keys = [*parameters, "turns"]
    for parameter in parameters:
        others = [name for name in group_keys if name != parameter]
        for indices in features.groupby(others, dropna=False).groups.values():
            ordered = sorted(indices, key=lambda index: (features.loc[index, parameter], index))
            if len(ordered) < 3:
                continue
            for position in range(1, len(ordered) - 1):
                left, current, right = ordered[position - 1:position + 2]
                x0, x1, x2 = (float(features.loc[index, parameter]) for index in (left, current, right))
                if x0 == x1 or x1 == x2:
                    continue
                slope_left = (throughput.loc[current] - throughput.loc[left]) / (x1 - x0)
                slope_right = (throughput.loc[right] - throughput.loc[current]) / (x2 - x1)
                score.loc[current] = max(score.loc[current], abs(slope_right - slope_left))
    return score


def label_pair(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    original_train_rows = len(train)
    train = _aggregate_rows(train)
    validation = _aggregate_rows(validation)
    test = _aggregate_rows(test)
    if train.empty:
        raise ValueError(
            "the training split contains no aggregate measurements; "
            "per-turn windows cannot define a throughput envelope"
        )
    throughput = _throughput(train)
    efficiency = throughput / (pd.to_numeric(train.get("tp", 1), errors="coerce").fillna(1).clip(lower=1))
    latency = _latency(train)
    curvature = _curvature(train)
    finite_efficiency = efficiency.replace([np.inf, -np.inf], np.nan).dropna()
    median_efficiency = float(finite_efficiency.median()) if not finite_efficiency.empty else 0.0
    efficiency_std = float(finite_efficiency.std(ddof=0)) if len(finite_efficiency) > 1 else 0.0
    latency_values = latency.replace([np.inf, -np.inf], np.nan).dropna()
    high_throughput_latency = latency[throughput > throughput.max() * 0.9].replace([np.inf, -np.inf], np.nan).dropna()
    latency_baseline = float(high_throughput_latency.median()) if not high_throughput_latency.empty else float(latency_values.median()) if not latency_values.empty else None
    curvature_threshold = float(curvature.quantile(0.75)) if (curvature > 0).any() else None
    rules = {"failure": "errored_requests > 0 or successful_requests == 0", "efficiency": "efficiency z-score < -1.5", "conservative_margin": "efficiency < 0.85 * train median", "latency": "latency > 3 * train high-throughput baseline when available", "knee": "local throughput second-difference > train 75th percentile", "micro_family": "aggregate benchmark rows only; per-turn windows excluded", "train_aggregate_rows": len(train), "train_rows_excluded": original_train_rows - len(train), "train_median_efficiency": median_efficiency, "train_efficiency_std": efficiency_std, "train_latency_baseline": latency_baseline, "train_curvature_threshold": curvature_threshold, "knee_observations": int((curvature > 0).sum())}

    def apply(frame: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        result = frame.copy()
        current_throughput = _throughput(frame)
        current_efficiency = current_throughput / pd.to_numeric(_column(frame, "tp", 1), errors="coerce").fillna(1).clip(lower=1)
        current_latency = _latency(frame)
        errors = pd.to_numeric(_column(frame, "errored_requests", 0), errors="coerce").fillna(0) > 0
        successful = pd.to_numeric(_column(frame, "successful_requests"), errors="coerce")
        failure = errors | (successful.notna() & (successful <= 0)) | current_throughput.isna()
        z_score = (current_efficiency - median_efficiency) / max(efficiency_std, 1e-12)
        flags = {"failure": failure, "efficiency": z_score < -1.5, "conservative_margin": current_efficiency < 0.85 * median_efficiency, "latency": current_latency > 3 * latency_baseline if latency_baseline is not None else pd.Series(False, index=frame.index)}
        # Curvature is computed within each split, but its threshold is fitted only
        # from training rows. This labels held-out points using the same rule without
        # allowing held-out data to tune the threshold.
        local_curvature = _curvature(frame)
        flags["knee"] = local_curvature > curvature_threshold if curvature_threshold is not None else pd.Series(False, index=frame.index)
        result["label_failure"] = flags["failure"].astype(int)
        result["label_efficiency"] = flags["efficiency"].astype(int)
        result["label_conservative_margin"] = flags["conservative_margin"].astype(int)
        result["label_latency"] = flags["latency"].astype(int)
        result["label_knee"] = flags["knee"].astype(int)
        result["within_tested_region"] = (~pd.concat(flags, axis=1).any(axis=1)).astype(int)
        return result

    return {"train": apply(train, True), "validation": apply(validation), "test": apply(test)}, rules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/tested-models/dataset-manifest.json"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/tested-models/stage5/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tested-models/stage5/labeled"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rules = {}
    for source in read_manifest(args.manifest):
        pair = str(source["pair_id"].iloc[0])
        parts = {split: pd.read_parquet(args.split_dir / f"{pair}.{split}.parquet") for split in ("train", "validation", "test")}
        try:
            labeled, pair_rules = label_pair(**parts)
        except ValueError as error:
            # A pair whose aggregate rows all landed in a held-out split cannot
            # fit an envelope. Record it and keep the rest of the run alive;
            # downstream stages treat a missing labeled file as skipped.
            if "no aggregate measurements" not in str(error):
                raise
            rules[pair] = {"status": "skipped_no_aggregate_rows", "reason": str(error)}
            # A previous run may have labeled this pair. Leaving those files
            # behind would let training fit an envelope the current split no
            # longer supports.
            for split in ("train", "validation", "test"):
                (args.output_dir / f"{pair}.{split}.parquet").unlink(missing_ok=True)
            continue
        rules[pair] = pair_rules
        for split, frame in labeled.items():
            frame.to_parquet(args.output_dir / f"{pair}.{split}.parquet", index=False)
    (args.output_dir / "label-rules.json").write_text(json.dumps(rules, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote labels for {len(rules)} pairs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

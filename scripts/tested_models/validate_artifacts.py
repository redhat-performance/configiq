#!/usr/bin/env python3
"""Validate a tested-model registry and its Triton artifact repository.

Registry schema v2 carries measurements even when a pair has no validated
classifier. Validation therefore checks two different contracts:

* ``validated`` pairs must pass the recorded quality gate and have complete
  Triton artifacts.
* ``measured_no_envelope`` pairs must not expose classifier artifacts, so the
  API can serve their graphs without emitting a verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VALIDATED = "validated"
MEASURED_NO_ENVELOPE = "measured_no_envelope"
SUPPORTED_STATUSES = {VALIDATED, MEASURED_NO_ENVELOPE}


def _gate_passes(pair: dict, gate: dict) -> bool:
    coverage = pair.get("coverage", {})
    test = (pair.get("metrics") or {}).get("test") or {}
    auc = test.get("roc_auc")
    f1 = test.get("f1")
    return (
        coverage.get("records", 0) >= gate["minRecords"]
        and coverage.get("concurrencyLevelCount", 0) >= gate["minConcurrencyLevels"]
        and auc is not None
        and auc >= gate["minTestAuc"]
        and f1 is not None
        and f1 >= gate["minTestF1"]
    )


def _validate_v2(registry: dict, triton_dir: Path) -> set[str]:
    gate = registry.get("gate")
    if not isinstance(gate, dict):
        raise TypeError("registry v2 is missing its validation gate")
    for key in ("minRecords", "minTestAuc", "minTestF1", "minConcurrencyLevels"):
        if not isinstance(gate.get(key), (int, float)):
            raise TypeError(f"registry validation gate is missing {key}")

    approved = registry.get("approvedModels")
    if not isinstance(approved, list) or not all(isinstance(model, str) for model in approved):
        raise ValueError("registry approvedModels must be a string array")

    pairs = registry.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("registry contains no measured pairs")

    ids = set()
    validated = 0
    for pair in pairs:
        pair_id = pair.get("id")
        if not isinstance(pair_id, str) or pair_id in ids:
            raise ValueError(f"invalid or duplicate pair id: {pair_id}")
        ids.add(pair_id)

        status = pair.get("status")
        if status not in SUPPORTED_STATUSES:
            raise ValueError(f"unsupported pair status for {pair_id}: {status}")
        coverage = pair.get("coverage")
        if not isinstance(coverage, dict):
            raise TypeError(f"missing coverage for {pair_id}")
        for key in ("records", "concurrencyLevelCount"):
            if not isinstance(coverage.get(key), int):
                raise TypeError(f"invalid coverage {key} for {pair_id}")

        if status == MEASURED_NO_ENVELOPE:
            leaked = [key for key in ("classifier", "tritonModel", "featureSchema") if key in pair]
            if leaked:
                raise ValueError(f"measured-only pair exposes classifier artifacts: {pair_id} ({leaked})")
            continue

        validated += 1
        if not _gate_passes(pair, gate):
            raise ValueError(f"validated pair does not pass the registry gate: {pair_id}")
        if pair.get("classifier") != "xgboost":
            raise ValueError(f"unsupported classifier: {pair_id}")
        model_dir = triton_dir / pair_id
        config_path = model_dir / "config.pbtxt"
        model_path = model_dir / "1" / "xgboost.json"
        for path in (config_path, model_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        config = config_path.read_text()
        feature_count = len(pair.get("featureSchema", {}).get("feature_names", []))
        if 'backend: "fil"' not in config or "is_classifier" not in config:
            raise ValueError(f"invalid FIL config: {config_path}")
        if f"dims: [ {feature_count} ]" not in config:
            raise ValueError(f"FIL input dimension does not match feature schema: {config_path}")

    print(f"validated {validated} classifier artifacts, {len(ids) - validated} measured-only pairs")
    return ids


def _validate_v1(registry: dict, triton_dir: Path) -> set[str]:
    pairs = registry.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("registry contains no classifier pairs")

    ids = set()
    for pair in pairs:
        pair_id = pair.get("id")
        if not isinstance(pair_id, str) or pair_id in ids:
            raise ValueError(f"invalid or duplicate pair id: {pair_id}")
        ids.add(pair_id)
        if pair.get("classifier") != "xgboost":
            raise ValueError(f"unsupported classifier: {pair_id}")
        model_dir = triton_dir / pair_id
        config_path = model_dir / "config.pbtxt"
        model_path = model_dir / "1" / "xgboost.json"
        for path in (config_path, model_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        config = config_path.read_text()
        feature_count = len(pair.get("featureSchema", {}).get("feature_names", []))
        if 'backend: "fil"' not in config or "is_classifier" not in config:
            raise ValueError(f"invalid FIL config: {config_path}")
        if f"dims: [ {feature_count} ]" not in config:
            raise ValueError(f"FIL input dimension does not match feature schema: {config_path}")

    print(f"validated {len(ids)} legacy schema-v1 classifier artifacts")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("data/tested-models/stage6/registry/registry.json"))
    parser.add_argument("--triton-dir", type=Path, default=Path("data/tested-models/stage6/registry/triton"))
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text())
    schema_version = registry.get("schemaVersion")
    if schema_version == 2:
        _validate_v2(registry, args.triton_dir)
    elif schema_version == 1:
        _validate_v1(registry, args.triton_dir)
    else:
        raise ValueError("unsupported registry schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

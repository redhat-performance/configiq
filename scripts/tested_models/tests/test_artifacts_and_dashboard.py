from __future__ import annotations

import json
import sys

import build_dashboard
import build_registry
import pandas as pd
import pytest
import validate_artifacts


def _write_labeled_pair(labeled_dir, pair: str, *, records: int = 600) -> None:
    labeled_dir.mkdir(parents=True, exist_ok=True)
    levels = (1, 2, 4, 8, 16)
    frame = pd.DataFrame(
        {
            "concurrency": [levels[index % len(levels)] for index in range(records)],
            "output_tok/sec": [float(index % 100) for index in range(records)],
            "request_latency_median": [float(index % 50) for index in range(records)],
            "label_knee": [index % 10 == 0 for index in range(records)],
            "within_tested_region": [index % 3 != 0 for index in range(records)],
            "backend_version": ["vLLM-0.10.2" for _ in range(records)],
        }
    )
    for split, part in (("train", frame.iloc[:400]), ("validation", frame.iloc[400:500]), ("test", frame.iloc[500:])):
        part.to_parquet(labeled_dir / f"{pair}.{split}.parquet", index=False)


def _write_artifacts(model_dir, triton_dir, dashboard_dir, pair: str, report_entry: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path = model_dir / "training-report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"pairs": {}}
    report["pairs"][pair] = report_entry
    report_path.write_text(json.dumps(report))
    (model_dir / f"{pair}.json").write_text("{}")
    (model_dir / f"{pair}.metadata.json").write_text("{}")
    (triton_dir / pair / "1").mkdir(parents=True, exist_ok=True)
    (triton_dir / pair / "1" / "xgboost.json").write_text("{}")
    (triton_dir / pair / "config.pbtxt").write_text(
        'backend: "fil"\ninput [ { dims: [ 1 ] } ]\nparameters: { key: "is_classifier" value: { string_value: "true" } }'
    )
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    (dashboard_dir / f"{pair}.json").write_text("{}")


def _run_validator(registry_dir) -> int:
    sys.argv = [
        "validate_artifacts.py",
        "--registry",
        str(registry_dir / "registry.json"),
        "--triton-dir",
        str(registry_dir / "triton"),
    ]
    return validate_artifacts.main()


def test_registry_v2_publishes_validated_pairs_and_keeps_weak_pairs_measured(tmp_path) -> None:
    validated_pair = "model__h200"
    weak_pair = "model__b200"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [
        {"id": validated_pair, "model_id": "model", "accelerator": "H200", "records": 600},
        {"id": weak_pair, "model_id": "model", "accelerator": "B200", "records": 600},
    ]}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"approvedModels": ["model", "approved-only"]}))
    labeled_dir = tmp_path / "labeled"
    _write_labeled_pair(labeled_dir, validated_pair)
    _write_labeled_pair(labeled_dir, weak_pair)
    model_dir = tmp_path / "models"
    triton_dir = tmp_path / "triton"
    dashboard_dir = tmp_path / "dashboard"
    _write_artifacts(model_dir, triton_dir, dashboard_dir, validated_pair, {
        "status": "trained",
        "model_format": "xgboost-json",
        "feature_schema": {"feature_names": ["tp"]},
        "validation": {"roc_auc": 0.91, "f1": 0.72},
        "test": {"roc_auc": 0.88, "f1": 0.66},
    })
    _write_artifacts(model_dir, triton_dir, dashboard_dir, weak_pair, {
        "status": "trained",
        "model_format": "xgboost-json",
        "feature_schema": {"feature_names": ["tp"]},
        "test": {"roc_auc": 0.27, "f1": 0.0},
    })

    registry_dir = tmp_path / "registry"
    registry = build_registry.build_registry(
        manifest, model_dir, triton_dir, registry_dir, dashboard_dir,
        labeled_dir=labeled_dir, config_path=config,
    )

    assert registry["schemaVersion"] == 2
    assert registry["gate"] == {
        "minRecords": 100,
        "minTestAuc": 0.70,
        "minTestF1": 0.30,
        "minConcurrencyLevels": 5,
    }
    assert registry["approvedModels"] == ["model", "approved-only"]
    by_id = {pair["id"]: pair for pair in registry["pairs"]}
    assert by_id[validated_pair]["status"] == "validated"
    assert by_id[validated_pair]["coverage"]["records"] == 600
    assert by_id[validated_pair]["coverage"]["concurrencyLevelCount"] == 5
    assert by_id[validated_pair]["coverage"]["backendVersions"] == ["vLLM-0.10.2"]
    assert by_id[validated_pair]["coverage"]["backends"] == ["vllm"]
    assert by_id[weak_pair]["status"] == "measured_no_envelope"
    assert by_id[weak_pair]["diagnostics"]["reasons"] == ["test_auc_below_minimum", "test_f1_below_minimum"]
    assert "classifier" not in by_id[weak_pair]
    assert (registry_dir / "triton" / validated_pair).is_dir()
    assert not (registry_dir / "triton" / weak_pair).exists()

    assert _run_validator(registry_dir) == 0

    # A validated pair stays strict about its Triton artifact.
    (registry_dir / "triton" / validated_pair / "config.pbtxt").unlink()
    with pytest.raises(FileNotFoundError):
        _run_validator(registry_dir)


def test_registry_keeps_pairs_that_were_never_trained(tmp_path) -> None:
    pair = "model__h200"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"id": pair, "model_id": "model", "accelerator": "H200", "records": 20}]}))
    labeled_dir = tmp_path / "labeled"
    _write_labeled_pair(labeled_dir, pair, records=20)
    model_dir = tmp_path / "models"
    triton_dir = tmp_path / "triton"
    dashboard_dir = tmp_path / "dashboard"
    _write_artifacts(model_dir, triton_dir, dashboard_dir, pair, {"status": "skipped_insufficient_groups"})

    registry_dir = tmp_path / "registry"
    registry = build_registry.build_registry(
        manifest, model_dir, triton_dir, registry_dir, dashboard_dir, labeled_dir=labeled_dir,
    )

    entry = registry["pairs"][0]
    assert entry["status"] == "measured_no_envelope"
    assert entry["diagnostics"]["trainingStatus"] == "skipped_insufficient_groups"
    assert entry["diagnostics"]["reasons"] == [
        "classifier_not_trained",
        "records_below_minimum",
        "test_auc_unavailable",
        "test_f1_unavailable",
    ]
    assert _run_validator(registry_dir) == 0


def test_registry_surfaces_split_warnings_and_selection_basis(tmp_path) -> None:
    pair = "model__h200"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"id": pair, "model_id": "model", "accelerator": "H200", "records": 120}]}))
    labeled_dir = tmp_path / "labeled"
    _write_labeled_pair(labeled_dir, pair, records=120)
    model_dir = tmp_path / "models"
    triton_dir = tmp_path / "triton"
    dashboard_dir = tmp_path / "dashboard"
    _write_artifacts(model_dir, triton_dir, dashboard_dir, pair, {
        "status": "trained",
        "warnings": ["validation_single_class", "test_single_class"],
        "selection_basis": "train_fallback",
        "test": {"roc_auc": None, "f1": 0.0},
    })

    registry_dir = tmp_path / "registry"
    registry = build_registry.build_registry(
        manifest, model_dir, triton_dir, registry_dir, dashboard_dir, labeled_dir=labeled_dir,
    )

    entry = registry["pairs"][0]
    assert entry["status"] == "measured_no_envelope"
    assert entry["diagnostics"]["warnings"] == ["validation_single_class", "test_single_class"]
    assert entry["diagnostics"]["selectionBasis"] == "train_fallback"
    # reasons mirror the gate only; split warnings are reported separately.
    assert entry["diagnostics"]["reasons"] == ["test_auc_unavailable", "test_f1_below_minimum"]
    assert _run_validator(registry_dir) == 0


def test_validated_pair_keeps_split_warnings_out_of_gate_reasons(tmp_path) -> None:
    pair = "model__h200"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"id": pair, "model_id": "model", "accelerator": "H200", "records": 600}]}))
    labeled_dir = tmp_path / "labeled"
    _write_labeled_pair(labeled_dir, pair)
    model_dir = tmp_path / "models"
    triton_dir = tmp_path / "triton"
    dashboard_dir = tmp_path / "dashboard"
    _write_artifacts(model_dir, triton_dir, dashboard_dir, pair, {
        "status": "trained",
        "model_format": "xgboost-json",
        "feature_schema": {"feature_names": ["tp"]},
        "warnings": ["validation_single_class"],
        "selection_basis": "train_fallback",
        "test": {"roc_auc": 0.88, "f1": 0.66},
    })

    registry_dir = tmp_path / "registry"
    registry = build_registry.build_registry(
        manifest, model_dir, triton_dir, registry_dir, dashboard_dir, labeled_dir=labeled_dir,
    )

    entry = registry["pairs"][0]
    assert entry["status"] == "validated"
    assert entry["gate"] == {"passed": True, "reasons": [], "thresholds": registry["gate"]}
    assert entry["diagnostics"]["warnings"] == ["validation_single_class"]
    assert entry["diagnostics"]["selectionBasis"] == "train_fallback"
    assert "reasons" not in entry["diagnostics"]
    assert _run_validator(registry_dir) == 0


def test_validator_rejects_measured_pairs_that_expose_a_classifier(tmp_path) -> None:
    pair = "model__h200"
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "registry.json").write_text(json.dumps({
        "schemaVersion": 2,
        "gate": {"minRecords": 100, "minTestAuc": 0.7, "minTestF1": 0.3, "minConcurrencyLevels": 5},
        "approvedModels": ["model"],
        "pairs": [{
            "id": pair,
            "modelId": "model",
            "systemId": "h200_sxm",
            "status": "measured_no_envelope",
            "coverage": {"records": 10, "concurrencyLevelCount": 1},
            "classifier": "xgboost",
        }],
    }))

    with pytest.raises(ValueError, match="exposes classifier artifacts"):
        _run_validator(registry_dir)


def test_dashboard_payload_contains_only_sanitized_derived_data(tmp_path, monkeypatch) -> None:
    pair = "model__h200"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"id": pair, "model_id": "model", "accelerator": "H200"}]}))
    labeled_dir = tmp_path / "labeled"
    model_dir = tmp_path / "models"
    labeled_dir.mkdir()
    model_dir.mkdir()
    levels = (1, 2, 4, 8, 16)
    rows = [
        {
            "concurrency": levels[index % len(levels)],
            "output_tok/sec": float(index % 100),
            "request_latency_median": float(index % 50),
            "label_knee": index % 10 == 0,
            "within_tested_region": index % 3 != 0,
            "runtime_args": "https://secret.invalid",
        }
        for index in range(120)
    ]
    frame = pd.DataFrame(rows)
    for split in ("train", "validation", "test"):
        frame.to_parquet(labeled_dir / f"{pair}.{split}.parquet", index=False)
    (model_dir / f"{pair}.metadata.json").write_text(
        json.dumps({
            "status": "trained",
            "feature_schema": {"feature_names": ["tp"]},
            "validation": {"roc_auc": 0.9, "f1": 0.75},
            "test": {"roc_auc": 0.88, "f1": 0.7},
        })
    )
    (model_dir / f"{pair}.json").write_text("{}")

    class FakeModel:
        feature_importances_: list[float]

        def __init__(self):
            self.feature_importances_ = [1.0]

        def load_model(self, path):
            assert path == model_dir / f"{pair}.json"

    monkeypatch.setattr(build_dashboard, "XGBClassifier", FakeModel)
    output_dir = tmp_path / "dashboard"
    assert build_dashboard.build_dashboard(manifest, labeled_dir, model_dir, output_dir) == 1

    payload = json.loads((output_dir / f"{pair}.json").read_text())
    serialized = json.dumps(payload)
    assert "runtime_args" not in serialized
    assert "secret.invalid" not in serialized
    assert payload["records"] == 360
    assert payload["hasEnvelope"] is True
    assert payload["labelDistribution"] == {"within": 240, "outside": 120}
    assert payload["featureImportance"] == [{"feature": "tp", "importance": 1.0}]
    assert payload["test"] == {"roc_auc": 0.88, "f1": 0.7}


def test_dashboard_hides_classifier_evidence_for_gate_failing_pairs(tmp_path, monkeypatch) -> None:
    pair = "model__h200"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"id": pair, "model_id": "model", "accelerator": "H200", "records": 600}]}))
    labeled_dir = tmp_path / "labeled"
    model_dir = tmp_path / "models"
    _write_labeled_pair(labeled_dir, pair)
    model_dir.mkdir(parents=True)
    (model_dir / f"{pair}.metadata.json").write_text(
        json.dumps({
            "status": "trained",
            "feature_schema": {"feature_names": ["tp"]},
            "validation": {"roc_auc": 0.55, "f1": 0.2},
            "test": {"roc_auc": 0.52, "f1": 0.18},
        })
    )
    (model_dir / f"{pair}.json").write_text("{}")

    class _UnexpectedModel:
        def __init__(self) -> None:
            raise AssertionError("a gate-failing pair must not load its classifier")

    monkeypatch.setattr(build_dashboard, "XGBClassifier", _UnexpectedModel)
    output_dir = tmp_path / "dashboard"
    assert build_dashboard.build_dashboard(manifest, labeled_dir, model_dir, output_dir) == 1

    payload = json.loads((output_dir / f"{pair}.json").read_text())
    assert payload["hasEnvelope"] is False
    assert "validation" not in payload
    assert "test" not in payload
    assert payload["featureImportance"] == []
    assert payload["throughputLatencyByConcurrency"]


def test_dashboard_keeps_measurements_when_no_classifier_was_trained(tmp_path) -> None:
    pair = "model__h200"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"id": pair, "model_id": "model", "accelerator": "H200"}]}))
    labeled_dir = tmp_path / "labeled"
    model_dir = tmp_path / "models"
    labeled_dir.mkdir()
    model_dir.mkdir()
    frame = pd.DataFrame([{"concurrency": 1, "within_tested_region": 1}])
    for split in ("train", "validation", "test"):
        frame.to_parquet(labeled_dir / f"{pair}.{split}.parquet", index=False)
    (model_dir / f"{pair}.metadata.json").write_text(json.dumps({"status": "skipped_single_class_train"}))

    output_dir = tmp_path / "dashboard"
    assert build_dashboard.build_dashboard(manifest, labeled_dir, model_dir, output_dir) == 1

    payload = json.loads((output_dir / f"{pair}.json").read_text())
    assert payload["hasEnvelope"] is False
    assert payload["trainingStatus"] == "skipped_single_class_train"
    assert payload["featureImportance"] == []
    assert payload["records"] == 3


def test_gate_rejects_non_finite_metrics() -> None:
    coverage = {
        "records": 600,
        "concurrencyLevels": [1, 2, 4, 8, 16],
        "concurrencyLevelCount": 5,
        "backendVersions": ["vLLM-0.10.2"],
    }
    for non_finite in (float("nan"), float("inf"), float("-inf")):
        gate = build_registry.evaluate_gate(
            {"status": "trained", "test": {"roc_auc": non_finite, "f1": 0.9}},
            coverage,
        )
        assert gate["passed"] is False
        assert "test_auc_unavailable" in gate["reasons"]

        gate = build_registry.evaluate_gate(
            {"status": "trained", "test": {"roc_auc": 0.9, "f1": non_finite}},
            coverage,
        )
        assert gate["passed"] is False
        assert "test_f1_unavailable" in gate["reasons"]

from __future__ import annotations

import json
import sys

import label_data
import pandas as pd
import pytest
import train_classifiers


def test_labels_do_not_flag_latency_when_training_baseline_is_missing() -> None:
    train = pd.DataFrame([{"model_id": "model", "accelerator": "H200", "tp": 1, "isl": 1000, "osl": 1000, "concurrency": 1, "measured rps": 1.0, "output_tok/sec": 10}])
    validation = train.copy()
    test = train.copy()

    labeled, rules = label_data.label_pair(train, validation, test)

    assert rules["train_latency_baseline"] is None
    assert labeled["train"]["label_latency"].tolist() == [0]
    assert labeled["train"]["within_tested_region"].tolist() == [1]


def test_labels_exclude_per_turn_windows_from_every_split() -> None:
    aggregate = {
        "model_id": "model", "accelerator": "H200", "tp": 1,
        "isl": 1000, "osl": 1000, "turns": 1, "concurrency": 10,
        "measured rps": 2.0, "output_tok/sec": 2000.0,
    }
    per_turn = {
        **aggregate,
        "turns": 5,
        "measured rps": None,
        "output_tok/sec": 10.0,
    }
    frame = pd.DataFrame([aggregate, per_turn])

    labeled, rules = label_data.label_pair(frame, frame, frame)

    for split in ("train", "validation", "test"):
        assert labeled[split]["output_tok/sec"].tolist() == [2000.0]
    assert rules["train_aggregate_rows"] == 1
    assert rules["train_rows_excluded"] == 1
    assert rules["train_median_efficiency"] == 2000.0


def test_labels_reject_a_split_without_aggregate_measurements() -> None:
    frame = pd.DataFrame([{
        "model_id": "model", "accelerator": "H200", "tp": 1,
        "isl": 1000, "osl": 1000, "turns": 5, "concurrency": 10,
        "measured rps": None, "output_tok/sec": 10.0,
    }])

    with pytest.raises(ValueError, match="no aggregate measurements"):
        label_data.label_pair(frame, frame, frame)


def test_labeling_skips_a_pair_without_aggregate_rows_and_removes_stale_files(tmp_path, monkeypatch) -> None:
    pair = "model__h200"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"id": pair}]}))
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    frame = pd.DataFrame([{
        "model_id": "model", "accelerator": "H200", "tp": 1,
        "isl": 1000, "osl": 1000, "turns": 5, "concurrency": 10,
        "measured rps": None, "output_tok/sec": 10.0,
    }])
    for split in ("train", "validation", "test"):
        frame.to_parquet(split_dir / f"{pair}.{split}.parquet", index=False)
    output_dir = tmp_path / "labeled"
    output_dir.mkdir()
    stale = output_dir / f"{pair}.train.parquet"
    frame.to_parquet(stale, index=False)

    monkeypatch.setattr(label_data, "read_manifest", lambda _: [pd.DataFrame({"pair_id": [pair]})])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "label_data.py",
            "--manifest",
            str(manifest),
            "--split-dir",
            str(split_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert label_data.main() == 0
    rules = json.loads((output_dir / "label-rules.json").read_text())
    assert rules[pair]["status"] == "skipped_no_aggregate_rows"
    # A previous run must not leave a frame that training would still consume.
    assert not stale.exists()


def test_training_skips_a_pair_without_labeled_files_and_removes_a_stale_model(tmp_path, monkeypatch) -> None:
    pair = "model__h200"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"id": pair}]}))
    labeled_dir = tmp_path / "labeled"
    labeled_dir.mkdir()
    (labeled_dir / "label-rules.json").write_text(json.dumps({pair: {"status": "skipped_no_aggregate_rows"}}))
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    stale = [model_dir / f"{pair}{suffix}" for suffix in (".json", ".metadata.json", ".config.pbtxt")]
    for path in stale:
        path.write_text("stale")

    monkeypatch.setattr(train_classifiers, "read_manifest", lambda _: [pd.DataFrame({"pair_id": [pair]})])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_classifiers.py",
            "--manifest",
            str(manifest),
            "--labeled-dir",
            str(labeled_dir),
            "--output-dir",
            str(model_dir),
        ],
    )

    assert train_classifiers.main() == 0
    report = json.loads((model_dir / "training-report.json").read_text())
    assert report["pairs"][pair]["status"] == "skipped_no_aggregate_rows"
    assert not any(path.exists() for path in stale)


def test_training_treats_unexpected_missing_labeled_files_as_an_error(tmp_path, monkeypatch) -> None:
    pair = "model__h200"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"id": pair}]}))
    labeled_dir = tmp_path / "labeled"
    labeled_dir.mkdir()
    (labeled_dir / "label-rules.json").write_text(json.dumps({pair: {}}))
    frame = pd.DataFrame([{"within_tested_region": 1}])
    frame.to_parquet(labeled_dir / f"{pair}.train.parquet", index=False)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    stale = [model_dir / f"{pair}{suffix}" for suffix in (".json", ".metadata.json", ".config.pbtxt")]
    for path in stale:
        path.write_text("stale")

    monkeypatch.setattr(train_classifiers, "read_manifest", lambda _: [pd.DataFrame({"pair_id": [pair]})])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_classifiers.py",
            "--manifest",
            str(manifest),
            "--labeled-dir",
            str(labeled_dir),
            "--output-dir",
            str(model_dir),
        ],
    )

    assert train_classifiers.main() == 1
    report = json.loads((model_dir / "training-report.json").read_text())
    assert report["pairs"][pair]["status"] == "error_missing_labeled_rows"
    assert report["pairs"][pair]["missing_splits"] == ["validation", "test"]
    # An incomplete run is not a skip: the previous bundle stays in place.
    assert all(path.exists() for path in stale)


def test_training_reports_single_class_train_data_without_fitting(tmp_path, monkeypatch) -> None:
    pair = "model__h200"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"id": pair}]}))
    labeled_dir = tmp_path / "labeled"
    labeled_dir.mkdir()
    frame = pd.DataFrame([{"within_tested_region": 1, "model_id": "model", "accelerator": "H200"}])
    for split in ("train", "validation", "test"):
        frame.to_parquet(labeled_dir / f"{pair}.{split}.parquet", index=False)
    (labeled_dir / "label-rules.json").write_text(json.dumps({pair: {}}))

    monkeypatch.setattr(train_classifiers, "read_manifest", lambda _: [pd.DataFrame({"pair_id": [pair]})])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_classifiers.py",
            "--manifest",
            str(manifest),
            "--labeled-dir",
            str(labeled_dir),
            "--output-dir",
            str(tmp_path / "models"),
        ],
    )

    assert train_classifiers.main() == 0
    report = json.loads((tmp_path / "models" / "training-report.json").read_text())
    assert report["pairs"][pair]["status"] == "skipped_single_class_train"

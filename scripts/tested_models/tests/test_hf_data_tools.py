from __future__ import annotations

import json
import os
from pathlib import Path

import download_ground_truth
import pandas as pd
import publish_ground_truth
import pytest
from dataset_common import file_sha256


def test_download_rejects_non_immutable_revision(monkeypatch) -> None:
    monkeypatch.setattr(download_ground_truth, "snapshot_download", lambda **_: pytest.fail("must not download"))
    monkeypatch.setattr(download_ground_truth, "parse_args", lambda: type("Args", (), {
        "repo_id": "org/data", "revision": "main", "output_dir": Path("data")
    })())
    with pytest.raises(SystemExit, match="full 40-character"):
        download_ground_truth.main()


def test_download_rejects_unsafe_pair_id(tmp_path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "pairs": [{"id": "../escape", "path": "pairs/x/ground_truth.parquet"}],
    }))
    monkeypatch.setattr(download_ground_truth, "snapshot_download", lambda **_: str(snapshot))
    monkeypatch.setattr(download_ground_truth, "parse_args", lambda: type("Args", (), {
        "repo_id": "org/data", "revision": "a" * 40, "output_dir": tmp_path / "output"
    })())
    with pytest.raises(ValueError, match="filesystem pair id"):
        download_ground_truth.main()


def test_publish_rejects_output_input_overlap(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    manifest.write_text(json.dumps({"schema_version": 1, "pairs": []}))
    with pytest.raises(ValueError, match="overlap input paths"):
        publish_ground_truth.prepare(manifest, datasets, tmp_path)


def test_publish_rejects_empty_manifest_before_cleanup(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    datasets = tmp_path / "datasets"
    output = tmp_path / "output"
    datasets.mkdir()
    output.mkdir()
    marker = output / "must-survive"
    marker.write_text("keep")
    manifest.write_text(json.dumps({"schema_version": 1, "pairs": []}))

    with pytest.raises(ValueError, match="at least one pair"):
        publish_ground_truth.prepare(manifest, datasets, output)
    assert marker.read_text() == "keep"


def test_publish_copies_each_source_pair_into_output_manifest(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    datasets = tmp_path / "datasets"
    output = tmp_path / "output"
    datasets.mkdir()
    frame = pd.DataFrame([{"model_id": "org/model", "accelerator": "H200", "pair_id": "model__h200"}])
    frame.to_parquet(datasets / "model__h200.parquet", index=False)
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "pairs": [{
            "id": "model__h200",
            "model_id": "org/model",
            "accelerator": "H200",
            "records": 1,
            "columns": list(frame.columns),
        }],
    }))

    result = publish_ground_truth.prepare(manifest, datasets, output)

    assert result["pairs"][0]["id"] == "model__h200"
    assert result["pairs"][0]["file_sha256"] == file_sha256(output / "pairs/model__h200/ground_truth.parquet")
    assert (output / "README.md").is_file()
    assert (output / "schema.json").is_file()


def test_download_verifies_checksum_and_replaces_stale_datasets(tmp_path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    source = snapshot / "pairs/model__h200/ground_truth.parquet"
    source.parent.mkdir(parents=True)
    frame = pd.DataFrame([{"model_id": "org/model", "accelerator": "H200", "pair_id": "model__h200"}])
    frame.to_parquet(source, index=False)
    (snapshot / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "pairs": [{
            "id": "model__h200",
            "model_id": "org/model",
            "accelerator": "H200",
            "records": 1,
            "columns": list(frame.columns),
            "path": "pairs/model__h200/ground_truth.parquet",
            "file_sha256": file_sha256(source),
        }],
    }))
    output = tmp_path / "output"
    stale = output / "datasets/stale.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    monkeypatch.setattr(download_ground_truth, "snapshot_download", lambda **_: str(snapshot))
    monkeypatch.setattr(download_ground_truth, "parse_args", lambda: type("Args", (), {
        "repo_id": "org/data", "revision": "a" * 40, "output_dir": output
    })())

    assert download_ground_truth.main() == 0
    assert not stale.exists()
    assert (output / "datasets/model__h200.parquet").is_file()


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    source = snapshot / "pairs/model__h200/ground_truth.parquet"
    source.parent.mkdir(parents=True)
    frame = pd.DataFrame([{"model_id": "org/model", "accelerator": "H200", "pair_id": "model__h200"}])
    frame.to_parquet(source, index=False)
    (snapshot / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "pairs": [{
            "id": "model__h200",
            "path": "pairs/model__h200/ground_truth.parquet",
            "records": 1,
            "columns": list(frame.columns),
            "file_sha256": "0" * 64,
        }],
    }))
    monkeypatch.setattr(download_ground_truth, "snapshot_download", lambda **_: str(snapshot))
    monkeypatch.setattr(download_ground_truth, "parse_args", lambda: type("Args", (), {
        "repo_id": "org/data", "revision": "a" * 40, "output_dir": tmp_path / "output"
    })())

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_ground_truth.main()


def test_publish_validates_all_pairs_before_replacing_output(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    datasets = tmp_path / "datasets"
    output = tmp_path / "output"
    datasets.mkdir()
    output.mkdir()
    marker = output / "must-survive"
    marker.write_text("keep")
    frame = pd.DataFrame([{"model_id": "org/model", "accelerator": "H200", "pair_id": "model__h200"}])
    frame.to_parquet(datasets / "model__h200.parquet", index=False)
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "pairs": [
            {"id": "model__h200", "model_id": "org/model", "accelerator": "H200"},
            {"id": "missing__h200", "model_id": "org/missing", "accelerator": "H200"},
        ],
    }))

    with pytest.raises(FileNotFoundError):
        publish_ground_truth.prepare(manifest, datasets, output)
    assert marker.read_text() == "keep"


def test_download_rolls_back_dataset_when_manifest_commit_fails(tmp_path) -> None:
    datasets = tmp_path / "datasets"
    staging = tmp_path / "datasets.staging"
    manifest = tmp_path / "dataset-manifest.json"
    staged_manifest = tmp_path / ".dataset-manifest.json.staging"
    datasets.mkdir()
    staging.mkdir()
    (datasets / "old.parquet").write_bytes(b"old")
    (staging / "new.parquet").write_bytes(b"new")
    manifest.write_text("old manifest")
    staged_manifest.write_text("new manifest")

    def fail_new_manifest(source, destination):
        if Path(source) == staged_manifest:
            raise OSError("injected manifest commit failure")
        os.replace(source, destination)

    with pytest.raises(OSError, match="injected"):
        download_ground_truth._commit_dataset(
            staging,
            datasets,
            staged_manifest,
            manifest,
            replace_file=fail_new_manifest,
        )

    assert (datasets / "old.parquet").read_bytes() == b"old"
    assert not (datasets / "new.parquet").exists()
    assert manifest.read_text() == "old manifest"

from __future__ import annotations

import json
import os
from pathlib import Path

import download_ground_truth
import pandas as pd
import publish_ground_truth
import pytest
from dataset_common import file_sha256


def test_resolve_revision_accepts_immutable_sha_without_network(monkeypatch) -> None:
    monkeypatch.setattr(download_ground_truth, "HfApi", lambda **_: pytest.fail("must not resolve a SHA"))
    assert download_ground_truth.resolve_revision("org/data", "a" * 40) == "a" * 40


def test_resolve_revision_pins_mutable_ref_to_commit_sha(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeApi:
        def repo_info(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return type("Info", (), {"sha": "b" * 40})()

    monkeypatch.setattr(download_ground_truth, "HfApi", lambda **_: _FakeApi())
    assert download_ground_truth.resolve_revision("org/data", "main") == "b" * 40
    assert captured["repo_id"] == "org/data"
    assert captured["repo_type"] == "dataset"
    assert captured["revision"] == "main"


def test_resolve_revision_uses_default_branch_when_unset(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeApi:
        def repo_info(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return type("Info", (), {"sha": "c" * 40})()

    monkeypatch.setattr(download_ground_truth, "HfApi", lambda **_: _FakeApi())
    assert download_ground_truth.resolve_revision("org/data", None) == "c" * 40
    assert captured["revision"] is None


def test_resolve_revision_rejects_unresolvable_sha(monkeypatch) -> None:
    class _FakeApi:
        def repo_info(self, **_: object) -> object:
            return type("Info", (), {"sha": "not-a-sha"})()

    monkeypatch.setattr(download_ground_truth, "HfApi", lambda **_: _FakeApi())
    with pytest.raises(SystemExit, match="full 40-character"):
        download_ground_truth.resolve_revision("org/data", "main")


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


def test_resolve_repo_id_prefers_explicit_flag(monkeypatch) -> None:
    monkeypatch.setenv(download_ground_truth.DATASET_REPO_ENV, "org/from-environment")

    assert download_ground_truth.resolve_repo_id("org/from-flag") == "org/from-flag"


def test_resolve_repo_id_falls_back_to_environment(monkeypatch) -> None:
    monkeypatch.setenv(download_ground_truth.DATASET_REPO_ENV, "org/from-environment")

    assert download_ground_truth.resolve_repo_id(None) == "org/from-environment"


def test_resolve_repo_id_ignores_blank_configuration(monkeypatch) -> None:
    # GitHub renders an unset repository variable as an empty string.
    monkeypatch.setenv(download_ground_truth.DATASET_REPO_ENV, "")

    assert download_ground_truth.resolve_repo_id(None) == download_ground_truth.DEFAULT_DATASET_REPO
    assert download_ground_truth.resolve_repo_id("   ") == download_ground_truth.DEFAULT_DATASET_REPO


def test_resolve_repo_id_defaults_to_upstream_dataset(monkeypatch) -> None:
    monkeypatch.delenv(download_ground_truth.DATASET_REPO_ENV, raising=False)

    assert download_ground_truth.resolve_repo_id(None) == "redhat-performance/configiq-performance-upstream-data"


def test_main_downloads_from_resolved_repo_id(tmp_path, monkeypatch) -> None:
    """A stale hardcoded repository ID broke every scheduled build for weeks."""
    captured: dict[str, object] = {}

    class _StopDownload(Exception):
        pass

    def fake_download(**kwargs: object) -> str:
        captured.update(kwargs)
        raise _StopDownload

    monkeypatch.setenv(download_ground_truth.DATASET_REPO_ENV, "org/from-environment")
    monkeypatch.setattr(download_ground_truth, "snapshot_download", fake_download)
    monkeypatch.setattr(download_ground_truth, "parse_args", lambda: type("Args", (), {
        "repo_id": None, "revision": "a" * 40, "output_dir": tmp_path / "output"
    })())

    with pytest.raises(_StopDownload):
        download_ground_truth.main()

    assert captured["repo_id"] == "org/from-environment"
    assert captured["token"] is False


def test_publish_uploads_to_resolved_repo_id(monkeypatch) -> None:
    """A stale hardcoded repository ID would publish revisions the downloader cannot find."""
    captured: dict[str, object] = {}

    class _StopPublish(Exception):
        pass

    class _FakeApi:
        def repo_info(self, repo_id: str, **_: object) -> None:
            captured["repo_id"] = repo_id
            raise _StopPublish

    monkeypatch.setenv(publish_ground_truth.DATASET_REPO_ENV, "org/from-environment")
    monkeypatch.setattr(publish_ground_truth, "HfApi", _FakeApi)
    monkeypatch.setattr(publish_ground_truth, "prepare", lambda *_: {"pairs": [{"records": 1}]})
    monkeypatch.setattr(publish_ground_truth, "parse_args", lambda: type("Args", (), {
        "manifest": Path("manifest.json"),
        "datasets_dir": Path("datasets"),
        "output_dir": Path("output"),
        "repo_id": None,
        "revision": None,
    })())

    with pytest.raises(_StopPublish):
        publish_ground_truth.main()

    assert captured["repo_id"] == "org/from-environment"


def test_publish_writes_workflow_outputs(tmp_path) -> None:
    output = tmp_path / "github-output"
    publish_ground_truth.write_github_output(
        output, changed="true", revision="a" * 40, check_only="false",
    )
    assert output.read_text().splitlines() == [
        "changed=true",
        f"revision={'a' * 40}",
        "check_only=false",
    ]


def test_publish_defaults_to_upstream_dataset(monkeypatch) -> None:
    monkeypatch.delenv(publish_ground_truth.DATASET_REPO_ENV, raising=False)

    assert publish_ground_truth.resolve_dataset_repo_id(None) == "redhat-performance/configiq-performance-upstream-data"


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

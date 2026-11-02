from __future__ import annotations

import csv
import json
from pathlib import Path

import import_s3_ground_truth
import publish_ground_truth
import pytest
from dataset_common import _backend_name


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=["model", "accelerator", "version"])
        writer.writeheader()
        writer.writerows(rows)


def _write_config(path: Path, models: list[str]) -> None:
    path.write_text(json.dumps({"testedModels": models}))


def test_import_keeps_only_plain_upstream_vllm_versions(tmp_path, capsys) -> None:
    csv_path = tmp_path / "aggregate.csv"
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "datasets"
    manifest_path = tmp_path / "dataset-manifest.json"
    _write_config(config_path, ["org/model"])
    _write_csv(csv_path, [
        {"model": "org/model", "accelerator": "H200", "version": "vLLM-0.10.2"},
        {"model": "org/model", "accelerator": "H200", "version": "vLLM-0.10.2-rc1"},
        {"model": "org/model", "accelerator": "H200", "version": "RHAIIS-3.2-vLLM-0.9.0"},
        {"model": "org/model", "accelerator": "B200", "version": ""},
        {"model": "org/model", "accelerator": "B200", "version": "vLLM-0.11.0"},
        {"model": "org/other", "accelerator": "H200", "version": "vLLM-0.10.2"},
    ])

    manifest = import_s3_ground_truth.import_csv(csv_path, config_path, output_dir, manifest_path)

    assert sorted(pair["id"] for pair in manifest["pairs"]) == ["org_model__b200", "org_model__h200"]
    assert manifest["approved_rows"] == 5
    assert manifest["upstream_rows"] == 2
    assert manifest["dropped_version_rows"] == 3
    assert manifest["dropped_versions"] == {
        "<blank>": 1,
        "RHAIIS-3.2-vLLM-0.9.0": 1,
        "vLLM-0.10.2-rc1": 1,
    }
    output = capsys.readouterr().out
    assert "Prepared 2 upstream rows" in output
    assert "Dropped 3 approved rows" in output


def test_import_fails_when_the_filter_removes_every_row(tmp_path) -> None:
    csv_path = tmp_path / "aggregate.csv"
    config_path = tmp_path / "config.json"
    _write_config(config_path, ["org/model"])
    _write_csv(csv_path, [
        {"model": "org/model", "accelerator": "H200", "version": "RHAIIS-3.2-vLLM-0.9.0"},
    ])

    with pytest.raises(ValueError, match="no approved model and accelerator pairs"):
        import_s3_ground_truth.import_csv(
            csv_path, config_path, tmp_path / "datasets", tmp_path / "dataset-manifest.json"
        )


def test_backend_name_prefers_productized_stack_labels() -> None:
    assert _backend_name("RHAIIS-3.2-vLLM-0.9.0") == "rhaiis"
    assert _backend_name("vLLM-0.10.2") == "vllm"


def test_missing_remote_pairs_lists_pairs_the_upload_would_delete() -> None:
    local = {"pairs": [{"id": "keep__h200"}]}
    remote = {"pairs": [{"id": "keep__h200"}, {"id": "gone__b200"}]}

    assert publish_ground_truth._missing_remote_pairs(local, remote) == ["gone__b200"]


def _pair(pair_id: str) -> dict[str, object]:
    return {"id": pair_id, "records": 1, "columns": [], "data_sha256": pair_id}


def test_publish_refuses_to_drop_remote_pairs(tmp_path, monkeypatch) -> None:
    remote_manifest = tmp_path / "manifest.json"
    remote_manifest.write_text(json.dumps({"pairs": [_pair("keep__h200"), _pair("gone__b200")]}))

    class _FakeApi:
        def repo_info(self, *_: object, **__: object) -> object:
            return type("Info", (), {"sha": "a" * 40})()

        def upload_folder(self, **_: object) -> object:
            raise AssertionError("an incomplete publish must not upload")

    monkeypatch.setattr(publish_ground_truth, "prepare", lambda *_: {"pairs": [_pair("keep__h200")]})
    monkeypatch.setattr(publish_ground_truth, "HfApi", _FakeApi)
    monkeypatch.setattr(publish_ground_truth, "hf_hub_download", lambda **_: str(remote_manifest))
    monkeypatch.setattr(publish_ground_truth, "parse_args", lambda: type("Args", (), {
        "manifest": Path("manifest.json"),
        "datasets_dir": Path("datasets"),
        "output_dir": Path("output"),
        "repo_id": "org/data",
        "revision": None,
        "check_only": False,
        "allow_removal": False,
    })())

    with pytest.raises(SystemExit, match="would remove pairs"):
        publish_ground_truth.main()


def test_publish_allows_an_explicit_removal(tmp_path, monkeypatch) -> None:
    remote_manifest = tmp_path / "manifest.json"
    remote_manifest.write_text(json.dumps({"pairs": [_pair("keep__h200"), _pair("gone__b200")]}))
    uploaded: dict[str, object] = {}

    class _FakeApi:
        def repo_info(self, *_: object, **__: object) -> object:
            return type("Info", (), {"sha": "a" * 40})()

        def upload_folder(self, **kwargs: object) -> object:
            uploaded.update(kwargs)
            return type("Commit", (), {"oid": "b" * 40})()

    monkeypatch.setattr(publish_ground_truth, "prepare", lambda *_: {"pairs": [_pair("keep__h200")]})
    monkeypatch.setattr(publish_ground_truth, "HfApi", _FakeApi)
    monkeypatch.setattr(publish_ground_truth, "hf_hub_download", lambda **_: str(remote_manifest))
    monkeypatch.setattr(publish_ground_truth, "parse_args", lambda: type("Args", (), {
        "manifest": Path("manifest.json"),
        "datasets_dir": Path("datasets"),
        "output_dir": Path("output"),
        "repo_id": "org/data",
        "revision": None,
        "check_only": False,
        "allow_removal": True,
    })())

    assert publish_ground_truth.main() == 0
    assert uploaded["repo_id"] == "org/data"

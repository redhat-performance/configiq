from __future__ import annotations

import csv
import json
from pathlib import Path

import import_s3_ground_truth
import pandas as pd
import publish_ground_truth
import pytest


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=["model", "accelerator", "version"])
        writer.writeheader()
        writer.writerows(rows)


def test_import_keeps_only_tested_upstream_vllm_rows(tmp_path) -> None:
    csv_path = tmp_path / "aggregate.csv"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"testedModels": ["org/model"]}))
    _write_csv(csv_path, [
        {"model": "org/model", "accelerator": "H200", "version": "vLLM-0.24.0"},
        {"model": "org/model", "accelerator": "H200", "version": "RHAIIS-3.5-EA2"},
        {"model": "org/model", "accelerator": "H200", "version": "vLLM-0.24.0-rerun"},
        {"model": "org/other", "accelerator": "H200", "version": "vLLM-0.24.0"},
    ])

    manifest = import_s3_ground_truth.import_csv(
        csv_path, config_path, tmp_path / "datasets", tmp_path / "manifest.json"
    )

    assert manifest["source_rows"] == 4
    assert manifest["approved_rows"] == 1
    assert manifest["excluded_version_rows"] == 2
    assert len(manifest["pairs"]) == 1
    frame = pd.read_parquet(tmp_path / "datasets" / f"{manifest['pairs'][0]['id']}.parquet")
    assert frame["version"].tolist() == ["vLLM-0.24.0"]
    assert frame["backend"].tolist() == ["vllm"]


def test_import_rejects_colliding_pair_ids_before_writing(tmp_path) -> None:
    csv_path = tmp_path / "aggregate.csv"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"testedModels": ["org/model"]}))
    _write_csv(csv_path, [
        {"model": "org/model", "accelerator": "H200 SXM", "version": "vLLM-0.24.0"},
        {"model": "org/model", "accelerator": "H200-SXM", "version": "vLLM-0.24.0"},
    ])
    output_dir = tmp_path / "datasets"

    with pytest.raises(ValueError, match="same ID"):
        import_s3_ground_truth.import_csv(csv_path, config_path, output_dir, tmp_path / "manifest.json")
    assert not output_dir.exists()


def test_publish_rejects_missing_remote_pairs(tmp_path, monkeypatch) -> None:
    remote_manifest = tmp_path / "remote-manifest.json"
    remote_manifest.write_text(json.dumps({
        "pairs": [
            {"id": "keep", "records": 1, "columns": [], "data_sha256": "a"},
            {"id": "missing", "records": 1, "columns": [], "data_sha256": "b"},
        ],
    }))

    class FakeApi:
        def repo_info(self, *_args, **_kwargs):
            return type("Info", (), {"sha": "a" * 40})()

        def upload_folder(self, **_kwargs):
            pytest.fail("an incomplete aggregate must not be uploaded")

    monkeypatch.setattr(publish_ground_truth, "prepare", lambda *_args: {
        "pairs": [{"id": "keep", "records": 1, "columns": [], "data_sha256": "a"}],
    })
    monkeypatch.setattr(publish_ground_truth, "HfApi", FakeApi)
    monkeypatch.setattr(publish_ground_truth, "hf_hub_download", lambda **_kwargs: str(remote_manifest))
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

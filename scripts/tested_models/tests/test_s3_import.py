from __future__ import annotations

import csv
import json
from pathlib import Path

import import_s3_ground_truth
import pandas as pd
import publish_ground_truth
import pytest
from dataset_common import _backend_name


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def _write_config(path: Path, models: list[str]) -> None:
    path.write_text(json.dumps({"approvedModels": models}))


def test_import_keeps_only_plain_upstream_vllm_versions(tmp_path, capsys) -> None:
    csv_path = tmp_path / "aggregate.csv"
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "datasets"
    manifest_path = tmp_path / "dataset-manifest.json"
    _write_config(config_path, ["org/model"])
    _write_csv(csv_path, [
        {"model": "org/model", "accelerator": "H200", "version": "vLLM-0.10.2", "intended concurrency": "1"},
        {"model": "org/model", "accelerator": "H200", "version": "vllm-0.6.2", "intended concurrency": "1"},
        {"model": "org/model", "accelerator": "H200", "version": "vLLM-0.10.2-rc1", "intended concurrency": "1"},
        {"model": "org/model", "accelerator": "H200", "version": "RHAIIS-3.2-vLLM-0.9.0", "intended concurrency": "1"},
        {"model": "org/model", "accelerator": "B200", "version": "", "intended concurrency": "1"},
        {"model": "org/model", "accelerator": "B200", "version": "vLLM-0.11.0", "intended concurrency": "1"},
        {"model": "org/other", "accelerator": "H200", "version": "vLLM-0.10.2", "intended concurrency": "1"},
    ])

    manifest = import_s3_ground_truth.import_csv(csv_path, config_path, output_dir, manifest_path)

    assert sorted(pair["id"] for pair in manifest["pairs"]) == ["org_model__b200", "org_model__h200"]
    assert manifest["approved_rows"] == 6
    assert manifest["upstream_rows"] == 3
    assert manifest["dropped_version_rows"] == 3
    assert manifest["dropped_versions"] == {
        "<blank>": 1,
        "RHAIIS-3.2-vLLM-0.9.0": 1,
        "vLLM-0.10.2-rc1": 1,
    }
    output = capsys.readouterr().out
    assert "Prepared 3 upstream summary rows" in output
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


def test_import_rejects_colliding_pair_ids_before_writing(tmp_path) -> None:
    csv_path = tmp_path / "aggregate.csv"
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "datasets"
    manifest_path = tmp_path / "dataset-manifest.json"
    _write_config(config_path, ["org/model"])
    _write_csv(csv_path, [
        {"model": "org/model", "accelerator": "H200 SXM", "version": "vLLM-0.24.0"},
        {"model": "org/model", "accelerator": "H200-SXM", "version": "vLLM-0.24.0"},
    ])

    with pytest.raises(ValueError, match="same ID"):
        import_s3_ground_truth.import_csv(csv_path, config_path, output_dir, manifest_path)
    assert not output_dir.exists()
    assert not manifest_path.exists()


def test_import_keeps_newest_summary_per_recorded_configuration(tmp_path) -> None:
    csv_path = tmp_path / "aggregate.csv"
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "datasets"
    _write_config(config_path, ["org/model"])
    base = {
        "model": "org/model", "accelerator": "H200", "version": "vLLM-0.31.0",
        "TP": "1", "prompt toks": "8192", "output toks": "1024",
        "prefix_caching": "yes", "runtime_args": "max-num-batched-tokens=8192",
        "env_vars": "",
        "turns": "5", "request_type": "/v1/chat/completions",
    }
    older = {**base, "uuid": "older", "guidellm_start_time_ms": "1000", "intended concurrency": "1", "ttft_median": "100", "turn_index": ""}
    newer = {**base, "uuid": "newer", "guidellm_start_time_ms": "2000", "intended concurrency": "1", "ttft_median": "80", "turn_index": ""}
    next_concurrency = {**base, "uuid": "newer", "guidellm_start_time_ms": "2000", "intended concurrency": "2", "ttft_median": "90", "turn_index": ""}
    turn_detail = {**newer, "turn_index": "0", "ttft_median": "40"}
    older_detail = {**older, "turn_index": "0", "ttft_median": "20"}
    hidden_variant = {**newer, "uuid": "variant", "guidellm_start_time_ms": "1500", "env_vars": '{"UNRELATED":"1"}'}
    aiter_variant = {**newer, "uuid": "aiter", "guidellm_start_time_ms": "3000", "env_vars": '{"VLLM_ROCM_USE_AITER":"1"}'}
    _write_csv(csv_path, [older, older_detail, newer, newer, next_concurrency, turn_detail, hidden_variant, aiter_variant])

    manifest = import_s3_ground_truth.import_csv(
        csv_path, config_path, output_dir, tmp_path / "manifest.json"
    )
    frame = pd.read_parquet(output_dir / "org_model__h200.parquet")

    assert manifest["upstream_rows"] == 8
    assert manifest["detail_rows_seen"] == 2
    assert manifest["detail_rows_dropped"] == 1
    assert manifest["detail_rows_kept"] == 1
    assert manifest["duplicate_source_rows_dropped"] == 1
    assert manifest["superseded_rows_dropped"] == 1
    assert manifest["public_duplicate_rows_dropped"] == 1
    assert manifest["published_rows"] == 4
    assert sorted(frame["ttft_median"].tolist()) == [40, 80, 80, 90]
    assert frame["max_num_batched_tokens"].tolist() == [8192, 8192, 8192, 8192]
    assert set(frame["vllm_rocm_use_aiter"].dropna()) == {"1"}
    identity = [field for field in import_s3_ground_truth.PUBLIC_CURVE_FIELDS if field in frame]
    assert not frame.duplicated([*identity, "concurrency", "turn_index"]).any()
    assert frame["turn_index"].notna().sum() == 1
    assert not {"uuid", "guidellm_start_time_ms", "runtime_args", "env_vars"} & set(frame)

    # A private source variant may disappear at the next refresh without
    # changing the published configuration fields.
    _write_csv(csv_path, [older, newer, next_concurrency])
    import_s3_ground_truth.import_csv(
        csv_path, config_path, tmp_path / "next-datasets", tmp_path / "next-manifest.json"
    )
    next_frame = pd.read_parquet(tmp_path / "next-datasets/org_model__h200.parquet")
    base_frame = frame[frame["vllm_rocm_use_aiter"].isna() & frame["turn_index"].isna()].reset_index(drop=True)
    assert [import_s3_ground_truth._public_curve_key(row) for row in next_frame.to_dict(orient="records")] == [
        import_s3_ground_truth._public_curve_key(row) for row in base_frame.to_dict(orient="records")
    ]


def test_import_refuses_to_choose_newest_without_start_time(tmp_path) -> None:
    csv_path = tmp_path / "aggregate.csv"
    config_path = tmp_path / "config.json"
    _write_config(config_path, ["org/model"])
    base = {"model": "org/model", "accelerator": "H200", "version": "vLLM-0.31.0", "intended concurrency": "1"}
    _write_csv(csv_path, [
        {**base, "uuid": "first", "ttft_median": "100"},
        {**base, "uuid": "second", "ttft_median": "80"},
    ])

    with pytest.raises(ValueError, match="valid GuideLLM start time"):
        import_s3_ground_truth.import_csv(
            csv_path, config_path, tmp_path / "datasets", tmp_path / "manifest.json"
        )


def test_import_keeps_distinct_parallel_serving_configurations(tmp_path) -> None:
    csv_path = tmp_path / "aggregate.csv"
    config_path = tmp_path / "config.json"
    _write_config(config_path, ["org/model"])
    base = {
        "model": "org/model", "accelerator": "H200", "version": "vLLM-0.31.0",
        "TP": "4", "prompt toks": "8192", "output toks": "1024",
        "intended concurrency": "8", "guidellm_start_time_ms": "1000", "turn_index": "",
    }
    _write_csv(csv_path, [
        {**base, "PP": "1", "TP_prefill": "2", "TP_decode": "4", "moe_tp_size": "2", "moe_ep_size": "2"},
        {**base, "PP": "2", "TP_prefill": "2", "TP_decode": "4", "moe_tp_size": "2", "moe_ep_size": "2"},
    ])

    manifest = import_s3_ground_truth.import_csv(
        csv_path, config_path, tmp_path / "datasets", tmp_path / "manifest.json"
    )
    frame = pd.read_parquet(tmp_path / "datasets/org_model__h200.parquet")

    assert manifest["published_rows"] == 2
    assert set(frame["pp"]) == {1, 2}
    assert set(frame["tp_prefill"]) == {2}
    assert set(frame["tp_decode"]) == {4}
    assert set(frame["moe_tp_size"]) == {2}
    assert set(frame["moe_ep_size"]) == {2}


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


def test_publish_refuses_unreviewed_pair_reduction(tmp_path, monkeypatch) -> None:
    remote_manifest = tmp_path / "manifest.json"
    remote_manifest.write_text(json.dumps({"pairs": [{**_pair("keep__h200"), "records": 2}]}))

    class _FakeApi:
        def repo_info(self, *_: object, **__: object) -> object:
            return type("Info", (), {"sha": "a" * 40})()

        def upload_folder(self, **_: object) -> object:
            raise AssertionError("an unreviewed reduction must not upload")

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
        "allow_row_reduction": False,
    })())

    with pytest.raises(SystemExit, match="would reduce rows in pairs"):
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
        "allow_row_reduction": True,
    })())

    assert publish_ground_truth.main() == 0
    assert uploaded["repo_id"] == "org/data"


def test_publish_bootstraps_a_dataset_without_a_remote_manifest(tmp_path, monkeypatch) -> None:
    uploaded: dict[str, object] = {}

    class _FakeApi:
        def repo_info(self, *_: object, **__: object) -> object:
            return type("Info", (), {"sha": "a" * 40})()

        def upload_folder(self, **kwargs: object) -> object:
            uploaded.update(kwargs)
            return type("Commit", (), {"oid": "b" * 40})()

    def _missing_manifest(**_: object) -> str:
        raise publish_ground_truth.EntryNotFoundError("manifest.json does not exist")

    monkeypatch.setattr(publish_ground_truth, "prepare", lambda *_: {"pairs": [_pair("keep__h200")]})
    monkeypatch.setattr(publish_ground_truth, "HfApi", _FakeApi)
    monkeypatch.setattr(publish_ground_truth, "hf_hub_download", _missing_manifest)
    monkeypatch.setattr(publish_ground_truth, "parse_args", lambda: type("Args", (), {
        "manifest": Path("manifest.json"),
        "datasets_dir": Path("datasets"),
        "output_dir": Path("output"),
        "repo_id": "org/data",
        "revision": None,
        "check_only": False,
        "allow_removal": False,
    })())

    assert publish_ground_truth.main() == 0
    assert uploaded["parent_commit"] == "a" * 40


def test_publish_propagates_unrelated_manifest_errors(monkeypatch) -> None:
    class _FakeApi:
        def repo_info(self, *_: object, **__: object) -> object:
            return type("Info", (), {"sha": "a" * 40})()

    def _broken_download(**_: object) -> str:
        raise RuntimeError("network exploded")

    monkeypatch.setattr(publish_ground_truth, "prepare", lambda *_: {"pairs": []})
    monkeypatch.setattr(publish_ground_truth, "HfApi", _FakeApi)
    monkeypatch.setattr(publish_ground_truth, "hf_hub_download", _broken_download)
    monkeypatch.setattr(publish_ground_truth, "parse_args", lambda: type("Args", (), {
        "manifest": Path("manifest.json"),
        "datasets_dir": Path("datasets"),
        "output_dir": Path("output"),
        "repo_id": "org/data",
        "revision": None,
        "check_only": True,
        "allow_removal": False,
    })())

    with pytest.raises(RuntimeError, match="network exploded"):
        publish_ground_truth.main()

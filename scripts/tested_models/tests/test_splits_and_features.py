from __future__ import annotations

import json
import sys

import pandas as pd
import split_dataset
from stage5_common import feature_matrix, normalize_backend_version, normalize_features


def test_split_is_deterministic_and_keeps_configuration_groups_together(tmp_path, monkeypatch) -> None:
    source_dir = tmp_path / "data" / "tested-models" / "datasets"
    source_dir.mkdir(parents=True)
    rows = [{"model_id": "model", "accelerator": "H200", "tp": 1, "concurrency": value} for value in range(10)]
    rows += [{"model_id": "model", "accelerator": "H200", "tp": 1, "concurrency": value} for value in range(10)]
    source = source_dir / "model__h200.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)
    manifest = tmp_path / "data" / "tested-models" / "dataset-manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"id": "model__h200", "model_id": "model", "path": str(source)}]}))

    def run(output_dir):
        monkeypatch.setattr(
            sys,
            "argv",
            ["split_dataset.py", "--manifest", str(manifest), "--output-dir", str(output_dir)],
        )
        assert split_dataset.main() == 0
        return {split: pd.read_parquet(output_dir / f"model__h200.{split}.parquet") for split in ("train", "validation", "test")}

    first = run(tmp_path / "splits-one")
    second = run(tmp_path / "splits-two")
    assert [frame["configuration_group"].tolist() for frame in first.values()] == [
        frame["configuration_group"].tolist() for frame in second.values()
    ]
    groups_by_split = {split: set(frame["configuration_group"]) for split, frame in first.items()}
    assert not groups_by_split["train"] & groups_by_split["validation"]
    assert not groups_by_split["train"] & groups_by_split["test"]
    assert not groups_by_split["validation"] & groups_by_split["test"]


def test_feature_matrix_fits_medians_and_categories_from_train_only() -> None:
    train = pd.DataFrame(
        [{"model_id": "a", "accelerator": "H200", "tp": 1, "concurrency": 2, "moe_num_experts": 2}, {"model_id": "a", "accelerator": "H200", "tp": 3, "concurrency": 4, "moe_num_experts": 4}]
    )
    other = pd.DataFrame([{"model_id": "unseen", "accelerator": "B200", "tp": None, "concurrency": 8, "moe_num_experts": None}])

    _, info = feature_matrix(train, other)

    assert info["numeric_medians"]["moe_num_experts"] == 3
    assert info["features"].loc[0, "moe_num_experts"] == 3
    assert "model_id=unseen" not in info["feature_names"]
    assert info["features"].loc[0, "model_id=a"] == 0
    assert "model_id=unseen" not in info["features"]


def test_normalize_backend_version_pools_runtime_and_training_labels() -> None:
    cases = {
        ("vLLM-0.24.0", "vllm"): "vLLM-0.24",
        ("0.24.0", "vllm"): "vLLM-0.24",
        ("AIC-0.10.0-vLLM-0.24.0", "vllm"): "vLLM-0.24",
        ("vLLM-0.24.0", "sglang"): "vLLM-0.24",
        ("sglang-0.5.18-recipe-latency-oriented", "sglang"): "sglang-0.5",
        ("0.5.18", "sglang"): "sglang-0.5",
        ("0.5.18-recipe-latency-oriented", "sglang"): "sglang-0.5",
        ("RHAIIS-3.5-GA", "rhaiis"): "rhaiis-3.5",
        ("RHAIIS-3.2-vLLM-0.9.0", "rhaiis"): "rhaiis-3.2",
        ("current", "vllm"): "unknown",
        (None, None): "unknown",
    }
    for (version, backend), expected in cases.items():
        assert normalize_backend_version(version, backend) == expected


def test_normalize_backend_version_handles_pathological_labels_without_backtracking() -> None:
    assert normalize_backend_version("0." * 5000, "vllm") == "vLLM-0.0"


def test_normalize_features_uses_the_backend_scoped_version_family() -> None:
    frame = pd.DataFrame(
        [
            {"version": "vLLM-0.24.0", "backend": "vllm"},
            {"version": "0.24.0", "backend": "vllm"},
            {"version": "sglang-0.5.18", "backend": "sglang"},
        ]
    )

    result = normalize_features(frame)

    assert result["version"].tolist() == ["vLLM-0.24", "vLLM-0.24", "sglang-0.5"]

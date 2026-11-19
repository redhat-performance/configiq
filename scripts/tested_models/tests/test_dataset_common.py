from __future__ import annotations

import json

import pandas as pd
import pytest
from dataset_common import load_approved_models, normalize_rows


def test_normalize_rows_keeps_allowlisted_fields_and_parses_runtime_values() -> None:
    rows = [
        {
            "model": "llama",
            "version": "vllm-0.6.2",
            "TP": 2,
            "intended concurrency": 8,
            "prompt toks": 512,
            "output toks": 128,
            "runtime_args": "dtype=float16 --max-num-seqs=32 max-model-len:8192 --max-num-batched-tokens=4096",
            "uuid": "run-secret",
            "mlflow_run_id": "mlflow-secret",
            "admin_note": "internal-only",
            "timestamp": "2026-01-01T00:00:00Z",
        }
    ]

    normalized = normalize_rows(rows, "org/model", "H200")

    assert normalized.loc[0, "tp"] == 2
    assert normalized.loc[0, "isl"] == 512
    assert normalized.loc[0, "max_num_seqs"] == "32"
    assert normalized.loc[0, "max_num_batched_tokens"] == "4096"
    assert normalized.loc[0, "max_seq_len"] == "8192"
    assert normalized.loc[0, "precision"] == "float16"
    assert not {"uuid", "mlflow_run_id", "runtime_args", "admin_note", "timestamp"} & set(normalized)
    assert set(normalized) <= {
        "model",
        "model_id",
        "accelerator",
        "pair_id",
        "version",
        "backend",
        "backend_version",
        "tp",
        "isl",
        "osl",
        "concurrency",
        "max_num_seqs",
        "max_num_batched_tokens",
        "pp", "dp", "tp_prefill", "tp_decode", "moe_tp_size", "moe_ep_size", "enable_expert_parallel",
        "max_seq_len",
        "precision",
        "gpu_memory_utilization",
        "chunked_prefill_enabled",
        "weight_quantization",
        "kv_quantization",
    }


def test_normalize_rows_keeps_explicit_serving_parallelism() -> None:
    frame = normalize_rows([
        {
            "model": "org/model", "version": "vLLM-0.31.0", "TP": "4",
            "PP": "2", "DP": "3", "TP_prefill": "2", "TP_decode": "4",
            "moe_tp_size": "2", "moe_ep_size": "2", "enable_expert_parallel": "true",
        },
        {
            "model": "org/model", "version": "vLLM-0.31.0", "TP": "4",
            "runtime_args": (
                "pipeline-parallel-size=2; data-parallel-size=3; "
                "prefill-tp-size=2; decode-tp-size=4; moe-tp-size=2; "
                "moe-ep-size=2; enable-expert-parallel=False"
            ),
        },
    ], "org/model", "H200")

    for column, expected in {
        "pp": "2", "dp": "3", "tp_prefill": "2", "tp_decode": "4",
        "moe_tp_size": "2", "moe_ep_size": "2",
    }.items():
        assert frame[column].tolist() == [expected, expected]
    assert frame["enable_expert_parallel"].tolist() == ["1", "0"]
    assert not {"PP", "DP", "TP_prefill", "TP_decode", "runtime_args"} & set(frame)


def test_normalize_rows_exposes_only_whitelisted_vllm_env_flags() -> None:
    frame = normalize_rows([{
        "model": "org/model", "version": "vLLM-0.31.0",
        "env_vars": json.dumps({
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_MOE": "0",
            "FLASHINFER_CUBIN_DIR": "/internal/location",
        }),
    }], "org/model", "MI355X")
    assert frame.loc[0, "vllm_rocm_use_aiter"] == "1"
    assert frame.loc[0, "vllm_rocm_use_aiter_moe"] == "0"
    assert pd.isna(frame.loc[0, "vllm_rocm_use_aiter_moe_situv2"])
    assert "env_vars" not in frame and "FLASHINFER_CUBIN_DIR" not in frame


@pytest.mark.parametrize(
    "value",
    ["https://example.com/token", "person@example.com", "worker.cluster"],
)
def test_normalize_rows_rejects_sensitive_values(value: str) -> None:
    with pytest.raises(ValueError, match="sensitive"):
        normalize_rows([{"model": value, "version": "vllm-0.6.2"}], "model", "H200")


def test_normalize_rows_returns_dataframe_for_empty_source() -> None:
    result = normalize_rows([], "model", "H200")

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result[["model_id", "accelerator", "pair_id"]].columns) == [
        "model_id",
        "accelerator",
        "pair_id",
    ]


def test_load_approved_models_prefers_the_new_name_and_accepts_the_transition(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"approvedModels": ["org/new"], "testedModels": ["org/old"]}))
    assert load_approved_models(config) == ["org/new"]

    config.write_text(json.dumps({"testedModels": ["org/old"]}))
    assert load_approved_models(config) == ["org/old"]


def test_load_approved_models_rejects_an_empty_or_duplicate_list(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"approvedModels": []}))
    with pytest.raises(ValueError, match="non-empty approvedModels"):
        load_approved_models(config)

    config.write_text(json.dumps({"approvedModels": ["org/model", "org/model"]}))
    with pytest.raises(ValueError, match="duplicate approved model"):
        load_approved_models(config)

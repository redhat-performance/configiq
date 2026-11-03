from __future__ import annotations

import pandas as pd
from dataset_common import normalize_rows
from quantization import canonical_quantization
from stage5_common import normalize_features


def test_canonical_quantization_maps_kernel_modes_and_runtime_flags() -> None:
    assert canonical_quantization("weight_quantization", "int4_wo") == "int4"
    assert canonical_quantization("weight_quantization", "awq") == "int4"
    assert canonical_quantization("weight_quantization", "fp8_e4m3") == "fp8"
    assert canonical_quantization("kv_quantization", "fp8_e4m3") == "fp8"
    assert canonical_quantization("kv_quantization", "auto") == "unknown"
    assert canonical_quantization("weight_quantization", None) == "unknown"
    assert canonical_quantization("weight_quantization", "compressed-tensors") == "compressed-tensors"


def test_normalize_rows_canonicalizes_runtime_quantization() -> None:
    rows = [{
        "model": "llama",
        "version": "vllm-0.6.2",
        "runtime_args": "quantization=awq kv-cache-dtype=fp8_e4m3",
    }]

    normalized = normalize_rows(rows, "org/model", "H200")

    assert normalized.loc[0, "weight_quantization"] == "int4"
    assert normalized.loc[0, "kv_quantization"] == "fp8"


def test_normalize_features_canonicalizes_existing_columns() -> None:
    frame = pd.DataFrame([{"weight_quantization": "gptq", "kv_quantization": "fp8_e5m2"}])

    result = normalize_features(frame)

    assert result.loc[0, "weight_quantization"] == "int4"
    assert result.loc[0, "kv_quantization"] == "fp8"

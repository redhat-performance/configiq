"""Canonical quantization vocabulary shared by ingestion and classifier features.

Serving runtimes describe quantization with different spellings: sizing requests
use kernel modes such as ``int4_wo``, while measured runtime arguments carry
values such as ``awq`` or ``fp8_e4m3``. Map both to one category so training and
inference encode the same configuration. The TypeScript counterpart lives in
``lib/tested-models/quantization.ts`` and must stay in sync.
"""

from __future__ import annotations

from typing import Any

UNKNOWN_VALUES = {"", "none", "auto", "unknown", "nan", "null"}

WEIGHT_ALIASES = {
    "fp8": "fp8",
    "fp8_e4m3": "fp8",
    "fp8_e5m2": "fp8",
    "fp8_dynamic": "fp8",
    "w8a8_fp8": "fp8",
    "int4": "int4",
    "int4_wo": "int4",
    "w4a16": "int4",
    "awq": "int4",
    "awq_marlin": "int4",
    "awq_marlin_24": "int4",
    "gptq": "int4",
    "gptq_marlin": "int4",
    "int8": "int8",
    "int8_wo": "int8",
    "w8a8": "int8",
    "w8a8_int8": "int8",
    "mxfp4": "mxfp4",
    "w4a16_mxfp4": "mxfp4",
    "w4a16_mxfp4_cutlass": "mxfp4",
    "w4a8_mxfp4_mxfp8": "mxfp4",
    "nvfp4": "nvfp4",
    "modelopt_fp4": "nvfp4",
    "w4a8_nvfp4": "nvfp4",
}

KV_ALIASES = {
    "fp8": "fp8",
    "fp8_e4m3": "fp8",
    "fp8_e5m2": "fp8",
    "fp8_dynamic": "fp8",
    "int8": "int8",
    "nvfp4": "nvfp4",
}

ALIASES = {
    "weight_quantization": WEIGHT_ALIASES,
    "kv_quantization": KV_ALIASES,
}


def canonical_quantization(field: str, value: Any) -> str:
    """Return the canonical category for one quantization field."""
    if value is None:
        return "unknown"
    normalized = str(value).strip().lower()
    if not normalized or normalized in UNKNOWN_VALUES:
        return "unknown"
    return ALIASES.get(field, {}).get(normalized, normalized)

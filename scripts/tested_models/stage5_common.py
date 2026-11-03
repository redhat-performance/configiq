"""Leakage-safe helpers for the tested-model performance envelope pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from quantization import canonical_quantization

MEASUREMENT_COLUMNS = {
    "measured rps", "output_tok/sec", "total_tok/sec", "prompt_token_count_mean",
    "prompt_token_count_p99", "output_token_count_mean", "output_token_count_p99",
    "ttft_median", "ttft_p95", "ttft_p99", "ttft_p999", "tpot_median", "tpot_p95",
    "tpot_p99", "tpot_p999", "itl_median", "itl_p95", "itl_p99", "itl_p999",
    "request_latency_median", "request_latency_min", "request_latency_max",
    "successful_requests", "errored_requests", "ttft_mean", "itl_mean", "ttft_p1",
    "tpot_p1", "itl_p1", "guidellm_start_time_ms", "guidellm_end_time_ms",
}
ID_COLUMNS = {"run", "uuid", "mlflow_run_id", "mlflow_experiment_id", "pair_id"}
CAT_FIELDS = ("model_id", "system_id", "backend", "version", "accelerator", "request_type", "precision", "weight_quantization", "kv_quantization")
NUM_FIELDS = ("tp", "pp", "dp", "ep", "cp", "isl", "osl", "concurrency", "shared_prefix", "prefix_caching", "prefix_tokens", "prefix_count", "moe_num_experts", "moe_top_k")


def read_manifest(manifest_path: Path) -> list[pd.DataFrame]:
    manifest = json.loads(manifest_path.read_text())
    frames = []
    for pair in manifest.get("pairs", []):
        path = Path(pair["path"])
        if not path.is_absolute():
            path = manifest_path.parent.parent.parent / path
        frame = pd.read_parquet(path)
        frame["pair_id"] = pair["id"]
        frame["model_id"] = frame.get("model_id", pair["model_id"])
        frame["accelerator"] = frame.get("accelerator", pair.get("accelerator", "unknown"))
        frames.append(frame)
    if not frames:
        raise ValueError("dataset manifest contains no pairs")
    return frames


def _runtime_value(text: str, *names: str) -> str | None:
    for name in names:
        match = re.search(rf"(?:^|[;\s]){re.escape(name)}\s*(?:=|:)\s*([^;\s]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip("'")
    return None


def _column(frame: pd.DataFrame, name: str, default: Any) -> pd.Series:
    value = frame.get(name)
    return value if value is not None else pd.Series(default, index=frame.index)


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only configuration features; benchmark outcomes are deliberately excluded."""
    result = pd.DataFrame(index=frame.index)
    model_id = _column(frame, "model_id", _column(frame, "model", "unknown"))
    result["model_id"] = model_id.fillna("unknown").astype(str)
    accelerator = _column(frame, "accelerator", "unknown")
    result["system_id"] = accelerator.fillna("unknown").map(_system_id)
    raw_version = _column(frame, "version", "unknown").fillna("unknown").astype(str)
    backend_values = _column(frame, "backend", np.nan)
    backend_source = backend_values.astype(object).where(backend_values.notna(), raw_version)
    result["backend"] = backend_source.map(_backend_name)
    result["version"] = [
        normalize_backend_version(version, backend)
        for version, backend in zip(raw_version, result["backend"])
    ]
    result["accelerator"] = _column(frame, "accelerator", "unknown").fillna("unknown").astype(str)
    result["request_type"] = _column(frame, "request_type", "unknown").fillna("unknown").astype(str)

    aliases = {"tp": ("tp", "TP", "tensor_parallel_size"), "pp": ("pp", "PP", "pipeline_parallel_size"),
               "dp": ("dp", "DP", "data_parallel_size"), "ep": ("ep", "EP", "expert_parallel_size"),
               "cp": ("cp", "CP", "context_parallel_size"), "isl": ("isl", "input_seq_len", "prompt toks"),
               "osl": ("osl", "output_seq_len", "output toks"), "concurrency": ("concurrency", "measured concurrency"),
               "prefix_tokens": ("prefix_tokens",), "prefix_count": ("prefix_count",)}
    for output, names in aliases.items():
        source = next((name for name in names if name in frame), None)
        result[output] = pd.to_numeric(frame[source], errors="coerce") if source else np.nan
    result["prefix_caching"] = _column(frame, "prefix_caching", np.nan).map(_to_boolean_number)
    result["shared_prefix"] = _column(frame, "shared_prefix", _column(frame, "prefix_tokens", np.nan)).map(_to_number)
    result["precision"] = _column(frame, "precision", "unknown").fillna("unknown").astype(str)
    result["weight_quantization"] = _column(frame, "weight_quantization", "unknown").fillna("unknown").astype(str).map(lambda value: canonical_quantization("weight_quantization", value))
    result["kv_quantization"] = _column(frame, "kv_quantization", "unknown").fillna("unknown").astype(str).map(lambda value: canonical_quantization("kv_quantization", value))
    for name in ("moe_num_experts", "moe_top_k"):
        result[name] = _column(frame, name, np.nan).map(_to_number)

    for column in ("tp", "pp", "dp", "ep", "cp"):
        result[column] = result[column].fillna(1)
    result["isl"] = result["isl"].fillna(0)
    result["osl"] = result["osl"].fillna(0)
    result["concurrency"] = result["concurrency"].fillna(0)
    result["token_budget"] = (result["isl"] + result["osl"]) * result["concurrency"].clip(lower=1)
    result["parallelism"] = result[["tp", "pp", "dp", "ep", "cp"]].prod(axis=1)
    result["memory_pressure"] = result["token_budget"] / result["parallelism"].clip(lower=1)
    result["prefix_pressure"] = result["shared_prefix"].fillna(0) * result["concurrency"].clip(lower=1)
    result["sequence_pressure"] = (result["isl"] + result["osl"]) / result["parallelism"].clip(lower=1)
    return result


def _to_number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _to_boolean_number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return float(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "on", "enabled", "1"}:
        return 1.0
    if normalized in {"false", "no", "off", "disabled", "0"}:
        return 0.0
    return np.nan


def _system_id(value: Any) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "h200": "h200_sxm",
        "b200": "b200_sxm",
        "b300": "b300_sxm",
        "mi300x": "mi300x",
        "mi355x": "mi355x",
    }
    return aliases.get(normalized, normalized or "unknown")


def _backend_name(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "unknown"
    lowered = text.lower()
    for name in ("vllm", "sglang", "trt-llm", "rhaiis", "nim", "aic"):
        if name in lowered:
            return name
    return text


_RUNTIME_FAMILY_LABEL = {
    "vllm": "vLLM",
    "sglang": "sglang",
    "trt-llm": "trt-llm",
    "rhaiis": "rhaiis",
    "nim": "nim",
    "aic": "aic",
    "tgi": "tgi",
    "lmdeploy": "lmdeploy",
}
_RUNTIME_TOKEN_FAMILY = {
    "vllm": "vllm",
    "sglang": "sglang",
    "tensorrt-llm": "trt-llm",
    "trt-llm": "trt-llm",
    "rhaiis": "rhaiis",
    "nim": "nim",
    "aic": "aic",
    "tgi": "tgi",
    "lmdeploy": "lmdeploy",
}
_RUNTIME_TOKEN_PATTERN = re.compile(
    r"(?i)\b("
    + "|".join(sorted((re.escape(token) for token in _RUNTIME_TOKEN_FAMILY), key=len, reverse=True))
    + r")-(\d+)(?:\.(\d+))?"
)
_BARE_VERSION_PATTERN = re.compile(r"^\s*(\d+)(?:\.(\d+))?(?:[-._+]\S*)?\s*$")


def backend_family(value: Any) -> str:
    """Return the canonical runtime family for an already-resolved backend name."""
    text = str(value or "").strip().lower()
    return text if text in _RUNTIME_FAMILY_LABEL else "unknown"


def _version_family(major: str, minor: str | None) -> str:
    # Only the major.minor family is a feature; patch and suffix detail stays
    # in coverage provenance. Parsing is bounded to avoid regex backtracking.
    return f"{major}.{minor}" if minor else major


def normalize_backend_version(value: Any, backend: Any = None) -> str:
    """Pool a raw version label into a backend-scoped major.minor family.

    Training rows carry labels such as ``vLLM-0.24.0`` while the runtime sends
    the catalog version ``0.24.0``. Both must map to the same feature category,
    and families from different runtimes must never collide.
    """
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown", "current"}:
        return "unknown"
    family = backend_family(backend)
    matches = _RUNTIME_TOKEN_PATTERN.findall(text)
    if matches:
        for token, major, minor in matches:
            resolved = _RUNTIME_TOKEN_FAMILY[token.lower()]
            if resolved == family:
                return f"{_RUNTIME_FAMILY_LABEL[resolved]}-{_version_family(major, minor or None)}"
        token, major, minor = matches[0]
        resolved = _RUNTIME_TOKEN_FAMILY[token.lower()]
        return f"{_RUNTIME_FAMILY_LABEL[resolved]}-{_version_family(major, minor or None)}"
    bare = _BARE_VERSION_PATTERN.match(text)
    if bare and family != "unknown":
        return f"{_RUNTIME_FAMILY_LABEL[family]}-{_version_family(bare.group(1), bare.group(2))}"
    return "unknown"


def configuration_group(frame: pd.DataFrame) -> pd.Series:
    features = normalize_features(frame)
    keys = [column for column in features.columns if column not in {"token_budget", "parallelism", "memory_pressure", "prefix_pressure", "sequence_pressure"}]
    values = features[keys].fillna("<NA>").astype(str).agg("|".join, axis=1)
    return values.map(lambda value: hashlib.sha256(value.encode()).hexdigest()[:16])


def feature_matrix(train: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_features, other_features = normalize_features(train), normalize_features(other)
    names: list[str] = []
    matrices: list[pd.DataFrame] = []
    medians: dict[str, float] = {}
    for column in train_features.columns:
        if not pd.api.types.is_numeric_dtype(train_features[column]):
            values = sorted(set(train_features[column].astype(str)))
            for value in values:
                name = f"{column}={value}"
                names.append(name)
                matrices.append(pd.DataFrame({name: (train_features[column].astype(str) == value).astype(float)}))
                matrices.append(pd.DataFrame({name: (other_features[column].astype(str) == value).astype(float)}))
        else:
            median = float(pd.to_numeric(train_features[column], errors="coerce").median()) if train_features[column].notna().any() else 0.0
            medians[column] = median
            names.append(column)
            matrices.append(pd.DataFrame({column: pd.to_numeric(train_features[column], errors="coerce").fillna(median).astype(float)}))
            matrices.append(pd.DataFrame({column: pd.to_numeric(other_features[column], errors="coerce").fillna(median).astype(float)}))
    train_parts = matrices[::2]
    other_parts = matrices[1::2]
    return pd.concat(train_parts, axis=1), {"features": pd.concat(other_parts, axis=1), "feature_names": names, "numeric_medians": medians}

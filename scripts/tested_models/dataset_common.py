"""Shared helpers for tested-model ground-truth extraction."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

SENSITIVE_COLUMN_PATTERN = re.compile(
    r"(?:email|e-mail|auth[_ -]?token|access[_ -]?token|secret|password|authorization|cookie|ip[_ -]?address|user[_ -]?id|hostname|host[_ -]?name)",
    re.IGNORECASE,
)
SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?:https?://|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|\b[a-z0-9-]+\.(?:internal|private|local|cluster)\b)",
    re.IGNORECASE,
)
CLASSIFICATION_COLUMNS = {
    "model", "model_id", "accelerator", "pair_id", "version", "backend", "backend_version",
    "tp", "pp", "dp", "ep", "cp", "isl", "osl", "concurrency",
    "measured concurrency", "measured rps", "output_tok/sec", "total_tok/sec",
    "prompt_token_count_mean", "prompt_token_count_p99", "output_token_count_mean",
    "output_token_count_p99", "ttft_median", "ttft_p95", "ttft_p1", "ttft_p999",
    "ttft_mean", "ttft_p99", "tpot_median", "tpot_p95", "tpot_p99", "tpot_p999",
    "tpot_p1", "itl_median", "itl_p95", "itl_p99", "itl_p999", "itl_p1", "itl_mean",
    "request_latency_median", "request_latency_min", "request_latency_max",
    "successful_requests", "errored_requests", "gpu_memory_utilization", "max_num_seqs",
    "chunked_prefill_enabled", "prefix_caching", "turns", "prefix_tokens", "prefix_count",
    "request_type", "spec_decoding", "precision", "weight_quantization", "kv_quantization",
    "moe_num_experts", "moe_top_k", "max_seq_len",
}


def _runtime_value(text: str, *names: str) -> str | None:
    for name in names:
        match = re.search(rf"(?:^|[;\s])(?:--)?{re.escape(name)}\s*(?:=|:)\s*([^;\s]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip("'")
    return None


def _backend_name(version: Any) -> str:
    text = str(version or "").lower()
    for name in ("vllm", "sglang", "trt-llm", "rhaiis", "nim", "aic"):
        if name in text:
            return name
    return "unknown"


def load_required_models(config_path: Path) -> list[str]:
    """Load the required tested-model IDs from ConfigIQ's public config."""
    data = json.loads(config_path.read_text())
    models = data.get("testedModels")
    if not isinstance(models, list) or not models or not all(isinstance(model, str) for model in models):
        raise ValueError(f"{config_path} must contain a non-empty testedModels string array")
    if len(models) != len(set(models)):
        raise ValueError(f"{config_path} contains duplicate tested model IDs")
    return models


def pair_id(model_id: str, accelerator: str) -> str:
    """Create a stable filesystem-safe identifier while preserving pair identity in metadata."""
    model_slug = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")
    accelerator_slug = re.sub(r"[^a-z0-9]+", "_", accelerator.lower()).strip("_")
    return f"{model_slug}__{accelerator_slug}"


def fetch_rows(
    session: requests.Session,
    api_url: str,
    model_id: str,
    accelerator: str | None = None,
    *,
    verify: bool = True,
) -> list[dict[str, Any]]:
    response = session.get(
        f"{api_url.rstrip('/')}/data",
        params={"model": model_id, **({"accelerator": accelerator} if accelerator else {})},
        timeout=120,
        verify=verify,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("ground-truth API response did not contain a data object array")
    return rows


def normalize_rows(rows: list[dict[str, Any]], model_id: str, accelerator: str) -> pd.DataFrame:
    """Normalize identity and stable parameter names without dropping API fields."""
    frame = pd.DataFrame(rows)
    sensitive_columns = [column for column in frame.columns if SENSITIVE_COLUMN_PATTERN.search(str(column))]
    if sensitive_columns:
        raise ValueError(f"source contains potentially sensitive columns: {sensitive_columns}")
    frame = frame.rename(
        columns={
            "TP": "tp",
            "intended concurrency": "concurrency",
            "prompt toks": "isl",
            "output toks": "osl",
        }
    )

    runtime = frame.get("runtime_args", pd.Series("", index=frame.index)).fillna("").astype(str)
    backend = frame.get("backend", frame.get("version", pd.Series("unknown", index=frame.index)))
    frame["backend"] = backend.map(_backend_name)
    frame["backend_version"] = frame.get("version", pd.Series("unknown", index=frame.index))
    runtime_aliases = {
        "gpu_memory_utilization": ("gpu-memory-utilization", "gpu_memory_utilization"),
        "max_num_seqs": ("max-num-seqs", "max_num_seqs"),
        "max_seq_len": ("max-model-len", "max_model_len"),
        "chunked_prefill_enabled": ("enable-chunked-prefill", "enable_chunked_prefill"),
        "precision": ("dtype", "precision"),
        "weight_quantization": ("quantization", "weight-quantization"),
        "kv_quantization": ("kv-cache-dtype", "kv-quantization"),
    }
    for column, names in runtime_aliases.items():
        if column not in frame:
            frame[column] = runtime.map(lambda value, aliases=names: _runtime_value(value, *aliases))
    frame = frame[[column for column in frame.columns if column in CLASSIFICATION_COLUMNS]]

    frame["model_id"] = model_id
    frame["accelerator"] = accelerator
    frame["pair_id"] = pair_id(model_id, accelerator)
    if not frame.empty:
        validate_public_frame(frame)
    return frame


def validate_public_frame(frame: pd.DataFrame) -> None:
    """Reject empty, unexpected, or potentially sensitive public dataset content."""
    if frame.empty:
        raise ValueError("ground-truth parquet must contain at least one row")
    unexpected = sorted(set(frame.columns) - CLASSIFICATION_COLUMNS)
    if unexpected:
        raise ValueError(f"ground-truth parquet contains non-public columns: {unexpected}")
    sensitive_columns = [column for column in frame.columns if SENSITIVE_COLUMN_PATTERN.search(str(column))]
    if sensitive_columns:
        raise ValueError(f"ground-truth parquet contains potentially sensitive columns: {sensitive_columns}")
    string_values = frame.select_dtypes(include=["object", "string"]).astype("string").melt(value_name="value")["value"].dropna().astype(str)
    sensitive_values = string_values[string_values.str.contains(SENSITIVE_VALUE_PATTERN, regex=True)]
    if not sensitive_values.empty:
        raise ValueError("ground-truth parquet contains potentially sensitive URL, email, or internal-host values")


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash a deterministic JSON representation for manifest provenance."""
    payload = frame.sort_index(axis=1).to_json(orient="records", date_format="iso", default_handler=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_parquet(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

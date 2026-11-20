"""Shared helpers for tested-model ground-truth extraction."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from quantization import canonical_quantization

DATASET_REPO_ENV = "TESTED_DATASET_REPO"
DEFAULT_DATASET_REPO = "redhat-performance/configiq-performance-upstream-data"
UPSTREAM_VLLM_VERSION_PATTERN = re.compile(r"vLLM-\d+\.\d+\.\d+", re.IGNORECASE)


def is_upstream_vllm_version(value: Any) -> bool:
    """Accept only a plain upstream vLLM release label."""
    return isinstance(value, str) and bool(UPSTREAM_VLLM_VERSION_PATTERN.fullmatch(value.strip()))

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
    "tp", "pp", "dp", "ep", "cp", "tp_prefill", "tp_decode",
    "moe_tp_size", "moe_ep_size", "enable_expert_parallel", "isl", "osl", "concurrency",
    "measured concurrency", "measured rps", "output_tok/sec", "total_tok/sec",
    "prompt_token_count_mean", "prompt_token_count_p99", "output_token_count_mean",
    "output_token_count_p99", "ttft_median", "ttft_p95", "ttft_p1", "ttft_p999",
    "ttft_mean", "ttft_p99", "tpot_median", "tpot_p95", "tpot_p99", "tpot_p999",
    "tpot_p1", "itl_median", "itl_p95", "itl_p99", "itl_p999", "itl_p1", "itl_mean",
    "request_latency_median", "request_latency_min", "request_latency_max",
    "successful_requests", "errored_requests", "gpu_memory_utilization", "max_num_seqs",
    "max_num_batched_tokens",
    "vllm_rocm_use_aiter", "vllm_rocm_use_aiter_moe", "vllm_rocm_use_aiter_moe_situv2",
    "chunked_prefill_enabled", "prefix_caching", "turns", "turn_index", "prefix_tokens", "prefix_count",
    "request_type", "spec_decoding", "precision", "weight_quantization", "kv_quantization",
    "moe_num_experts", "moe_top_k", "max_seq_len",
    "runtime_args", "env_vars", "guidellm_version",
}

# Public values require review because these source fields may also contain
# credentials, hostnames, or local paths. Unreviewed or unsafe entries are
# omitted rather than publishing values that cannot be used to reproduce a run.
PUBLIC_RUNTIME_VALUE_KEYS = {
    "tensor-parallel-size", "trust-remote-code", "gpu-memory-utilization",
    "no-enable-log-requests", "uvicorn-log-level", "enable-prefix-caching",
    "no-enable-prefix-caching", "kv-cache-dtype", "reasoning-parser",
    "tool-call-parser", "enable-auto-tool-choice", "max-model-len",
    "disable-log-requests", "tokenizer-mode", "dtype", "moe-backend",
    "max-num-batched-tokens", "max-num-seqs", "max-cudagraph-capture-size",
    "AMDGCN_USE_BUFFER_OPS", "max_num_seqs", "attention-backend",
    "block-size", "performance-mode", "optimization-level",
    "mm-encoder-tp-mode", "prefix_caching", "enable-expert-parallel",
    "VLLM_ROCM_USE_AITER_TRITON_ROPE", "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION",
    "VLLM_ROCM_USE_AITER",
}
PUBLIC_ENV_VALUE_KEYS = {
    "VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE",
    "VLLM_ROCM_USE_AITER_MOE_SITUV2",
}
SETTING_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,99}\Z")
SETTING_VALUE_PATTERN = re.compile(r"[A-Za-z0-9_.+-]{1,40}\Z")
GUIDELLM_VERSION_PATTERN = re.compile(r"v?\d+(?:\.\d+){1,3}[A-Za-z0-9_.+-]{0,40}\Z")
RUNTIME_PAIR_PATTERN = re.compile(
    r"(?:^|[;\s])(?:--)?([A-Za-z_][A-Za-z0-9_-]*)\s*(?:=|:)\s*([^;\s]+)"
)
def _public_setting_value(name: str, value: object, allowed: set[str]) -> str | None:
    text = str(value)
    if name not in allowed or not SETTING_VALUE_PATTERN.fullmatch(text):
        return None
    return text


def _public_runtime_args(raw: str) -> str:
    """Keep only recorded options with reviewed, simple public values."""
    values: dict[str, str] = {}
    for match in RUNTIME_PAIR_PATTERN.finditer(raw):
        name, value = match.groups()
        if SETTING_NAME_PATTERN.fullmatch(name):
            public_value = _public_setting_value(name, value, PUBLIC_RUNTIME_VALUE_KEYS)
            if public_value is not None:
                values[name] = public_value
    for name in re.findall(r"(?:^|[;\s])--([A-Za-z_][A-Za-z0-9_-]*)(?=\s|;|$)", raw):
        if SETTING_NAME_PATTERN.fullmatch(name):
            public_value = _public_setting_value(name, "true", PUBLIC_RUNTIME_VALUE_KEYS)
            if public_value is not None:
                values.setdefault(name, public_value)
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _public_env_vars(raw: str) -> str:
    """Keep recorded vLLM-related settings with reviewed, safe values."""
    try:
        source = json.loads(raw or "{}")
    except json.JSONDecodeError as error:
        raise ValueError("env_vars is not valid JSON") from error
    if not isinstance(source, dict):
        raise TypeError("env_vars must be a JSON object")
    values: dict[str, str] = {}
    for name, value in source.items():
        if not isinstance(name, str) or not SETTING_NAME_PATTERN.fullmatch(name):
            continue
        if not name.startswith(("VLLM_", "FLASHINFER_")):
            continue
        public_value = _public_setting_value(name, value, PUBLIC_ENV_VALUE_KEYS)
        if public_value is not None:
            values[name] = public_value
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def resolve_dataset_repo_id(cli_value: str | None) -> str:
    """Resolve the dataset repository: explicit flag, then environment, then default.

    Publishing and downloading share this resolution so a published revision
    stays discoverable. Blank values fall through because GitHub renders an
    unset repository variable as an empty string.
    """
    for candidate in (cli_value, os.environ.get(DATASET_REPO_ENV)):
        if candidate and candidate.strip():
            return candidate.strip()
    return DEFAULT_DATASET_REPO


def _runtime_value(text: str, *names: str) -> str | None:
    for name in names:
        match = re.search(rf"(?:^|[;\s])(?:--)?{re.escape(name)}\s*(?:=|:)\s*([^;\s]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip("'")
    return None


def _env_flags(text: str) -> dict[str, str | None]:
    """Expose only explicit, non-sensitive vLLM ROCm kernel settings."""
    names = {
        "VLLM_ROCM_USE_AITER": "vllm_rocm_use_aiter",
        "VLLM_ROCM_USE_AITER_MOE": "vllm_rocm_use_aiter_moe",
    }
    try:
        source = json.loads(text or "{}")
    except json.JSONDecodeError as error:
        raise ValueError("env_vars is not valid JSON") from error
    if not isinstance(source, dict):
        raise TypeError("env_vars must be a JSON object")
    result = {}
    for source_name, public_name in names.items():
        value = source.get(source_name)
        if value is None:
            result[public_name] = None
        elif str(value).lower() in ("1", "true"):
            result[public_name] = "1"
        elif str(value).lower() in ("0", "false"):
            result[public_name] = "0"
        else:
            raise ValueError(f"unsupported value for {source_name}")
    situv2 = source.get("VLLM_ROCM_USE_AITER_MOE_SITUV2")
    if situv2 is None:
        result["vllm_rocm_use_aiter_moe_situv2"] = None
    else:
        mode = str(situv2).strip().lower()
        if mode in ("true", "false"):
            mode = "1" if mode == "true" else "0"
        if mode not in {"0", "1", "auto", "a4w4", "a8w4", "a16w4"}:
            raise ValueError("unsupported value for VLLM_ROCM_USE_AITER_MOE_SITUV2")
        result["vllm_rocm_use_aiter_moe_situv2"] = mode
    return result


def _explicit_bool(value: Any) -> str | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes"):
        return "1"
    if normalized in ("0", "false", "no"):
        return "0"
    raise ValueError(f"unsupported explicit boolean value: {value!r}")


def _backend_name(version: Any) -> str:
    text = str(version or "").lower()
    # Productized builds sometimes embed "vllm" in their version string, so
    # check the more specific stacks first.
    for name in ("rhaiis", "sglang", "trt-llm", "nim", "aic", "vllm"):
        if name in text:
            return name
    return "unknown"


def load_approved_models(config_path: Path) -> list[str]:
    """Load models approved for measurement from ConfigIQ's public config.

    ``approvedModels`` is the durable name. ``testedModels`` is accepted during
    the transition because the approved list must never be mistaken for the
    registry's classifier-validated list.
    """
    data = json.loads(config_path.read_text())
    models = data.get("approvedModels", data.get("testedModels"))
    if not isinstance(models, list) or not models or not all(isinstance(model, str) for model in models):
        raise ValueError(f"{config_path} must contain a non-empty approvedModels string array")
    if len(models) != len(set(models)):
        raise ValueError(f"{config_path} contains duplicate approved model IDs")
    return models


# Kept for callers that predate the approved/validated distinction.
load_required_models = load_approved_models


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
            "PP": "pp",
            "DP": "dp",
            "EP": "ep",
            "CP": "cp",
            "TP_prefill": "tp_prefill",
            "TP_decode": "tp_decode",
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
        "max_num_batched_tokens": ("max-num-batched-tokens", "max_num_batched_tokens"),
        "pp": ("pipeline-parallel-size", "pipeline_parallel_size"),
        "dp": ("data-parallel-size", "data_parallel_size"),
        "tp_prefill": ("tp-prefill", "tp_prefill", "prefill-tp-size", "prefill_tp_size", "prefill-tensor-parallel-size"),
        "tp_decode": ("tp-decode", "tp_decode", "decode-tp-size", "decode_tp_size", "decode-tensor-parallel-size"),
        "moe_tp_size": ("moe-tp-size", "moe_tp_size"),
        "moe_ep_size": ("moe-ep-size", "moe_ep_size"),
        "enable_expert_parallel": ("enable-expert-parallel", "enable_expert_parallel"),
        "max_seq_len": ("max-model-len", "max_model_len"),
        "chunked_prefill_enabled": ("enable-chunked-prefill", "enable_chunked_prefill"),
        "precision": ("dtype", "precision"),
        "weight_quantization": ("quantization", "weight-quantization"),
        "kv_quantization": ("kv-cache-dtype", "kv-quantization"),
    }
    for column, names in runtime_aliases.items():
        parsed = runtime.map(lambda value, aliases=names: _runtime_value(value, *aliases))
        if column not in frame:
            frame[column] = parsed
        elif column in {"pp", "dp", "tp_prefill", "tp_decode", "moe_tp_size", "moe_ep_size", "enable_expert_parallel"}:
            recorded = frame[column].replace("", None)
            frame[column] = recorded.combine_first(parsed)
    frame["runtime_args"] = runtime.map(_public_runtime_args)
    frame["enable_expert_parallel"] = frame["enable_expert_parallel"].map(_explicit_bool)
    env_raw = frame.get("env_vars", pd.Series("", index=frame.index)).fillna("").astype(str)
    if "env_vars" in frame:
        flags = env_raw.map(_env_flags)
        for column in ("vllm_rocm_use_aiter", "vllm_rocm_use_aiter_moe", "vllm_rocm_use_aiter_moe_situv2"):
            frame[column] = flags.map(lambda value, name=column: value[name])
    frame["env_vars"] = env_raw.map(_public_env_vars)
    if "guidellm_version" in frame:
        versions = frame["guidellm_version"].fillna("").astype(str).str.strip()
        invalid = versions.ne("") & ~versions.map(lambda value: bool(GUIDELLM_VERSION_PATTERN.fullmatch(value)))
        if invalid.any():
            raise ValueError("guidellm_version is not a public version string")
        frame["guidellm_version"] = versions.replace("", None)
    for column in ("weight_quantization", "kv_quantization"):
        if column in frame:
            frame[column] = frame[column].map(
                lambda value, name=column: canonical_quantization(name, value)
            )
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

"""Build request-scoped BLIS Scenario and Deployment documents."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .manifest import BlisUnsupportedRequest, ModelResolution
from .runtime import blis_release

#: The vLLM releases BLIS models. Upstream refuses any other engine_version before a
#: run, so the API refuses it first and names the alternatives in a 422 rather than
#: letting BLIS's fatal error surface as a 502.
MODELED_ENGINE_VERSIONS = ("0.29.0", "0.31.0", "0.32.0")
#: The release a request gets when it names none. 0.31.0 is the newest release that
#: models prefill/decode disaggregation *and* accepts max_model_len: auto.
DEFAULT_ENGINE_VERSION = "0.31.0"
#: What each modeled release can express. 0.32.0 runs blis-vllm's translated vLLM
#: engine, which models colocated serving only and requires max_model_len stated.
ENGINE_VERSION_SUPPORT: dict[str, dict[str, bool]] = {
    "0.29.0": {"disaggregated": True, "max_model_len_auto": True},
    "0.31.0": {"disaggregated": True, "max_model_len_auto": True},
    "0.32.0": {"disaggregated": False, "max_model_len_auto": False},
}
#: Releases that can run a prefill/decode topology, newest last.
DISAGGREGATED_ENGINE_VERSIONS = tuple(
    version for version, support in ENGINE_VERSION_SUPPORT.items() if support["disaggregated"]
)
_VERSION_ALIASES = frozenset({"current", "previous", "next"})

_COEFFICIENTS = [
    "cost-model-primitives",
    "cost-model-collectives",
    "cost-model-host-overheads",
    "cost-model-attention",
    "cost-model-recurrent",
    "cost-model-memory",
]
_WEIGHT_QUANTIZATION = {
    "bf16": None,
    "fp16": None,
    "half": None,
    "fp8": "fp8",
    "int8": "int8",
    "int4": "int4",
    "mxfp4": "mxfp4",
    "nvfp4": "nvfp4",
}
_CACHE_DTYPE = {
    "half": None,
    "bf16": "bf16",
    "fp16": "fp16",
    "fp8": "fp8",
    "fp8_e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
}


@dataclass(frozen=True)
class ScenarioPoolRequest:
    role: str
    nodes: int
    tp: int
    pp: int
    dp: int
    moe_tp_size: int | None
    moe_ep_size: int | None
    enable_expert_parallel: bool
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float | None
    weight_quantization: str | None
    moe_quantization: str | None
    cache_dtype: str | None
    prefix_caching: bool | None
    block_size: int = 64


@dataclass(frozen=True)
class ScenarioRequest:
    model: ModelResolution
    hardware: str
    system: str
    gpus_per_node: int
    engine_version: str
    tp: int
    pp: int
    dp: int
    moe_tp_size: int | None
    moe_ep_size: int | None
    enable_expert_parallel: bool
    isl: int
    osl: int
    concurrency: int
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float | None = None
    weight_quantization: str | None = None
    moe_quantization: str | None = None
    cache_dtype: str | None = None
    prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    fmha_quantization: str | None = None
    fabric: str | None = None
    prefill_pool: ScenarioPoolRequest | None = None
    decode_pool: ScenarioPoolRequest | None = None


def _scenario_name(request: ScenarioRequest) -> str:
    canonical = json.dumps(
        asdict(request),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    safe_model = re.sub(r"[^a-z0-9-]+", "-", request.model.catalog_name.lower()).strip("-")
    return f"configiq-{safe_model}-{digest}"


def _optional_quantization(raw: str | None, supported: dict[str, str | None], name: str) -> str | None:
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if not normalized:
        return None
    if normalized not in supported:
        choices = ", ".join(sorted(supported))
        raise BlisUnsupportedRequest(f"BLIS cannot represent {name}={raw!r}; supported values: {choices}")
    return supported[normalized]


def normalize_weight_quantization(raw: str | None) -> str | None:
    """Return the BLIS scenario value for a supported weight-precision setting."""
    return _optional_quantization(raw, _WEIGHT_QUANTIZATION, "weight quantization")


def _pool_engine(pool: ScenarioPoolRequest) -> dict[str, Any]:
    raw_weight_quant = pool.weight_quantization.strip().lower() if pool.weight_quantization else None
    raw_moe_quant = pool.moe_quantization.strip().lower() if pool.moe_quantization else None
    if raw_moe_quant and raw_weight_quant != raw_moe_quant:
        raise BlisUnsupportedRequest("BLIS scenarios cannot separate MoE and dense weight quantization")
    weight_quant = normalize_weight_quantization(pool.weight_quantization)
    moe_quant = _optional_quantization(pool.moe_quantization, _WEIGHT_QUANTIZATION, "MoE quantization")
    if moe_quant and moe_quant != weight_quant:
        raise BlisUnsupportedRequest("BLIS scenarios cannot separate MoE and dense weight quantization")
    cache_dtype = _optional_quantization(pool.cache_dtype, _CACHE_DTYPE, "KV-cache dtype")
    engine: dict[str, Any] = {
        "block_size": pool.block_size,
        "max_num_batched_tokens": pool.max_num_batched_tokens,
        "max_num_seqs": pool.max_num_seqs,
        "max_model_len": pool.max_model_len,
        "cudagraph_mode": "PIECEWISE",
        "gpu_memory_utilization": pool.gpu_memory_utilization if pool.gpu_memory_utilization is not None else 0.92,
    }
    if weight_quant:
        engine["quantization"] = weight_quant
    if cache_dtype:
        engine["cache_dtype"] = cache_dtype
    engine["enable_prefix_caching"] = pool.prefix_caching if pool.prefix_caching is not None else True
    return engine


def disaggregation_refusal(version: str) -> str:
    """Explain why a release cannot run a prefill/decode topology."""
    modeled = ", ".join(DISAGGREGATED_ENGINE_VERSIONS)
    return (
        f"BLIS {blis_release()} does not model prefill/decode disaggregation for vLLM {version}; "
        f"use {modeled}, or run a colocated deployment"
    )


def _build_disagg_documents(request: ScenarioRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    if not engine_version_support(request.engine_version)["disaggregated"]:
        raise BlisUnsupportedRequest(disaggregation_refusal(request.engine_version))
    prefill = request.prefill_pool
    decode = request.decode_pool
    if prefill is None or decode is None:
        raise BlisUnsupportedRequest("BLIS disaggregated scenarios require both prefill and decode pools")
    if not request.fabric:
        raise BlisUnsupportedRequest("BLIS disaggregated scenarios require an explicit inter-node fabric")
    if prefill.role != "prefill" or decode.role != "decode":
        raise BlisUnsupportedRequest("BLIS disaggregated pools must have prefill and decode roles")
    if min(request.isl, request.osl, request.concurrency) < 1:
        raise BlisUnsupportedRequest("BLIS parallelism and workload values must be positive integers")

    pools: list[dict[str, Any]] = []
    for pool in (prefill, decode):
        if min(
            pool.nodes,
            pool.tp,
            pool.pp,
            pool.dp,
            pool.max_model_len,
            pool.max_num_seqs,
            pool.max_num_batched_tokens,
            pool.block_size,
        ) < 1:
            raise BlisUnsupportedRequest(f"BLIS {pool.role} pool dimensions must be positive integers")
        if pool.dp != 1:
            raise BlisUnsupportedRequest("BLIS disaggregated predictions currently require DP=1 in both pools")
        if pool.tp > request.gpus_per_node and pool.tp % request.gpus_per_node != 0:
            raise BlisUnsupportedRequest(
                f"BLIS can span {pool.role} TP across nodes only in whole-node multiples of "
                f"{request.gpus_per_node} GPUs"
            )
        if pool.moe_tp_size not in (None, 0, pool.tp):
            raise BlisUnsupportedRequest(
                f"BLIS derives {pool.role} expert tensor parallelism from TP; "
                f"it cannot represent moe_tp_size={pool.moe_tp_size} with tp_size={pool.tp}"
            )
        derived_ep_size = pool.tp if pool.enable_expert_parallel else 1
        if pool.moe_ep_size not in (None, 0, derived_ep_size):
            raise BlisUnsupportedRequest(
                f"BLIS derives {pool.role} expert parallel width from TP; "
                f"it cannot represent moe_ep_size={pool.moe_ep_size} with tp_size={pool.tp}"
            )
        if pool.nodes * request.gpus_per_node < pool.tp * pool.pp:
            raise BlisUnsupportedRequest(
                f"BLIS {pool.role} pool has {pool.nodes * request.gpus_per_node} GPU slots, "
                f"but TP×PP requires {pool.tp * pool.pp}"
            )
        pools.append({
            "role": pool.role,
            "nodes": pool.nodes,
            "parallel": {
                "tp": pool.tp,
                "pp": pool.pp,
                "dp": pool.dp,
                "enable_expert_parallel": pool.enable_expert_parallel,
            },
            "engine": _pool_engine(pool),
        })

    name = _scenario_name(request)
    scenario = {
        "kind": "Scenario",
        "name": name,
        "engine_version": request.engine_version,
        "model": request.model.catalog_name,
        "coefficients": list(_COEFFICIENTS),
        "cluster": {
            "hardware": request.hardware,
            "fabric": request.fabric,
            "nodes": prefill.nodes + decode.nodes,
            "gpus_per_node": request.gpus_per_node,
        },
    }
    deployment = {
        "kind": "Deployment",
        "name": name,
        "pools": pools,
        "pd_transfer": {"connector": "NixlConnector"},
        "router": {"endpoint_picker_config": "router/pd-always.yaml"},
    }
    return scenario, deployment


def build_documents(request: ScenarioRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return validated-shape Scenario and Deployment documents for the requested topology."""
    if not re.fullmatch(r"\d+(?:\.\d+){1,2}(?:[-+][A-Za-z0-9.-]+)?", request.engine_version):
        raise BlisUnsupportedRequest(f"BLIS requires a numeric engine version, got {request.engine_version!r}")
    if request.prefill_pool is not None or request.decode_pool is not None:
        if request.gpus_per_node < 1:
            raise BlisUnsupportedRequest(f"BLIS has no valid node layout for system {request.system!r}")
        if request.enable_chunked_prefill is not None:
            raise BlisUnsupportedRequest("BLIS scenarios do not expose the vLLM enable_chunked_prefill setting")
        if request.fmha_quantization:
            raise BlisUnsupportedRequest("BLIS scenarios do not expose an independent FMHA quantization setting")
        return _build_disagg_documents(request)
    if min(request.tp, request.pp, request.dp, request.isl, request.osl, request.concurrency) < 1:
        raise BlisUnsupportedRequest("BLIS parallelism and workload values must be positive integers")
    if request.gpus_per_node < 1:
        raise BlisUnsupportedRequest(f"BLIS has no valid node layout for system {request.system!r}")
    total_gpus = request.tp * request.pp * request.dp
    if total_gpus > request.gpus_per_node:
        raise BlisUnsupportedRequest(
            f"BLIS scenario generation is currently single-node: {total_gpus} GPUs requested, "
            f"but {request.system} has {request.gpus_per_node} GPUs per node"
        )
    if request.moe_tp_size not in (None, 0, request.tp):
        raise BlisUnsupportedRequest(
            "BLIS derives expert tensor parallelism from the pool TP size; "
            f"it cannot represent moe_tp_size={request.moe_tp_size} with tp_size={request.tp}"
        )
    derived_ep_size = request.tp * request.dp if request.enable_expert_parallel else 1
    if request.moe_ep_size not in (None, 0, derived_ep_size):
        raise BlisUnsupportedRequest(
            "BLIS derives expert parallel width from TP × DP; "
            f"it cannot represent moe_ep_size={request.moe_ep_size} for TP={request.tp}, DP={request.dp}"
        )
    if request.enable_chunked_prefill is not None:
        raise BlisUnsupportedRequest("BLIS scenarios do not expose the vLLM enable_chunked_prefill setting")
    if request.fmha_quantization:
        raise BlisUnsupportedRequest("BLIS scenarios do not expose an independent FMHA quantization setting")
    raw_weight_quant = request.weight_quantization.strip().lower() if request.weight_quantization else None
    raw_moe_quant = request.moe_quantization.strip().lower() if request.moe_quantization else None
    if raw_moe_quant and raw_weight_quant != raw_moe_quant:
        raise BlisUnsupportedRequest("BLIS scenarios cannot separate MoE and dense weight quantization")
    moe_quant = _optional_quantization(request.moe_quantization, _WEIGHT_QUANTIZATION, "MoE quantization")

    name = _scenario_name(request)
    # 0.32.0 refuses an `auto` context window; state the same limit the response
    # reports as applied instead.
    max_model_len: int | str = (
        request.max_model_len
        if request.max_model_len is not None
        else (
            "auto"
            if engine_version_support(request.engine_version)["max_model_len_auto"]
            else request.isl + request.osl
        )
    )
    engine: dict[str, Any] = {
        "block_size": 16,
        "max_num_batched_tokens": request.max_num_batched_tokens or max(8192, request.isl),
        "max_num_seqs": request.max_num_seqs or request.concurrency,
        "max_model_len": max_model_len,
        "cudagraph_mode": "PIECEWISE",
        "gpu_memory_utilization": (
            request.gpu_memory_utilization
            if request.gpu_memory_utilization is not None
            else 0.92
        ),
    }
    quantization = normalize_weight_quantization(request.weight_quantization)
    cache_dtype = _optional_quantization(request.cache_dtype, _CACHE_DTYPE, "KV-cache dtype")
    if moe_quant and moe_quant != quantization:
        raise BlisUnsupportedRequest("BLIS scenarios cannot separate MoE and dense weight quantization")
    if quantization:
        engine["quantization"] = quantization
    if cache_dtype:
        engine["cache_dtype"] = cache_dtype
    # vLLM's schema default is prefix caching enabled. State it explicitly so
    # the emitted Scenario and normalized response agree on the applied setting.
    engine["enable_prefix_caching"] = request.prefix_caching if request.prefix_caching is not None else True

    scenario = {
        "kind": "Scenario",
        "name": name,
        "engine_version": request.engine_version,
        "model": request.model.catalog_name,
        "coefficients": list(_COEFFICIENTS),
        "cluster": {
            "hardware": request.hardware,
            "nodes": 1,
            "gpus_per_node": request.gpus_per_node,
        },
    }
    deployment = {
        "kind": "Deployment",
        "name": name,
        "pools": [
            {
                "role": "colocated",
                "nodes": 1,
                "parallel": {
                    "tp": request.tp,
                    "pp": request.pp,
                    "dp": request.dp,
                    "enable_expert_parallel": request.enable_expert_parallel,
                },
                "engine": engine,
            }
        ],
    }
    return scenario, deployment


def write_scenario(directory: str | Path, request: ScenarioRequest) -> str:
    """Write a uniquely named two-document YAML scenario and return its filename."""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    scenario, deployment = build_documents(request)
    filename = f"{scenario['name']}.yaml"
    target = target_dir / filename
    target.write_text(
        yaml.safe_dump_all(
            [scenario, deployment],
            explicit_start=False,
            sort_keys=False,
            allow_unicode=False,
        )
    )
    if request.prefill_pool is not None or request.decode_pool is not None:
        router_directory = target_dir / "router"
        router_directory.mkdir(parents=True, exist_ok=True)
        router_config = {
            "apiVersion": "llm-d.ai/v1",
            "kind": "EndpointPickerConfig",
            "plugins": [
                {"type": "approx-prefix-cache-producer", "parameters": {"maxPrefixTokensToMatch": 16384, "lruCapacityPerServer": 31250}},
                {"type": "prefix-cache-scorer"},
                {"type": "queue-scorer"},
                {"type": "prefill-filter"},
                {"type": "decode-filter"},
                {"type": "max-score-picker"},
                {"type": "always-disagg-pd-decider"},
                {"type": "disagg-profile-handler", "parameters": {"deciders": {"prefill": "always-disagg-pd-decider"}}},
            ],
            "schedulingProfiles": [
                {"name": "prefill", "plugins": [
                    {"pluginRef": "prefill-filter"},
                    {"pluginRef": "max-score-picker"},
                    {"pluginRef": "prefix-cache-scorer", "weight": 2},
                    {"pluginRef": "queue-scorer", "weight": 1},
                ]},
                {"name": "decode", "plugins": [
                    {"pluginRef": "decode-filter"},
                    {"pluginRef": "max-score-picker"},
                    {"pluginRef": "prefix-cache-scorer", "weight": 2},
                    {"pluginRef": "queue-scorer", "weight": 1},
                ]},
            ],
        }
        (router_directory / "pd-always.yaml").write_text(
            yaml.safe_dump(router_config, sort_keys=False, allow_unicode=False)
        )
    return filename


def engine_version_support(version: str) -> dict[str, bool]:
    """What BLIS models for ``version``; an unmodeled release supports nothing."""
    return dict(
        ENGINE_VERSION_SUPPORT.get(
            version, {"disaggregated": False, "max_model_len_auto": False}
        )
    )


def resolve_engine_version(backend_version: str | None) -> str:
    """Return the vLLM release to simulate, refusing ones BLIS does not model.

    A blank ``backend_version`` means "the image default". An explicit value must be
    an exact release BLIS models: upstream treats ``engine_version`` as a check, not
    a selector, so silently substituting a different release would change fidelity
    without saying so.
    """
    version = (
        (backend_version or "").strip()
        or os.environ.get("BLIS_ENGINE_VERSION", "").strip()
        or DEFAULT_ENGINE_VERSION
    )
    if version in _VERSION_ALIASES:
        raise BlisUnsupportedRequest(
            "BLIS requires an exact backend_version; it cannot resolve AISimulate version aliases"
        )
    if version not in ENGINE_VERSION_SUPPORT:
        modeled = ", ".join(MODELED_ENGINE_VERSIONS)
        raise BlisUnsupportedRequest(
            f"BLIS {blis_release()} models vLLM {modeled} only; requested backend_version "
            f"{version!r}. Leave backend_version blank to use {DEFAULT_ENGINE_VERSION}."
        )
    return version

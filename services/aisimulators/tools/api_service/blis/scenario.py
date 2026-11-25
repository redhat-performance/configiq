"""Build request-scoped BLIS Scenario and Deployment documents.

BLIS reads one Scenario and one Deployment per run. A Scenario states the
immutable problem — model, available cluster hardware, and the coefficient sets
the prediction is fitted to — and no engine release: the build fixes the engine,
so a document that names one is refused. `engine.kind` is the only engine fact a
scenario carries, and it is the engine kind the request selected. Each kind names
its settings in its own vocabulary, so the engine block is built per kind rather
than translated into one engine's field names.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .catalog import ModelResolution
from .errors import BlisUnsupportedRequest

#: Coefficient sets one run applies. v0.19 splits fabric and offload-tier pricing
#: out of the pre-0.19 sets, so a scenario must name all of them; a set fitted for
#: a different cost model does not belong to this one.
_COEFFICIENTS = [
    "cost-model-primitives",
    "cost-model-collectives",
    "cost-model-host-overheads",
    "cost-model-attention",
    "cost-model-recurrent",
    "cost-model-memory",
    "cost-model-fabric",
    "cost-model-offload-tiers",
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
#: The API's KV-cache spellings, mapped to the names ``engine.cache_dtype``
#: accepts. ``bf16`` and ``fp16`` are aliases the API accepts; the engine names
#: the formats ``bfloat16`` and ``float16``, and a document that says otherwise is
#: refused, so the alias is resolved here rather than forwarded.
_CACHE_DTYPE = {
    "half": None,
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp8": "fp8",
    "fp8_e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
    "nvfp4": "nvfp4",
}
#: SGLang names the same two settings in its own vocabulary and accepts a smaller
#: set of values. A request the engine cannot name is refused rather than mapped
#: onto the nearest one.
_SGLANG_WEIGHT_QUANTIZATION = {
    "bf16": None,
    "fp16": None,
    "half": None,
    "fp8": "fp8",
    "nvfp4": "modelopt_fp4",
}
_SGLANG_CACHE_DTYPE = {
    "half": None,
    "fp16": None,
    "bf16": "bf16",
    "fp8": "fp8_e4m3",
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
    #: The engine kind the request selected, named as BLIS names it.
    engine_kind: str
    model: ModelResolution
    hardware: str
    system: str
    gpus_per_node: int
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


def _separate_weight_quantization(
    weight_quantization: str | None,
    moe_quantization: str | None,
) -> None:
    """Refuse a request that names a different dense and MoE weight precision."""
    raw_weight = weight_quantization.strip().lower() if weight_quantization else None
    raw_moe = moe_quantization.strip().lower() if moe_quantization else None
    if raw_moe and raw_weight != raw_moe:
        raise BlisUnsupportedRequest("BLIS scenarios cannot separate MoE and dense weight quantization")


def _vllm_pool_engine(pool: ScenarioPoolRequest) -> dict[str, Any]:
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
        # The block takes vLLM's own setting names, nested as vLLM nests them;
        # a top-level `cudagraph_mode` is refused with the nested name.
        "compilation_config": {"cudagraph_mode": "PIECEWISE"},
    }
    # Only settings the request actually asked for are stated; the rest stay at
    # the engine's own defaults, which the build applies.
    if pool.gpu_memory_utilization is not None:
        engine["gpu_memory_utilization"] = pool.gpu_memory_utilization
    if weight_quant:
        engine["quantization"] = weight_quant
    if cache_dtype:
        engine["cache_dtype"] = cache_dtype
    if pool.prefix_caching is not None:
        engine["enable_prefix_caching"] = pool.prefix_caching
    return engine


def _sglang_pool_engine(pool: ScenarioPoolRequest) -> dict[str, Any]:
    weight_quant = _optional_quantization(
        pool.weight_quantization, _SGLANG_WEIGHT_QUANTIZATION, "weight quantization"
    )
    moe_quant = _optional_quantization(
        pool.moe_quantization, _SGLANG_WEIGHT_QUANTIZATION, "MoE quantization"
    )
    if moe_quant and moe_quant != weight_quant:
        raise BlisUnsupportedRequest("BLIS scenarios cannot separate MoE and dense weight quantization")
    cache_dtype = _optional_quantization(pool.cache_dtype, _SGLANG_CACHE_DTYPE, "KV-cache dtype")
    engine: dict[str, Any] = {
        "max_running_requests": pool.max_num_seqs,
        # SGLang's own name for the served window. The block must state it: a
        # graph that states no window leaves the engine none to resolve.
        "context_length": pool.max_model_len,
    }
    # Only settings the request actually asked for are stated; the rest stay at
    # the engine's own defaults, which the build applies and reports back.
    if pool.gpu_memory_utilization is not None:
        engine["mem_fraction_static"] = pool.gpu_memory_utilization
    if weight_quant:
        engine["quantization"] = weight_quant
    if cache_dtype:
        engine["kv_cache_dtype"] = cache_dtype
    if pool.prefix_caching is not None:
        engine["disable_radix_cache"] = not pool.prefix_caching
    return engine


#: How one engine kind names a pool's settings.
_POOL_ENGINE = {
    "vllm": _vllm_pool_engine,
    "sglang": _sglang_pool_engine,
}


def _pool_engine(pool: ScenarioPoolRequest, engine_kind: str) -> dict[str, Any]:
    builder = _POOL_ENGINE.get(engine_kind)
    if builder is None:
        available = ", ".join(sorted(_POOL_ENGINE))
        raise BlisUnsupportedRequest(
            f"BLIS scenario generation does not model engine kind {engine_kind!r}; "
            f"it can generate scenarios for: {available}"
        )
    return builder(pool)


def _vllm_colocated_engine(request: ScenarioRequest, max_model_len: int) -> dict[str, Any]:
    engine: dict[str, Any] = {
        "block_size": 16,
        "max_num_batched_tokens": request.max_num_batched_tokens or max(8192, request.isl),
        "max_num_seqs": request.max_num_seqs or request.concurrency,
        "max_model_len": max_model_len,
        "compilation_config": {"cudagraph_mode": "PIECEWISE"},
    }
    quantization = normalize_weight_quantization(request.weight_quantization)
    cache_dtype = _optional_quantization(request.cache_dtype, _CACHE_DTYPE, "KV-cache dtype")
    if quantization:
        engine["quantization"] = quantization
    if cache_dtype:
        engine["cache_dtype"] = cache_dtype
    # Only settings the request actually asked for are stated; the engine's own
    # defaults apply to the rest and the build reports back what it resolved.
    if request.gpu_memory_utilization is not None:
        engine["gpu_memory_utilization"] = request.gpu_memory_utilization
    if request.prefix_caching is not None:
        engine["enable_prefix_caching"] = request.prefix_caching
    return engine


def _sglang_colocated_engine(request: ScenarioRequest, max_model_len: int) -> dict[str, Any]:
    return _sglang_pool_engine(
        ScenarioPoolRequest(
            role="colocated",
            nodes=1,
            tp=request.tp,
            pp=request.pp,
            dp=request.dp,
            moe_tp_size=request.moe_tp_size,
            moe_ep_size=request.moe_ep_size,
            enable_expert_parallel=request.enable_expert_parallel,
            max_model_len=max_model_len,
            max_num_seqs=request.max_num_seqs or request.concurrency,
            max_num_batched_tokens=request.max_num_batched_tokens or max(8192, request.isl),
            gpu_memory_utilization=request.gpu_memory_utilization,
            weight_quantization=request.weight_quantization,
            moe_quantization=request.moe_quantization,
            cache_dtype=request.cache_dtype,
            prefix_caching=request.prefix_caching,
        )
    )


#: How one engine kind builds the aggregate pool's settings.
_COLOCATED_ENGINE = {
    "vllm": _vllm_colocated_engine,
    "sglang": _sglang_colocated_engine,
}


def _colocated_engine(request: ScenarioRequest, max_model_len: int) -> dict[str, Any]:
    builder = _COLOCATED_ENGINE.get(request.engine_kind)
    if builder is None:
        available = ", ".join(sorted(_COLOCATED_ENGINE))
        raise BlisUnsupportedRequest(
            f"BLIS scenario generation does not model engine kind {request.engine_kind!r}; "
            f"it can generate scenarios for: {available}"
        )
    _separate_weight_quantization(request.weight_quantization, request.moe_quantization)
    return builder(request, max_model_len)


def _build_disagg_documents(request: ScenarioRequest) -> tuple[dict[str, Any], dict[str, Any]]:
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
            "engine": _pool_engine(pool, request.engine_kind),
        })

    name = _scenario_name(request)
    scenario = {
        "kind": "Scenario",
        "name": name,
        "engine": {"kind": request.engine_kind},
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
    if request.prefill_pool is not None or request.decode_pool is not None:
        if request.gpus_per_node < 1:
            raise BlisUnsupportedRequest(f"BLIS has no valid node layout for system {request.system!r}")
        if request.enable_chunked_prefill is not None:
            raise BlisUnsupportedRequest("BLIS scenarios do not expose a chunked-prefill setting")
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
        raise BlisUnsupportedRequest("BLIS scenarios do not expose a chunked-prefill setting")
    if request.fmha_quantization:
        raise BlisUnsupportedRequest("BLIS scenarios do not expose an independent FMHA quantization setting")

    name = _scenario_name(request)
    # BLIS resolves the served window from the model graph, but some catalog
    # graphs omit the window. State the window the run serves instead: the
    # requested limit, the catalog model's native context, or the workload's
    # maximum sequence.
    max_model_len = (
        request.max_model_len
        if request.max_model_len is not None
        else request.model.context_length or request.isl + request.osl
    )


    scenario = {
        "kind": "Scenario",
        "name": name,
        "engine": {"kind": request.engine_kind},
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
                "engine": _colocated_engine(request, max_model_len),
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

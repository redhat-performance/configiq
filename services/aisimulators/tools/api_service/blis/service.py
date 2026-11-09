"""BLIS request orchestration, support gating, and bounded recommendation search."""

from __future__ import annotations

import math
import os
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from aisimulate_core.sdk.perf_database import load_system_spec

from .manifest import (
    BlisUnsupportedRequest,
    ModelResolution,
    SupportManifest,
    load_support_manifest,
)
from .runner import BlisExecutionError, RunRequest, run_blis
from .runtime import BlisRuntime, blis_release, resolve_runtime
from .scenario import (
    ScenarioPoolRequest,
    ScenarioRequest,
    disaggregation_refusal,
    engine_version_support,
    normalize_weight_quantization,
    resolve_engine_version,
    write_scenario,
)

_MAX_RECOMMEND_BUDGET_SECONDS = 60.0
_DEFAULT_RUN_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class EstimateInput:
    model_path: str
    system: str
    backend: str
    backend_version: str | None
    isl: int
    osl: int
    concurrency: int
    tp: int
    pp: int
    dp: int
    moe_tp_size: int | None = None
    moe_ep_size: int | None = None
    max_seq_len: int | None = None
    max_num_seqs: int | None = None
    gpu_memory_utilization: float | None = None
    gemm_quant_mode: str | None = None
    moe_quant_mode: str | None = None
    kvcache_quant_mode: str | None = None
    fmha_quant_mode: str | None = None
    prefix: int = 0
    prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    mode: str = "agg"
    decode_system: str | None = None
    fabric: str | None = None
    prefill_max_seq_len: int | None = None
    decode_max_seq_len: int | None = None
    prefill_gpu_memory_utilization: float | None = None
    decode_gpu_memory_utilization: float | None = None
    prefill_tp_size: int | None = None
    prefill_pp_size: int | None = None
    prefill_moe_tp_size: int | None = None
    prefill_moe_ep_size: int | None = None
    prefill_batch_size: int | None = None
    prefill_num_workers: int | None = None
    decode_tp_size: int | None = None
    decode_pp_size: int | None = None
    decode_moe_tp_size: int | None = None
    decode_moe_ep_size: int | None = None
    decode_batch_size: int | None = None
    decode_num_workers: int | None = None
    model_config_provided: bool = False
    unsupported_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecommendInput:
    model_path: str
    system: str
    backend: str
    backend_version: str | None
    isl: int
    osl: int
    concurrency: int | None
    request_rate: float | None
    ttft: float
    tpot: float
    request_latency: float | None
    prefix: int
    min_candidate_gpus: int | None
    max_candidate_gpus: int | None
    top_n: int
    search_budget_seconds: int
    max_seq_len: int | None = None
    prefill_max_seq_len: int | None = None
    decode_max_seq_len: int | None = None
    fabric: str | None = None
    database_mode: str = "HYBRID"
    inclusive_tpot: bool = False
    model_config_provided: bool = False


@dataclass(frozen=True)
class BlisPrediction:
    summary: dict[str, float | None]
    provenance: dict[str, str | None]
    serving_config: dict[str, Any] | None
    warnings: list[str]
    prefill_config: dict[str, Any] | None = None
    decode_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class BlisRecommendation:
    configs: list[dict[str, Any]]
    search_summary: dict[str, Any]
    provenance: dict[str, str | None]


@dataclass(frozen=True)
class _RecommendCandidate:
    mode: str
    total_gpus: int
    tp: int | None = None
    prefill_tp: int | None = None
    prefill_workers: int = 1
    decode_tp: int | None = None
    decode_workers: int = 1


@dataclass(frozen=True)
class _ResolvedRequest:
    manifest: SupportManifest
    runtime: BlisRuntime
    model: ModelResolution
    hardware: str
    gpus_per_node: int
    engine_version: str


def _resolve_request(model_path: str, system: str, backend: str, backend_version: str | None) -> _ResolvedRequest:
    if backend != "vllm":
        raise BlisUnsupportedRequest(
            f"BLIS scenarios currently model vLLM deployments, not backend {backend!r}; "
            "select vLLM or choose AISimulate explicitly"
        )
    manifest = load_support_manifest()
    model = manifest.resolve_model(model_path)
    hardware = manifest.resolve_hardware(system)
    try:
        system_spec = load_system_spec(system)
    except Exception as exc:
        raise BlisUnsupportedRequest(f"BLIS cannot load the GPU node layout for {system!r}: {exc}") from exc
    node = system_spec.get("node", {}) if isinstance(system_spec, dict) else {}
    gpus_per_node = int(node.get("num_gpus_per_node", 0)) if isinstance(node, dict) else 0
    if gpus_per_node < 1:
        raise BlisUnsupportedRequest(f"BLIS has no single-node GPU layout for {system!r}")
    return _ResolvedRequest(
        manifest=manifest,
        runtime=resolve_runtime(),
        model=model,
        hardware=hardware,
        gpus_per_node=gpus_per_node,
        engine_version=resolve_engine_version(backend_version),
    )


def _provenance(resolved: _ResolvedRequest, model_path: str, system: str) -> dict[str, str | None]:
    return {
        "simulator": "blis",
        "blis_version": blis_release(),
        "requested_model": model_path,
        "resolved_model": resolved.model.catalog_name,
        "model_source": resolved.model.origin,
        "system": system,
        "hardware": resolved.hardware,
        "model_config_relation": resolved.model.config_relation,
        "engine_version": resolved.engine_version,
    }


def _require_disaggregation(resolved: _ResolvedRequest) -> None:
    """Refuse a P/D topology before a run when the release cannot model one."""
    if not engine_version_support(resolved.engine_version)["disaggregated"]:
        raise BlisUnsupportedRequest(disaggregation_refusal(resolved.engine_version))


def supported_model_ids(manifest: SupportManifest | None = None) -> list[str]:
    """AISimulate model IDs BLIS can price, for the catalog endpoints.

    Only entries a picker can actually offer are listed: an upstream catalog model
    with no AISimulate ID is reachable by its catalog name, but ConfigIQ never
    offers that name, so advertising it would be a dead option.
    """
    manifest = manifest or load_support_manifest()
    return sorted(
        entry["aisimulate_model_id"]
        for entry in manifest.models.values()
        if entry.get("graph") is True and isinstance(entry.get("aisimulate_model_id"), str)
    )


def supported_system_ids(systems: Iterable[str], manifest: SupportManifest | None = None) -> list[str]:
    """The subset of ``systems`` whose hardware BLIS has a catalog chip for."""
    manifest = manifest or load_support_manifest()
    return [system for system in systems if manifest.supports_hardware(system)]


def _model_warnings(resolved: _ResolvedRequest, *, model_config_provided: bool = False) -> list[str]:
    warnings: list[str] = []
    relation = resolved.model.config_relation
    if relation and relation != "identical":
        warnings.append(f"The selected simulator uses a model configuration marked {relation}.")
    if resolved.model.origin == "aisimulate":
        warnings.append(
            "BLIS priced this model from a catalog entry derived at image build time "
            "from the AISimulate model configuration."
        )
    if model_config_provided:
        warnings.append("The selected simulator used its resolved catalog model; the supplied model_config was not applied.")
    return warnings


def _weight_quantization(gemm: str | None, moe: str | None) -> tuple[str | None, str | None]:
    normalized_gemm = gemm.strip().lower() if gemm else None
    normalized_moe = moe.strip().lower() if moe else None
    # The default UI MoE mode names the MXFP4 weight format; scenario engines
    # accept the weight format, not the serving kernel implementation.
    if normalized_moe == "w4a16_mxfp4":
        normalized_moe = "mxfp4"
    elif normalized_moe and normalized_moe not in {"fp16", "bf16", "half", "mxfp4", "nvfp4", "fp8"}:
        raise BlisUnsupportedRequest(
            f"BLIS cannot represent the requested MoE kernel quantization {moe!r}"
        )
    if normalized_moe and normalized_gemm != normalized_moe:
        raise BlisUnsupportedRequest("BLIS cannot represent separate GEMM and MoE quantization settings")
    return normalized_gemm, normalized_moe


def _scenario_request(
    resolved: _ResolvedRequest,
    *,
    system: str,
    isl: int,
    osl: int,
    concurrency: int,
    tp: int,
    pp: int,
    dp: int,
    moe_tp_size: int | None,
    moe_ep_size: int | None,
    max_seq_len: int | None,
    max_num_seqs: int | None,
    gpu_memory_utilization: float | None,
    gemm_quant_mode: str | None = None,
    moe_quant_mode: str | None = None,
    kvcache_quant_mode: str | None = None,
    fmha_quant_mode: str | None = None,
    prefix_caching: bool | None = None,
    enable_chunked_prefill: bool | None = None,
) -> ScenarioRequest:
    gemm_quant_mode, moe_quant_mode = _weight_quantization(gemm_quant_mode, moe_quant_mode)
    return ScenarioRequest(
        model=resolved.model,
        hardware=resolved.hardware,
        system=system,
        gpus_per_node=resolved.gpus_per_node,
        engine_version=resolved.engine_version,
        tp=tp,
        pp=pp,
        dp=dp,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        enable_expert_parallel=moe_ep_size is not None and moe_ep_size > 1,
        isl=isl,
        osl=osl,
        concurrency=concurrency,
        max_num_seqs=max_num_seqs,
        max_model_len=max_seq_len if max_seq_len is not None else isl + osl,
        max_num_batched_tokens=max(8192, isl),
        gpu_memory_utilization=gpu_memory_utilization,
        weight_quantization=gemm_quant_mode,
        moe_quantization=moe_quant_mode,
        cache_dtype=kvcache_quant_mode,
        prefix_caching=prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        fmha_quantization=fmha_quant_mode,
    )


def _metrics_summary(metrics: dict[str, Any], total_gpus: int) -> dict[str, float | None]:
    completed = metrics.get("completed_requests")
    if not isinstance(completed, (int, float)) or completed <= 0:
        dropped = metrics.get("dropped_unservable", 0)
        raise BlisUnsupportedRequest(
            f"BLIS completed no requests (dropped_unservable={dropped}); no performance estimate is available"
        )

    def number(name: str) -> float | None:
        value = metrics.get(name)
        return float(value) if isinstance(value, (int, float)) else None

    summary = {
        "ttft_ms": number("ttft_mean_ms"),
        "tpot_ms": number("itl_mean_ms"),
        "e2e_latency_ms": number("e2e_mean_ms"),
        "output_throughput_tok_s": number("tokens_per_sec"),
        "num_total_gpus": float(total_gpus),
        "power_w": None,
    }
    if summary["ttft_ms"] is None or summary["tpot_ms"] is None:
        raise BlisExecutionError("BLIS completed requests but omitted TTFT or ITL metrics")
    return summary


def _serving_config(scenario: ScenarioRequest) -> dict[str, Any]:
    weight_quantization = normalize_weight_quantization(scenario.weight_quantization)
    return {
        "backend": "vllm",
        "tensor_parallel_size": scenario.tp,
        "max_model_len": scenario.max_model_len or (scenario.isl + scenario.osl),
        "max_num_seqs": scenario.max_num_seqs or scenario.concurrency,
        "max_num_batched_tokens": scenario.max_num_batched_tokens or max(8192, scenario.isl),
        "gpu_memory_utilization": (
            scenario.gpu_memory_utilization
            if scenario.gpu_memory_utilization is not None
            else 0.92
        ),
        # Chunked prefill is not a Scenario field. Leave it unknown rather than
        # claiming a default that the generated document did not specify.
        "enable_chunked_prefill": None,
        "enable_prefix_caching": scenario.prefix_caching if scenario.prefix_caching is not None else True,
        "quantization": weight_quantization or "model default",
    }


def _run_scenario(
    resolved: _ResolvedRequest,
    scenario: ScenarioRequest,
    *,
    prefix: int,
    timeout_seconds: float,
    prefill_instances: int = 0,
    decode_instances: int = 0,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="configiq-blis-") as scenario_directory:
        filename = write_scenario(scenario_directory, scenario)
        return run_blis(
            resolved.runtime,
            RunRequest(
                scenario=filename,
                scenario_directory=scenario_directory,
                isl=scenario.isl,
                osl=scenario.osl,
                concurrency=scenario.concurrency,
                # BLIS treats concurrency as virtual users and num-requests as
                # the total simulation slice. The API uses a bounded smoke run
                # of one request per virtual user, matching its current input.
                num_requests=scenario.concurrency,
                prefix_tokens=prefix,
                prefill_instances=prefill_instances,
                decode_instances=decode_instances,
            ),
            timeout_seconds=timeout_seconds,
        )


def _estimate_disagg(request: EstimateInput, resolved: _ResolvedRequest) -> BlisPrediction:
    if request.decode_system and request.decode_system != request.system:
        raise BlisUnsupportedRequest(
            "BLIS scenarios use one cluster hardware type; decode_system must match system"
        )
    if not request.fabric:
        raise BlisUnsupportedRequest("BLIS disaggregated predictions require an explicit fabric")
    fabric = resolved.manifest.resolve_network(request.fabric)
    if request.dp != 1:
        raise BlisUnsupportedRequest("BLIS disaggregated predictions currently require attention_dp_size=1")
    if request.prefix > 0 and request.prefix_caching is False:
        raise BlisUnsupportedRequest("a shared prefix requires prefix caching to be enabled in BLIS")
    gemm_quant_mode, moe_quant_mode = _weight_quantization(request.gemm_quant_mode, request.moe_quant_mode)

    def make_pool(role: str) -> ScenarioPoolRequest:
        is_prefill = role == "prefill"
        tp = (
            (request.prefill_tp_size if is_prefill else request.decode_tp_size)
            or request.tp
        )
        pp = (
            (request.prefill_pp_size if is_prefill else request.decode_pp_size)
            or request.pp
        )
        moe_tp = (
            (request.prefill_moe_tp_size if is_prefill else request.decode_moe_tp_size)
            or request.moe_tp_size
        )
        moe_ep = (
            (request.prefill_moe_ep_size if is_prefill else request.decode_moe_ep_size)
            or request.moe_ep_size
        )
        workers = (request.prefill_num_workers if is_prefill else request.decode_num_workers) or 1
        batch_size = (request.prefill_batch_size if is_prefill else request.decode_batch_size) or request.concurrency
        context_length = (
            (request.prefill_max_seq_len if is_prefill else request.decode_max_seq_len)
            or request.max_seq_len
            or request.isl + request.osl
        )
        memory_utilization = (
            (request.prefill_gpu_memory_utilization if is_prefill else request.decode_gpu_memory_utilization)
            if (request.prefill_gpu_memory_utilization if is_prefill else request.decode_gpu_memory_utilization) is not None
            else request.gpu_memory_utilization
        )
        if min(tp, pp, workers, batch_size, context_length) < 1:
            raise BlisUnsupportedRequest(f"BLIS {role} pool values must be positive integers")
        nodes = math.ceil(workers * tp * pp / resolved.gpus_per_node)
        return ScenarioPoolRequest(
            role=role,
            nodes=nodes,
            tp=tp,
            pp=pp,
            dp=request.dp,
            moe_tp_size=moe_tp,
            moe_ep_size=moe_ep,
            enable_expert_parallel=moe_ep is not None and moe_ep > 1,
            max_model_len=context_length,
            max_num_seqs=request.max_num_seqs or batch_size,
            max_num_batched_tokens=max(8192, request.isl),
            gpu_memory_utilization=memory_utilization,
            weight_quantization=gemm_quant_mode,
            moe_quantization=moe_quant_mode,
            cache_dtype=request.kvcache_quant_mode,
            prefix_caching=True if request.prefix > 0 and request.prefix_caching is None else request.prefix_caching,
        )

    prefill = make_pool("prefill")
    decode = make_pool("decode")
    base = ScenarioRequest(
        model=resolved.model,
        hardware=resolved.hardware,
        system=request.system,
        gpus_per_node=resolved.gpus_per_node,
        engine_version=resolved.engine_version,
        tp=1,
        pp=1,
        dp=1,
        moe_tp_size=None,
        moe_ep_size=None,
        enable_expert_parallel=False,
        isl=request.isl,
        osl=request.osl,
        concurrency=request.concurrency,
        fabric=fabric,
        prefill_pool=prefill,
        decode_pool=decode,
        enable_chunked_prefill=request.enable_chunked_prefill,
        fmha_quantization=request.fmha_quant_mode,
    )
    scenario = base
    metrics = _run_scenario(
        resolved,
        scenario,
        prefix=request.prefix,
        timeout_seconds=float(os.environ.get("BLIS_RUN_TIMEOUT_SECONDS", _DEFAULT_RUN_TIMEOUT_SECONDS)),
        prefill_instances=request.prefill_num_workers or 1,
        decode_instances=request.decode_num_workers or 1,
    )
    total_gpus = (
        prefill.tp * prefill.pp * (request.prefill_num_workers or 1)
        + decode.tp * decode.pp * (request.decode_num_workers or 1)
    )

    def pool_config(pool: ScenarioPoolRequest, workers: int, batch_size: int | None) -> dict[str, Any]:
        return {
            "tp": pool.tp,
            "pp": pool.pp,
            "dp": pool.dp,
            "moe_tp": pool.moe_tp_size,
            "moe_ep": pool.moe_ep_size,
            "num_workers": workers,
            "batch_size": batch_size or request.concurrency,
            "backend_version": resolved.engine_version,
        }

    return BlisPrediction(
        summary=_metrics_summary(metrics, total_gpus),
        provenance=_provenance(resolved, request.model_path, request.system),
        serving_config=None,
        warnings=_model_warnings(resolved, model_config_provided=request.model_config_provided) + (
            [f"Simulator marked {int(metrics['dropped_unservable'])} request(s) unservable."]
            if isinstance(metrics.get("dropped_unservable"), (int, float)) and metrics["dropped_unservable"] > 0
            else []
        ),
        prefill_config=pool_config(prefill, request.prefill_num_workers or 1, request.prefill_batch_size),
        decode_config=pool_config(decode, request.decode_num_workers or 1, request.decode_batch_size),
    )


def estimate(request: EstimateInput) -> BlisPrediction:
    if request.unsupported_options:
        options = ", ".join(request.unsupported_options)
        raise BlisUnsupportedRequest(f"BLIS cannot represent these request options: {options}")
    if request.mode == "disagg":
        if request.decode_system and request.decode_system != request.system:
            raise BlisUnsupportedRequest(
                "BLIS scenarios use one cluster hardware type; decode_system must match system"
            )
        if not request.fabric:
            raise BlisUnsupportedRequest("BLIS disaggregated predictions require an explicit fabric")
        resolved = _resolve_request(request.model_path, request.system, request.backend, request.backend_version)
        _require_disaggregation(resolved)
        return _estimate_disagg(request, resolved)
    resolved = _resolve_request(request.model_path, request.system, request.backend, request.backend_version)
    if request.prefix > 0 and request.prefix_caching is False:
        raise BlisUnsupportedRequest("a shared prefix requires prefix caching to be enabled in BLIS")
    scenario = _scenario_request(
        resolved,
        system=request.system,
        isl=request.isl,
        osl=request.osl,
        concurrency=request.concurrency,
        tp=request.tp,
        pp=request.pp,
        dp=request.dp,
        moe_tp_size=request.moe_tp_size,
        moe_ep_size=request.moe_ep_size,
        max_seq_len=request.max_seq_len,
        max_num_seqs=request.max_num_seqs,
        gpu_memory_utilization=request.gpu_memory_utilization,
        gemm_quant_mode=request.gemm_quant_mode,
        moe_quant_mode=request.moe_quant_mode,
        kvcache_quant_mode=request.kvcache_quant_mode,
        fmha_quant_mode=request.fmha_quant_mode,
        prefix_caching=True if request.prefix > 0 and request.prefix_caching is None else request.prefix_caching,
        enable_chunked_prefill=request.enable_chunked_prefill,
    )
    metrics = _run_scenario(
        resolved,
        scenario,
        prefix=request.prefix,
        timeout_seconds=float(os.environ.get("BLIS_RUN_TIMEOUT_SECONDS", _DEFAULT_RUN_TIMEOUT_SECONDS)),
    )
    total_gpus = request.tp * request.pp * request.dp
    return BlisPrediction(
        summary=_metrics_summary(metrics, total_gpus),
        provenance=_provenance(resolved, request.model_path, request.system),
        serving_config=_serving_config(scenario),
        warnings=_model_warnings(resolved, model_config_provided=request.model_config_provided) + (
            [f"Simulator marked {int(metrics['dropped_unservable'])} request(s) unservable."]
            if isinstance(metrics.get("dropped_unservable"), (int, float)) and metrics["dropped_unservable"] > 0
            else []
        ),
    )


def _power_of_two_values(upper: int) -> list[int]:
    values: list[int] = []
    value = 1
    while value <= upper:
        values.append(value)
        value *= 2
    return values


def _pd_pool_shapes(gpus_per_node: int) -> list[tuple[int, int]]:
    """Return (TP, workers) shapes whose complete pool fits on one node."""
    return [
        (tp, workers)
        for tp in _power_of_two_values(gpus_per_node)
        for workers in _power_of_two_values(gpus_per_node // tp)
    ]


def _recommend_disagg_scenario(
    resolved: _ResolvedRequest,
    request: RecommendInput,
    *,
    concurrency: int,
    prefill_tp: int,
    prefill_workers: int,
    decode_tp: int,
    decode_workers: int,
    fabric: str,
) -> ScenarioRequest:
    def pool(role: str, tp: int, workers: int) -> ScenarioPoolRequest:
        role_context_length = request.prefill_max_seq_len if role == "prefill" else request.decode_max_seq_len
        context_length = role_context_length or request.max_seq_len or request.isl + request.osl
        return ScenarioPoolRequest(
            role=role,
            nodes=1,
            tp=tp,
            pp=1,
            dp=1,
            moe_tp_size=None,
            moe_ep_size=None,
            enable_expert_parallel=False,
            max_model_len=context_length,
            max_num_seqs=concurrency,
            max_num_batched_tokens=max(8192, request.isl),
            gpu_memory_utilization=None,
            weight_quantization=None,
            moe_quantization=None,
            cache_dtype=None,
            prefix_caching=True if request.prefix > 0 else None,
        )

    return ScenarioRequest(
        model=resolved.model,
        hardware=resolved.hardware,
        system=request.system,
        gpus_per_node=resolved.gpus_per_node,
        engine_version=resolved.engine_version,
        tp=1,
        pp=1,
        dp=1,
        moe_tp_size=None,
        moe_ep_size=None,
        enable_expert_parallel=False,
        isl=request.isl,
        osl=request.osl,
        concurrency=concurrency,
        fabric=fabric,
        prefill_pool=pool("prefill", prefill_tp, prefill_workers),
        decode_pool=pool("decode", decode_tp, decode_workers),
    )


def recommend(request: RecommendInput) -> BlisRecommendation:
    if request.request_rate is not None:
        raise BlisUnsupportedRequest(
            "BLIS recommendation currently searches concurrency workloads; choose a target concurrency "
            "or select AISimulate for request-rate search"
        )
    if request.database_mode != "HYBRID":
        raise BlisUnsupportedRequest("BLIS does not accept the AISimulate database_mode option")
    concurrency = request.concurrency or 1
    resolved = _resolve_request(request.model_path, request.system, request.backend, request.backend_version)
    include_disagg = bool(request.fabric)
    if include_disagg:
        _require_disaggregation(resolved)
    fabric = resolved.manifest.resolve_network(request.fabric) if request.fabric else None
    hardware_max = resolved.gpus_per_node * (2 if include_disagg else 1)
    upper = min(request.max_candidate_gpus or hardware_max, hardware_max)
    lower = request.min_candidate_gpus or 1
    if lower > upper:
        topology_limit = (
            f"one node per pool ({resolved.gpus_per_node} GPUs each)"
            if include_disagg
            else f"single-node hardware limit {resolved.gpus_per_node} GPUs"
        )
        raise BlisUnsupportedRequest(
            f"Requested minimum GPU count {lower} exceeds the effective BLIS search ceiling {upper} "
            f"(configured limit {request.max_candidate_gpus}, {topology_limit})"
        )

    candidates = [
        _RecommendCandidate(mode="agg", total_gpus=tp, tp=tp)
        for tp in _power_of_two_values(min(resolved.gpus_per_node, upper))
        if lower <= tp <= upper
    ]
    if include_disagg and fabric is not None:
        pool_shapes = _pd_pool_shapes(resolved.gpus_per_node)
        candidates.extend(
            _RecommendCandidate(
                mode="disagg",
                total_gpus=prefill_tp * prefill_workers + decode_tp * decode_workers,
                prefill_tp=prefill_tp,
                prefill_workers=prefill_workers,
                decode_tp=decode_tp,
                decode_workers=decode_workers,
            )
            for prefill_tp, prefill_workers in pool_shapes
            for decode_tp, decode_workers in pool_shapes
            if lower <= prefill_tp * prefill_workers + decode_tp * decode_workers <= upper
        )
    candidates.sort(key=lambda item: (
        item.total_gpus,
        0 if item.mode == "agg" else 1,
        abs(
            (item.prefill_tp or 0) * item.prefill_workers
            - (item.decode_tp or 0) * item.decode_workers
        ),
        item.prefill_tp or item.tp or 0,
        item.decode_tp or 0,
    ))

    total_budget_seconds = min(max(request.search_budget_seconds, 1), int(_MAX_RECOMMEND_BUDGET_SECONDS))
    started = time.monotonic()
    evaluated: list[_RecommendCandidate] = []
    feasible: list[dict[str, Any]] = []
    timed_out = False
    execution_failures: list[BlisExecutionError] = []

    for index, candidate in enumerate(candidates):
        remaining = total_budget_seconds - (time.monotonic() - started)
        if remaining <= 0:
            timed_out = True
            break
        timeout = min(
            float(os.environ.get("BLIS_RUN_TIMEOUT_SECONDS", _DEFAULT_RUN_TIMEOUT_SECONDS)),
            remaining,
        )
        if candidate.mode == "agg":
            scenario = _scenario_request(
                resolved,
                system=request.system,
                isl=request.isl,
                osl=request.osl,
                concurrency=concurrency,
                tp=candidate.tp or 1,
                pp=1,
                dp=1,
                moe_tp_size=None,
                moe_ep_size=None,
                max_seq_len=request.max_seq_len,
                max_num_seqs=concurrency,
                gpu_memory_utilization=None,
                prefix_caching=True if request.prefix > 0 else None,
            )
            prefill_instances = decode_instances = 0
        else:
            assert fabric is not None
            scenario = _recommend_disagg_scenario(
                resolved,
                request,
                concurrency=concurrency,
                prefill_tp=candidate.prefill_tp or 1,
                prefill_workers=candidate.prefill_workers,
                decode_tp=candidate.decode_tp or 1,
                decode_workers=candidate.decode_workers,
                fabric=fabric,
            )
            prefill_instances = candidate.prefill_workers
            decode_instances = candidate.decode_workers
        try:
            metrics = _run_scenario(
                resolved,
                scenario,
                prefix=request.prefix,
                timeout_seconds=timeout,
                prefill_instances=prefill_instances,
                decode_instances=decode_instances,
            )
        except BlisExecutionError as exc:
            evaluated.append(candidate)
            execution_failures.append(exc)
            continue
        evaluated.append(candidate)
        completed = metrics.get("completed_requests")
        if not isinstance(completed, (int, float)) or completed <= 0 or metrics.get("dropped_unservable", 0):
            continue
        ttft = metrics.get("ttft_mean_ms")
        itl = metrics.get("itl_mean_ms")
        e2e = metrics.get("e2e_mean_ms")
        if not all(isinstance(value, (int, float)) for value in (ttft, itl, e2e)):
            continue
        if request.request_latency is not None:
            if e2e > request.request_latency:
                continue
        elif ttft > request.ttft or itl > request.tpot:
            continue

        tokens_per_second = metrics.get("tokens_per_sec")
        reported_tpot = (
            (float(ttft) + float(itl) * (request.osl - 1)) / request.osl
            if request.inclusive_tpot and request.osl > 0
            else float(itl)
        )
        common_config: dict[str, Any] = {
            "total_gpus_needed": candidate.total_gpus,
            "replicas_needed": 1,
            "num_total_gpus": candidate.total_gpus,
            "bs": concurrency,
            "ttft": float(ttft),
            "tpot": reported_tpot,
            "request_latency": float(e2e),
            "tokens_per_second": float(tokens_per_second) if isinstance(tokens_per_second, (int, float)) else None,
            "tokens_per_second_per_gpu": (
                float(tokens_per_second) / candidate.total_gpus
                if isinstance(tokens_per_second, (int, float)) and candidate.total_gpus > 0
                else None
            ),
            "tokens_per_second_per_user": (
                float(tokens_per_second) / concurrency
                if isinstance(tokens_per_second, (int, float)) and concurrency > 0
                else None
            ),
            "concurrency": concurrency,
            "model": request.model_path,
            "system": request.system,
            "backend": request.backend,
            "backend_version": resolved.engine_version,
            "mode": candidate.mode,
            "simulator": "blis",
            "provenance": _provenance(resolved, request.model_path, request.system),
            "warnings": _model_warnings(resolved, model_config_provided=request.model_config_provided),
        }
        if candidate.mode == "agg":
            common_config.update({"tp": candidate.tp, "pp": 1, "dp": 1})
            if not include_disagg:
                common_config["warnings"].append(
                    "No network fabric was selected; BLIS P/D candidates were not searched."
                )
        else:
            common_config.update({
                "prefill_config": {
                    "tp": candidate.prefill_tp,
                    "pp": 1,
                    "dp": 1,
                    "num_workers": candidate.prefill_workers,
                    "batch_size": concurrency,
                    "backend_version": resolved.engine_version,
                },
                "decode_config": {
                    "tp": candidate.decode_tp,
                    "pp": 1,
                    "dp": 1,
                    "num_workers": candidate.decode_workers,
                    "batch_size": concurrency,
                    "backend_version": resolved.engine_version,
                },
            })
        feasible.append(common_config)

        at_end_of_gpu_group = index + 1 == len(candidates) or candidates[index + 1].total_gpus > candidate.total_gpus
        if at_end_of_gpu_group and len(feasible) >= request.top_n:
            break

    if not feasible and candidates and len(execution_failures) == len(evaluated):
        raise execution_failures[-1]
    feasible.sort(key=lambda config: (
        config["total_gpus_needed"],
        -(config["tokens_per_second"] or 0),
        config["mode"],
    ))
    feasible = feasible[:request.top_n]
    modes_searched = ["agg", "disagg"] if include_disagg else ["agg"]
    return BlisRecommendation(
        configs=feasible,
        provenance=_provenance(resolved, request.model_path, request.system),
        search_summary={
            "strategy": "aggregate_tp_and_pd_grid" if include_disagg else "tensor_parallel_only",
            "modes_searched": modes_searched,
            "candidate_gpus": sorted({candidate.total_gpus for candidate in candidates}),
            "evaluated_gpus": sorted({candidate.total_gpus for candidate in evaluated}),
            "max_candidates": len(candidates),
            "candidate_configurations": len(candidates),
            "evaluated_configurations": len(evaluated),
            "configured_max_gpus": request.max_candidate_gpus or hardware_max,
            "hardware_max_gpus": hardware_max,
            "effective_max_gpus": upper,
            "time_budget_seconds": total_budget_seconds,
            "budget_exhausted": timed_out,
        },
    )

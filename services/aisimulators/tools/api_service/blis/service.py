"""BLIS request orchestration, support gating, and bounded recommendation search."""

from __future__ import annotations

import math
import os
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from aisimulate_core.sdk.perf_database import load_system_spec

from .manifest import (
    BlisCatalog,
    BlisUnsupportedRequest,
    ModelResolution,
    load_catalog,
)
from .runner import (
    BlisExecutionError,
    BlisTimeoutError,
    RunRequest,
    normalize_result,
    run_blis,
)
from .runtime import BlisRuntime, blis_release, resolve_runtime
from .scenario import (
    ScenarioPoolRequest,
    ScenarioRequest,
    disaggregation_refusal,
    engine_version_support,
    resolve_engine_version,
    write_scenario,
)

_MAX_RECOMMEND_BUDGET_SECONDS = 60.0
_DEFAULT_RUN_TIMEOUT_SECONDS = 60.0
#: Length of the open-loop slice replayed for a throughput-demand candidate.
#: The request count scales with the offered rate so the measured window is the
#: same at every target instead of shrinking as the rate rises.
_DEMAND_WINDOW_SECONDS = 5.0
#: Safety bound on the generated slice. A very high target still measures over
#: the actual run window; this only keeps one candidate from ballooning.
_DEFAULT_DEMAND_MAX_REQUESTS = 5000


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _demand_slice(rate: float) -> tuple[float, int]:
    """Return `(window_seconds, num_requests)` for an open-loop demand run."""
    window = _positive_float_env("BLIS_DEMAND_WINDOW_SECONDS", _DEMAND_WINDOW_SECONDS)
    cap = int(_positive_float_env("BLIS_DEMAND_MAX_REQUESTS", _DEFAULT_DEMAND_MAX_REQUESTS))
    return window, min(max(4, round(rate * window)), cap)


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
    ttft: float | None
    tpot: float | None
    request_latency: float | None
    prefix: int
    min_candidate_gpus: int | None
    max_candidate_gpus: int | None
    top_n: int
    search_budget_seconds: int
    # Open-loop demand sizing: replay `request_rate`, ignore latency targets.
    demand: bool = False
    # Memory-residency sizing: hold `concurrency` full-context sequences.
    capacity: bool = False
    # Enumerate topology/load combinations and return the efficiency frontier.
    pareto: bool = False
    # Whether latency bounds gate feasibility (pareto only).
    strict_sla: bool = False
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
class BlisDemandOutcome:
    target_rps: float
    achieved_rps: float | None
    met: bool
    observed_concurrency: float | None
    window_seconds: float | None
    completed_requests: int | None
    evaluated_configurations: int | None
    best_observed_rps: float | None
    simulator: str = "blis"


@dataclass(frozen=True)
class BlisCapacityOutcome:
    """Memory-residency outcome reported by the BLIS capacity search.

    Capacity is read from the KV pool BLIS reports for the run
    (``engine.cluster.kv_capacity_tokens_per_replica``), then multiplied by
    the replicas needed to hold the requested sequences. No KV bytes are
    re-derived from weights or a memory fraction, and no performance value is
    invented: the search does not simulate the target concurrency.
    """

    target_concurrency: int
    context_per_sequence_tokens: int
    capacity_tokens: int
    capacity_sequences: int
    fits: bool
    shortfall_sequences: int | None
    shortfall_tokens: int | None
    minimum_gpus: int | None
    per_rank_tokens: int
    replicas: int
    attention_dp: int
    evaluated_shapes: int
    feasible_shapes: int
    exhaustive: bool = True
    performance_available: bool = False
    performance_unavailable_reason: str | None = None
    mode: str = "agg"
    simulator: str = "blis"


@dataclass(frozen=True)
class BlisRecommendation:
    configs: list[dict[str, Any]]
    search_summary: dict[str, Any]
    provenance: dict[str, str | None]
    demand: BlisDemandOutcome | None = None
    capacity: BlisCapacityOutcome | None = None
    #: Pareto ledger entries shaped like the API's CandidateResponse.
    candidates: list[dict[str, Any]] = field(default_factory=list)
    #: Candidate-ID lists for the simulator-computed Pareto views.
    views: dict[str, list[str]] | None = None


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
    catalog: BlisCatalog
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
    catalog = load_catalog()
    model = catalog.resolve_model(model_path)
    hardware = catalog.resolve_hardware(system)
    try:
        system_spec = load_system_spec(system)
    except Exception as exc:
        raise BlisUnsupportedRequest(f"BLIS cannot load the GPU node layout for {system!r}: {exc}") from exc
    node = system_spec.get("node", {}) if isinstance(system_spec, dict) else {}
    gpus_per_node = int(node.get("num_gpus_per_node", 0)) if isinstance(node, dict) else 0
    if gpus_per_node < 1:
        raise BlisUnsupportedRequest(f"BLIS has no single-node GPU layout for {system!r}")
    return _ResolvedRequest(
        catalog=catalog,
        runtime=resolve_runtime(),
        model=model,
        hardware=hardware,
        gpus_per_node=gpus_per_node,
        engine_version=resolve_engine_version(backend_version),
    )


def _provenance(resolved: _ResolvedRequest, model_path: str, system: str) -> dict[str, str | None]:
    return {
        "simulator": "blis",
        "simulator_version": blis_release(),
        "requested_model": model_path,
        "resolved_model": resolved.model.catalog_name,
        "model_source": resolved.model.origin,
        "system": system,
        "hardware": resolved.hardware,
        "model_config_relation": None,
        "engine_version": resolved.engine_version,
    }


def _require_disaggregation(resolved: _ResolvedRequest) -> None:
    """Refuse a P/D topology before a run when the release cannot model one."""
    if not engine_version_support(resolved.engine_version)["disaggregated"]:
        raise BlisUnsupportedRequest(disaggregation_refusal(resolved.engine_version))


def supported_model_ids(catalog: BlisCatalog | None = None) -> list[str]:
    """Model identifiers the BLIS catalog can price, for the catalog endpoints."""
    catalog = catalog or load_catalog()
    return catalog.model_ids()


def supported_system_ids(systems: Iterable[str], catalog: BlisCatalog | None = None) -> list[str]:
    """The subset of ``systems`` whose hardware BLIS has a catalog chip for."""
    catalog = catalog or load_catalog()
    return [system for system in systems if catalog.supports_hardware(system)]


def _model_warnings(resolved: _ResolvedRequest, *, model_config_provided: bool = False) -> list[str]:
    warnings: list[str] = []
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
        max_model_len=max_seq_len,
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


def _deployment_pools(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The Deployment pools BLIS echoed in `Result.input.deployment`."""
    input_document = result.get("input") if isinstance(result.get("input"), dict) else {}
    deployment = input_document.get("deployment") if isinstance(input_document.get("deployment"), dict) else {}
    pools = deployment.get("pools")
    return [pool for pool in pools if isinstance(pool, dict)] if isinstance(pools, list) else []


def _engine_serving_config(engine: dict[str, Any], parallel: dict[str, Any], backend_version: str) -> dict[str, Any]:
    """Report the engine settings BLIS resolved, not a reconstruction of them."""
    max_model_len = engine.get("max_model_len")
    max_model_len_policy: str | None = None
    if max_model_len == "auto":
        # `auto` is the simulator's own policy, so report the policy rather than
        # inventing a token window the run did not state.
        max_model_len = None
        max_model_len_policy = "auto"
    return {
        "backend": "vllm",
        "backend_version": backend_version,
        "tensor_parallel_size": parallel.get("tp") or 1,
        "max_model_len": max_model_len if isinstance(max_model_len, int) else None,
        "max_model_len_policy": max_model_len_policy,
        "max_num_seqs": engine.get("max_num_seqs"),
        "max_num_batched_tokens": engine.get("max_num_batched_tokens"),
        "gpu_memory_utilization": engine.get("gpu_memory_utilization"),
        # Chunked prefill is not a Scenario field. Leave it unknown rather than
        # claiming a default that the generated document did not specify.
        "enable_chunked_prefill": None,
        "enable_prefix_caching": engine.get("enable_prefix_caching"),
        "quantization": engine.get("quantization") or "model default",
    }


def _serving_config(result: dict[str, Any]) -> dict[str, Any] | None:
    pools = _deployment_pools(result)
    if not pools:
        return None
    pool = pools[0]
    engine = pool.get("engine") if isinstance(pool.get("engine"), dict) else {}
    parallel = pool.get("parallel") if isinstance(pool.get("parallel"), dict) else {}
    provenance_version = result.get("revisions") if isinstance(result.get("revisions"), dict) else {}
    engine_revision = provenance_version.get("engine") if isinstance(provenance_version.get("engine"), dict) else {}
    backend_version = engine_revision.get("upstream_version")
    return _engine_serving_config(engine, parallel, backend_version if isinstance(backend_version, str) else "unknown")


def _run_scenario(
    resolved: _ResolvedRequest,
    scenario: ScenarioRequest,
    *,
    prefix: int,
    timeout_seconds: float,
    prefill_instances: int = 0,
    decode_instances: int = 0,
    rate: float | None = None,
    num_requests: int | None = None,
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
                # Closed-loop runs use concurrency as virtual users. Open-loop
                # demand runs send only the arrival rate, because BLIS refuses
                # a command that sets both.
                concurrency=None if rate is not None else scenario.concurrency,
                rate=rate,
                # BLIS treats concurrency as virtual users and num-requests as
                # the total simulation slice. Closed-loop recommendations use a
                # bounded smoke run of one request per virtual user.
                num_requests=num_requests if num_requests is not None else scenario.concurrency,
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
    fabric = resolved.catalog.resolve_network(request.fabric)
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
    result = _run_scenario(
        resolved,
        scenario,
        prefix=request.prefix,
        timeout_seconds=float(os.environ.get("BLIS_RUN_TIMEOUT_SECONDS", _DEFAULT_RUN_TIMEOUT_SECONDS)),
        prefill_instances=request.prefill_num_workers or 1,
        decode_instances=request.decode_num_workers or 1,
    )
    metrics = normalize_result(result)
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
    result = _run_scenario(
        resolved,
        scenario,
        prefix=request.prefix,
        timeout_seconds=float(os.environ.get("BLIS_RUN_TIMEOUT_SECONDS", _DEFAULT_RUN_TIMEOUT_SECONDS)),
    )
    metrics = normalize_result(result)
    total_gpus = request.tp * request.pp * request.dp
    return BlisPrediction(
        summary=_metrics_summary(metrics, total_gpus),
        provenance=_provenance(resolved, request.model_path, request.system),
        serving_config=_serving_config(result),
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


def _search_summary(
    request: RecommendInput,
    candidates: list[_RecommendCandidate],
    evaluated: list[_RecommendCandidate],
    include_disagg: bool,
    hardware_max: int,
    upper: int,
    total_budget_seconds: int,
    timed_out: bool,
) -> dict[str, Any]:
    return {
        "strategy": "aggregate_tp_and_pd_grid" if include_disagg else "tensor_parallel_only",
        "modes_searched": ["agg", "disagg"] if include_disagg else ["agg"],
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
    }


def _run_timeout_seconds() -> float:
    return float(os.environ.get("BLIS_RUN_TIMEOUT_SECONDS", _DEFAULT_RUN_TIMEOUT_SECONDS))


def _recommend_candidates(
    resolved: _ResolvedRequest,
    *,
    include_disagg: bool,
    lower: int,
    upper: int,
) -> list[_RecommendCandidate]:
    """The bounded aggregate-TP and (optionally) P/D search grid."""
    candidates = [
        _RecommendCandidate(mode="agg", total_gpus=tp, tp=tp)
        for tp in _power_of_two_values(min(resolved.gpus_per_node, upper))
        if lower <= tp <= upper
    ]
    if include_disagg:
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
    return candidates


def _candidate_scenario(
    resolved: _ResolvedRequest,
    request: RecommendInput,
    candidate: _RecommendCandidate,
    *,
    concurrency: int,
    fabric: str | None,
) -> tuple[ScenarioRequest, int, int]:
    """Build one candidate's scenario and its per-pool instance counts."""
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
        return scenario, 0, 0
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
    return scenario, candidate.prefill_workers, candidate.decode_workers


def _window_bounds(
    request: RecommendInput,
    resolved: _ResolvedRequest,
    *,
    include_disagg: bool,
    capacity: bool = False,
) -> tuple[int, int, int]:
    """Resolve the ``(lower, upper, hardware_max)`` GPU window."""
    hardware_max = resolved.gpus_per_node * (2 if include_disagg else 1)
    if capacity:
        # Capacity answers in replicas, so its window totals GPUs across
        # replicas instead of being capped at one node. Only one BLIS run per
        # shape is needed: that run reports the per-replica KV pool, and the
        # replica count is computed. The configured ceiling bounds the answer.
        upper = request.max_candidate_gpus or hardware_max
    else:
        upper = min(request.max_candidate_gpus or hardware_max, hardware_max)
    lower = request.min_candidate_gpus or 1
    if lower > upper:
        topology_limit = (
            "capacity replicas may span multiple nodes"
            if capacity
            else f"one node per pool ({resolved.gpus_per_node} GPUs each)"
            if include_disagg
            else f"single-node hardware limit {resolved.gpus_per_node} GPUs"
        )
        raise BlisUnsupportedRequest(
            f"Requested minimum GPU count {lower} exceeds the effective BLIS search ceiling {upper} "
            f"(configured limit {request.max_candidate_gpus}, {topology_limit})"
        )
    return lower, upper, hardware_max


def _capacity_config(
    request: RecommendInput,
    resolved: _ResolvedRequest,
    *,
    tp: int,
    replicas: int,
    target_concurrency: int,
) -> dict[str, Any]:
    total_gpus = tp * replicas
    return {
        "total_gpus_needed": total_gpus,
        "replicas_needed": replicas,
        "num_total_gpus": total_gpus,
        "tp": tp,
        "pp": 1,
        "dp": 1,
        "bs": target_concurrency,
        "concurrency": target_concurrency,
        "model": request.model_path,
        "system": request.system,
        "backend": request.backend,
        "backend_version": resolved.engine_version,
        "mode": "agg",
        "simulator": "blis",
        "provenance": _provenance(resolved, request.model_path, request.system),
        "warnings": _model_warnings(
            resolved, model_config_provided=request.model_config_provided
        ) + [
            "BLIS capacity sizing reads the engine's KV pool and does not simulate the target concurrency."
        ],
    }


def _recommend_capacity(request: RecommendInput) -> BlisRecommendation:
    """Size the fewest aggregate GPUs whose BLIS KV pool holds the target.

    BLIS reports the engine's KV pool in tokens. Capacity is that pool times
    the replicas needed for the requested sequences, never a KV byte estimate
    reconstructed from weights. The search is aggregate-only: a disaggregated
    pool holds resident sequences on its decode side, which BLIS does not
    expose as a capacity input, so the service refuses rather than inventing
    one.
    """
    if request.request_rate is not None:
        raise BlisUnsupportedRequest(
            "BLIS concurrency-capacity sizing does not accept target_request_rate"
        )
    if request.concurrency is None or request.concurrency < 1:
        raise BlisUnsupportedRequest(
            "BLIS concurrency-capacity sizing requires a positive target_concurrency"
        )
    if request.request_latency is not None or request.ttft is not None or request.tpot is not None:
        raise BlisUnsupportedRequest(
            "BLIS concurrency-capacity sizing does not apply latency targets; "
            "omit ttft, tpot, and request_latency"
        )
    if request.database_mode != "HYBRID":
        raise BlisUnsupportedRequest("BLIS does not accept the AISimulate database_mode option")
    resolved = _resolve_request(
        request.model_path, request.system, request.backend, request.backend_version
    )
    context_tokens = int(request.max_seq_len or (request.isl + request.osl))
    if context_tokens < 1:
        raise BlisUnsupportedRequest(
            "BLIS concurrency-capacity sizing requires a positive per-sequence context length"
        )
    target = int(request.concurrency)
    lower, upper, hardware_max = _window_bounds(
        request, resolved, include_disagg=False, capacity=True
    )
    # A shape narrower than the window floor can still reach it with replicas,
    # so enumerate every aggregate shape and resolve replicas per shape. The
    # floor is applied by the replica math below, not by dropping the shape.
    candidates = _recommend_candidates(
        resolved, include_disagg=False, lower=1, upper=upper
    )
    if not candidates:
        raise BlisUnsupportedRequest(
            f"No aggregate BLIS shape fits the requested {lower}-{upper} GPU window"
        )

    total_budget_seconds = min(max(request.search_budget_seconds, 1), int(_MAX_RECOMMEND_BUDGET_SECONDS))
    started = time.monotonic()
    evaluated: list[_RecommendCandidate] = []
    measured: list[tuple[_RecommendCandidate, int]] = []
    execution_failures: list[BlisExecutionError] = []
    timed_out = False

    for candidate in candidates:
        remaining = total_budget_seconds - (time.monotonic() - started)
        if remaining <= 0:
            timed_out = True
            break
        scenario, prefill_instances, decode_instances = _candidate_scenario(
            resolved,
            request,
            candidate,
            # The KV pool is a property of the deployment shape, not of the
            # offered load. One request is enough to have BLIS construct and
            # report the engine's pool.
            concurrency=1,
            fabric=None,
        )
        try:
            metrics = normalize_result(_run_scenario(
                resolved,
                scenario,
                prefix=request.prefix,
                timeout_seconds=min(_run_timeout_seconds(), remaining),
                prefill_instances=prefill_instances,
                decode_instances=decode_instances,
            ))
        except BlisExecutionError as exc:
            evaluated.append(candidate)
            execution_failures.append(exc)
            continue
        evaluated.append(candidate)
        capacity_tokens = metrics.get("kv_capacity_tokens_per_replica")
        if not isinstance(capacity_tokens, (int, float)) or capacity_tokens <= 0:
            execution_failures.append(BlisExecutionError(
                "BLIS did not report engine.cluster.kv_capacity_tokens_per_replica"
            ))
            continue
        measured.append((candidate, int(capacity_tokens)))

    if not measured:
        if execution_failures and len(execution_failures) >= len(evaluated):
            raise execution_failures[-1]
        raise BlisUnsupportedRequest(
            "BLIS reported no KV pool within the capacity search budget"
            if timed_out
            else "BLIS did not report a KV pool for any aggregate shape in the requested window"
        )

    # Resolve each shape's replicas inside the GPU window exactly like the
    # AISimulate capacity search: the smallest deployment that holds the target,
    # and the largest deployment of each shape as the best-effort fallback.
    fitting: list[dict[str, Any]] = []
    best_effort: list[dict[str, Any]] = []
    minimum_gpus: int | None = None
    feasible_shapes = 0
    for candidate, capacity_tokens in measured:
        tp = candidate.total_gpus
        if tp <= 0 or tp > upper:
            continue
        min_replicas = max(1, math.ceil(lower / tp))
        if tp * min_replicas > upper:
            continue
        max_replicas = upper // tp
        can_hold_one = capacity_tokens >= context_tokens
        if can_hold_one:
            feasible_shapes += 1
            single_gpus = tp * min_replicas
            if minimum_gpus is None or single_gpus < minimum_gpus:
                minimum_gpus = single_gpus
        needed_replicas = max(
            1, math.ceil((target * context_tokens) / capacity_tokens)
        )
        first_fitting = max(min_replicas, needed_replicas)
        row = {
            "tp": tp,
            "replicas": 0,
            "capacity_tokens": 0,
            "capacity_sequences": 0,
        }
        if first_fitting <= max_replicas:
            fitting.append({
                **row,
                "replicas": first_fitting,
                "capacity_tokens": capacity_tokens * first_fitting,
                "capacity_sequences": (capacity_tokens * first_fitting) // context_tokens,
            })
        if max_replicas >= min_replicas:
            best_effort.append({
                **row,
                "replicas": max_replicas,
                "capacity_tokens": capacity_tokens * max_replicas,
                "capacity_sequences": (capacity_tokens * max_replicas) // context_tokens,
            })

    if minimum_gpus is None:
        raise BlisUnsupportedRequest(
            f"BLIS found no aggregate configuration that holds one {context_tokens}-token "
            f"sequence within {lower}-{upper} GPUs"
        )

    if fitting:
        winner = min(
            fitting,
            key=lambda row: (
                row["tp"] * row["replicas"],
                -row["capacity_sequences"],
                row["tp"],
                row["replicas"],
            ),
        )
    else:
        winner = max(
            best_effort,
            key=lambda row: (
                row["capacity_sequences"],
                -row["tp"] * row["replicas"],
                row["tp"],
            ),
        )

    fits = winner["capacity_sequences"] >= target
    needed_tokens = target * context_tokens
    capacity = BlisCapacityOutcome(
        target_concurrency=target,
        context_per_sequence_tokens=context_tokens,
        capacity_tokens=winner["capacity_tokens"],
        capacity_sequences=winner["capacity_sequences"],
        fits=fits,
        shortfall_sequences=None if fits else max(0, target - winner["capacity_sequences"]),
        shortfall_tokens=None if fits else max(0, needed_tokens - winner["capacity_tokens"]),
        minimum_gpus=minimum_gpus,
        # With attention DP=1 the aggregate pool BLIS reports is the replica's
        # whole KV capacity, so the per-rank figure equals the per-replica one.
        per_rank_tokens=winner["capacity_tokens"] // winner["replicas"],
        replicas=winner["replicas"],
        attention_dp=1,
        evaluated_shapes=len(evaluated),
        feasible_shapes=feasible_shapes,
        # The enumeration only counts as exhaustive when every shape ran and
        # reported a pool; a timeout or failed run leaves the minimum unknown.
        exhaustive=not timed_out and not execution_failures,
        performance_available=False,
        performance_unavailable_reason=(
            "BLIS capacity sizing reads the engine's KV pool; it does not simulate "
            "the target concurrency."
        ),
        mode="agg",
        simulator="blis",
    )
    summary = _search_summary(
        request,
        candidates,
        evaluated,
        False,
        hardware_max,
        upper,
        total_budget_seconds,
        timed_out,
    )
    summary["strategy"] = "aggregate_tp_capacity"
    # Capacity bounds total GPUs across replicas; `hardware_max_gpus` is the
    # node width one replica can use.
    return BlisRecommendation(
        configs=[_capacity_config(
            request,
            resolved,
            tp=winner["tp"],
            replicas=winner["replicas"],
            target_concurrency=target,
        )],
        search_summary=summary,
        provenance=_provenance(resolved, request.model_path, request.system),
        capacity=capacity,
    )


#: Load levels a BLIS Pareto sweep evaluates for every topology. BLIS drives
#: closed-loop concurrency, so these are the sweep's load points; the service
#: enumerates the grid rather than sampling it.
_PARETO_CONCURRENCY_LEVELS = (1, 2, 4, 8, 16, 32, 64)


def _pareto_concurrency_levels() -> list[int]:
    raw = os.environ.get("BLIS_PARETO_CONCURRENCY_LEVELS")
    if raw:
        values = sorted({
            int(part)
            for part in raw.split(",")
            if part.strip().isdigit() and int(part) > 0
        })
        if values:
            return values[:12]
    return list(_PARETO_CONCURRENCY_LEVELS)


def _pareto_candidate_id(candidate: _RecommendCandidate, concurrency: int) -> str:
    if candidate.mode == "agg":
        return f"agg-tp{candidate.tp or 1}-c{concurrency}"
    return (
        f"pd-p{candidate.prefill_tp or 1}x{candidate.prefill_workers}"
        f"-d{candidate.decode_tp or 1}x{candidate.decode_workers}-c{concurrency}"
    )


def _pareto_topology(candidate: _RecommendCandidate, fabric: str | None) -> dict[str, Any]:
    if candidate.mode == "agg":
        return {
            "deployment_mode": "agg",
            "tp": candidate.tp or 1,
            "pp": 1,
            "attention_dp": 1,
            "replicas": 1,
        }
    return {
        "deployment_mode": "disagg",
        "prefill_tp": candidate.prefill_tp or 1,
        "prefill_pp": 1,
        "prefill_attention_dp": 1,
        "prefill_replicas": candidate.prefill_workers,
        "decode_tp": candidate.decode_tp or 1,
        "decode_pp": 1,
        "decode_attention_dp": 1,
        "decode_replicas": candidate.decode_workers,
        "fabric": fabric,
    }


def _pareto_metrics(metrics: dict[str, Any], concurrency: int) -> dict[str, float | None]:
    def number(name: str) -> float | None:
        value = metrics.get(name)
        return float(value) if isinstance(value, (int, float)) else None

    return {
        "ttft_ms": number("ttft_mean_ms"),
        "tpot_ms": number("itl_mean_ms"),
        "request_latency_ms": number("e2e_mean_ms"),
        "tokens_per_second": number("tokens_per_sec"),
        "concurrency": float(concurrency),
    }


def _pareto_config(
    request: RecommendInput,
    resolved: _ResolvedRequest,
    candidate: _RecommendCandidate,
    *,
    concurrency: int,
    metrics: dict[str, float | None],
) -> dict[str, Any]:
    ttft = metrics.get("ttft_ms")
    tpot = metrics.get("tpot_ms")
    e2e = metrics.get("request_latency_ms")
    tokens_per_second = metrics.get("tokens_per_second")
    total_gpus = candidate.total_gpus
    config: dict[str, Any] = {
        "total_gpus_needed": total_gpus,
        "replicas_needed": 1,
        "num_total_gpus": total_gpus,
        "bs": concurrency,
        "ttft": ttft,
        "tpot": tpot,
        "request_latency": e2e,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_per_gpu": (
            tokens_per_second / total_gpus
            if isinstance(tokens_per_second, (int, float)) and total_gpus > 0
            else None
        ),
        "tokens_per_second_per_user": (
            tokens_per_second / concurrency
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
        "warnings": _model_warnings(
            resolved, model_config_provided=request.model_config_provided
        ),
    }
    if candidate.mode == "agg":
        config.update({"tp": candidate.tp, "pp": 1, "dp": 1})
    else:
        config.update({
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
    return config


def _pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-dominated rows for (throughput per user, throughput per GPU)."""
    frontier: list[dict[str, Any]] = []
    for row in rows:
        objectives = row["candidate"].get("objectives") or {}
        user = objectives.get("throughput_per_user")
        gpu = objectives.get("throughput_per_gpu")
        if not isinstance(user, (int, float)) or not isinstance(gpu, (int, float)):
            continue
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_objectives = other["candidate"].get("objectives") or {}
            other_user = other_objectives.get("throughput_per_user")
            other_gpu = other_objectives.get("throughput_per_gpu")
            if not isinstance(other_user, (int, float)) or not isinstance(other_gpu, (int, float)):
                continue
            if (
                other_user >= user
                and other_gpu >= gpu
                and (other_user > user or other_gpu > gpu)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return frontier


def _recommend_pareto(request: RecommendInput) -> BlisRecommendation:
    """Enumerate topology/load candidates and return BLIS's efficiency frontier.

    BLIS has no Pareto command, so the service runs the bounded candidate grid
    itself and computes the non-dominated set over throughput per user and
    throughput per GPU. The frontier is therefore the simulator's own measured
    output, not a client-side re-derivation.
    """
    if request.request_rate is not None:
        raise BlisUnsupportedRequest(
            "BLIS Pareto sweeps drive their own closed-loop load; "
            "target_request_rate is not accepted"
        )
    if request.concurrency is not None:
        raise BlisUnsupportedRequest(
            "BLIS Pareto sweeps drive their own load; target_concurrency is not accepted"
        )
    if request.database_mode != "HYBRID":
        raise BlisUnsupportedRequest("BLIS does not accept the AISimulate database_mode option")
    resolved = _resolve_request(
        request.model_path, request.system, request.backend, request.backend_version
    )
    include_disagg = bool(request.fabric)
    if include_disagg:
        _require_disaggregation(resolved)
    fabric = resolved.catalog.resolve_network(request.fabric) if include_disagg and request.fabric else None
    lower, upper, hardware_max = _window_bounds(
        request, resolved, include_disagg=include_disagg and fabric is not None
    )
    candidates = _recommend_candidates(
        resolved,
        include_disagg=include_disagg and fabric is not None,
        lower=lower,
        upper=upper,
    )
    levels = _pareto_concurrency_levels()
    if not candidates:
        raise BlisUnsupportedRequest(
            f"No BLIS topology fits the requested {lower}-{upper} GPU window"
        )

    total_budget_seconds = min(max(request.search_budget_seconds, 1), int(_MAX_RECOMMEND_BUDGET_SECONDS))
    started = time.monotonic()
    ledger: list[dict[str, Any]] = []
    evaluated: list[_RecommendCandidate] = []
    timed_out = False
    stop = False

    for concurrency in levels:
        for candidate in candidates:
            remaining = total_budget_seconds - (time.monotonic() - started)
            if remaining <= 0:
                timed_out = True
                stop = True
                break
            scenario, prefill_instances, decode_instances = _candidate_scenario(
                resolved,
                request,
                candidate,
                concurrency=concurrency,
                fabric=fabric,
            )
            evaluated.append(candidate)
            candidate_id = _pareto_candidate_id(candidate, concurrency)
            topology = _pareto_topology(candidate, fabric)
            try:
                raw = normalize_result(_run_scenario(
                    resolved,
                    scenario,
                    prefix=request.prefix,
                    timeout_seconds=min(_run_timeout_seconds(), remaining),
                    prefill_instances=prefill_instances,
                    decode_instances=decode_instances,
                    num_requests=concurrency,
                ))
            except BlisTimeoutError:
                ledger.append({
                    "candidate": {
                        "candidate_id": candidate_id,
                        "status": "timed_out",
                        "reason_category": "timeout",
                        "mode": candidate.mode,
                        "topology": topology,
                        "used_gpus": candidate.total_gpus,
                        "metrics": {},
                        "objectives": None,
                        "transfer_delay_modeled": candidate.mode == "disagg",
                        "provenance_sources": ["blis"],
                        "simulator": "blis",
                    },
                    "config": None,
                })
                continue
            except BlisExecutionError:
                ledger.append({
                    "candidate": {
                        "candidate_id": candidate_id,
                        "status": "failed",
                        "reason_category": "execution_error",
                        "mode": candidate.mode,
                        "topology": topology,
                        "used_gpus": candidate.total_gpus,
                        "metrics": {},
                        "objectives": None,
                        "transfer_delay_modeled": candidate.mode == "disagg",
                        "provenance_sources": ["blis"],
                        "simulator": "blis",
                    },
                    "config": None,
                })
                continue

            metrics = _pareto_metrics(raw, concurrency)
            ttft = metrics["ttft_ms"]
            tpot = metrics["tpot_ms"]
            e2e = metrics["request_latency_ms"]
            tokens_per_second = metrics["tokens_per_second"]
            completed = raw.get("completed_requests")
            dropped = raw.get("dropped_unservable", 0)
            dropped = dropped if isinstance(dropped, (int, float)) else 0
            status = "feasible"
            reason: str | None = None
            if not isinstance(completed, (int, float)) or completed <= 0:
                status, reason = "infeasible", "no_completed_requests"
            elif dropped > 0:
                status, reason = "infeasible", "dropped_unservable"
            elif not isinstance(tokens_per_second, (int, float)) or tokens_per_second <= 0:
                status, reason = "infeasible", "no_throughput"
            elif request.strict_sla:
                if request.request_latency is not None:
                    if not isinstance(e2e, (int, float)) or e2e > request.request_latency:
                        status, reason = "infeasible", "latency_sla"
                elif (
                    request.ttft is None
                    or request.tpot is None
                    or not isinstance(ttft, (int, float))
                    or not isinstance(tpot, (int, float))
                    or ttft > request.ttft
                    or tpot > request.tpot
                ):
                    status, reason = "infeasible", "latency_sla"

            objectives: dict[str, float] | None = None
            if status == "feasible" and isinstance(tokens_per_second, (int, float)):
                objectives = {
                    "throughput_per_user": tokens_per_second / concurrency,
                    "throughput_per_gpu": tokens_per_second / candidate.total_gpus,
                }
            ledger.append({
                "candidate": {
                    "candidate_id": candidate_id,
                    "status": status,
                    "reason_category": reason,
                    "mode": candidate.mode,
                    "topology": topology,
                    "used_gpus": candidate.total_gpus,
                    "metrics": metrics,
                    "objectives": objectives,
                    "transfer_delay_modeled": candidate.mode == "disagg",
                    "provenance_sources": ["blis"],
                    "simulator": "blis",
                },
                "config": (
                    _pareto_config(
                        request,
                        resolved,
                        candidate,
                        concurrency=concurrency,
                        metrics=metrics,
                    )
                    if objectives is not None
                    else None
                ),
            })
        if stop:
            break

    if not ledger:
        raise BlisUnsupportedRequest(
            "BLIS did not evaluate any Pareto candidate within the search budget"
        )

    feasible_rows = [
        row
        for row in ledger
        if row["candidate"]["status"] == "feasible" and row["candidate"]["objectives"] is not None
    ]
    frontier_rows = _pareto_frontier(feasible_rows)
    frontier_rows.sort(key=lambda row: (
        -(row["candidate"]["objectives"] or {}).get("throughput_per_user", 0.0),
        -(row["candidate"]["objectives"] or {}).get("throughput_per_gpu", 0.0),
    ))
    top_rows = sorted(
        feasible_rows,
        key=lambda row: (
            -(row["candidate"]["objectives"] or {}).get("throughput_per_gpu", 0.0),
            -(row["candidate"]["objectives"] or {}).get("throughput_per_user", 0.0),
        ),
    )[:request.top_n]
    summary = _search_summary(
        request,
        candidates,
        evaluated,
        include_disagg and fabric is not None,
        hardware_max,
        upper,
        total_budget_seconds,
        timed_out,
    )
    summary["strategy"] = "enumerated_topology_and_concurrency_grid"
    summary["candidate_configurations"] = len(candidates) * len(levels)
    summary["max_candidates"] = summary["candidate_configurations"]
    summary["evaluated_configurations"] = len(ledger)
    return BlisRecommendation(
        configs=[row["config"] for row in top_rows if row["config"] is not None],
        search_summary=summary,
        provenance=_provenance(resolved, request.model_path, request.system),
        candidates=[row["candidate"] for row in ledger],
        views={
            "pareto_front": [
                row["candidate"]["candidate_id"] for row in frontier_rows
            ],
            "top_n": [row["candidate"]["candidate_id"] for row in top_rows],
        },
    )


def recommend(request: RecommendInput) -> BlisRecommendation:
    if request.capacity:
        return _recommend_capacity(request)
    if request.pareto:
        return _recommend_pareto(request)
    demand = request.demand
    if request.request_rate is not None and not demand:
        raise BlisUnsupportedRequest(
            "BLIS recommendation currently searches concurrency workloads; choose a target concurrency "
            "or select AISimulate for request-rate search"
        )
    if demand:
        if request.request_rate is None or request.request_rate <= 0:
            raise BlisUnsupportedRequest("BLIS throughput-demand sizing requires a positive request rate")
        if request.concurrency is not None:
            raise BlisUnsupportedRequest(
                "BLIS throughput-demand sizing derives load from the request rate; "
                "target_concurrency is not accepted"
            )
        if request.request_latency is not None or request.ttft is not None or request.tpot is not None:
            raise BlisUnsupportedRequest(
                "BLIS throughput-demand sizing does not apply latency targets; "
                "omit ttft, tpot, and request_latency"
            )
    if request.database_mode != "HYBRID":
        raise BlisUnsupportedRequest("BLIS does not accept the AISimulate database_mode option")
    resolved = _resolve_request(request.model_path, request.system, request.backend, request.backend_version)
    # Demand runs replay open-loop traffic and are aggregate-only: the adapter
    # cannot attach a P/D transfer fabric to a rate-driven run it did not search.
    include_disagg = bool(request.fabric) and not demand
    if include_disagg:
        _require_disaggregation(resolved)
    fabric = resolved.catalog.resolve_network(request.fabric) if include_disagg and request.fabric else None
    lower, upper, hardware_max = _window_bounds(
        request, resolved, include_disagg=include_disagg
    )

    demand_requests = 0
    concurrency: int | None = None
    if demand:
        _, demand_requests = _demand_slice(float(request.request_rate))
        # The scheduler admission cap is the run's whole request slice, so it
        # never limits admission below what the open-loop generator can inject.
        # This is a run bound, not a claim about the production engine's cap.
        concurrency = demand_requests
    else:
        concurrency = request.concurrency or 1
    assert concurrency is not None

    candidates = _recommend_candidates(
        resolved,
        include_disagg=include_disagg and fabric is not None,
        lower=lower,
        upper=upper,
    )

    total_budget_seconds = min(max(request.search_budget_seconds, 1), int(_MAX_RECOMMEND_BUDGET_SECONDS))
    started = time.monotonic()
    evaluated: list[_RecommendCandidate] = []
    feasible: list[dict[str, Any]] = []
    demand_measured: list[dict[str, Any]] = []
    timed_out = False
    execution_failures: list[BlisExecutionError] = []

    for index, candidate in enumerate(candidates):
        remaining = total_budget_seconds - (time.monotonic() - started)
        if remaining <= 0:
            timed_out = True
            break
        timeout = min(_run_timeout_seconds(), remaining)
        scenario, prefill_instances, decode_instances = _candidate_scenario(
            resolved,
            request,
            candidate,
            concurrency=concurrency,
            fabric=fabric,
        )
        try:
            metrics = normalize_result(_run_scenario(
                resolved,
                scenario,
                prefix=request.prefix,
                timeout_seconds=timeout,
                prefill_instances=prefill_instances,
                decode_instances=decode_instances,
                rate=request.request_rate if demand else None,
                num_requests=demand_requests if demand else None,
            ))
        except BlisExecutionError as exc:
            evaluated.append(candidate)
            execution_failures.append(exc)
            continue
        evaluated.append(candidate)
        completed = metrics.get("completed_requests")
        if not isinstance(completed, (int, float)) or completed <= 0:
            continue
        dropped = metrics.get("dropped_unservable", 0)
        dropped = dropped if isinstance(dropped, (int, float)) else 0
        # A closed-loop recommendation treats a dropped request as a failed
        # candidate. Demand runs keep measuring it so the shortfall can be
        # reported, but a drop still fails demand compliance below.
        if dropped and not demand:
            continue
        ttft = metrics.get("ttft_mean_ms")
        itl = metrics.get("itl_mean_ms")
        e2e = metrics.get("e2e_mean_ms")
        if not all(isinstance(value, (int, float)) for value in (ttft, itl, e2e)):
            continue
        if not demand:
            if request.request_latency is not None:
                if e2e > request.request_latency:
                    continue
            elif (
                request.ttft is None
                or request.tpot is None
                or ttft > request.ttft
                or itl > request.tpot
            ):
                continue

        tokens_per_second = metrics.get("tokens_per_sec")
        reported_tpot = (
            (float(ttft) + float(itl) * (request.osl - 1)) / request.osl
            if request.inclusive_tpot and request.osl > 0
            else float(itl)
        )
        # Demand runs measure request throughput. BLIS reports it directly;
        # fall back to output tokens per second divided by the fixed OSL.
        achieved_rps: float | None = None
        window_seconds: float | None = None
        observed_concurrency: float | None = None
        demand_met = False
        if demand:
            responded = metrics.get("responses_per_sec")
            if isinstance(responded, (int, float)) and float(responded) > 0:
                measured_rps = float(responded)
            elif isinstance(tokens_per_second, (int, float)) and request.osl > 0:
                measured_rps = float(tokens_per_second) / request.osl
            else:
                measured_rps = None
            duration = metrics.get("vllm_estimated_duration_s")
            duration_s = (
                float(duration)
                if isinstance(duration, (int, float)) and float(duration) > 0
                else None
            )
            if duration_s is not None:
                # BLIS times a run to its last completion, so the measured
                # window appends one response latency to the offered traffic.
                # Removing one mean latency recovers the arrival window; without
                # that, every candidate with a non-zero latency looks short of
                # the target it actually sustained.
                arrival_s = duration_s - float(e2e) / 1000.0
                if arrival_s > 0:
                    window_seconds = arrival_s
                    achieved_rps = float(completed) / arrival_s
                else:
                    window_seconds = duration_s
                    achieved_rps = measured_rps
            elif measured_rps is not None and measured_rps > 0:
                window_seconds = float(completed) / measured_rps
                achieved_rps = measured_rps
            else:
                achieved_rps = measured_rps
            observed_concurrency = (
                achieved_rps * float(e2e) / 1000.0 if achieved_rps is not None else None
            )
            target_rps = float(request.request_rate or 0.0)
            # A compliant run must sustain the offered rate over the arrival
            # window and actually serve every request it generated; a dropped
            # request is not service.
            demand_met = (
                achieved_rps is not None
                and achieved_rps * (1 + 1e-3) >= target_rps
                and dropped == 0
                and completed >= demand_requests
            )
        batch_size = (
            max(1, round(observed_concurrency))
            if demand and observed_concurrency
            else concurrency
        )
        common_config: dict[str, Any] = {
            "total_gpus_needed": candidate.total_gpus,
            "replicas_needed": 1,
            "num_total_gpus": candidate.total_gpus,
            "bs": batch_size,
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
                float(tokens_per_second) / batch_size
                if isinstance(tokens_per_second, (int, float)) and batch_size > 0
                else None
            ),
            "concurrency": batch_size,
            "request_rate": achieved_rps if demand else None,
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
            if demand:
                common_config["warnings"].append(
                    "BLIS throughput-demand sizing is aggregate-only; P/D candidates were not searched."
                )
            elif not include_disagg:
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
        if demand:
            demand_measured.append({
                "candidate": candidate,
                "config": common_config,
                "achieved_rps": achieved_rps,
                "met": demand_met,
                "observed_concurrency": observed_concurrency,
                "window_seconds": window_seconds,
                "completed_requests": int(completed),
                "e2e_ms": float(e2e),
            })
        else:
            feasible.append(common_config)

        at_end_of_gpu_group = index + 1 == len(candidates) or candidates[index + 1].total_gpus > candidate.total_gpus
        if not demand and at_end_of_gpu_group and len(feasible) >= request.top_n:
            break

    if demand:
        if not demand_measured:
            if candidates and len(execution_failures) == len(evaluated):
                raise execution_failures[-1]
            return BlisRecommendation(
                configs=[],
                provenance=_provenance(resolved, request.model_path, request.system),
                search_summary=_search_summary(request, candidates, evaluated, include_disagg, hardware_max, upper, total_budget_seconds, timed_out),
            )
        target_rps = float(request.request_rate or 0.0)
        best_observed = max(
            (entry["achieved_rps"] for entry in demand_measured if entry["achieved_rps"] is not None),
            default=None,
        )
        compliant = [entry for entry in demand_measured if entry["met"]]
        if compliant:
            # Fewest GPUs that sustained the rate, then the highest measured
            # rate, then the lowest end-to-end latency.
            winner = min(
                compliant,
                key=lambda entry: (
                    entry["candidate"].total_gpus,
                    -(entry["achieved_rps"] or 0.0),
                    entry["e2e_ms"],
                ),
            )
            met = True
        else:
            # Report the best observed shortfall at 200 rather than a bare 422.
            winner = min(
                demand_measured,
                key=lambda entry: (
                    -(entry["achieved_rps"] or 0.0),
                    entry["candidate"].total_gpus,
                ),
            )
            met = False
        outcome = BlisDemandOutcome(
            target_rps=target_rps,
            achieved_rps=(
                round(winner["achieved_rps"], 3)
                if winner["achieved_rps"] is not None
                else None
            ),
            met=met,
            observed_concurrency=(
                round(winner["observed_concurrency"], 1)
                if winner["observed_concurrency"] is not None
                else None
            ),
            window_seconds=(
                round(winner["window_seconds"], 3)
                if winner["window_seconds"] is not None
                else None
            ),
            completed_requests=winner["completed_requests"],
            evaluated_configurations=len(demand_measured),
            best_observed_rps=round(best_observed, 3) if best_observed is not None else None,
        )
        return BlisRecommendation(
            configs=[winner["config"]],
            provenance=_provenance(resolved, request.model_path, request.system),
            search_summary=_search_summary(request, candidates, evaluated, include_disagg, hardware_max, upper, total_budget_seconds, timed_out),
            demand=outcome,
        )

    if not feasible and candidates and len(execution_failures) == len(evaluated):
        raise execution_failures[-1]
    feasible.sort(key=lambda config: (
        config["total_gpus_needed"],
        -(config["tokens_per_second"] or 0),
        config["mode"],
    ))
    feasible = feasible[:request.top_n]
    return BlisRecommendation(
        configs=feasible,
        provenance=_provenance(resolved, request.model_path, request.system),
        search_summary=_search_summary(request, candidates, evaluated, include_disagg, hardware_max, upper, total_budget_seconds, timed_out),
    )

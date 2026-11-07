# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISimulators REST API.

Minimal service wrapping the aisimulate SDK for GPU recommendation,
single-point performance estimation, and memory estimation.
See docs/api/openapi.yaml for the full spec.
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import uvicorn
from configiq.systems import load_device_names_from_perf_data, supported_systems
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, model_validator

from aisimulate_core.sdk.backends.factory import get_backend
from aisimulate_core.sdk.common import get_default_models
from aisimulate_core.sdk.errors import NoFeasibleConfigError
from aisimulate_core.sdk.memory import estimate_kv_cache
from aisimulate_core.sdk.perf_database import get_supported_databases, get_version_slots, load_system_spec
from aisimulate_core.sdk.utils import get_model_config_from_model_path

from .blis.manifest import BlisManifestError, BlisUnsupportedRequest, load_support_manifest
from .blis.runner import BlisExecutionError, BlisTimeoutError
from .blis.runtime import blis_release
from .blis.scenario import DEFAULT_ENGINE_VERSION, ENGINE_VERSION_SUPPORT, MODELED_ENGINE_VERSIONS
from .blis.service import EstimateInput as BlisEstimateInput
from .blis.service import RecommendInput as BlisRecommendInput
from .blis.service import estimate as run_blis_estimate
from .blis.service import recommend as run_blis_recommend
from .blis.service import supported_model_ids, supported_system_ids

# Optional observability + MCP, provided by the shared configiq package
# (configiq[otel,mcp]). Kept out of the base install/container because otel
# tracing opens an OTLP exporter eagerly at import; every use is guarded so the
# service runs unchanged when the extras are absent.
try:
    from configiq import observability
    _OBS = True
except ImportError:
    _OBS = False

try:
    from configiq import mcp as mcp_support
    _MCP = True
except ImportError:
    _MCP = False

logger = logging.getLogger(__name__)

# ─── Pydantic models ────────────────────────────────────────────────────────


class RecommendRequest(BaseModel):
    model_path: str = Field(examples=["Qwen/Qwen3-32B"], description="HuggingFace model path or SDK model key.")
    system: str = Field(examples=["h200_sxm"], description="GPU system identifier.")
    backend: str = Field(default="vllm", description="Inference backend.")
    backend_version: str | None = Field(
        default=None,
        examples=[None],
        description=(
            "Backend version. Omit (or send `current`) to resolve the system "
            "default published in /backends system_defaults and let the service "
            "retry newer versions when the default has no performance data for "
            "this model. Literal versions, `previous`, and `next` are pins and are "
            "never substituted; if the data cannot serve them the request returns "
            "422 naming the versions the SDK ships."
        ),
    )
    target_request_rate: float | None = Field(
        default=None,
        examples=[None, 10],
        description=(
            "Offered request rate in requests/second. With objective='min_gpus' this is a "
            "minimum-goodput constraint on closed-loop or rate traffic. With "
            "objective='throughput_demand' it is the open-loop arrival rate the cluster "
            "must sustain."
        ),
    )
    target_concurrency: float | None = Field(
        default=None,
        examples=[32],
        description=(
            "Target concurrent users (closed-loop load). With "
            "objective='concurrency_capacity' this is the number of sequences the "
            "deployment must hold at the per-sequence context length; it must be a "
            "positive whole number and is a design input, not a derived demand."
        ),
    )
    isl: int = Field(default=4000, description="Input sequence length.")
    osl: int = Field(default=1000, description="Output sequence length.")
    max_seq_len: int | None = Field(default=None, gt=0, description="Maximum sequence length for KV cache allocation. Defaults to isl + osl.")
    prefill_max_seq_len: int | None = Field(default=None, gt=0, description="Optional maximum sequence length for disaggregated prefill workers.")
    decode_max_seq_len: int | None = Field(default=None, gt=0, description="Optional maximum sequence length for disaggregated decode workers.")
    ttft: float | None = Field(default=None, description="TTFT target (ms). Required when sla_mode='enforce' unless request_latency is set.")
    tpot: float | None = Field(default=None, description="TPOT target (ms). Required when sla_mode='enforce' unless request_latency is set.")
    request_latency: float | None = Field(default=None, description="E2E latency target (ms).")
    prefix: int = Field(default=0, description="Prefix cache length.")
    database_mode: str = Field(default="HYBRID", description="Perf database mode.")
    top_n: int = Field(default=5, ge=1, le=20, examples=[2], description="Number of configs to return.")
    search_budget_seconds: int = Field(
        default=60,
        ge=1,
        le=60,
        description="BLIS total search budget in seconds. Capped to keep the search within gateway timeouts.",
    )
    min_candidate_gpus: int | None = Field(default=None, gt=0, description="Lower bound for the recommendation GPU window.")
    max_candidate_gpus: int | None = Field(default=None, gt=0, description="Upper bound for the recommendation GPU window.")
    inclusive_tpot: bool = Field(
        default=False,
        description="Report TPOT as (ttft + tpot * (osl - 1)) / osl, spreading TTFT across all output tokens. "
        "Useful for comparing with benchmarks that report inclusive TPOT (e.g. GuideLLM).",
    )
    model_config_data: dict | None = Field(
        default=None,
        alias="model_config",
        examples=[None],
        description="Pre-fetched HuggingFace model config.json. Skips HF download. "
        "Leave null to let the service resolve the model from HuggingFace.",
    )
    fabric: str | None = Field(
        default=None,
        min_length=1,
        description="Inter-node fabric for P/D candidates. BLIS defaults to roce-200g; explicit null searches aggregate candidates only. AISimulate does not consume this field.",
    )
    objective: Literal["min_gpus", "throughput_demand", "pareto", "concurrency_capacity"] = Field(
        default="min_gpus",
        description=(
            "Optimization objective. 'min_gpus' sizes subject to the latency SLA. "
            "'throughput_demand' finds the fewest GPUs that sustain target_request_rate "
            "with no latency target (sla_mode must be 'none'). 'concurrency_capacity' "
            "finds the fewest GPUs whose weights plus KV cache hold target_concurrency "
            "sequences at the per-sequence context length, with no latency target "
            "(sla_mode must be 'none'; AISimulate only). 'pareto' sweeps the "
            "throughput_per_gpu vs throughput_per_user frontier (AISimulate only)."
        ),
    )
    sla_mode: Literal["enforce", "none"] = Field(
        default="enforce",
        description=(
            "Whether latency targets constrain the result. 'enforce' requires a TTFT/TPOT "
            "or E2E bound and is required by objective='min_gpus'. 'none' rejects any "
            "latency bound so no SLA is fabricated; it is required by "
            "objective='throughput_demand' and objective='concurrency_capacity', and "
            "optional for objective='pareto'."
        ),
    )
    kv_transfer_bandwidth_gb_per_second: float | None = Field(
        default=None,
        gt=0,
        description=(
            "P/D KV transfer bandwidth to model when objective='pareto'. Omit (or send null) to "
            "apply AISimulate's default: no modeled P/D transfer delay. Omission is not a fabric "
            "default; set this only when the topology assumption is intentional."
        ),
    )
    model_config = {"populate_by_name": True}
    simulator: Literal["aisimulate", "blis"] = Field(default="aisimulate", description="Simulator to use.")

    @model_validator(mode="after")
    def validate_request(self):
        has_rate = self.target_request_rate is not None
        has_conc = self.target_concurrency is not None
        has_latency = self.request_latency is not None
        token_form = self.ttft is not None or self.tpot is not None

        if self.objective == "pareto" and self.simulator == "blis":
            raise ValueError("objective='pareto' is not supported by the BLIS simulator; use simulator='aisimulate' or objective='min_gpus'.")
        if self.objective == "concurrency_capacity" and self.simulator == "blis":
            raise ValueError(
                "objective='concurrency_capacity' is not supported by the BLIS simulator; "
                "use simulator='aisimulate'."
            )

        # A demand run asks whether a cluster can absorb an arrival rate. It has
        # no latency target, so any latency bound must be rejected rather than
        # silently dropped: accepting one would imply a constraint the search
        # never applies.
        if self.objective == "throughput_demand":
            if self.sla_mode != "none":
                raise ValueError("objective='throughput_demand' requires sla_mode='none'.")
            if has_latency or token_form:
                raise ValueError("objective='throughput_demand' does not accept ttft, tpot, or request_latency.")
            if not has_rate or has_conc:
                raise ValueError("objective='throughput_demand' requires target_request_rate and no target_concurrency.")
        elif self.objective == "concurrency_capacity":
            # Capacity is a memory-residency question: N sequences of a known
            # context length must fit. It is neither an arrival rate nor a
            # latency target, so both are rejected rather than reinterpreted.
            if self.sla_mode != "none":
                raise ValueError("objective='concurrency_capacity' requires sla_mode='none'.")
            if has_rate:
                raise ValueError("objective='concurrency_capacity' does not accept target_request_rate.")
            if not has_conc:
                raise ValueError("objective='concurrency_capacity' requires target_concurrency.")
            concurrency = self.target_concurrency
            if (
                concurrency is None
                or not math.isfinite(concurrency)
                or concurrency < 1
                or concurrency != int(concurrency)
            ):
                raise ValueError(
                    "objective='concurrency_capacity' requires target_concurrency to be a positive whole number."
                )
            if has_latency or token_form:
                raise ValueError(
                    "objective='concurrency_capacity' does not accept ttft, tpot, or request_latency."
                )
        elif self.objective == "min_gpus":
            if self.sla_mode != "enforce":
                raise ValueError("objective='min_gpus' requires sla_mode='enforce'.")
            if has_rate == has_conc:
                raise ValueError("Exactly one of target_request_rate or target_concurrency must be provided.")
        else:
            # A pareto sweep drives load internally via kv_capacity_fraction.
            if has_rate or has_conc:
                raise ValueError("objective='pareto' does not accept target_request_rate or target_concurrency.")

        if self.sla_mode == "none":
            if has_latency or token_form:
                raise ValueError("sla_mode='none' does not accept ttft, tpot, or request_latency.")
        else:
            if has_latency and token_form:
                raise ValueError("request_latency is mutually exclusive with ttft and tpot.")
            if not has_latency and not (self.ttft is not None and self.tpot is not None):
                raise ValueError("sla_mode='enforce' requires request_latency or both ttft and tpot.")

        # P/D transfer bandwidth is a Pareto sweep input. Accepting it for other
        # objectives would imply a topology the search does not model.
        if self.kv_transfer_bandwidth_gb_per_second is not None and self.objective != "pareto":
            raise ValueError("kv_transfer_bandwidth_gb_per_second is only supported with objective='pareto'.")

        if (
            self.min_candidate_gpus is not None
            and self.max_candidate_gpus is not None
            and self.min_candidate_gpus > self.max_candidate_gpus
        ):
            raise ValueError("min_candidate_gpus cannot exceed max_candidate_gpus.")
        if self.simulator == "aisimulate" and self.fabric is not None:
            raise ValueError("AISimulate does not currently consume a network-fabric selection")
        if self.simulator == "blis" and "fabric" not in self.model_fields_set:
            self.fabric = "roce-200g"
        return self


class EstimateRequest(BaseModel):
    model_path: str = Field(examples=["Qwen/Qwen3-32B"], description="HuggingFace model path or SDK model key.")
    system: str = Field(examples=["h200_sxm"], description="GPU system identifier.")
    backend: str = Field(default="vllm", description="Inference backend.")
    backend_version: str | None = Field(
        default=None,
        examples=[None],
        description=(
            "Backend version. Omit (or send `current`) to resolve the system "
            "default published in /backends system_defaults and let the service "
            "retry newer versions when the default has no performance data for "
            "this model. Literal versions, `previous`, and `next` are pins and are "
            "never substituted; if the data cannot serve them the request returns "
            "422 naming the versions the SDK ships."
        ),
    )
    isl: int = Field(default=4000, description="Input sequence length.")
    osl: int = Field(default=1000, description="Output sequence length.")
    max_seq_len: int | None = Field(default=None, gt=0, description="Maximum sequence length for KV cache allocation.")
    prefill_max_seq_len: int | None = Field(default=None, gt=0, description="Prefill worker sequence-length override.")
    decode_max_seq_len: int | None = Field(default=None, gt=0, description="Decode worker sequence-length override.")
    prefix: int = Field(default=0, ge=0, description="Cached prefix tokens.")
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1, description="GPU memory fraction for KV-cache capacity.")
    prefill_gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    decode_gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    max_num_seqs: int | None = Field(default=None, gt=0, description="Maximum scheduler sequences.")
    enable_chunked_prefill: bool | None = Field(default=None)
    prefix_caching: bool | None = Field(default=None)
    tp_size: int = Field(default=1, description="Tensor parallel size.")
    pp_size: int = Field(default=1, description="Pipeline parallel size.")
    batch_size: int = Field(default=128, description="Batch size (max concurrent requests).")
    requests_per_load_unit: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Optional simulated requests per concurrency slot. When omitted, stop after one total request.",
    )
    database_mode: str = Field(default="HYBRID", description="Perf database mode.")
    gemm_quant_mode: str | None = Field(default=None)
    moe_quant_mode: str | None = Field(default=None)
    kvcache_quant_mode: str | None = Field(default=None)
    fmha_quant_mode: str | None = Field(default=None)
    moe_tp_size: int | None = Field(default=None, gt=0)
    moe_ep_size: int | None = Field(default=None, gt=0)
    attention_dp_size: int = Field(default=1)
    # Serving mode. When 'disagg', the prefill_*/decode_* fields below drive
    # separate prefill and decode pools; the top-level tp/pp/batch fields act
    # as fallbacks for any per-role field left unset.
    mode: Literal["agg", "disagg"] = Field(default="agg", description="Serving mode: 'agg' or 'disagg'.")
    decode_system: str | None = Field(default=None, description="GPU system for disagg decode workers; defaults to `system`.")
    prefill_tp_size: int | None = Field(default=None)
    prefill_pp_size: int | None = Field(default=None)
    prefill_moe_tp_size: int | None = Field(default=None, gt=0)
    prefill_moe_ep_size: int | None = Field(default=None, gt=0)
    prefill_batch_size: int | None = Field(default=None)
    prefill_num_workers: int | None = Field(default=None)
    decode_tp_size: int | None = Field(default=None)
    decode_pp_size: int | None = Field(default=None)
    decode_moe_tp_size: int | None = Field(default=None, gt=0)
    decode_moe_ep_size: int | None = Field(default=None, gt=0)
    decode_batch_size: int | None = Field(default=None)
    decode_num_workers: int | None = Field(default=None)
    inclusive_tpot: bool = Field(
        default=False,
        description="Report TPOT as (ttft + tpot * (osl - 1)) / osl, spreading TTFT across all output tokens. "
        "Useful for comparing with benchmarks that report inclusive TPOT (e.g. GuideLLM).",
    )
    model_config_data: dict | None = Field(
        default=None,
        alias="model_config",
        examples=[None],
        description="Pre-fetched HuggingFace model config.json. Skips HF download. "
        "Leave null to let the service resolve the model from HuggingFace.",
    )
    fabric: str | None = Field(
        default=None,
        min_length=1,
        description="Inter-node fabric for P/D transfer. BLIS uses this field; AISimulate does not consume it.",
    )

    model_config = {"populate_by_name": True}
    simulator: Literal["aisimulate", "blis"] = Field(default="aisimulate", description="Simulator to use.")

    @model_validator(mode="after")
    def reject_aisimulate_fabric(self):
        if self.simulator == "aisimulate" and self.fabric is not None:
            raise ValueError("AISimulate does not currently consume a network-fabric selection")
        return self


class MemoryBreakdown(BaseModel):
    weights_bytes: int = 0
    activations_bytes: int = 0
    runtime_overhead_bytes: int = 0
    comm_overhead_bytes: int = 0
    kv_cache_bytes: int = 0


class ServingConfig(BaseModel):
    backend: str
    tensor_parallel_size: int
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float
    enable_chunked_prefill: bool | None
    enable_prefix_caching: bool
    quantization: str
    memory_fraction: float | None = None
    memory_fraction_kind: str | None = None
    runtime_memory_field: str | None = None


class WorkerConfig(BaseModel):
    """Parallelism and serving config for one worker role in a disagg deployment."""
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    cp: int | None = None
    moe_tp: int | None = None
    moe_ep: int | None = None
    num_workers: int | None = None
    batch_size: int | None = None
    memory_gb: float | None = None
    gemm: str | None = None
    kvcache: str | None = None
    fmha: str | None = None
    moe: str | None = None
    comm: str | None = None
    backend_version: str | None = None
    memory_breakdown: MemoryBreakdown | None = None


class SimulatorProvenance(BaseModel):
    simulator: Literal["aisimulate", "blis"]
    requested_model: str | None = None
    resolved_model: str | None = None
    model_source: str | None = None
    system: str | None = None
    hardware: str | None = None
    model_config_relation: str | None = None
    engine_version: str | None = None


class CandidateResponse(BaseModel):
    """One candidate from an aisimulate pareto sweep — a subset of the SDK CandidateRecord.

    Every field is copied verbatim from the SDK record; nothing is synthesized.
    `topology` carries the SDK's flat parallelism keys (deployment_mode, tp, pp,
    prefill_tp, decode_tp, prefill_replicas, …) exactly as emitted.
    """
    candidate_id: str
    status: Literal["feasible", "infeasible", "unsupported", "timed_out", "failed", "resource_limited"]
    reason_category: str | None = None
    mode: str
    topology: dict[str, Any]
    used_gpus: int | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)
    objectives: dict[str, float] | None = None
    transfer_delay_modeled: bool = False
    provenance_sources: list[str] = Field(default_factory=list)
    simulator: Literal["aisimulate"] = "aisimulate"


class CandidateViews(BaseModel):
    """Candidate-ID lists for SDK-computed views (pareto front, top-N)."""
    pareto_front: list[str] = Field(default_factory=list)
    top_n: list[str] = Field(default_factory=list)


class DemandOutcome(BaseModel):
    """Measured outcome of a throughput-demand sizing run.

    Every value is read from the selected candidate's replay report; nothing is
    inferred from the requested target. `met` is the best of the evaluated
    candidates, not a guarantee.
    """
    target_rps: float
    achieved_rps: float | None = None
    met: bool
    observed_concurrency: float | None = None
    window_seconds: float | None = None
    completed_requests: int | None = None
    evaluated_configurations: int | None = None
    best_observed_rps: float | None = None
    simulator: Literal["aisimulate", "blis"] = "aisimulate"


class CapacityOutcome(BaseModel):
    """Memory-residency outcome of a concurrency-capacity sizing run.

    Capacity is decided by the KV-cache estimate for the selected deployment,
    not by a replay. For a disaggregated option the decode pool holds the
    resident sequences and the prefill pool only holds one transiently. The
    performance fields on the option config are a separate informational
    simulation at the requested concurrency; they never gate ``fits``.

    ``capacity_sequences`` is the largest whole number of
    ``context_per_sequence_tokens``-long sequences the deployment's KV budget
    holds. When the target does not fit, the response is still 200 and reports
    the best achievable configuration plus the shortfall.
    """
    target_concurrency: int
    context_per_sequence_tokens: int
    capacity_tokens: int
    capacity_sequences: int
    fits: bool
    shortfall_sequences: int | None = None
    shortfall_tokens: int | None = None
    #: Fewest GPUs, within the requested window, that hold one sequence.
    minimum_gpus: int | None = None
    per_rank_tokens: int
    replicas: int
    attention_dp: int
    evaluated_shapes: int
    feasible_shapes: int
    exhaustive: bool = True
    performance_available: bool = False
    performance_unavailable_reason: str | None = None
    #: Serving mode the option uses. For disagg, ``replicas``,
    #: ``attention_dp``, and ``per_rank_tokens`` describe the decode pool;
    #: prefill only needs to hold one sequence transiently.
    mode: Literal["agg", "disagg"] = "agg"
    prefill_replicas: int | None = None
    decode_replicas: int | None = None
    prefill_gpus: int | None = None
    decode_gpus: int | None = None
    simulator: Literal["aisimulate"] = "aisimulate"


class SweepMeta(BaseModel):
    """Run-control facts for a companion sweep.

    `budget_exhausted` means the configured trial budget was spent; it does not
    mean the search was exhaustive.
    """
    run_id: str | None = None
    algorithm: str | None = None
    seed: int | None = None
    max_trials: int | None = None
    evaluated: int | None = None
    feasible: int | None = None
    timed_out: int | None = None
    budget_exhausted: bool = False
    strict_sla: bool | None = None
    window_min_gpus: int | None = None
    window_max_gpus: int | None = None


class RecommendConfig(BaseModel):
    total_gpus_needed: int | None = None
    replicas_needed: int | None = None
    num_total_gpus: int | None = None
    # Parallelism (agg only; None for disagg — see prefill/decode_config)
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    moe_tp: int | None = None
    moe_ep: int | None = None
    cp: int | None = None
    bs: int | None = None
    ttft: float | None = None
    tpot: float | None = None
    request_latency: float | None = None
    concurrency: int | None = None
    request_rate: float | None = None
    tokens_per_second: float | None = None
    tokens_per_second_per_gpu: float | None = None
    tokens_per_second_per_user: float | None = None
    memory: float | None = None
    model: str | None = None
    system: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    gemm: str | None = None
    kvcache: str | None = None
    fmha: str | None = None
    moe: str | None = None
    comm: str | None = None
    power_w: float | None = None
    # Optional detail sections (include=config / include=memory)
    serving_config: ServingConfig | None = None
    memory_breakdown: MemoryBreakdown | None = None
    # Disagg detail (present for disagg results)
    prefill_config: WorkerConfig | None = None
    decode_config: WorkerConfig | None = None
    mode: str = "agg"
    simulator: Literal["aisimulate", "blis"] = "aisimulate"
    provenance: SimulatorProvenance | None = None
    warnings: list[str] = Field(default_factory=list)


class CapacityOption(BaseModel):
    """One ranked deployment alternative for a concurrency-capacity run.

    The first option is the primary recommendation and is also returned as
    ``configs[0]``. ``configs`` remains a single-item list so clients that do
    not understand options keep their existing behaviour.
    """
    option_id: str
    rank: int
    role: Literal["minimum", "same_gpus", "larger", "disaggregated", "shortfall"]
    gpus_over_minimum: int
    config: RecommendConfig
    capacity: CapacityOutcome


class RecommendResponse(BaseModel):
    configs: list[RecommendConfig]
    chosen_mode: str
    simulator: Literal["aisimulate", "blis"] = "aisimulate"
    search_summary: dict[str, Any] | None = None
    candidates: list[CandidateResponse] = Field(default_factory=list)
    views: CandidateViews | None = None
    demand: DemandOutcome | None = None
    capacity: CapacityOutcome | None = None
    capacity_options: list[CapacityOption] = Field(default_factory=list)
    sweep: SweepMeta | None = None


class EstimateResponse(BaseModel):
    """Single-point performance estimate for a given parallelism configuration."""
    ttft: float
    tpot: float
    request_latency: float | None = None
    tokens_per_second: float | None = None
    tokens_per_second_per_gpu: float | None = None
    tokens_per_second_per_user: float | None = None
    memory: float | None = None
    concurrency: int | None = None
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    moe_tp: int | None = None
    moe_ep: int | None = None
    system: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    gemm: str | None = None
    kvcache: str | None = None
    power_w: float | None = None
    serving_config: ServingConfig | None = None
    memory_breakdown: MemoryBreakdown | None = None
    # Serving mode + disagg detail (present when mode='disagg')
    mode: str = "agg"
    prefill_config: WorkerConfig | None = None
    decode_config: WorkerConfig | None = None
    simulator: Literal["aisimulate", "blis"] = "aisimulate"
    provenance: SimulatorProvenance | None = None
    memory_unavailable_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


class MemoryRequest(BaseModel):
    model_path: str = Field(examples=["Qwen/Qwen3-32B"], description="HuggingFace model path or SDK model key.")
    system: str = Field(examples=["h200_sxm"], description="GPU system identifier.")
    backend: str = Field(default="vllm", description="Inference backend.")
    backend_version: str | None = Field(default=None, examples=[None])
    max_num_tokens: int = Field(default=8192)
    max_batch_size: int = Field(default=128)
    memory_fraction_kind: str = Field(default="of_total")
    memory_fraction_value: float = Field(default=1.0, ge=0.0, le=1.0)
    tp_size: int = Field(default=1)
    pp_size: int = Field(default=1)
    attention_dp_size: int = Field(default=1)
    moe_tp_size: int | None = Field(default=None)
    moe_ep_size: int | None = Field(default=None)
    gemm_quant_mode: str | None = Field(default=None)
    moe_quant_mode: str | None = Field(default=None)
    kvcache_quant_mode: str | None = Field(default=None)
    fmha_quant_mode: str | None = Field(default=None)
    comm_quant_mode: str | None = Field(default=None)
    tolerance_fraction: float | None = Field(default=None)
    model_config_data: dict | None = Field(
        default=None,
        alias="model_config",
        examples=[None],
        description="Pre-fetched HuggingFace model config.json. Skips HF download. "
        "Leave null to let the service resolve the model from HuggingFace.",
    )

    model_config = {"populate_by_name": True}


class MemoryResponse(BaseModel):
    total_gpu_capacity_bytes: int
    total_kv_size_bytes: int
    kv_size_per_token_bytes: int
    total_kv_size_tokens: int
    source: str
    memory_breakdown: MemoryBreakdown
    tolerance_adjusted: dict[str, Any] | None = None
    simulator: Literal["aisimulate", "blis"] = "aisimulate"


class _PredictionResult:
    """Small adapter around the SDK metrics consumed by ``post_predict``."""

    def __init__(self, summary: dict[str, Any]):
        self.summary = summary
        self.provenance: dict[str, Any] | None = None
        self.serving_config: dict[str, Any] | None = None
        self.warnings: list[str] = []


class SystemDetail(BaseModel):
    id: str
    name: str
    vendor: str
    architecture: str
    memory_bytes: int
    tdp_watts: float
    gpus_per_node: int


# ─── Helpers ─────────────────────────────────────────────────────────────────

_COLUMN_MAP = {
    "tokens/s": "tokens_per_second",
    "tokens/s/gpu": "tokens_per_second_per_gpu",
    "tokens/s/user": "tokens_per_second_per_user",
    "num_total_gpus": "num_total_gpus",
    "version": "backend_version",
}

_INT_FIELDS = frozenset({
    "total_gpus_needed", "replicas_needed", "num_total_gpus",
    "tp", "pp", "dp", "moe_tp", "moe_ep", "cp", "bs", "concurrency",
})

_SM_ARCHITECTURE = {
    70: "volta",
    75: "turing",
    80: "ampere",
    86: "ampere",
    89: "ada-lovelace",
    90: "hopper",
    100: "blackwell",
    103: "blackwell",
    120: "blackwell",
}

_DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
_DEFAULT_MAX_CANDIDATE_GPUS = 1024
_BACKEND_PUBLIC_IDS = {"vllm": "vllm", "sglang": "sglang", "trtllm": "tensorrt-llm"}
_BACKEND_SDK_IDS = {public: sdk for sdk, public in _BACKEND_PUBLIC_IDS.items()}


def _sdk_backend_id(backend: str) -> str:
    return _BACKEND_SDK_IDS.get(backend, backend)


def _backend_memory_fraction(backend: str, backend_version: str | None = None) -> float:
    from aisimulate_core.sdk.backends.factory import get_backend

    value = get_backend(_sdk_backend_id(backend)).get_default_free_gpu_memory_fraction(backend_version)
    return value if value is not None else _DEFAULT_GPU_MEMORY_UTILIZATION


def _backend_memory_fraction_kind(backend: str) -> str:
    from aisimulate_core.sdk.backends.factory import get_backend

    return "of_free" if get_backend(_sdk_backend_id(backend)).memory_fraction_of_free() else "of_total"


def _backend_runtime_memory_field(backend: str) -> str:
    return {
        "vllm": "gpu_memory_utilization",
        "sglang": "mem_fraction_static",
        "trtllm": "free_gpu_memory_fraction",
        "tensorrt-llm": "free_gpu_memory_fraction",
    }.get(backend, "memory_fraction")


def _version_sort_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def _current_or_newest(slots: dict[str, str] | None, versions: list[str]) -> str | None:
    """The version a system completed, not the newest one anyone declared.

    Slots are absent when the SDK ships no version gate, where the newest
    declared version is the best available answer.
    """
    if slots and slots.get("current"):
        return slots["current"]
    return max(versions, key=_version_sort_key) if versions else None


def _declared_versions(system: str, backend: str) -> list[str]:
    """Literal backend versions declared for a system, oldest first.

    Mirrors the per-system list ``/backends`` publishes, so a fallback can
    never offer a version the SDK does not ship data for.
    """
    try:
        supported = get_supported_databases().get(system, {})
    except Exception:
        logger.warning("Could not read the supported databases for %s", system, exc_info=True)
        return []
    return sorted(supported.get(_sdk_backend_id(backend), []) or [], key=_version_sort_key)


def _system_default_version(system: str, backend: str) -> str | None:
    """The literal version an omitted ``backend_version`` resolves to.

    ``/backends`` documents this as ``system_defaults[system]``. Resolving it
    here with the same rule lets a prediction report the version it actually
    used instead of echoing the caller's blank.
    """
    versions = _declared_versions(system, backend)
    if not versions:
        return None
    try:
        slots = get_version_slots(system, _sdk_backend_id(backend))
    except Exception:
        slots = None
    return _current_or_newest(slots, versions)


class _noop_context:
    def __init__(self, model_path: str):
        self._path = model_path

    def __enter__(self):
        return self._path

    def __exit__(self, *args):
        pass


class _tempdir_context:
    def __init__(self, model_path: str, config_dict: dict):
        self._original = model_path
        self._config = config_dict
        self._tmpdir: Any = None

    def __enter__(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        config_path = Path(self._tmpdir.name) / "config.json"
        config_path.write_text(json.dumps(self._config))
        return self._tmpdir.name

    def __exit__(self, *args):
        if self._tmpdir:
            self._tmpdir.cleanup()


def _with_model_config(model_path: str, config_dict: dict | None):
    # An empty dict carries no usable config — treat anything falsy as
    # "not supplied" so the SDK resolves the model itself instead of loading a
    # bogus config.json. (A non-empty but invalid config, such as Swagger UI's
    # {"additionalProp1": {}} placeholder, still reaches the SDK and surfaces as
    # a clear 422 via the KeyError branch in _common_error_handler.)
    if not config_dict:
        return _noop_context(model_path)
    return _tempdir_context(model_path, config_dict)


def _coerce_int(val: Any) -> int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _coerce_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _inclusive_tpot(ttft: float | None, tpot: float | None, osl: int) -> float | None:
    """Spread TTFT across all output tokens: (ttft + tpot * (osl - 1)) / osl.

    Presentation-only transform for benchmark-comparable TPOT (e.g. GuideLLM).
    Returns ``tpot`` unchanged when inputs are insufficient.
    """
    if ttft is None or tpot is None or osl <= 0:
        return tpot
    return (ttft + tpot * (osl - 1)) / osl


def _aisimulate_runner_factory():
    from aisimulate.stack import resolve_runner_factory

    return resolve_runner_factory("engine")


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive integer env var, falling back on missing/invalid values.

    A malformed or non-positive setting must not fail every request that reads
    it (the same contract as AISIMULATORS_MAX_CANDIDATE_GPUS).
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _max_candidate_gpus() -> int:
    return _positive_int_env("AISIMULATORS_MAX_CANDIDATE_GPUS", _DEFAULT_MAX_CANDIDATE_GPUS)


#: Largest GPU count an omitted Pareto window will reach automatically. The
#: auto window starts at the smallest shape that holds one full-context
#: sequence and doubles that floor up to this cap, so a model that needs more
#: than this must pass explicit `min_candidate_gpus` / `max_candidate_gpus`.
_DEFAULT_PARETO_MAX_WINDOW_GPUS = 64


def _pareto_max_window_gpus() -> int:
    return _positive_int_env("PARETO_MAX_WINDOW_GPUS", _DEFAULT_PARETO_MAX_WINDOW_GPUS)


#: Simulated traffic per demand run: `requests_per_load_unit × offered rate`.
#: The replay window must scale with the rate; a fixed request count would
#: measure a different (shorter) window at every target.
_DEMAND_REQUESTS_PER_LOAD_UNIT = 5
#: Wall-clock budget for the whole demand search. A demand run grows its GPU
#: window rather than sampling one enormous window, so several bounded sweeps
#: share this budget. Kept below the gateway timeout so the caller still gets
#: the best observed result instead of a proxy timeout.
_DEMAND_SEARCH_BUDGET_S = 50
#: First window a demand search tries. The window doubles until the offered
#: rate is met, the effective ceiling is reached, or the time budget is spent.
#: This is a search starting point, not a GPU cap: the ceiling still applies.
_DEMAND_WINDOW_START_GPUS = 8
#: Do not start another demand window with less time than this remaining.
_DEMAND_MIN_WINDOW_S = 5
_DEFAULT_SWEEP_SEED = 42


def _sweep_algorithm() -> str:
    """Sweep sampler. `random` is the default: the budgets here are far too
    small for a GP to learn much, and the local GP designer flips JAX's global
    float64 setting in a long-lived service process. Override for A/B tests."""
    raw = os.environ.get("PARETO_ALGORITHM", "").strip().lower()
    return raw if raw in {"bayesian", "random"} else "random"


def _sweep_seed() -> int:
    raw = os.environ.get("SWEEP_SEED")
    if raw is None:
        return _DEFAULT_SWEEP_SEED
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_SWEEP_SEED
    return value if value >= 0 else _DEFAULT_SWEEP_SEED


def _demand_requests_per_load_unit() -> int:
    return _positive_int_env("DEMAND_REQUESTS_PER_LOAD_UNIT", _DEMAND_REQUESTS_PER_LOAD_UNIT)


def _demand_search_budget_seconds() -> int:
    return _positive_int_env("DEMAND_SEARCH_BUDGET_S", _DEMAND_SEARCH_BUDGET_S)


def _demand_start_window_gpus() -> int:
    return _positive_int_env("DEMAND_WINDOW_START_GPUS", _DEMAND_WINDOW_START_GPUS)


def _effective_gpu_window(req: "RecommendRequest") -> tuple[int, int]:
    server_max = _max_candidate_gpus()
    window_max = min(req.max_candidate_gpus, server_max) if req.max_candidate_gpus else server_max
    return req.min_candidate_gpus or 1, window_max


def _demand_gpu_windows(req: "RecommendRequest") -> list[tuple[int, int]]:
    """Windows a demand search tries, smallest first.

    A demand sweep cannot use the SDK's smallest-first ordering: `min_gpus` is
    rejected without an SLA, so the search is driven by `throughput_per_gpu`
    and samples the whole GPU range uniformly. Sampling 16 trials from a
    thousand-shape space effectively randomises the cluster size. Bounding the
    first window keeps every candidate near the low end, and doubling it only
    when the rate is unmet grows the search instead of skipping over it.

    An explicit `max_candidate_gpus` pins a single window: the caller has
    already chosen the range it wants measured.
    """
    lower, ceiling = _effective_gpu_window(req)
    if req.max_candidate_gpus is not None:
        return [(lower, ceiling)]
    upper = min(ceiling, max(lower, _demand_start_window_gpus()))
    windows: list[tuple[int, int]] = []
    while True:
        windows.append((lower, upper))
        if upper >= ceiling:
            return windows
        upper = min(ceiling, upper * 2)


def _is_no_viable_config(error: Exception) -> bool:
    """True when a GPU window is simply too small to hold the model."""
    if isinstance(error, NoFeasibleConfigError) or error.__class__.__name__ == "NoViableParallelConfig":
        return True
    message = str(error)
    return (
        "NoViableParallelConfig" in message
        or "no deployment_mode has a viable parallel config" in message
    )


def _recommendation_trial_budget(
    objective: str,
    window_max_gpus: int,
    *,
    has_requested_max: bool,
) -> tuple[int, int, int]:
    """Return `(max_trials, parallelism, candidate_timeout_seconds)`.

    The trial budget is split across the searched deployment modes, so these
    are totals, not per-mode counts.
    """
    narrow_window = window_max_gpus == 1
    if objective == "pareto":
        # A one-GPU window has no room for P/D, so the full sweep budget would
        # only repeat equivalent aggregate trials.
        return (
            1 if narrow_window else _positive_int_env("PARETO_MAX_TRIALS", 20),
            1 if narrow_window else _positive_int_env("PARETO_PARALLELISM", 8),
            _positive_int_env("PARETO_CANDIDATE_TIMEOUT_S", 8),
        )
    if objective == "throughput_demand":
        # Demand runs replay open-loop traffic, which is heavier per candidate
        # than a closed-loop sizing trial, so they get their own budget.
        return (
            1 if narrow_window else _positive_int_env("DEMAND_MAX_TRIALS", 16),
            1 if narrow_window else _positive_int_env("DEMAND_PARALLELISM", 4),
            _positive_int_env("DEMAND_CANDIDATE_TIMEOUT_S", 15),
        )
    return (
        1 if window_max_gpus == 1 else 4 if has_requested_max else 8,
        1 if window_max_gpus == 1 else 2 if has_requested_max else 4,
        15 if has_requested_max else 30,
    )


#: Pareto sweeps are synchronous and can outlive the request that started them.
#: Bound how many run at once per worker process so abandoned sweeps cannot
#: occupy an unbounded number of threadpool slots. The limit is read lazily so
#: a deployment (or a test) can change it without a restart.
_PARETO_SLOT_LOCK = threading.Lock()
_PARETO_SLOTS: threading.BoundedSemaphore | None = None
_PARETO_SLOTS_SIZE = 0


def _pareto_slots() -> threading.BoundedSemaphore:
    global _PARETO_SLOTS, _PARETO_SLOTS_SIZE
    size = _positive_int_env("PARETO_MAX_CONCURRENCY", 2)
    with _PARETO_SLOT_LOCK:
        if _PARETO_SLOTS is None or _PARETO_SLOTS_SIZE != size:
            _PARETO_SLOTS = threading.BoundedSemaphore(size)
            _PARETO_SLOTS_SIZE = size
        return _PARETO_SLOTS


def _aisimulate_recommendation_config(
    req: RecommendRequest,
    *,
    model_path: str | None = None,
    max_trials_override: int | None = None,
    candidate_timeout_override: int | None = None,
) -> Any:
    from aisimulate.config.cli import CoreRecommendationConfig

    context_length = req.max_seq_len or req.isl + req.osl
    window_min_gpus, window_max_gpus = _effective_gpu_window(req)
    if window_min_gpus > window_max_gpus:
        raise ValueError("min_candidate_gpus cannot exceed the effective max_candidate_gpus.")
    # A one-GPU total window cannot contain a disaggregated deployment: it
    # requires at least one prefill and one decode GPU. Narrow windows also
    # need a small trial budget so incremental search does not spend the full
    # per-window timeout on redundant optimizer trials.
    is_pareto = req.objective == "pareto"
    is_demand = req.objective == "throughput_demand"
    narrow_window = window_max_gpus == 1
    modes = ["aggregated"] if narrow_window else ["aggregated", "disaggregated"]
    max_trials, parallelism, candidate_timeout_seconds = _recommendation_trial_budget(
        req.objective,
        window_max_gpus,
        has_requested_max=req.max_candidate_gpus is not None,
    )
    # A demand ladder shares one wall-clock budget across several windows and
    # shrinks each window's trial budget to fit the time that remains.
    if max_trials_override is not None:
        max_trials = max(1, max_trials_override)
    if candidate_timeout_override is not None:
        candidate_timeout_seconds = max(1, candidate_timeout_override)
    concurrency = int(req.target_concurrency or 1)
    has_rate = req.target_request_rate is not None
    if is_pareto:
        load: dict[str, Any] = {"type": "kv_capacity_fraction", "fraction": {"range": {"min": 0.1, "max": 0.9}}}
        stop: dict[str, Any] = {"requests_per_load_unit": _demand_requests_per_load_unit()}
    elif has_rate:
        # Open-loop traffic: the request count must scale with the offered
        # rate. A concurrency-derived stop sends a fixed handful of requests,
        # which measures almost nothing at production rates.
        load = {"type": "constant_rate", "requests_per_second": req.target_request_rate}
        stop = {"requests_per_load_unit": _demand_requests_per_load_unit()}
    else:
        load = {"type": "concurrency", "concurrency": concurrency}
        stop = {"requests": max(4, min(32, concurrency * 4))}
    workers: dict[str, Any] = {
        "aggregated": {"parallelism": {"preset": "default"}},
    }
    if "disaggregated" in modes:
        workers.update({
            "prefill": {
                "parallelism": {"preset": "default"},
                **({"context_length": req.prefill_max_seq_len} if req.prefill_max_seq_len else {}),
            },
            "decode": {
                "parallelism": {"preset": "default"},
                **({"context_length": req.decode_max_seq_len} if req.decode_max_seq_len else {}),
            },
        })
    engine: dict[str, Any] = {
        "model": model_path or req.model_path,
        "hardware": req.system,
        "backend": _sdk_backend_id(req.backend),
        "backend_version": req.backend_version,
        "database_mode": req.database_mode,
        "context_length": context_length,
        "mode": {"choices": modes},
        "workers": workers,
    }
    # Omission means AISimulate models no P/D transfer delay. Only add the
    # engine block when the caller explicitly chose a bandwidth; never invent a
    # fabric default.
    if is_pareto and req.kv_transfer_bandwidth_gb_per_second is not None:
        engine["kv_transfer"] = {
            "bandwidth_gb_per_second": req.kv_transfer_bandwidth_gb_per_second,
            "timing_mode": "destination_missing",
        }
    optimization: dict[str, Any]
    if is_pareto:
        optimization = {
            "target": "pareto",
            # SLO-mode frontiers only contain candidates that clear the SLA;
            # demand-mode frontiers sweep capacity and must not filter on one.
            "strict_sla": req.sla_mode == "enforce",
            "constraints": {
                "min_candidate_gpus": req.min_candidate_gpus,
                "max_candidate_gpus": window_max_gpus,
            },
        }
    elif is_demand:
        optimization = {
            # A fixed offered rate is explored best by efficiency: compliant
            # configs all report the offered throughput, so treating GPUs as the
            # denominator ranks the smallest compliant cluster first. The
            # service still filters the ledger by achieved rate before picking.
            "target": "throughput_per_gpu",
            "strict_sla": False,
            "constraints": {
                "min_candidate_gpus": req.min_candidate_gpus,
                "max_candidate_gpus": window_max_gpus,
            },
        }
    else:
        optimization = {
            "target": "min_gpus",
            "strict_sla": True,
            "constraints": {
                "min_candidate_gpus": req.min_candidate_gpus,
                "max_candidate_gpus": window_max_gpus,
                **({"min_goodput_rps": req.target_request_rate} if req.target_request_rate is not None else {}),
            },
        }
    evaluation: dict[str, Any] = {}
    if req.sla_mode == "enforce":
        evaluation["sla"] = (
            {"e2e_ms": req.request_latency}
            if req.request_latency is not None
            else {"ttft_ms": req.ttft, "itl_ms": req.tpot}
        )
    raw: dict[str, Any] = {
        "traffic": {
            "source": {
                "type": "synthetic",
                "input_tokens": req.isl,
                "output_tokens": req.osl,
                "cached_prefix_tokens": req.prefix,
            },
            "load": load,
            "stop": stop,
        },
        "engine": engine,
        "evaluation": evaluation,
        "optimization": optimization,
        "optimizer": {
            "algorithm": _sweep_algorithm(),
            "seed": _sweep_seed(),
            "max_trials": max_trials,
            "parallelism": parallelism,
            "candidate_timeout_seconds": candidate_timeout_seconds,
        },
    }
    return CoreRecommendationConfig.model_validate(raw)


def _resolve_moe_widths(
    tp: int | None,
    attention_dp: int | None,
    moe_tp: int | None,
    moe_ep: int | None,
) -> tuple[int, int]:
    """Resolve the FFN/expert width pair for one worker role.

    AISimulate requires ``tp * attention_dp * cp == moe_tp * moe_ep`` for every
    model family that can host MoE layers, including dense variants of those
    families (dense Gemma 4 is one). Treating two omitted widths as ``1 x 1``
    therefore fails at any attention width greater than one.

    When the caller supplies neither width, state the default layout
    explicitly: tensor-shard the FFN/expert block across the attention group
    with no expert parallelism. Explicit values are preserved. When only one
    width is supplied, derive the other when it divides the attention width;
    otherwise keep 1 so the SDK reports the precise invariant error rather than
    a validation failure.
    """
    attention_width = (tp or 1) * (attention_dp or 1)
    if moe_tp is None and moe_ep is None:
        return attention_width, 1
    if moe_tp is None:
        assert moe_ep is not None
        return (
            attention_width // moe_ep if attention_width % moe_ep == 0 else 1,
            moe_ep,
        )
    if moe_ep is None:
        return (
            moe_tp,
            attention_width // moe_tp if attention_width % moe_tp == 0 else 1,
        )
    return moe_tp, moe_ep


def _aisimulate_prediction_config(req: EstimateRequest) -> dict[str, Any]:
    from aisimulate.config.cli import CorePredictionConfig

    def worker(role: str) -> dict[str, Any]:
        if role == "agg":
            tp, pp, dp, moe_tp, moe_ep, batch = (
                req.tp_size, req.pp_size, req.attention_dp_size, req.moe_tp_size, req.moe_ep_size, req.batch_size
            )
            context = req.max_seq_len or req.isl + req.osl
        else:
            tp = req.prefill_tp_size if role == "prefill" else req.decode_tp_size
            pp = req.prefill_pp_size if role == "prefill" else req.decode_pp_size
            dp = req.attention_dp_size
            moe_tp = (
                req.prefill_moe_tp_size if role == "prefill" else req.decode_moe_tp_size
            ) or req.moe_tp_size
            moe_ep = (
                req.prefill_moe_ep_size if role == "prefill" else req.decode_moe_ep_size
            ) or req.moe_ep_size
            batch = req.prefill_batch_size if role == "prefill" else req.decode_batch_size
            context = (
                req.prefill_max_seq_len if role == "prefill" else req.decode_max_seq_len
            ) or req.max_seq_len or req.isl + req.osl
        tp = tp or req.tp_size
        dp = dp or req.attention_dp_size
        moe_tp, moe_ep = _resolve_moe_widths(tp, dp, moe_tp, moe_ep)
        result = {
            "parallelism": {
                "replicas": (req.prefill_num_workers if role == "prefill" else req.decode_num_workers) or 1,
                "tensor": tp,
                "pipeline": pp or req.pp_size,
                "attention_data": dp,
                "moe_tensor": moe_tp,
                "moe_expert": moe_ep,
            },
            "scheduler": {"max_sequences": req.max_num_seqs or batch or req.batch_size},
        }
        if req.prefix_caching is not None:
            result["kv_cache"] = {"prefix_caching": req.prefix_caching}
        memory_fraction = req.gpu_memory_utilization
        if role == "prefill" and req.prefill_gpu_memory_utilization is not None:
            memory_fraction = req.prefill_gpu_memory_utilization
        elif role == "decode" and req.decode_gpu_memory_utilization is not None:
            memory_fraction = req.decode_gpu_memory_utilization
        if memory_fraction is not None:
            result["kv_cache"] = {
                **(result.get("kv_cache") or {}),
                "capacity": {"type": "default", "memory_fraction": memory_fraction},
            }
        if role != "agg":
            result["context_length"] = context
        return result

    mode = "disaggregated" if req.mode == "disagg" else "aggregated"
    workers = {"prefill": worker("prefill"), "decode": worker("decode")} if req.mode == "disagg" else {
        "aggregated": worker("agg")
    }
    raw = {
        "traffic": {
            "source": {
                "type": "synthetic",
                "input_tokens": req.isl,
                "output_tokens": req.osl,
                "cached_prefix_tokens": req.prefix,
            },
            "load": {"type": "concurrency", "concurrency": req.batch_size},
            "stop": (
                {"requests": 1}
                if req.requests_per_load_unit is None
                else {"requests_per_load_unit": req.requests_per_load_unit}
            ),
        },
        "engine": {
            "mode": mode,
            "model": req.model_path,
            "hardware": req.system,
            "backend": _sdk_backend_id(req.backend),
            "backend_version": req.backend_version,
            "database_mode": req.database_mode,
            "context_length": req.max_seq_len or req.isl + req.osl,
            "workers": workers,
            "gemm_quant_mode": req.gemm_quant_mode,
            "moe_quant_mode": req.moe_quant_mode,
            "kvcache_quant_mode": req.kvcache_quant_mode,
            "enable_chunked_prefill": req.enable_chunked_prefill,
        },
    }
    return CorePredictionConfig.model_validate(raw)


def _metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            return _coerce_float(value)
    return None


def _recommendation_concurrency(metrics: dict[str, Any], req: RecommendRequest) -> int | None:
    """Return reported concurrency, or derive it from Little's law.

    The engine runner reports aggregate request throughput and latency, but does
    not currently publish the legacy ``concurrency`` field used by this API.
    """
    reported = _coerce_int(metrics.get("concurrency") or metrics.get("concurrent_users"))
    if reported is not None:
        return reported

    request_rate = _measured_request_rate(metrics, open_loop=req.target_request_rate is not None)
    request_latency = _metric(metrics, "mean_e2e_latency_ms", "e2e_latency_ms", "request_latency")
    if request_rate is not None and request_latency is not None:
        return max(0, round(request_rate * request_latency / 1000.0))

    # This is only a last-resort compatibility fallback when a runner provides
    # neither the legacy field nor enough measurements for Little's law.
    return round(req.target_concurrency) if req.target_concurrency is not None else None


def _per_gpu_throughput(metrics: dict[str, Any], total_gpus: int | None) -> float | None:
    reported = _metric(metrics, "output_throughput_tok_s_per_gpu")
    if reported is not None:
        return reported
    throughput = _metric(metrics, "output_throughput_tok_s", "tokens_per_second")
    if throughput is None or not total_gpus or total_gpus <= 0:
        return None
    return throughput / total_gpus


def _per_user_throughput(metrics: dict[str, Any], concurrency: int | None) -> float | None:
    reported = _metric(
        metrics,
        "output_throughput_tok_s_per_user",
        "tokens_per_second_per_user",
        "mean_output_token_throughput_per_user",
    )
    if reported is not None:
        return reported
    throughput = _metric(metrics, "output_throughput_tok_s", "tokens_per_second")
    if throughput is None or not concurrency or concurrency <= 0:
        return None
    return throughput / concurrency


def _prediction_total_gpus(summary: dict[str, Any], req: EstimateRequest) -> int | None:
    reported = _coerce_int(summary.get("num_total_gpus"))
    if reported is not None and reported > 0:
        return reported
    if req.mode == "agg":
        return req.tp_size * req.pp_size * req.attention_dp_size
    prefill_gpus = (
        (req.prefill_tp_size or req.tp_size)
        * (req.prefill_pp_size or req.pp_size)
        * req.attention_dp_size
        * (req.prefill_num_workers or 1)
    )
    decode_gpus = (
        (req.decode_tp_size or req.tp_size)
        * (req.decode_pp_size or req.pp_size)
        * req.attention_dp_size
        * (req.decode_num_workers or 1)
    )
    return prefill_gpus + decode_gpus


def _aisimulate_worker_config(raw: dict[str, Any], role: str, req: RecommendRequest) -> WorkerConfig | None:
    worker = (raw.get("engine") or {}).get("workers", {}).get(role)
    if not isinstance(worker, dict):
        return None
    parallel = worker.get("parallelism") or {}
    scheduler = worker.get("scheduler") or {}
    return WorkerConfig(
        tp=parallel.get("tensor"), pp=parallel.get("pipeline"), dp=parallel.get("attention_data"),
        cp=parallel.get("context") or 1,
        moe_tp=parallel.get("moe_tensor"), moe_ep=parallel.get("moe_expert"),
        num_workers=parallel.get("replicas"), batch_size=scheduler.get("max_sequences"),
        backend_version=(raw.get("engine") or {}).get("backend_version") or req.backend_version,
    )


def _aisimulate_candidate_config(candidate: Any, req: RecommendRequest) -> RecommendConfig:
    prediction = candidate.prediction_config or {}
    engine = prediction.get("engine") or {}
    workers = engine.get("workers") or {}
    metrics = candidate.metrics or {}
    mode = "disagg" if engine.get("mode") == "disaggregated" else "agg"
    agg = workers.get("aggregated") or {}
    parallel = agg.get("parallelism") or {}
    concurrency = _recommendation_concurrency(metrics, req)
    replicas = int(parallel.get("replicas") or 1)
    # candidate.used_gpus is cluster-wide. num_total_gpus is the size of one
    # aggregated replica/worker, which consumers use as their scalable unit.
    # Keeping these distinct avoids multiplying a multi-replica deployment a
    # second time downstream.
    gpus_per_agg_replica = math.prod(
        int(parallel.get(dimension) or 1)
        for dimension in ("tensor", "pipeline", "attention_data", "context")
    )
    config = RecommendConfig(
        total_gpus_needed=candidate.used_gpus,
        replicas_needed=replicas,
        num_total_gpus=(gpus_per_agg_replica if mode == "agg" else candidate.used_gpus),
        tp=parallel.get("tensor"), pp=parallel.get("pipeline"), dp=parallel.get("attention_data"),
        cp=parallel.get("context"),
        moe_tp=parallel.get("moe_tensor"), moe_ep=parallel.get("moe_expert"),
        bs=(agg.get("scheduler") or {}).get("max_sequences"),
        ttft=_metric(metrics, "ttft_ms", "mean_ttft_ms"),
        tpot=_metric(metrics, "tpot_ms", "mean_tpot_ms", "itl_ms"),
        request_latency=_metric(metrics, "e2e_latency_ms", "mean_e2e_latency_ms"),
        tokens_per_second=_metric(metrics, "output_throughput_tok_s", "tokens_per_second"),
        tokens_per_second_per_gpu=_per_gpu_throughput(metrics, candidate.used_gpus),
        tokens_per_second_per_user=_per_user_throughput(metrics, concurrency),
        memory=_metric(metrics, "memory_gb", "memory", "peak_memory_gb"),
        concurrency=concurrency,
        request_rate=_measured_request_rate(metrics, open_loop=req.target_request_rate is not None),
        power_w=_metric(metrics, "power_w", "mean_power_w"),
        gemm=metrics.get("gemm") or engine.get("gemm_quant_mode"),
        kvcache=metrics.get("kvcache") or engine.get("kvcache_quant_mode"),
        model=engine.get("model", req.model_path), system=engine.get("hardware", req.system),
        backend=engine.get("backend", req.backend), backend_version=engine.get("backend_version", req.backend_version),
        mode=mode,
    )
    if mode == "disagg":
        config.prefill_config = _aisimulate_worker_config(prediction, "prefill", req)
        config.decode_config = _aisimulate_worker_config(prediction, "decode", req)
    return config


def _demand_config_from_record(record: Any, req: RecommendRequest) -> RecommendConfig:
    """Convert a demand winner from the SDK candidate ledger into a config.

    `result.selected_candidates` is ranked by the SDK's own objective, so the
    service's demand winner is not necessarily in that list. Ledger records
    carry the deployment shape as a flat `provenance.topology` plus measured
    metrics; every value below is copied from that record.
    """
    topology = (
        dict(record.provenance.topology)
        if record.provenance is not None and record.provenance.topology
        else {}
    )
    metrics = record.metrics or {}
    mode = "disagg" if topology.get("deployment_mode") == "disagg" else "agg"
    concurrency = _recommendation_concurrency(metrics, req)
    replicas = int(topology.get("replicas") or 1)
    config = RecommendConfig(
        total_gpus_needed=record.used_gpus,
        replicas_needed=replicas,
        # One aggregate replica's GPU count; a disaggregated candidate is one
        # complete multi-pool deployment, so its cluster total is the unit.
        num_total_gpus=(
            math.prod(
                int(topology.get(key) or 1)
                for key in ("tp", "pp", "attention_dp", "context")
            )
            if mode == "agg"
            else record.used_gpus
        ),
        tp=topology.get("tp"),
        pp=topology.get("pp"),
        dp=topology.get("attention_dp"),
        cp=topology.get("context"),
        moe_tp=topology.get("moe_tp"),
        moe_ep=topology.get("moe_ep"),
        ttft=_metric(metrics, "ttft_ms", "mean_ttft_ms"),
        tpot=_metric(metrics, "tpot_ms", "mean_tpot_ms", "itl_ms"),
        request_latency=_metric(metrics, "e2e_latency_ms", "mean_e2e_latency_ms"),
        tokens_per_second=_metric(metrics, "output_throughput_tok_s", "tokens_per_second"),
        tokens_per_second_per_gpu=_per_gpu_throughput(metrics, record.used_gpus),
        tokens_per_second_per_user=_per_user_throughput(metrics, concurrency),
        memory=_metric(metrics, "memory_gb", "memory", "peak_memory_gb"),
        concurrency=concurrency,
        request_rate=_measured_request_rate(metrics, open_loop=req.target_request_rate is not None),
        power_w=_metric(metrics, "power_w", "mean_power_w"),
        model=req.model_path,
        system=req.system,
        backend=req.backend,
        backend_version=req.backend_version,
        mode=mode,
    )
    if mode == "disagg":
        config.prefill_config = WorkerConfig(
            tp=topology.get("prefill_tp"),
            pp=topology.get("prefill_pp"),
            dp=topology.get("prefill_attention_dp"),
            cp=topology.get("prefill_context") or 1,
            moe_tp=topology.get("prefill_moe_tp"),
            moe_ep=topology.get("prefill_moe_ep"),
            num_workers=topology.get("prefill_replicas"),
            backend_version=req.backend_version,
        )
        config.decode_config = WorkerConfig(
            tp=topology.get("decode_tp"),
            pp=topology.get("decode_pp"),
            dp=topology.get("decode_attention_dp"),
            cp=topology.get("decode_context") or 1,
            moe_tp=topology.get("decode_moe_tp"),
            moe_ep=topology.get("decode_moe_ep"),
            num_workers=topology.get("decode_replicas"),
            backend_version=req.backend_version,
        )
    return config


def _candidate_record_to_response(record: Any, *, transfer_delay_modeled: bool) -> CandidateResponse:
    topology = dict(record.provenance.topology) if record.provenance and record.provenance.topology else {}
    mode = topology.get("deployment_mode", "agg")
    perf_data = []
    if record.provenance and record.provenance.performance_data:
        perf_data = record.provenance.performance_data
    # Preserve first-seen order; a set would make the response non-deterministic.
    provenance_sources = list(dict.fromkeys(
        str(entry.get("source", "")) for entry in perf_data if entry.get("source")
    ))
    return CandidateResponse(
        candidate_id=record.candidate_id,
        status=str(record.status.value) if hasattr(record.status, "value") else str(record.status),
        reason_category=str(record.reason_category.value) if record.reason_category and hasattr(record.reason_category, "value") else (str(record.reason_category) if record.reason_category else None),
        mode=mode,
        topology=topology,
        used_gpus=record.used_gpus,
        metrics=dict(record.metrics) if record.metrics else {},
        objectives=dict(record.objectives) if record.objectives else None,
        transfer_delay_modeled=transfer_delay_modeled,
        provenance_sources=provenance_sources,
    )


def _finite_metric(metrics: dict[str, float | None], *names: str) -> float | None:
    """Return the first finite numeric metric, ignoring booleans and nulls."""
    for name in names:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _demand_sustained_rate(metrics: dict[str, Any]) -> float | None:
    """Offered request rate a replay sustained, excluding the final drain tail.

    A replay is timed from the first arrival to the last completion, so a
    cluster that keeps up still appends one request's latency to the measured
    window. `completed / duration` consequently reports a rate below the
    offered one for every candidate with a non-zero latency, and comparing that
    value with the target marks correctly sized clusters unmet. Removing one
    mean latency recovers the arrival window. A saturated cluster keeps its
    queue well past that point, so its sustained rate still falls short.
    """
    achieved = _finite_metric(
        metrics, "request_throughput_rps", "requests_per_second", "request_rate"
    )
    completed = _finite_metric(metrics, "completed_requests")
    if completed is not None and completed <= 0:
        return None
    duration_ms = _finite_metric(metrics, "duration_ms")
    if duration_ms is None:
        duration_s = _finite_metric(metrics, "vllm_estimated_duration_s")
        duration_ms = duration_s * 1000.0 if duration_s is not None else None
    if achieved is None or completed is None or duration_ms is None or duration_ms <= 0:
        return achieved
    e2e_ms = _finite_metric(
        metrics, "mean_e2e_latency_ms", "e2e_latency_ms", "request_latency"
    ) or 0.0
    arrival_ms = duration_ms - e2e_ms
    if arrival_ms <= 0:
        return achieved
    return completed * 1000.0 / arrival_ms


def _measured_request_rate(metrics: dict[str, Any], *, open_loop: bool) -> float | None:
    """Request rate for reported configs, preferring the arrival window.

    `completed / duration` includes the final response drain, so using it for
    the config's `request_rate` (and for Little's-law concurrency) would
    understate a cluster that kept up. The correction only holds for open-loop
    traffic, where arrivals are spread over a window; a closed-loop replay
    starts every request together, so its duration is roughly one wave and
    subtracting a mean latency would inflate the rate.
    """
    if open_loop:
        sustained = _demand_sustained_rate(metrics)
        if sustained is not None:
            return sustained
    return _metric(
        metrics, "request_rate", "requests_per_second", "request_throughput_rps"
    )


def _demand_window_seconds(metrics: dict[str, Any]) -> float | None:
    """Arrival window a demand replay covered, in seconds."""
    duration = _finite_metric(metrics, "vllm_estimated_duration_s")
    if duration is not None and duration > 0:
        duration_ms = duration * 1000.0
    else:
        duration_ms = _finite_metric(metrics, "duration_ms")
    if duration_ms is None or duration_ms <= 0:
        return None
    e2e_ms = _finite_metric(
        metrics, "mean_e2e_latency_ms", "e2e_latency_ms", "request_latency"
    ) or 0.0
    arrival_ms = duration_ms - e2e_ms
    if arrival_ms <= 0:
        return duration_ms / 1000.0
    return arrival_ms / 1000.0


def _gpu_count_for_ranking(record: Any) -> float:
    """GPU count for ordering; an unknown count must not look cheapest."""
    used = getattr(record, "used_gpus", None)
    if isinstance(used, bool) or not isinstance(used, (int, float)) or used <= 0:
        return math.inf
    return float(used)


def _demand_record_signature(record: Any) -> str:
    """Stable identity for a candidate shape across separate demand windows."""
    topology = getattr(getattr(record, "provenance", None), "topology", None) or {}
    return json.dumps(
        [getattr(record, "used_gpus", None), topology], sort_keys=True, default=str
    )


def _candidate_status(record: Any) -> str:
    status = getattr(record, "status", "")
    return str(getattr(status, "value", status))


def _select_demand_record(
    records: list[Any],
    target_rps: float,
) -> tuple[Any | None, bool, float | None, int]:
    """Pick the demand winner from the SDK candidate ledger.

    Compliant candidates are ranked by fewest GPUs (matching the SDK's
    `min_gpus` ordering), then highest sustained rate, then lowest E2E latency.
    When nothing meets the target, the highest sustained rate is returned
    instead so the caller can report the shortfall at 200 rather than a bare
    422. `met` is the best of the evaluated configurations, not a guarantee.

    Compliance uses the sustained arrival-window rate rather than
    `completed / duration`, which includes the final response drain and would
    mark every candidate with a non-zero latency as unmet.
    """
    measured: list[tuple[Any, float]] = []
    for record in records:
        if _candidate_status(record) != "feasible":
            continue
        sustained = _demand_sustained_rate(record.metrics or {})
        if sustained is None:
            continue
        measured.append((record, sustained))
    if not measured:
        return None, False, None, 0

    # A small relative tolerance absorbs float noise from an exactly-served rate.
    tolerance = max(target_rps * 1e-3, 1e-9)
    compliant = [(record, rps) for record, rps in measured if rps + tolerance >= target_rps]
    if compliant:
        winner, _ = min(
            compliant,
            key=lambda item: (
                _gpu_count_for_ranking(item[0]),
                -item[1],
                _finite_metric(item[0].metrics or {}, "mean_e2e_latency_ms") or math.inf,
                getattr(item[0], "candidate_id", ""),
            ),
        )
        best = max(rps for _, rps in compliant)
        return winner, True, best, len(measured)

    winner, best = min(
        measured,
        key=lambda item: (
            -item[1],
            _gpu_count_for_ranking(item[0]),
            getattr(item[0], "candidate_id", ""),
        ),
    )
    return winner, False, best, len(measured)


def _demand_outcome(
    record: Any,
    *,
    target_rps: float,
    met: bool,
    best_observed_rps: float | None,
    evaluated: int,
) -> DemandOutcome:
    """Build the demand block from the selected candidate's replay metrics.

    `achieved_rps` is the rate sustained over the arrival window, so it is the
    value comparable to `target_rps`. `window_seconds` is that arrival window,
    excluding the final response drain.
    """
    metrics = record.metrics or {}
    achieved = _demand_sustained_rate(metrics)
    e2e_ms = _finite_metric(
        metrics, "mean_e2e_latency_ms", "e2e_latency_ms", "request_latency"
    )
    completed = _finite_metric(metrics, "completed_requests")
    # Little's law: only meaningful on a run that completed work.
    observed = (
        achieved * e2e_ms / 1000.0
        if achieved is not None and e2e_ms is not None and (completed is None or completed > 0)
        else None
    )
    window_seconds = _demand_window_seconds(metrics)
    return DemandOutcome(
        target_rps=float(target_rps),
        achieved_rps=round(achieved, 3) if achieved is not None else None,
        met=met,
        observed_concurrency=round(observed, 1) if observed is not None else None,
        window_seconds=round(window_seconds, 3) if window_seconds is not None else None,
        completed_requests=int(completed) if completed is not None else None,
        evaluated_configurations=evaluated,
        best_observed_rps=(
            round(best_observed_rps, 3)
            if best_observed_rps is not None
            else (round(achieved, 3) if achieved is not None else None)
        ),
        simulator="aisimulate",
    )


def _sweep_meta(result: Any, req: RecommendRequest) -> SweepMeta:
    """Summarise a sweep's run control from SDK counts/provenance."""
    counts = getattr(result, "counts", None)
    provenance = getattr(result, "provenance", None)
    evaluated = getattr(counts, "evaluated", None)
    if not isinstance(evaluated, int):
        evaluated = len(getattr(result, "candidates", []) or [])
    _, window_max_gpus = _effective_gpu_window(req)
    max_trials, _, _ = _recommendation_trial_budget(
        req.objective,
        window_max_gpus,
        has_requested_max=req.max_candidate_gpus is not None,
    )
    return SweepMeta(
        run_id=getattr(provenance, "run_id", None),
        algorithm=_sweep_algorithm(),
        seed=_sweep_seed(),
        max_trials=max_trials,
        evaluated=evaluated,
        feasible=getattr(counts, "feasible", None),
        timed_out=getattr(counts, "timed_out", None),
        # The budget being spent means the search stopped at its trial limit,
        # never that it enumerated the whole space.
        budget_exhausted=evaluated >= max_trials,
        strict_sla=req.sla_mode == "enforce",
        window_min_gpus=max(req.min_candidate_gpus or 1, 1),
        window_max_gpus=window_max_gpus,
    )


def _run_demand_windows(req: RecommendRequest) -> tuple[list[Any], SweepMeta]:
    """Run the demand GPU-window ladder and merge the measured ledger.

    A window that cannot hold the model is skipped, and the ladder stops as
    soon as a measured candidate sustains the offered rate or the search budget
    is spent. Trial and timeout overrides keep each window inside the remaining
    wall-clock budget, because the SDK's own budget covers one call only.
    """
    windows = _demand_gpu_windows(req)
    max_trials, parallelism, candidate_timeout = _recommendation_trial_budget(
        req.objective,
        windows[0][1],
        has_requested_max=req.max_candidate_gpus is not None,
    )
    deadline = time.monotonic() + _demand_search_budget_seconds()
    target_rps = float(req.target_request_rate or 0.0)
    records: list[Any] = []
    seen: set[str] = set()
    attempted: list[tuple[int, int]] = []
    searched = False
    offered_trials = 0
    evaluated = 0
    feasible = 0
    timed_out = 0
    budget_exhausted = False
    run_id: str | None = None
    for lower, upper in windows:
        remaining = deadline - time.monotonic()
        if attempted and remaining < _DEMAND_MIN_WINDOW_S:
            budget_exhausted = True
            break
        attempted.append((lower, upper))
        window_timeout = max(1, min(candidate_timeout, int(remaining)))
        window_trials = max(
            1,
            min(max_trials, max(1, int(remaining // window_timeout)) * parallelism),
        )
        window_req = req.model_copy(
            update={"min_candidate_gpus": lower, "max_candidate_gpus": upper}
        )
        offered_trials += window_trials
        try:
            result = _run_aisimulate_recommendation(
                window_req,
                max_trials_override=window_trials,
                candidate_timeout_override=window_timeout,
            )
        except Exception as exc:
            # The window is simply below the model's fit; the next, larger
            # window is the point of the ladder.
            if _is_no_viable_config(exc):
                continue
            raise
        searched = True
        counts = getattr(result, "counts", None)
        window_evaluated = getattr(counts, "evaluated", None)
        if not isinstance(window_evaluated, int):
            window_evaluated = len(getattr(result, "candidates", []) or [])
        evaluated += window_evaluated
        feasible += int(getattr(counts, "feasible", 0) or 0)
        timed_out += int(getattr(counts, "timed_out", 0) or 0)
        budget_exhausted = budget_exhausted or window_evaluated >= window_trials
        run_id = getattr(getattr(result, "provenance", None), "run_id", None) or run_id
        for record in getattr(result, "candidates", []) or []:
            signature = _demand_record_signature(record)
            if signature in seen:
                continue
            seen.add(signature)
            records.append(record)
        # Stop growing the window once a measured candidate sustains the rate.
        if _select_demand_record(records, target_rps)[1]:
            break
    last = attempted[-1] if attempted else windows[0]
    return records, SweepMeta(
        run_id=run_id,
        algorithm=_sweep_algorithm(),
        seed=_sweep_seed(),
        max_trials=offered_trials or max_trials,
        evaluated=evaluated,
        feasible=feasible if searched else None,
        timed_out=timed_out if searched else None,
        budget_exhausted=budget_exhausted,
        strict_sla=False,
        window_min_gpus=last[0],
        window_max_gpus=last[1],
    )


def _capacity_context_tokens(req: RecommendRequest) -> int:
    """Peak context one sequence occupies: ``max_seq_len`` or ``ISL + OSL``.

    Capacity is a residency question, so this is the full sequence length with
    no prefix-cache credit. A shorter average prompt does not reduce the KV a
    resident sequence can grow to.
    """
    return int(req.max_seq_len or (req.isl + req.osl))


def _gpu_window_label(lower: int, upper: int) -> str:
    return str(lower) if lower == upper else f"{lower}-{upper}"


#: Capacity comparisons share a wall-clock budget across all performance
#: simulations. Capacity itself is already known from the KV estimate, so an
#: option that misses this budget stays promotable with an explicit reason.
_CAPACITY_OPTION_SIMULATION_BUDGET_S = 40
_CAPACITY_OPTION_SIMULATION_WORKERS = 2
#: One process-wide budget for capacity simulations. A simulation that outlives
#: its request keeps its slot until it actually returns, so abandoned work
#: cannot multiply across requests and saturate the CPU.
_CAPACITY_SIMULATION_SLOTS = threading.BoundedSemaphore(_CAPACITY_OPTION_SIMULATION_WORKERS)
_CAPACITY_OPTION_DEFAULT_COUNT = 8
_CAPACITY_OPTION_DEFAULT_COUNT_NO_DISAGG = 6


@dataclass(frozen=True)
class _CapacityShape:
    """One aggregated worker shape and its per-rank KV token capacity."""

    tp: int
    dp: int
    moe_tp: int
    moe_ep: int
    per_rank_tokens: int

    @property
    def gpus_per_worker(self) -> int:
        return self.tp * self.dp

    @property
    def tokens_per_replica(self) -> int:
        # Each attention-DP rank has its own KV budget, so a replica holds
        # per-rank tokens on every DP rank. TP splits each token across ranks
        # and is already reflected in the per-rank figure.
        return self.per_rank_tokens * self.dp


@dataclass(frozen=True)
class _CapacityCandidate:
    """A worker shape plus the replica count needed to reach a token target."""

    shape: _CapacityShape
    replicas: int
    context_tokens: int

    @property
    def total_gpus(self) -> int:
        return self.shape.gpus_per_worker * self.replicas

    @property
    def capacity_tokens(self) -> int:
        return self.shape.tokens_per_replica * self.replicas

    @property
    def capacity_sequences(self) -> int:
        return self.capacity_tokens // self.context_tokens


@dataclass(frozen=True)
class _DisaggCapacityCandidate:
    """A prefill/decode pair whose decode pool holds the requested sequences.

    Prefill holds one sequence transiently, so it contributes GPUs but not
    resident capacity. The decode pool is the capacity-bearing side.
    """

    prefill: _CapacityCandidate
    decode: _CapacityCandidate

    @property
    def total_gpus(self) -> int:
        return self.prefill.total_gpus + self.decode.total_gpus

    @property
    def capacity_tokens(self) -> int:
        return self.decode.capacity_tokens

    @property
    def capacity_sequences(self) -> int:
        return self.decode.capacity_sequences


def _capacity_shapes(
    req: RecommendRequest,
    model_path: str,
    *,
    backend: str,
    backend_version: str,
    context_tokens: int,
    gpu_budget: int,
) -> list[_CapacityShape]:
    """Enumerate every aggregated shape the SDK's search space allows.

    This mirrors ``parallel_configs_for(..., deployment_mode="agg")`` — the
    same worker-shape ladder, backend filters, wideEP heuristic, and
    KV-feasibility filter — but keeps the per-shape token capacity that
    function discards. Replica counts are added by the caller: they only
    multiply capacity, so the search does not need one entry per replica.

    Aggregated only: a disaggregated deployment replicates the full weights in
    each pool, so it cannot need fewer GPUs to hold weights plus KV than an
    aggregated deployment with the same worker shape.
    """
    from aisimulate.sweeper.kv_estimate import feasible_shape_tokens
    from aisimulate.sweeper.model_hw import resolve_model_hardware
    from aisimulate.sweeper.parallel_enum import enumerate_parallel_configs

    facts = resolve_model_hardware(model_path, req.system, backend=backend)
    replica_configs = enumerate_parallel_configs(
        is_moe=facts.is_moe,
        backend=backend,
        gpu_budget=gpu_budget,
        min_gpus_per_worker=1,
        enable_wideep=facts.enable_wideep,
        allow_moe_pure_tp=True,
    )
    # `feasible_shape_tokens` estimates each distinct shape once, so collapse
    # the replica ladder first and preserve its deterministic first-seen order.
    worker_shapes = list(dict.fromkeys(config.shape for config in replica_configs))
    capacities = feasible_shape_tokens(
        worker_shapes,
        model_name=model_path,
        hardware_sku=req.system,
        backend=backend,
        backend_version=backend_version,
        max_seq_len=context_tokens,
        # Reserve for one sequence at full context and the requested number of
        # resident sequences. Both are the workload's own bounds, not invented
        # defaults; they match the memory breakdown the response reports.
        max_num_tokens=context_tokens,
        max_batch_size=int(req.target_concurrency or 1),
        memory_fraction=_backend_memory_fraction(req.backend, backend_version),
    )
    return [
        _CapacityShape(
            tp=shape.tp,
            dp=shape.dp,
            moe_tp=shape.moe_tp,
            moe_ep=shape.moe_ep,
            per_rank_tokens=tokens,
        )
        for shape, tokens in capacities.items()
        if shape.gpus_per_worker <= gpu_budget
    ]


def _pareto_auto_window(
    req: RecommendRequest,
    *,
    backend: str,
    backend_version: str | None,
) -> tuple[int, int]:
    """Pick a Pareto GPU window from the model's own fit floor.

    An omitted Pareto window should not start at one GPU and should not reach
    the server ceiling: both waste trials on shapes the model cannot run. This
    reuses the aggregated KV-feasibility enumeration (`_capacity_shapes`) to
    find the smallest shape that holds one sequence at the workload context,
    then searches from that floor up to twice the floor, at least 16 GPUs, and
    never past the Pareto window cap or the server ceiling.

    Aggregated is a valid floor for both deployment modes: a disaggregated
    deployment replicates the full weights in the prefill and decode pools, so
    it cannot need fewer GPUs than the aggregated shape.

    Raises ``ValueError`` when no shape fits inside the cap, so the caller
    returns an explicit no-result response instead of running a useless sweep.
    """
    server_ceiling = _max_candidate_gpus()
    ceiling = min(server_ceiling, _pareto_max_window_gpus())
    context_tokens = _capacity_context_tokens(req)
    try:
        with _with_model_config(req.model_path, req.model_config_data) as model_path:
            shapes = _capacity_shapes(
                req,
                model_path,
                backend=backend,
                backend_version=backend_version,
                context_tokens=context_tokens,
                gpu_budget=ceiling,
            )
    except Exception as error:  # a failed preflight must not mask the sweep's own error
        fallback_upper = min(16, ceiling)
        logger.warning(
            "Pareto auto-window preflight failed for %s on %s (%s); "
            "using the bounded 1-%d window: %s",
            req.model_path,
            req.system,
            req.backend,
            fallback_upper,
            error,
        )
        return 1, max(1, fallback_upper)
    if not shapes:
        raise ValueError(
            f"{req.model_path} on {req.system}: no aggregated parallel configuration "
            f"holds a {context_tokens}-token sequence within {ceiling} GPUs "
            f"({req.backend} KV-cache estimate). Pass explicit min_candidate_gpus / "
            "max_candidate_gpus to widen the Pareto window."
        )
    floor = min(shape.gpus_per_worker for shape in shapes)
    upper = min(ceiling, max(16, floor * 2))
    return floor, max(floor, upper)


def _capacity_aggregate_candidates(
    shapes: list[_CapacityShape],
    *,
    target_concurrency: int,
    context_tokens: int,
    lower_gpus: int,
    upper_gpus: int,
) -> tuple[list[_CapacityCandidate], list[_CapacityCandidate], int | None]:
    """Enumerate aggregate deployments around the minimum for each shape.

    Returns ``(fitting, best_effort, minimum_gpus)``. ``fitting`` includes the
    smallest deployment that holds the target plus nearby larger ones, so the
    comparison keeps same-GPU ties and larger alternatives without enumerating
    every replica count. ``best_effort`` is the largest deployment of each
    shape inside the window, used only when nothing holds the target.
    """
    needed_tokens = target_concurrency * context_tokens
    minimum_gpus: int | None = None
    fitting: list[_CapacityCandidate] = []
    best_effort: list[_CapacityCandidate] = []

    for shape in shapes:
        per_worker = shape.gpus_per_worker
        if per_worker <= 0 or per_worker > upper_gpus:
            continue
        min_replicas = max(1, math.ceil(lower_gpus / per_worker))
        if per_worker * min_replicas > upper_gpus:
            continue
        single_gpus = per_worker * min_replicas
        if minimum_gpus is None or single_gpus < minimum_gpus:
            minimum_gpus = single_gpus

        max_replicas = upper_gpus // per_worker
        needed_replicas = max(1, math.ceil(needed_tokens / shape.tokens_per_replica))
        first_fitting = max(min_replicas, needed_replicas)
        if first_fitting <= max_replicas:
            # Keep the minimum and the next two replica counts, plus the largest
            # deployment in the window. This gives larger options even when a
            # single shape is feasible.
            for replicas in sorted({first_fitting, first_fitting + 1, first_fitting + 2, max_replicas}):
                if first_fitting <= replicas <= max_replicas:
                    fitting.append(_CapacityCandidate(shape, replicas, context_tokens))
        if max_replicas >= min_replicas:
            best_effort.append(_CapacityCandidate(shape, max_replicas, context_tokens))

    # Collapse duplicate shape/replica pairs while preserving deterministic order.
    fitting = list(dict.fromkeys(fitting))
    best_effort = list(dict.fromkeys(best_effort))
    return fitting, best_effort, minimum_gpus


def _capacity_disagg_candidates(
    req: RecommendRequest,
    model_path: str,
    *,
    backend: str,
    backend_version: str,
    context_tokens: int,
    target_concurrency: int,
    lower_gpus: int,
    upper_gpus: int,
    minimum_gpus: int | None,
) -> list[_DisaggCapacityCandidate]:
    """Enumerate disaggregated options from the SDK's P/D search space.

    The SDK enumerates prefill/decode shapes and replica counts together; this
    keeps the pairs whose decode pool holds the requested resident sequences.
    Prefill only needs to hold one sequence transiently, so its KV capacity is
    estimated at batch size 1. Weights are replicated per pool, so a P/D pair is
    only offered as an option larger than the aggregated minimum.
    """
    from aisimulate.sweeper.kv_estimate import feasible_shape_tokens
    from aisimulate.sweeper.model_hw import resolve_model_hardware
    from aisimulate.sweeper.parallel_enum import enumerate_disagg_configs

    try:
        facts = resolve_model_hardware(model_path, req.system, backend=backend)
        pairs = enumerate_disagg_configs(
            is_moe=facts.is_moe,
            backend=backend,
            gpu_budget=upper_gpus,
            min_gpu_budget=lower_gpus,
            min_gpus_per_worker=1,
            enable_wideep=facts.enable_wideep,
            allow_moe_pure_tp=True,
        )
        if not pairs:
            return []
        prefill_shapes = list(dict.fromkeys(pair.prefill.shape for pair in pairs))
        decode_shapes = list(dict.fromkeys(pair.decode.shape for pair in pairs))
        memory_fraction = _backend_memory_fraction(req.backend, backend_version)
        prefill_tokens = feasible_shape_tokens(
            prefill_shapes,
            model_name=model_path,
            hardware_sku=req.system,
            backend=backend,
            backend_version=backend_version,
            max_seq_len=context_tokens,
            max_num_tokens=context_tokens,
            # Prefill only needs to hold one sequence transiently.
            max_batch_size=1,
            memory_fraction=memory_fraction,
        )
        decode_tokens = feasible_shape_tokens(
            decode_shapes,
            model_name=model_path,
            hardware_sku=req.system,
            backend=backend,
            backend_version=backend_version,
            max_seq_len=context_tokens,
            max_num_tokens=context_tokens,
            max_batch_size=max(1, target_concurrency),
            memory_fraction=memory_fraction,
        )
    except Exception as error:  # disagg is an optional comparison set
        logger.warning(
            "Disaggregated capacity options unavailable for %s on %s: %s",
            req.model_path,
            req.system,
            error,
        )
        return []

    if not prefill_tokens or not decode_tokens:
        return []

    needed_tokens = target_concurrency * context_tokens

    def candidate_for(worker: Any, tokens: dict[Any, int]) -> _CapacityCandidate:
        shape = worker.shape
        return _CapacityCandidate(
            shape=_CapacityShape(
                tp=shape.tp,
                dp=shape.dp,
                moe_tp=shape.moe_tp,
                moe_ep=shape.moe_ep,
                per_rank_tokens=tokens[shape],
            ),
            replicas=worker.replicas,
            context_tokens=context_tokens,
        )

    candidates: list[_DisaggCapacityCandidate] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    for pair in pairs:
        if pair.prefill.shape not in prefill_tokens or pair.decode.shape not in decode_tokens:
            continue
        prefill = candidate_for(pair.prefill, prefill_tokens)
        decode = candidate_for(pair.decode, decode_tokens)
        if decode.capacity_tokens < needed_tokens:
            continue
        total = prefill.total_gpus + decode.total_gpus
        if total > upper_gpus or total < lower_gpus:
            continue
        if minimum_gpus is not None and total <= minimum_gpus:
            continue
        key = (
            prefill.shape.tp,
            prefill.shape.dp,
            prefill.replicas,
            decode.shape.tp,
            decode.shape.dp,
            decode.replicas,
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(_DisaggCapacityCandidate(prefill, decode))
    return candidates


def _disagg_capacity_order(candidate: _DisaggCapacityCandidate) -> tuple[int, ...]:
    prefill = candidate.prefill.shape
    decode = candidate.decode.shape
    return (
        candidate.total_gpus,
        -candidate.capacity_sequences,
        -decode.per_rank_tokens,
        candidate.prefill.total_gpus,
        prefill.tp,
        prefill.dp,
        decode.tp,
        decode.dp,
        candidate.decode.replicas,
    )


def _agg_candidate_signature(candidate: _CapacityCandidate) -> tuple[int, int, int, int, int]:
    shape = candidate.shape
    return (shape.tp, shape.dp, shape.moe_tp, shape.moe_ep, candidate.replicas)


def _disagg_candidate_signature(
    candidate: _DisaggCapacityCandidate,
) -> tuple[int, int, int, int, int, int, int]:
    prefill = candidate.prefill.shape
    decode = candidate.decode.shape
    return (
        prefill.tp,
        prefill.dp,
        prefill.moe_tp,
        prefill.moe_ep,
        decode.tp,
        decode.dp,
        candidate.decode.replicas,
    )


def _select_disagg_representatives(
    candidates: list[_DisaggCapacityCandidate],
    *,
    limit: int = 2,
) -> list[_DisaggCapacityCandidate]:
    """Pick the smallest P/D option plus one at a larger GPU count."""
    if not candidates or limit <= 0:
        return []
    ordered = sorted(candidates, key=_disagg_capacity_order)
    selected = [ordered[0]]
    for candidate in ordered[1:]:
        if len(selected) >= limit:
            break
        if candidate.total_gpus > selected[0].total_gpus:
            selected.append(candidate)
    if len(selected) < limit:
        for candidate in ordered[1:]:
            if len(selected) >= limit:
                break
            if candidate not in selected:
                selected.append(candidate)
    return selected[:limit]


def _select_capacity_options(
    *,
    aggregate_fitting: list[_CapacityCandidate],
    aggregate_best_effort: list[_CapacityCandidate],
    disagg_candidates: list[_DisaggCapacityCandidate],
    option_count: int,
) -> list[tuple[str, _CapacityCandidate | _DisaggCapacityCandidate]]:
    """Build the ranked option list: minimum, ties, larger, then P/D."""
    if option_count <= 0:
        return []
    if not aggregate_fitting:
        ordered = sorted(aggregate_best_effort, key=_capacity_shortfall_order)
        return [("shortfall", candidate) for candidate in ordered[:option_count]]

    minimum = min(aggregate_fitting, key=_capacity_order)
    selected: list[tuple[str, _CapacityCandidate | _DisaggCapacityCandidate]] = [
        ("minimum", minimum)
    ]
    chosen_agg = {_agg_candidate_signature(minimum)}
    minimum_gpus = minimum.total_gpus

    ties = [
        candidate
        for candidate in aggregate_fitting
        if candidate.total_gpus == minimum_gpus
        and _agg_candidate_signature(candidate) not in chosen_agg
    ]
    for candidate in sorted(ties, key=_capacity_order)[:3]:
        selected.append(("same_gpus", candidate))
        chosen_agg.add(_agg_candidate_signature(candidate))

    larger_source = sorted(
        (
            candidate
            for candidate in aggregate_fitting
            if candidate.total_gpus > minimum_gpus
            and _agg_candidate_signature(candidate) not in chosen_agg
        ),
        key=_capacity_order,
    )
    larger: list[_CapacityCandidate] = []
    seen_gpu_counts: set[int] = set()
    for candidate in larger_source:
        if len(larger) >= 2:
            break
        if candidate.total_gpus in seen_gpu_counts:
            continue
        larger.append(candidate)
        seen_gpu_counts.add(candidate.total_gpus)
    if len(larger) < 2:
        for candidate in larger_source:
            if len(larger) >= 2:
                break
            if candidate in larger:
                continue
            larger.append(candidate)
    for candidate in larger:
        selected.append(("larger", candidate))
        chosen_agg.add(_agg_candidate_signature(candidate))

    disagg_selected = _select_disagg_representatives(disagg_candidates)
    for candidate in disagg_selected:
        selected.append(("disaggregated", candidate))

    if len(selected) < option_count:
        for candidate in sorted(aggregate_fitting, key=_capacity_order):
            if len(selected) >= option_count:
                break
            signature = _agg_candidate_signature(candidate)
            if signature in chosen_agg:
                continue
            role = "same_gpus" if candidate.total_gpus == minimum_gpus else "larger"
            selected.append((role, candidate))
            chosen_agg.add(signature)
    if len(selected) < option_count:
        chosen_disagg = {_disagg_candidate_signature(candidate) for candidate in disagg_selected}
        for candidate in sorted(disagg_candidates, key=_disagg_capacity_order):
            if len(selected) >= option_count:
                break
            if _disagg_candidate_signature(candidate) in chosen_disagg:
                continue
            selected.append(("disaggregated", candidate))
            chosen_disagg.add(_disagg_candidate_signature(candidate))
    return selected[:option_count]


def _capacity_option_id(
    candidate: _CapacityCandidate | _DisaggCapacityCandidate,
    *,
    used: set[str],
) -> str:
    if isinstance(candidate, _DisaggCapacityCandidate):
        prefill = candidate.prefill.shape
        decode = candidate.decode.shape
        base = (
            f"disagg-pt{prefill.tp}-pd{prefill.dp}-pr{candidate.prefill.replicas}"
            f"-dt{decode.tp}-dd{decode.dp}-dr{candidate.decode.replicas}"
            f"-me{decode.moe_ep}"
        )
    else:
        shape = candidate.shape
        base = f"agg-t{shape.tp}-d{shape.dp}-mt{shape.moe_tp}-me{shape.moe_ep}-r{candidate.replicas}"
    option_id = base
    suffix = 2
    while option_id in used:
        option_id = f"{base}-{suffix}"
        suffix += 1
    used.add(option_id)
    return option_id



def _capacity_order(candidate: _CapacityCandidate) -> tuple[int, ...]:
    shape = candidate.shape
    return (
        candidate.total_gpus,
        -candidate.capacity_sequences,
        -shape.per_rank_tokens,
        candidate.shape.gpus_per_worker,
        shape.tp,
        shape.dp,
        shape.moe_tp,
        shape.moe_ep,
        candidate.replicas,
    )


def _capacity_shortfall_order(candidate: _CapacityCandidate) -> tuple[int, ...]:
    shape = candidate.shape
    return (
        -candidate.capacity_sequences,
        candidate.total_gpus,
        -shape.per_rank_tokens,
        candidate.shape.gpus_per_worker,
        shape.tp,
        shape.dp,
        shape.moe_tp,
        shape.moe_ep,
        candidate.replicas,
    )


def _capacity_prediction(
    req: RecommendRequest,
    model_path: str,
    *,
    backend_version: str,
    candidate: _CapacityCandidate | _DisaggCapacityCandidate,
    target_concurrency: int,
) -> tuple[Any | None, str | None]:
    """Simulate one capacity option at the requested concurrency.

    Capacity does not depend on this replay, so a coverage gap in the timing
    data must not fail the sizing answer. The caller reports the capacity block
    with ``performance_available=false`` and a reason instead.
    """
    if isinstance(candidate, _DisaggCapacityCandidate):
        prefill = candidate.prefill
        decode = candidate.decode
        prediction_req = EstimateRequest(
            model_path=model_path,
            system=req.system,
            backend=req.backend,
            backend_version=backend_version,
            isl=req.isl,
            osl=req.osl,
            max_seq_len=req.max_seq_len,
            max_num_seqs=max(1, math.ceil(target_concurrency / decode.replicas)),
            mode="disagg",
            prefill_tp_size=prefill.shape.tp,
            prefill_pp_size=1,
            prefill_moe_tp_size=prefill.shape.moe_tp,
            prefill_moe_ep_size=prefill.shape.moe_ep,
            prefill_num_workers=prefill.replicas,
            prefill_batch_size=max(1, math.ceil(target_concurrency / prefill.replicas)),
            decode_tp_size=decode.shape.tp,
            decode_pp_size=1,
            decode_moe_tp_size=decode.shape.moe_tp,
            decode_moe_ep_size=decode.shape.moe_ep,
            decode_num_workers=decode.replicas,
            decode_batch_size=max(1, math.ceil(target_concurrency / decode.replicas)),
            attention_dp_size=decode.shape.dp,
            batch_size=target_concurrency,
            requests_per_load_unit=_demand_requests_per_load_unit(),
            database_mode=req.database_mode,
            gpu_memory_utilization=_backend_memory_fraction(req.backend, backend_version),
            simulator="aisimulate",
        )
    else:
        replicas = candidate.replicas
        per_replica = max(1, math.ceil(target_concurrency / replicas))
        prediction_req = EstimateRequest(
            model_path=model_path,
            system=req.system,
            backend=req.backend,
            backend_version=backend_version,
            isl=req.isl,
            osl=req.osl,
            max_seq_len=req.max_seq_len,
            max_num_seqs=per_replica,
            tp_size=candidate.shape.tp,
            pp_size=1,
            attention_dp_size=candidate.shape.dp,
            moe_tp_size=candidate.shape.moe_tp,
            moe_ep_size=candidate.shape.moe_ep,
            batch_size=per_replica,
            # One request per concurrency slot is enough to measure steady state;
            # the /predict default stops after a single total request instead.
            requests_per_load_unit=_demand_requests_per_load_unit(),
            database_mode=req.database_mode,
            gpu_memory_utilization=_backend_memory_fraction(req.backend, backend_version),
            simulator="aisimulate",
        )
    try:
        return _run_aisimulate_prediction(prediction_req, set()), None
    except Exception as error:  # capacity does not gate on timing data
        logger.warning(
            "Capacity performance simulation unavailable for %s on %s: %s",
            req.model_path,
            req.system,
            error,
        )
        reason = str(error) or error.__class__.__name__
        return None, reason[:500]


def _capacity_config(
    req: RecommendRequest,
    *,
    backend_version: str,
    candidate: _CapacityCandidate,
    target_concurrency: int,
    prediction: Any | None,
) -> RecommendConfig:
    summary = prediction.summary if prediction is not None else {}
    replicas = candidate.replicas
    total_gpus = candidate.total_gpus
    worker_throughput = _metric(summary, "output_throughput_tok_s", "tokens_per_second")
    cluster_throughput = worker_throughput * replicas if worker_throughput is not None else None
    worker_power = _metric(summary, "power_w", "mean_power_w")
    gemm = summary.get("gemm") or summary.get("gemm_quant_mode")
    kvcache = summary.get("kvcache") or summary.get("kvcache_quant_mode")
    return RecommendConfig(
        total_gpus_needed=total_gpus,
        replicas_needed=replicas,
        # One replica's GPU count is the scalable unit; total_gpus_needed is
        # the cluster total. The two stay distinct so a cost calculator does
        # not multiply the deployment a second time.
        num_total_gpus=candidate.shape.gpus_per_worker,
        tp=candidate.shape.tp,
        pp=1,
        dp=candidate.shape.dp,
        cp=1,
        moe_tp=candidate.shape.moe_tp,
        moe_ep=candidate.shape.moe_ep,
        bs=target_concurrency,
        ttft=_metric(summary, "ttft_ms", "mean_ttft_ms"),
        tpot=_metric(summary, "tpot_ms", "mean_tpot_ms", "itl_ms"),
        request_latency=_metric(summary, "e2e_latency_ms", "mean_e2e_latency_ms"),
        concurrency=target_concurrency,
        tokens_per_second=cluster_throughput,
        tokens_per_second_per_gpu=(
            cluster_throughput / total_gpus
            if cluster_throughput is not None and total_gpus > 0
            else None
        ),
        tokens_per_second_per_user=(
            cluster_throughput / target_concurrency
            if cluster_throughput is not None and target_concurrency > 0
            else None
        ),
        memory=_metric(summary, "memory_gb", "memory", "peak_memory_gb"),
        power_w=worker_power * replicas if worker_power is not None else None,
        model=req.model_path,
        system=req.system,
        backend=req.backend,
        backend_version=backend_version,
        gemm=str(gemm) if gemm is not None else None,
        kvcache=str(kvcache) if kvcache is not None else None,
        mode="agg",
        simulator="aisimulate",
    )


def _capacity_disagg_config(
    req: RecommendRequest,
    *,
    backend_version: str,
    candidate: _DisaggCapacityCandidate,
    target_concurrency: int,
    prediction: Any | None,
) -> RecommendConfig:
    summary = prediction.summary if prediction is not None else {}
    total_gpus = candidate.total_gpus
    throughput = _metric(summary, "output_throughput_tok_s", "tokens_per_second")
    power = _metric(summary, "power_w", "mean_power_w")
    gemm = summary.get("gemm") or summary.get("gemm_quant_mode")
    kvcache = summary.get("kvcache") or summary.get("kvcache_quant_mode")
    prefill_batch = max(1, math.ceil(target_concurrency / candidate.prefill.replicas))
    decode_batch = max(1, math.ceil(target_concurrency / candidate.decode.replicas))
    return RecommendConfig(
        total_gpus_needed=total_gpus,
        # A P/D deployment is one scalable unit; its prefill and decode pools
        # are sized independently and are not replicated as a single block.
        replicas_needed=1,
        num_total_gpus=total_gpus,
        bs=target_concurrency,
        ttft=_metric(summary, "ttft_ms", "mean_ttft_ms"),
        tpot=_metric(summary, "tpot_ms", "mean_tpot_ms", "itl_ms"),
        request_latency=_metric(summary, "e2e_latency_ms", "mean_e2e_latency_ms"),
        concurrency=target_concurrency,
        tokens_per_second=throughput,
        tokens_per_second_per_gpu=(
            throughput / total_gpus if throughput is not None and total_gpus > 0 else None
        ),
        tokens_per_second_per_user=(
            throughput / target_concurrency
            if throughput is not None and target_concurrency > 0
            else None
        ),
        memory=_metric(summary, "memory_gb", "memory", "peak_memory_gb"),
        power_w=power,
        model=req.model_path,
        system=req.system,
        backend=req.backend,
        backend_version=backend_version,
        gemm=str(gemm) if gemm is not None else None,
        kvcache=str(kvcache) if kvcache is not None else None,
        mode="disagg",
        prefill_config=WorkerConfig(
            tp=candidate.prefill.shape.tp,
            pp=1,
            dp=candidate.prefill.shape.dp,
            cp=1,
            moe_tp=candidate.prefill.shape.moe_tp,
            moe_ep=candidate.prefill.shape.moe_ep,
            num_workers=candidate.prefill.replicas,
            batch_size=prefill_batch,
            backend_version=backend_version,
        ),
        decode_config=WorkerConfig(
            tp=candidate.decode.shape.tp,
            pp=1,
            dp=candidate.decode.shape.dp,
            cp=1,
            moe_tp=candidate.decode.shape.moe_tp,
            moe_ep=candidate.decode.shape.moe_ep,
            num_workers=candidate.decode.replicas,
            batch_size=decode_batch,
            backend_version=backend_version,
        ),
        simulator="aisimulate",
    )


def _capacity_option_config(
    req: RecommendRequest,
    *,
    backend_version: str,
    candidate: _CapacityCandidate | _DisaggCapacityCandidate,
    target_concurrency: int,
    prediction: Any | None,
) -> RecommendConfig:
    if isinstance(candidate, _DisaggCapacityCandidate):
        config = _capacity_disagg_config(
            req,
            backend_version=backend_version,
            candidate=candidate,
            target_concurrency=target_concurrency,
            prediction=prediction,
        )
    else:
        config = _capacity_config(
            req,
            backend_version=backend_version,
            candidate=candidate,
            target_concurrency=target_concurrency,
            prediction=prediction,
        )
    if prediction is not None:
        config.warnings.extend(str(w) for w in getattr(prediction, "warnings", []) or [])
    if req.inclusive_tpot:
        config.tpot = _inclusive_tpot(config.ttft, config.tpot, req.osl)
    return config


def _capacity_option_outcome(
    *,
    candidate: _CapacityCandidate | _DisaggCapacityCandidate,
    context_tokens: int,
    target_concurrency: int,
    minimum_gpus: int | None,
    evaluated_shapes: int,
    feasible_shapes: int,
    prediction: Any | None,
    unavailable_reason: str | None,
) -> CapacityOutcome:
    needed_tokens = target_concurrency * context_tokens
    fits = candidate.capacity_tokens >= needed_tokens
    common = {
        "target_concurrency": target_concurrency,
        "context_per_sequence_tokens": context_tokens,
        "capacity_tokens": candidate.capacity_tokens,
        "capacity_sequences": candidate.capacity_sequences,
        "fits": fits,
        "shortfall_sequences": (
            None if fits else max(0, target_concurrency - candidate.capacity_sequences)
        ),
        "shortfall_tokens": (
            None if fits else max(0, needed_tokens - candidate.capacity_tokens)
        ),
        "minimum_gpus": minimum_gpus,
        "evaluated_shapes": evaluated_shapes,
        "feasible_shapes": feasible_shapes,
        "exhaustive": True,
        "performance_available": prediction is not None,
        "performance_unavailable_reason": unavailable_reason,
        "simulator": "aisimulate",
    }
    if isinstance(candidate, _DisaggCapacityCandidate):
        return CapacityOutcome(
            **common,
            mode="disagg",
            # The decode pool holds resident sequences; the prefill pool only
            # holds one sequence transiently.
            per_rank_tokens=candidate.decode.shape.per_rank_tokens,
            replicas=candidate.decode.replicas,
            attention_dp=candidate.decode.shape.dp,
            prefill_replicas=candidate.prefill.replicas,
            decode_replicas=candidate.decode.replicas,
            prefill_gpus=candidate.prefill.total_gpus,
            decode_gpus=candidate.decode.total_gpus,
        )
    return CapacityOutcome(
        **common,
        per_rank_tokens=candidate.shape.per_rank_tokens,
        replicas=candidate.replicas,
        attention_dp=candidate.shape.dp,
    )


def _attach_capacity_config_extras(
    req: RecommendRequest,
    model_path: str,
    *,
    config: RecommendConfig,
    candidate: _CapacityCandidate | _DisaggCapacityCandidate,
    backend_version: str,
    target_concurrency: int,
    include: set[str],
) -> None:
    """Add optional serving-config and memory detail to one capacity option."""
    if isinstance(candidate, _DisaggCapacityCandidate):
        if "memory" in include:
            for worker, context_length in (
                (config.prefill_config, req.prefill_max_seq_len or req.max_seq_len),
                (config.decode_config, req.decode_max_seq_len or req.max_seq_len),
            ):
                if worker and worker.tp:
                    worker.memory_breakdown = _build_memory_breakdown(
                        model_path,
                        req.system,
                        req.backend,
                        backend_version,
                        worker.tp,
                        worker.pp or 1,
                        req.isl,
                        req.osl,
                        worker.batch_size or 1,
                        moe_tp=worker.moe_tp,
                        moe_ep=worker.moe_ep,
                        max_seq_len=context_length,
                        memory_fraction=_backend_memory_fraction(req.backend, backend_version),
                        attention_dp=worker.dp or 1,
                    )
            if config.memory is None:
                config.memory = max(
                    (
                        memory_gb
                        for worker in (config.prefill_config, config.decode_config)
                        if worker is not None
                        for memory_gb in [_memory_breakdown_gb(worker.memory_breakdown)]
                        if memory_gb is not None
                    ),
                    default=None,
                )
        return

    if "config" in include:
        config.serving_config = _build_serving_config(
            req.backend,
            candidate.shape.tp,
            req.isl,
            req.osl,
            target_concurrency,
            config.gemm,
            req.prefix,
            max_seq_len=req.max_seq_len,
            backend_version=backend_version,
            max_num_seqs=target_concurrency,
        )
    if "memory" in include:
        config.memory_breakdown = _build_memory_breakdown(
            model_path,
            req.system,
            req.backend,
            backend_version,
            candidate.shape.tp,
            1,
            req.isl,
            req.osl,
            target_concurrency,
            moe_tp=candidate.shape.moe_tp,
            moe_ep=candidate.shape.moe_ep,
            max_seq_len=req.max_seq_len,
            attention_dp=candidate.shape.dp,
        )
        if config.memory is None:
            config.memory = _memory_breakdown_gb(config.memory_breakdown)


def _simulate_capacity_option_predictions(
    req: RecommendRequest,
    model_path: str,
    *,
    backend_version: str,
    candidates: list[_CapacityCandidate | _DisaggCapacityCandidate],
    target_concurrency: int,
) -> dict[int, tuple[Any | None, str | None]]:
    """Simulate options in priority order under one shared wall-clock budget.

    Capacity is already decided, so missing a simulation is not an error. The
    primary option is submitted first, then the P/D representatives, then the
    remaining alternatives. At most two simulations run concurrently.
    """
    if not candidates:
        return {}
    deadline = time.monotonic() + _CAPACITY_OPTION_SIMULATION_BUDGET_S
    results: dict[int, tuple[Any | None, str | None]] = {}
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=_CAPACITY_OPTION_SIMULATION_WORKERS
    )
    pending: dict[concurrent.futures.Future[tuple[Any | None, str | None]], int] = {}
    next_index = 0
    try:
        while next_index < len(candidates) or pending:
            while next_index < len(candidates) and len(pending) < _CAPACITY_OPTION_SIMULATION_WORKERS:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # Hold a process-wide slot until this simulation actually
                # finishes, even if the response budget expires first. New
                # requests cannot start work the abandoned simulation is still
                # consuming CPU for.
                if not _CAPACITY_SIMULATION_SLOTS.acquire(timeout=remaining):
                    break
                index = next_index
                next_index += 1
                try:
                    future = executor.submit(
                        _capacity_prediction,
                        req,
                        model_path,
                        backend_version=backend_version,
                        candidate=candidates[index],
                        target_concurrency=target_concurrency,
                    )
                except BaseException:
                    _CAPACITY_SIMULATION_SLOTS.release()
                    raise
                future.add_done_callback(
                    lambda _future: _CAPACITY_SIMULATION_SLOTS.release()
                )
                pending[future] = index
            if not pending:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            done, _ = concurrent.futures.wait(
                pending,
                timeout=remaining,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                index = pending.pop(future)
                try:
                    results[index] = future.result()
                except Exception as error:  # a failed option must not fail the run
                    reason = str(error) or error.__class__.__name__
                    results[index] = (None, reason[:500])
    finally:
        # Do not block the response on a simulation that outlived the budget.
        executor.shutdown(wait=False, cancel_futures=True)
    for index in range(len(candidates)):
        results.setdefault(
            index,
            (None, "The shared performance-simulation budget ended before this option ran."),
        )
    return results


def _run_capacity_recommendation(
    req: RecommendRequest,
    *,
    include: set[str],
) -> tuple[RecommendConfig, CapacityOutcome, SweepMeta, list[CapacityOption]]:
    """Size a deployment for ``target_concurrency`` sequences and rank options.

    Selection is a deterministic enumeration of the SDK's search space
    (``algorithm="enumerate"``), not a sampled sweep: an SLA-free residency
    question has no sweep objective, and sampling could miss the true minimum.
    ``configs[0]`` remains the primary recommendation; the comparison set is
    returned separately as ``capacity_options``.
    """
    from aisimulate.sweeper.kv_estimate import resolve_backend_version

    backend = _sdk_backend_id(req.backend)
    target_concurrency = int(req.target_concurrency or 0)
    context_tokens = _capacity_context_tokens(req)
    lower_gpus, upper_gpus = _effective_gpu_window(req)
    explicit_top_n = "top_n" in getattr(req, "model_fields_set", set())
    option_count = int(req.top_n) if explicit_top_n else _CAPACITY_OPTION_DEFAULT_COUNT

    with _with_model_config(req.model_path, req.model_config_data) as model_path:
        backend_version = req.backend_version or resolve_backend_version(req.system, backend)
        shapes = _capacity_shapes(
            req,
            model_path,
            backend=backend,
            backend_version=backend_version,
            context_tokens=context_tokens,
            gpu_budget=upper_gpus,
        )
        if not shapes:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{req.model_path} on {req.system}: no aggregated parallel configuration "
                    f"holds a {context_tokens}-token sequence within "
                    f"{_gpu_window_label(lower_gpus, upper_gpus)} GPUs "
                    f"({req.backend} KV-cache estimate)."
                ),
            )
        aggregate_fitting, aggregate_best_effort, minimum_gpus = _capacity_aggregate_candidates(
            shapes,
            target_concurrency=target_concurrency,
            context_tokens=context_tokens,
            lower_gpus=lower_gpus,
            upper_gpus=upper_gpus,
        )
        if minimum_gpus is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{req.model_path} on {req.system}: no aggregated parallel configuration "
                    f"holds even one {context_tokens}-token sequence within "
                    f"{_gpu_window_label(lower_gpus, upper_gpus)} GPUs "
                    f"({req.backend} KV-cache estimate)."
                ),
            )

        disagg_candidates = _capacity_disagg_candidates(
            req,
            model_path,
            backend=backend,
            backend_version=backend_version,
            context_tokens=context_tokens,
            target_concurrency=target_concurrency,
            lower_gpus=lower_gpus,
            upper_gpus=upper_gpus,
            minimum_gpus=minimum_gpus,
        )
        if not explicit_top_n and not disagg_candidates:
            option_count = min(option_count, _CAPACITY_OPTION_DEFAULT_COUNT_NO_DISAGG)
        selected = _select_capacity_options(
            aggregate_fitting=aggregate_fitting,
            aggregate_best_effort=aggregate_best_effort,
            disagg_candidates=disagg_candidates,
            option_count=option_count,
        )
        if not selected:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{req.model_path} on {req.system}: no capacity deployment fits within "
                    f"{_gpu_window_label(lower_gpus, upper_gpus)} GPUs "
                    f"({req.backend} KV-cache estimate)."
                ),
            )

        # Simulate the primary first, then the P/D representatives, then the
        # remaining alternatives, under one shared budget. The response keeps
        # the comparison ranking; only simulation priority changes.
        priority = (
            [0]
            + [index for index, (role, _candidate) in enumerate(selected) if role == "disaggregated"]
            + [
                index
                for index, (role, _candidate) in enumerate(selected)
                if index != 0 and role != "disaggregated"
            ]
        )
        priority_results = _simulate_capacity_option_predictions(
            req,
            model_path,
            backend_version=backend_version,
            candidates=[selected[index][1] for index in priority],
            target_concurrency=target_concurrency,
        )
        predictions = {
            selected_index: priority_results[position]
            for position, selected_index in enumerate(priority)
        }

        options: list[CapacityOption] = []
        used_ids: set[str] = set()
        # Rank 1 is the fewest-GPU option that holds the target, which can need
        # more GPUs than the fewest that hold a single sequence. Measure "over
        # minimum" from rank 1 when the target fits, so the primary reports 0.
        # A shortfall still measures from the single-sequence minimum.
        baseline_gpus = selected[0][1].total_gpus if aggregate_fitting else minimum_gpus
        for index, (role, candidate) in enumerate(selected):
            prediction, unavailable_reason = predictions[index]
            config = _capacity_option_config(
                req,
                backend_version=backend_version,
                candidate=candidate,
                target_concurrency=target_concurrency,
                prediction=prediction,
            )
            if unavailable_reason is not None:
                config.warnings.append(
                    "Performance simulation was unavailable for this configuration; "
                    "the capacity result does not depend on it."
                )
            _attach_capacity_config_extras(
                req,
                model_path,
                config=config,
                candidate=candidate,
                backend_version=backend_version,
                target_concurrency=target_concurrency,
                include=include,
            )
            outcome = _capacity_option_outcome(
                candidate=candidate,
                context_tokens=context_tokens,
                target_concurrency=target_concurrency,
                minimum_gpus=minimum_gpus,
                evaluated_shapes=len(shapes),
                feasible_shapes=len(shapes),
                prediction=prediction,
                unavailable_reason=unavailable_reason,
            )
            options.append(
                CapacityOption(
                    option_id=_capacity_option_id(candidate, used=used_ids),
                    rank=index + 1,
                    role=role,  # type: ignore[arg-type]
                    gpus_over_minimum=max(0, candidate.total_gpus - baseline_gpus),
                    config=config,
                    capacity=outcome,
                )
            )

        primary = options[0]
        sweep = SweepMeta(
            algorithm="enumerate",
            # Every enumerated shape was evaluated by the KV estimate; the
            # budget cannot be "exhausted" because the search is exhaustive.
            max_trials=len(shapes),
            evaluated=len(shapes),
            feasible=len(shapes),
            timed_out=0,
            budget_exhausted=False,
            strict_sla=False,
            window_min_gpus=lower_gpus,
            window_max_gpus=upper_gpus,
        )
        return primary.config, primary.capacity, sweep, options


def _run_aisimulate_recommendation(
    req: RecommendRequest,
    *,
    max_trials_override: int | None = None,
    candidate_timeout_override: int | None = None,
):
    from aisimulate.recommend import run_recommendation

    with _with_model_config(req.model_path, req.model_config_data) as effective_path:
        return run_recommendation(
            _aisimulate_recommendation_config(
                req,
                model_path=effective_path,
                max_trials_override=max_trials_override,
                candidate_timeout_override=candidate_timeout_override,
            ),
            stack="engine",
            runner_factory=_aisimulate_runner_factory(),
            show_progress=False,
        )


def _run_pareto_recommendation(req: RecommendRequest, *, auto_window: bool):
    """Run a Pareto sweep, resolving an omitted GPU window from the model's fit.

    Resolution happens inside the version-fallback closure so the preflight
    estimates KV capacity with the same backend version the sweep will use.
    An explicit bound, on either side, keeps the caller's current semantics.
    """
    if auto_window:
        lower, upper = _pareto_auto_window(
            req,
            backend=_sdk_backend_id(req.backend),
            backend_version=req.backend_version,
        )
        req.min_candidate_gpus = lower
        req.max_candidate_gpus = upper
    return _run_aisimulate_recommendation(req)


def _run_aisimulate_prediction(req: EstimateRequest, include: set[str]):
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config.cli import CorePredictionConfig
    from aisimulate.sweeper.replay import ReplayOutputRequirements

    with _with_model_config(req.model_path, req.model_config_data) as effective_path:
        prediction_req = req.model_copy(update={"model_path": effective_path})
        config = CorePredictionConfig.model_validate(
            _aisimulate_prediction_config(prediction_req)
        )
        spec = prediction_to_replay_spec(config)
        runner = _aisimulate_runner_factory().create(worker_id=0)
        report = runner.run(
            spec,
            output_requirements=ReplayOutputRequirements(
                include_raw_report=True,
                capture_memory_diagnostics="memory" in include,
            ),
        )
        return _PredictionResult(dict(report.metrics))


#: One cached fallback per request shape, so a blank-version request pays the
#: failed attempt once instead of on every call. The shape is part of the key:
#: a version can cover one shape and miss another, and a cached fallback must
#: not skip a default that works for a different request. Least-recently-used
#: entries are evicted so sustained traffic cannot grow it without bound.
#: Synchronous handlers run in a threadpool, so cache access is locked.
_VERSION_FALLBACK_CACHE_LOCK = threading.Lock()
_VERSION_FALLBACK_CACHE: OrderedDict[tuple[str, str, str, str, str], str] = OrderedDict()
_VERSION_FALLBACK_CACHE_MAX = 512

#: Bound the extra replay work: the system default plus the two newest versions.
_VERSION_FALLBACK_ATTEMPTS = 3

#: The SDK slot alias that means "the system default" (see
#: ``resolve_query_version``). ``previous`` and ``next`` name specific slots.
_DEFAULT_VERSION_ALIAS = "current"


def _clear_version_fallback_cache() -> None:
    """Drop the resolved-version cache (used by tests and after data reloads)."""
    with _VERSION_FALLBACK_CACHE_LOCK:
        _VERSION_FALLBACK_CACHE.clear()


def _fallback_cache_key(req: Any, default: str | None) -> tuple[str, str, str, str, str]:
    """Model, system, backend, resolved default, and the request shape.

    Coverage gaps depend on the compute shape, so keying on the model alone
    would let a fallback for one workload skip the default for another. The
    resolved default is part of the key too: when the SDK moves its ``current``
    slot, a cached fallback must not bypass the new default. The shape is
    hashed so a large ``model_config`` does not stay resident.
    """
    shape = req.model_dump(mode="json", exclude={"backend_version"})
    digest = hashlib.sha256(
        json.dumps(shape, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
    return (req.model_path, req.system, _sdk_backend_id(req.backend), default or "", digest)


def _fallback_cache_get(cache_key: tuple[str, str, str, str, str]) -> str | None:
    """Read a cached fallback and mark it most recently used."""
    with _VERSION_FALLBACK_CACHE_LOCK:
        cached = _VERSION_FALLBACK_CACHE.get(cache_key)
        if cached is not None:
            _VERSION_FALLBACK_CACHE.move_to_end(cache_key)
        return cached


def _fallback_cache_put(cache_key: tuple[str, str, str, str, str], version: str) -> bool:
    """Store a resolved fallback, evicting least-recently-used entries.

    Returns whether this was a new entry, so the caller logs each model once
    rather than on every request that hits the cache.
    """
    with _VERSION_FALLBACK_CACHE_LOCK:
        is_new = cache_key not in _VERSION_FALLBACK_CACHE
        _VERSION_FALLBACK_CACHE[cache_key] = version
        _VERSION_FALLBACK_CACHE.move_to_end(cache_key)
        while len(_VERSION_FALLBACK_CACHE) > _VERSION_FALLBACK_CACHE_MAX:
            _VERSION_FALLBACK_CACHE.popitem(last=False)
        return is_new


def _run_with_version_fallback(req: Any, run: Callable[[], Any]) -> tuple[Any, str | None]:
    """Resolve an omitted ``backend_version`` to a version that can serve the model.

    A blank version (or the ``current`` alias, which means the same thing) is
    resolved to a literal version. The system default is right for most models,
    but a model can be missing from that version's op-level data while a newer
    version covers it (dense Gemma-4 31B is absent on vLLM 0.24.0 and present on
    0.31.0). Retry only on a performance-data gap, and never for a version the
    caller pinned literally: ``previous`` and ``next`` name specific slots.

    Returns ``(result, version_the_default_could_not_serve)``.
    """
    requested = (req.backend_version or "").strip() or None
    # ``current`` is an alias for the system default, not a pin on a literal
    # version; ``previous`` and ``next`` name specific slots and are honoured.
    open_version = requested is None or requested == _DEFAULT_VERSION_ALIAS
    if not open_version:
        try:
            return run(), None
        except Exception as error:
            if _is_perf_data_gap(error):
                raise _PerfDataGap(
                    model=req.model_path,
                    system=req.system,
                    backend=req.backend,
                    requested_version=requested,
                    attempted=[requested],
                    cause=error,
                ) from error
            raise

    declared = _declared_versions(req.system, req.backend)
    default = _system_default_version(req.system, req.backend)
    cache_key = _fallback_cache_key(req, default)
    # Ignore a cached fallback that a data update no longer declares: replaying
    # it would fail as an unlisted version instead of finding coverage again.
    cached = _fallback_cache_get(cache_key)
    first = cached if cached in declared else default
    ordered = [first] + [version for version in reversed(declared) if version != first]

    attempted: list[str | None] = []
    cause: BaseException | None = None
    for version in ordered[:_VERSION_FALLBACK_ATTEMPTS]:
        req.backend_version = version
        attempted.append(version)
        try:
            result = run()
        except Exception as error:
            if not _is_perf_data_gap(error):
                raise
            cause = error
            continue
        if version and default and version != default:
            if _fallback_cache_put(cache_key, version):
                logger.warning(
                    "No performance data for %s with %s %s on %s; resolved to %s",
                    req.model_path, req.backend, default, req.system, version,
                )
            return result, default
        return result, None

    raise _PerfDataGap(
        model=req.model_path,
        system=req.system,
        backend=req.backend,
        requested_version=None,
        attempted=attempted,
        cause=cause or RuntimeError("no performance data for this model"),
    )


def _worker_config_from_row(row: pd.Series, prefix: str, req: RecommendRequest) -> WorkerConfig | None:
    def g(col: str) -> Any:
        v = row.get(f"({prefix}){col}")
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

    tp = _coerce_int(g("tp"))
    if tp is None:
        return None
    return WorkerConfig(
        tp=tp,
        pp=_coerce_int(g("pp")),
        dp=_coerce_int(g("dp")),
        cp=_coerce_int(g("cp")),
        moe_tp=_coerce_int(g("moe_tp")),
        moe_ep=_coerce_int(g("moe_ep")),
        num_workers=_coerce_int(g("workers")),
        batch_size=_coerce_int(g("bs")),
        memory_gb=_coerce_float(g("memory")),
        gemm=g("gemm"),
        kvcache=g("kvcache"),
        fmha=g("fmha"),
        moe=g("moe"),
        comm=g("comm"),
        backend_version=row.get(f"({prefix})version") or req.backend_version,
    )


def _row_to_config(row: pd.Series, req: RecommendRequest) -> RecommendConfig:
    is_disagg = "(p)tp" in row.index

    d: dict[str, Any] = {}
    for col, val in row.items():
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        col_str = str(col)
        if col_str.startswith(("(p)", "(d)", "(e)")):
            continue
        key = _COLUMN_MAP.get(col_str, col_str)
        if isinstance(val, float) and val == int(val) and key in _INT_FIELDS:
            val = int(val)
        d[key] = val

    d.setdefault("system", req.system)
    d.setdefault("backend", req.backend)
    d.setdefault("backend_version", req.backend_version)

    cfg = RecommendConfig.model_validate(d)

    if is_disagg:
        cfg.prefill_config = _worker_config_from_row(row, "p", req)
        cfg.decode_config = _worker_config_from_row(row, "d", req)
        if cfg.memory is None:
            # (p)/(d)/(e)memory are each per-GPU peak usage (GB) for that worker
            # type — each is checked against a single GPU's capacity, so they are
            # NOT additive. The meaningful single figure is the worst-case
            # per-GPU across all pools (prefill, decode, and encode).
            phase_mems = [
                m for m in (
                    _coerce_float(row.get("(p)memory")),
                    _coerce_float(row.get("(d)memory")),
                    _coerce_float(row.get("(e)memory")),
                )
                if m is not None
            ]
            if phase_mems:
                cfg.memory = max(phase_mems)

    return cfg


def _build_serving_config(
    backend: str,
    tp: int,
    isl: int,
    osl: int,
    concurrency: int,
    gemm: str | None,
    prefix: int = 0,
    max_seq_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    backend_version: str | None = None,
    max_num_seqs: int | None = None,
    enable_chunked_prefill: bool | None = None,
    prefix_caching: bool | None = None,
) -> ServingConfig:
    quant_map = {"fp8": "fp8", "fp8_block": "fp8", "int8": "int8"}
    quantization = quant_map.get(gemm or "", "auto")
    return ServingConfig(
        backend=backend,
        tensor_parallel_size=tp,
        max_model_len=max_seq_len or isl + osl,
        max_num_seqs=max_num_seqs if max_num_seqs is not None else min(concurrency, 256),
        gpu_memory_utilization=gpu_memory_utilization or _backend_memory_fraction(backend, backend_version),
        enable_chunked_prefill=enable_chunked_prefill if enable_chunked_prefill is not None else isl >= 4096 or concurrency >= 64,
        enable_prefix_caching=prefix_caching if prefix_caching is not None else prefix > 0,
        quantization=quantization,
        memory_fraction=gpu_memory_utilization or _backend_memory_fraction(backend, backend_version),
        memory_fraction_kind=_backend_memory_fraction_kind(backend),
        runtime_memory_field=_backend_runtime_memory_field(backend),
    )


def _build_memory_breakdown(
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None,
    tp: int,
    pp: int,
    isl: int,
    osl: int,
    concurrency: int,
    gemm_quant: str | None = None,
    kvcache_quant: str | None = None,
    moe_tp: int | None = None,
    moe_ep: int | None = None,
    max_seq_len: int | None = None,
    memory_fraction: float | None = None,
    attention_dp: int = 1,
) -> MemoryBreakdown | None:
    try:
        raw = estimate_kv_cache(
            model_path=model_path,
            system=system,
            backend=_sdk_backend_id(backend),
            backend_version=backend_version,
            max_num_tokens=max_seq_len or isl + osl,
            max_batch_size=concurrency,
            memory_fraction_kind=_backend_memory_fraction_kind(backend),
            memory_fraction_value=memory_fraction or _backend_memory_fraction(backend, backend_version),
            tp_size=tp,
            pp_size=pp,
            attention_dp_size=attention_dp,
            moe_tp_size=moe_tp,
            moe_ep_size=moe_ep,
            gemm_quant_mode=gemm_quant if gemm_quant and gemm_quant != "half" else None,
            kvcache_quant_mode=kvcache_quant if kvcache_quant and kvcache_quant != "half" else None,
        )
    except Exception:
        logger.debug("memory breakdown unavailable for %s on %s", model_path, system)
        return None

    breakdown = raw.get("memory_breakdown") or {}
    return MemoryBreakdown(
        weights_bytes=int(breakdown.get("weights_bytes", 0)),
        activations_bytes=int(breakdown.get("activations_bytes", 0)),
        runtime_overhead_bytes=int(breakdown.get("runtime_overhead_bytes", 0)),
        comm_overhead_bytes=int(breakdown.get("comm_overhead_bytes", 0)),
        kv_cache_bytes=int(raw.get("total_kv_size_bytes", 0)),
    )


def _memory_breakdown_gb(breakdown: MemoryBreakdown | None) -> float | None:
    if breakdown is None:
        return None
    total_bytes = sum((
        breakdown.weights_bytes,
        breakdown.activations_bytes,
        breakdown.runtime_overhead_bytes,
        breakdown.comm_overhead_bytes,
        breakdown.kv_cache_bytes,
    ))
    return total_bytes / (1024 ** 3)


#: Version rejections the engine runner wraps as RuntimeError("ValueError: ..."),
#: so they arrive by text rather than by exception type.
_UNSERVABLE_MESSAGE = re.compile(
    r"is not a queryable version|no queryable versions defined|no timing data for"
)

#: A replay that reaches a compute shape the selected performance data never
#: measured. The runner wraps these in "AISimulate replay failed: engine error:
#: ...", so they are matched by text (and by exception type) rather than class.
#: This is a coverage gap for this model/version, not a service failure: a 500
#: told users the API was down when another backend_version would have worked.
_PERF_DATA_GAP_MESSAGE = re.compile(
    r"EmpiricalNotImplementedError"
    r"|empirical estimation not implemented"
    r"|No empirical utilisation data"
    r"|no own-shape, cross-shape, or sibling transfer reference"
    r"|perf database error: (?:data missing|data empty|data unavailable|no rows in"
    r"|has no compatible data|has data for|no data to anchor)",
    re.IGNORECASE,
)

_PERF_DATA_GAP_TYPES = frozenset({"EmpiricalNotImplementedError", "PerfDataNotAvailableError"})

#: The op query the estimator could not price, e.g. ``[32.0, 2048.0, 1.0]``.
_SHAPE_QUERY = re.compile(r"at query=\[([^\]]*)\]")


def _exception_chain_text(error: BaseException) -> str:
    """Every message in an exception's cause chain, innermost text included."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(parts) < 6:
        seen.add(id(current))
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return " | ".join(parts)


def _is_perf_data_gap(error: BaseException) -> bool:
    """Whether an error means "this data set has no measurement", not "we broke"."""
    if error.__class__.__name__ in _PERF_DATA_GAP_TYPES:
        return True
    return bool(_PERF_DATA_GAP_MESSAGE.search(_exception_chain_text(error)))


def _perf_gap_shape(error: BaseException) -> str:
    """Name the missing coverage in the estimator's own terms, without the noise."""
    text = _exception_chain_text(error)
    match = _SHAPE_QUERY.search(text)
    if match is not None:
        shape = ", ".join(
            part.strip().removesuffix(".0") or part.strip()
            for part in match.group(1).split(",")
        )
        return f"no measurement for compute shape [{shape}]"
    if "no own-shape, cross-shape, or sibling transfer reference" in text:
        return "no measurement, and no transferable reference shape"
    return "no measurement for a compute shape this model needs"


class _PerfDataGap(Exception):
    """No empirical performance data for this model on the selected backend."""

    def __init__(
        self,
        *,
        model: str,
        system: str,
        backend: str,
        requested_version: str | None,
        attempted: list[str | None],
        cause: BaseException,
    ) -> None:
        self.model = model
        self.system = system
        self.backend = backend
        self.requested_version = requested_version
        self.attempted = attempted
        self.cause = cause
        super().__init__(self.detail())

    def detail(self) -> str:
        shape = _perf_gap_shape(self.cause)
        declared = _declared_versions(self.system, self.backend)
        listed = ", ".join(declared) if declared else "none"
        if self.requested_version:
            return (
                f"No empirical performance data for model={self.model} with "
                f"{self.backend} {self.requested_version} on {self.system}: {shape}. "
                f"Versions the SDK ships for {self.system}: {listed}. Retry with one of "
                "those, or omit backend_version to let the service choose."
            )
        tried = ", ".join(version or "the system default" for version in self.attempted)
        return (
            f"No empirical performance data for model={self.model} with {self.backend} on "
            f"{self.system}: {shape}. Tried {tried or 'the system default'}. The estimator "
            "has no coverage for this model on this system; try another backend or GPU system."
        )


def _unwrapped_detail(msg: str) -> str:
    return re.sub(r"^[A-Za-z_][A-Za-z0-9_]*Error: ", "", msg, count=1)


def _common_error_handler(e: Exception, op: str, model_path: str, backend: str, system: str) -> None:
    # A handler that already chose a status and detail (for example the
    # capacity enumeration's no-fit 422) must keep it; re-wrapping it as a 500
    # would report a client-actionable answer as a server failure.
    if isinstance(e, HTTPException):
        raise e
    msg = str(e)
    # A pinned version that the performance data cannot serve, or a blank
    # version that no declared version can serve. Both are client-actionable.
    if isinstance(e, _PerfDataGap):
        raise HTTPException(status_code=422, detail=e.detail())
    # aisimulate's recommendation stack uses NoViableParallelConfig when a
    # bounded GPU window is too small to hold the model. That is an expected
    # no-result response (the caller may try a larger window), not a server
    # failure. Avoid a hard import of this optimizer-internal exception so the
    # wrapper remains compatible across SDK releases.
    if isinstance(e, NoFeasibleConfigError) or e.__class__.__name__ == "NoViableParallelConfig":
        raise HTTPException(status_code=422, detail=msg)
    # Some SDK versions wrap NoViableParallelConfig while crossing the
    # supervised-process boundary. It still means this GPU window is
    # infeasible and must not terminate an incremental search.
    if "NoViableParallelConfig" in msg or "no deployment_mode has a viable parallel config" in msg:
        raise HTTPException(status_code=422, detail=msg)
    # The engine runner reports the same two rejections as RuntimeError
    # ("ValueError: ..."), so they are matched by text as well as by type.
    if _UNSERVABLE_MESSAGE.search(msg):
        raise HTTPException(status_code=422, detail=_unwrapped_detail(msg))
    # Defence for call sites that do not run the version-fallback wrapper
    # (/memory, legacy callers). Predict and recommend raise _PerfDataGap,
    # which carries the versions already tried.
    if _is_perf_data_gap(e):
        declared = _declared_versions(system, backend)
        listed = (
            f" Versions the SDK ships for {system}: {', '.join(declared)}."
            if declared
            else ""
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"No empirical performance data for model={model_path} with {backend} "
                f"on {system}: {_perf_gap_shape(e)}.{listed} Try another backend_version, "
                "backend or GPU system."
            ),
        )
    if isinstance(e, KeyError):
        # A bare KeyError here almost always means a supplied model_config is
        # incomplete (e.g. missing 'architectures'); str(KeyError('x')) is "'x'".
        detail = (
            f"Invalid or incomplete model_config for model={model_path}: missing key {msg}. "
            "Omit model_config (or send null) to resolve the model from HuggingFace."
        )
        raise HTTPException(status_code=422, detail=detail)
    if isinstance(e, (ValueError, AttributeError)):
        if "system_spec" in msg or "NoneType" in msg or "unsupported model" in msg.lower():
            detail = f"No performance data available for model={model_path}, backend={backend}, system={system}."
            raise HTTPException(status_code=422, detail=detail)
        raise HTTPException(status_code=422, detail=msg)
    logger.exception("%s failed", op)
    raise HTTPException(status_code=500, detail=msg)


def _architecture_from_sm(sm_version: int) -> str:
    sm_arch = _SM_ARCHITECTURE.get(sm_version, f"sm_{sm_version}")
    if sm_arch == f"sm_{sm_version}":
        return "Other"
    return sm_arch


# Cache of system_id -> vendor device name, populated at startup from the
# aisimulate SDK (via configiq.systems). aicostings loads the same map from
# the same source so the two services never drift on GPU naming.
# Only systems with genuine perf-data display names appear here; systems without
# perf data are absent and get hidden from /systems (see get_systems). Startup
# fails outright if this ends up empty, so the map is never empty at request time.
_DEVICE_DISPLAY_NAMES: dict[str, str] = {}


def _parse_include(include: str | None) -> set[str]:
    if not include:
        return set()
    return {s.strip().lower() for s in include.split(",")}


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AISimulators API",
    description="GPU recommendation, performance estimation, and memory estimation for LLM inference.",
    version="1.0.0",
    default_response_class=ORJSONResponse,
)

# Initialize OpenTelemetry (tracing + metrics + HTTP middleware) if the optional
# extra is present.
if _OBS:
    observability.enable(app, service_name="aisimulators", service_version="1.0.0",
                         meter_name="aisimulators.api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _blis_backends() -> dict[str, Any]:
    """BLIS backend/version support, in the same shape as the AISimulate branch.

    BLIS versions are properties of the BLIS release, not of a GPU system, and BLIS
    takes no version-scoped memory fraction from the SDK. ``systems`` and
    ``system_defaults`` are therefore empty rather than filled with invented
    per-system rows; ``version_support`` is the extra field that lets a client
    disable a topology the selected release cannot model.
    ``disaggregation_requires_fabric`` tells a client whether a P/D topology
    needs an inter-node fabric it must collect from the user first.
    """
    return {
        "simulator": "blis",
        "blis_version": blis_release(),
        # The sizing goals this simulator can answer. A client gates its goal
        # selector on this list instead of on the simulator's name.
        "objectives": ["min_gpus", "throughput_demand"],
        "backends": [
            {
                "id": "vllm",
                "aisimulate_id": "vllm",
                "versions": list(MODELED_ENGINE_VERSIONS),
                # A literal, not AISimulate's `current` alias: BLIS refuses aliases.
                "default_version": DEFAULT_ENGINE_VERSION,
                "memory_fraction": None,
                "memory_fraction_kind": "of_total",
                "runtime_field": "gpu_memory_utilization",
                "systems": {},
                "system_defaults": {},
                # A P/D run transfers KV between pools, and BLIS prices that
                # transfer through a named fabric, so the request must choose one.
                "disaggregation_requires_fabric": True,
                # BLIS cannot run the capacity objective, so it has no
                # disaggregated capacity comparison options either.
                "capacity_options": {"disaggregated": False},
                "version_support": {
                    version: dict(support) for version, support in ENGINE_VERSION_SUPPORT.items()
                },
            }
        ],
    }


@app.get("/backends")
def get_backends(
    simulator: Literal["aisimulate", "blis"] = Query(
        default="aisimulate",
        description="Simulator whose backend and version support is reported.",
    ),
):
    """Return backend/version support and effective memory defaults."""
    if simulator == "blis":
        return _blis_backends()
    supported = get_supported_databases()
    backend_data: dict[str, dict[str, Any]] = {}
    for system, systems_backends in supported.items():
        for sdk_id, versions in systems_backends.items():
            public_id = _BACKEND_PUBLIC_IDS.get(sdk_id, sdk_id)
            entry = backend_data.setdefault(public_id, {
                "id": public_id,
                "aisimulate_id": sdk_id,
                "versions": set(),
                "systems": {},
            })
            entry["versions"].update(versions)
            entry["systems"][system] = sorted(versions)

    result = []
    for public_id, entry in sorted(backend_data.items()):
        backend = get_backend(entry["aisimulate_id"])
        versions = sorted(entry["versions"], key=_version_sort_key)
        system_defaults: dict[str, str | None] = {}
        gate_enabled = bool(entry["systems"])
        any_gate_enabled = False
        for system, system_versions in sorted(entry["systems"].items()):
            slots = get_version_slots(system, entry["aisimulate_id"])
            has_slots = bool(slots)
            gate_enabled = gate_enabled and has_slots
            any_gate_enabled = any_gate_enabled or has_slots
            system_defaults[system] = _current_or_newest(slots, system_versions)
        # Fleet-wide defaults describe the version most systems run, so a single
        # backend-level figure does not silently advertise a version that only
        # one override system declares.
        counts = Counter(version for version in system_defaults.values() if version)
        fleet_version = max(
            counts.items(), key=lambda item: (item[1], _version_sort_key(item[0]))
        )[0] if counts else None
        result.append({
            "id": public_id,
            "aisimulate_id": entry["aisimulate_id"],
            "versions": versions,
            # The alias resolves per system, so it is safe only when every
            # system has slots. Mixed backends use the fleet fallback; with no
            # gate anywhere, retain the newest declared version.
            "default_version": "current" if gate_enabled else (
                fleet_version if any_gate_enabled else (versions[-1] if versions else None)
            ),
            "memory_fraction": backend.get_default_free_gpu_memory_fraction(fleet_version),
            "memory_fraction_kind": "of_free" if backend.memory_fraction_of_free() else "of_total",
            "runtime_field": {
                "vllm": "gpu_memory_utilization",
                "sglang": "mem_fraction_static",
                "trtllm": "free_gpu_memory_fraction",
            }.get(entry["aisimulate_id"]),
            "systems": entry["systems"],
            "system_defaults": {
                system: {
                    "version": version,
                    "memory_fraction": backend.get_default_free_gpu_memory_fraction(version),
                }
                for system, version in system_defaults.items()
            },
            # AISimulate's P/D search takes no fabric; no extra input is needed.
            "disaggregation_requires_fabric": False,
            # Capacity runs can compare aggregated and disaggregated options.
            "capacity_options": {"disaggregated": True},
        })
    return {
        "simulator": "aisimulate",
        # The sizing goals this simulator can answer. A client gates its goal
        # selector on this list instead of on the simulator's name.
        "objectives": ["min_gpus", "throughput_demand", "pareto", "concurrency_capacity"],
        "backends": result,
    }


@app.on_event("startup")
def startup_event():
    """Load GPU display names from the aisimulate SDK at startup.

    Systems without a loaded display name (no perf benchmark data) are excluded
    from /systems responses (see get_systems). If no display names load at all,
    startup fails — the API refuses to serve without valid perf data. Logs
    coverage gaps for ops visibility.
    """
    global _DEVICE_DISPLAY_NAMES
    _DEVICE_DISPLAY_NAMES = load_device_names_from_perf_data()
    if not _DEVICE_DISPLAY_NAMES:
        raise RuntimeError(
            "No GPU display names loaded from perf data; refusing to start "
            "without valid performance data."
        )

    # Log which supported systems lack perf data (they are hidden from /systems).
    missing = supported_systems() - set(_DEVICE_DISPLAY_NAMES.keys())
    if missing:
        logger.warning(f"No display names for {len(missing)} GPU systems: {missing}")


@app.post("/recommend")
def post_recommend(
    req: RecommendRequest,
    include: str | None = Query(default=None, examples=["config,memory"], description="Comma-separated extras: config, memory."),
):
    """Find optimal GPU configuration for a workload."""
    if req.simulator == "blis":
        try:
            outcome = _blis_recommend(req)
        except (BlisUnsupportedRequest, BlisManifestError, BlisExecutionError) as e:
            _handle_blis_error(e)
        except Exception as e:
            _common_error_handler(e, "recommend", req.model_path, req.backend, req.system)
        if not outcome.configs:
            raise HTTPException(status_code=422, detail="No configuration meets the specified requirements.")
        # A non-demand BLIS outcome (or a test double) may not carry a demand
        # block at all.
        blis_demand = getattr(outcome, "demand", None)
        demand = (
            DemandOutcome(
                target_rps=blis_demand.target_rps,
                achieved_rps=blis_demand.achieved_rps,
                met=blis_demand.met,
                observed_concurrency=blis_demand.observed_concurrency,
                window_seconds=blis_demand.window_seconds,
                completed_requests=blis_demand.completed_requests,
                evaluated_configurations=blis_demand.evaluated_configurations,
                best_observed_rps=blis_demand.best_observed_rps,
                simulator="blis",
            )
            if blis_demand is not None
            else None
        )
        # BLIS enumerates its candidate grid instead of sampling it. Reporting
        # that as a sweep keeps `sweep` present for both simulators without
        # inventing a sampler identity.
        summary = outcome.search_summary or {}
        return RecommendResponse(
            configs=[RecommendConfig.model_validate(config) for config in outcome.configs],
            chosen_mode=str(outcome.configs[0].get("mode", "agg")),
            simulator="blis",
            search_summary=outcome.search_summary,
            demand=demand,
            sweep=SweepMeta(
                algorithm="enumerate",
                max_trials=summary.get("candidate_configurations"),
                evaluated=summary.get("evaluated_configurations"),
                budget_exhausted=bool(summary.get("budget_exhausted")),
                strict_sla=req.sla_mode == "enforce",
                window_min_gpus=req.min_candidate_gpus or 1,
                window_max_gpus=summary.get("effective_max_gpus"),
            ),
        )

    # A client abort cannot stop a synchronous sweep, so bound how many Pareto
    # sweeps can occupy threadpool slots at once. Reject rather than queue: a
    # queued request still holds its own worker thread.
    pareto_slots = _pareto_slots() if req.objective == "pareto" else None
    if pareto_slots is not None and not pareto_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="A Pareto sweep is already running on this worker; retry shortly.",
            headers={"Retry-After": "5"},
        )
    demand: DemandOutcome | None = None
    capacity: CapacityOutcome | None = None
    capacity_options: list[CapacityOption] = []
    result = None
    capacity_result: tuple[RecommendConfig, CapacityOutcome, SweepMeta, list[CapacityOption]] | None = None
    fell_back_from: str | None = None
    demand_records: list[Any] = []
    sweep: SweepMeta | None = None
    includes = _parse_include(include)
    # An omitted Pareto window is auto-fitted to the model. Resolve it inside
    # the version-fallback closure (below) so the preflight and the sweep share
    # the same backend version. Either explicit bound keeps the old semantics.
    pareto_auto_window = (
        req.objective == "pareto"
        and req.min_candidate_gpus is None
        and req.max_candidate_gpus is None
    )
    try:
        if req.objective == "throughput_demand":
            # Demand sizing cannot rely on the SDK's smallest-first search
            # (min_gpus is rejected without an SLA), so the service walks a
            # growing GPU window and stops at the first one that sustains
            # the offered rate.
            (demand_records, sweep), fell_back_from = _run_with_version_fallback(
                req, lambda: _run_demand_windows(req)
            )
        elif req.objective == "concurrency_capacity":
            # Capacity is a deterministic enumeration over the aggregated
            # search space: no SLA-free min-GPU sweep objective exists, so
            # there is no window ladder or sampler to drive.
            capacity_result, fell_back_from = _run_with_version_fallback(
                req, lambda: _run_capacity_recommendation(req, include=includes)
            )
        elif req.objective == "pareto":
            result, fell_back_from = _run_with_version_fallback(
                req,
                lambda: _run_pareto_recommendation(req, auto_window=pareto_auto_window),
            )
        else:
            result, fell_back_from = _run_with_version_fallback(
                req, lambda: _run_aisimulate_recommendation(req)
            )
    except Exception as e:
        _common_error_handler(e, "recommend", req.model_path, req.backend, req.system)
    finally:
        if pareto_slots is not None:
            pareto_slots.release()

    if req.objective == "throughput_demand":
        # Select the winner from the merged ledger instead of the SDK's
        # throughput-per-GPU pick: cheapest config that actually sustained the
        # target, or the best observed rate when none did.
        target_rps = float(req.target_request_rate or 0.0)
        winner, met, best_observed, measured = _select_demand_record(
            demand_records, target_rps
        )
        if winner is None:
            raise HTTPException(
                status_code=422,
                detail="No feasible configuration completed the demand traffic.",
            )
        demand = _demand_outcome(
            winner,
            target_rps=target_rps,
            met=met,
            best_observed_rps=best_observed,
            evaluated=measured,
        )
        # The winner is a ledger record, not a `selected_candidates` entry, so
        # it is converted through its flat `provenance.topology` rather than a
        # prediction config the SDK may not keep on the ledger.
        configs = [_demand_config_from_record(winner, req)]
    elif req.objective == "concurrency_capacity":
        assert capacity_result is not None
        configs = [capacity_result[0]]
        capacity = capacity_result[1]
        sweep = capacity_result[2]
        capacity_options = capacity_result[3]
    else:
        assert result is not None
        candidates = result.selected_candidates[: req.top_n]
        # Pareto returns the frontier via selected_candidates; the full ledger
        # is in result.candidates. Fail only when the sweep found nothing.
        if req.objective == "pareto":
            if not result.candidates:
                raise HTTPException(status_code=422, detail="No configuration meets the specified requirements.")
        elif not candidates:
            raise HTTPException(status_code=422, detail="No configuration meets the specified requirements.")
        configs = [_aisimulate_candidate_config(candidate, req) for candidate in candidates]
        sweep = _sweep_meta(result, req)

    if fell_back_from and req.backend_version:
        fallback_warning = (
            f"No performance data for {req.backend} {fell_back_from} on {req.system}; "
            f"used {req.backend_version}."
        )
        for cfg in configs:
            cfg.warnings.insert(0, fallback_warning)
    for cfg in configs:
        if req.objective == "concurrency_capacity":
            # `_run_capacity_recommendation` already built the winning shape's
            # extras, including `inclusive_tpot`. Rebuilding them here would
            # drop attention-DP and MoE widths, disagree with the capacity
            # block, and apply the TPOT transform twice to the primary config,
            # which is shared with `capacity_options[0].config`.
            continue
        if req.inclusive_tpot:
            cfg.tpot = _inclusive_tpot(cfg.ttft, cfg.tpot, req.osl)
        if cfg.prefill_config is not None or cfg.decode_config is not None:
            for worker, context_length in (
                (cfg.prefill_config, req.prefill_max_seq_len or req.max_seq_len),
                (cfg.decode_config, req.decode_max_seq_len or req.max_seq_len),
            ):
                if worker and worker.tp and "memory" in includes:
                    worker.memory_breakdown = _build_memory_breakdown(
                        req.model_path, req.system, req.backend, worker.backend_version,
                        worker.tp, worker.pp or 1, req.isl, req.osl, worker.num_workers or 1,
                        max_seq_len=context_length,
                    )
            if "memory" in includes and cfg.memory is None:
                cfg.memory = max(
                    (
                        memory_gb
                        for worker in (cfg.prefill_config, cfg.decode_config)
                        if worker is not None
                        for memory_gb in [_memory_breakdown_gb(worker.memory_breakdown)]
                        if memory_gb is not None
                    ),
                    default=None,
                )
        else:
            if "config" in includes:
                cfg.serving_config = _build_serving_config(
                    req.backend, cfg.tp or 1, req.isl, req.osl,
                    cfg.bs or req.target_concurrency or cfg.concurrency or 1, cfg.gemm, req.prefix,
                    max_seq_len=req.max_seq_len, backend_version=cfg.backend_version,
                )
            if "memory" in includes:
                cfg.memory_breakdown = _build_memory_breakdown(
                    req.model_path, req.system, req.backend, cfg.backend_version,
                    cfg.tp or 1, cfg.pp or 1, req.isl, req.osl,
                    cfg.bs or req.target_concurrency or cfg.concurrency or 1, max_seq_len=req.max_seq_len,
                )
                if cfg.memory is None:
                    cfg.memory = _memory_breakdown_gb(cfg.memory_breakdown)
    chosen_mode = "disagg" if configs and configs[0].prefill_config is not None else "agg"
    candidate_responses: list[CandidateResponse] = []
    pareto_views: CandidateViews | None = None
    if req.objective == "pareto":
        transfer_modeled = req.kv_transfer_bandwidth_gb_per_second is not None
        for record in result.candidates:
            topology = dict(record.provenance.topology) if record.provenance else {}
            is_disagg = topology.get("deployment_mode") == "disagg"
            # Failed/timed-out records keep a disagg topology but have no replay
            # metrics, so transfer delay was not actually modeled for them.
            candidate_responses.append(
                _candidate_record_to_response(
                    record,
                    transfer_delay_modeled=is_disagg and transfer_modeled and bool(record.metrics),
                )
            )
        # ResultViews.pareto_front / top_n are plain candidate-ID string lists.
        # Guard a missing view object: an SDK build may return no views.
        if result.views is not None:
            pareto_views = CandidateViews(
                pareto_front=list(result.views.pareto_front),
                top_n=list(result.views.top_n),
            )
    return RecommendResponse(
        configs=configs,
        chosen_mode=chosen_mode,
        simulator="aisimulate",
        candidates=candidate_responses,
        views=pareto_views,
        demand=demand,
        capacity=capacity,
        capacity_options=capacity_options,
        sweep=sweep,
    )


def _handle_blis_error(error: Exception) -> None:
    if isinstance(error, BlisUnsupportedRequest):
        raise HTTPException(status_code=422, detail=str(error)) from error
    if isinstance(error, BlisManifestError):
        raise HTTPException(status_code=503, detail=str(error)) from error
    if isinstance(error, BlisTimeoutError):
        raise HTTPException(status_code=504, detail=str(error)) from error
    if isinstance(error, BlisExecutionError):
        raise HTTPException(status_code=502, detail=str(error)) from error
    raise error


def _blis_estimate(req: EstimateRequest):
    unsupported_values: dict[str, Any] = {
        "database_mode": req.database_mode if req.database_mode != "HYBRID" else None,
        "requests_per_load_unit": req.requests_per_load_unit,
    }
    if req.mode != "disagg":
        unsupported_values.update({
            "decode_system": req.decode_system,
            "fabric": req.fabric,
            "prefill_max_seq_len": req.prefill_max_seq_len,
            "decode_max_seq_len": req.decode_max_seq_len,
            "prefill_gpu_memory_utilization": req.prefill_gpu_memory_utilization,
            "decode_gpu_memory_utilization": req.decode_gpu_memory_utilization,
            "prefill_tp_size": req.prefill_tp_size,
            "prefill_pp_size": req.prefill_pp_size,
            "prefill_moe_tp_size": req.prefill_moe_tp_size,
            "prefill_moe_ep_size": req.prefill_moe_ep_size,
            "prefill_batch_size": req.prefill_batch_size,
            "prefill_num_workers": req.prefill_num_workers,
            "decode_tp_size": req.decode_tp_size,
            "decode_pp_size": req.decode_pp_size,
            "decode_moe_tp_size": req.decode_moe_tp_size,
            "decode_moe_ep_size": req.decode_moe_ep_size,
            "decode_batch_size": req.decode_batch_size,
            "decode_num_workers": req.decode_num_workers,
        })
    unsupported_options = tuple(
        name
        for name, value in unsupported_values.items()
        if value is not None
    )
    return run_blis_estimate(BlisEstimateInput(
        model_path=req.model_path,
        system=req.system,
        backend=req.backend,
        backend_version=req.backend_version,
        isl=req.isl,
        osl=req.osl,
        concurrency=req.batch_size,
        tp=req.tp_size,
        pp=req.pp_size,
        dp=req.attention_dp_size,
        moe_tp_size=req.moe_tp_size,
        moe_ep_size=req.moe_ep_size,
        max_seq_len=req.max_seq_len,
        max_num_seqs=req.max_num_seqs,
        gpu_memory_utilization=req.gpu_memory_utilization,
        gemm_quant_mode=req.gemm_quant_mode,
        moe_quant_mode=req.moe_quant_mode,
        kvcache_quant_mode=req.kvcache_quant_mode,
        fmha_quant_mode=req.fmha_quant_mode,
        prefix=req.prefix,
        prefix_caching=req.prefix_caching,
        enable_chunked_prefill=req.enable_chunked_prefill,
        mode=req.mode,
        decode_system=req.decode_system,
        fabric=req.fabric,
        prefill_max_seq_len=req.prefill_max_seq_len,
        decode_max_seq_len=req.decode_max_seq_len,
        prefill_gpu_memory_utilization=req.prefill_gpu_memory_utilization,
        decode_gpu_memory_utilization=req.decode_gpu_memory_utilization,
        prefill_tp_size=req.prefill_tp_size,
        prefill_pp_size=req.prefill_pp_size,
        prefill_moe_tp_size=req.prefill_moe_tp_size,
        prefill_moe_ep_size=req.prefill_moe_ep_size,
        prefill_batch_size=req.prefill_batch_size,
        prefill_num_workers=req.prefill_num_workers,
        decode_tp_size=req.decode_tp_size,
        decode_pp_size=req.decode_pp_size,
        decode_moe_tp_size=req.decode_moe_tp_size,
        decode_moe_ep_size=req.decode_moe_ep_size,
        decode_batch_size=req.decode_batch_size,
        decode_num_workers=req.decode_num_workers,
        model_config_provided=req.model_config_data is not None,
        unsupported_options=unsupported_options,
    ))


def _blis_recommend(req: RecommendRequest):
    configured_max = _max_candidate_gpus()
    requested_max = req.max_candidate_gpus
    effective_max = min(requested_max, configured_max) if requested_max is not None else configured_max
    return run_blis_recommend(BlisRecommendInput(
        model_path=req.model_path,
        system=req.system,
        backend=req.backend,
        backend_version=req.backend_version,
        isl=req.isl,
        osl=req.osl,
        concurrency=int(req.target_concurrency) if req.target_concurrency is not None else None,
        request_rate=req.target_request_rate,
        ttft=req.ttft,
        tpot=req.tpot,
        request_latency=req.request_latency,
        prefix=req.prefix,
        min_candidate_gpus=req.min_candidate_gpus,
        max_candidate_gpus=effective_max,
        top_n=req.top_n,
        search_budget_seconds=req.search_budget_seconds,
        demand=req.objective == "throughput_demand",
        max_seq_len=req.max_seq_len,
        prefill_max_seq_len=req.prefill_max_seq_len,
        decode_max_seq_len=req.decode_max_seq_len,
        fabric=req.fabric,
        database_mode=req.database_mode,
        inclusive_tpot=req.inclusive_tpot,
        model_config_provided=req.model_config_data is not None,
    ))

def post_predict(
    req: EstimateRequest,
    include: str | None = Query(default=None, examples=["config,memory"], description="Comma-separated extras: config, memory."),
):
    """Single-point performance prediction for a given parallelism configuration.

    Given a model, GPU system, backend, and explicit parallelism settings
    (TP/PP/batch_size), returns predicted TTFT, TPOT, throughput, and memory.
    Use this when you know your deployment configuration and want to predict
    its performance. Use /recommend when you want to find the optimal config.
    """
    is_disagg = req.mode == "disagg"

    # Resolve per-role parallelism, falling back to the top-level agg fields.
    p_tp = req.prefill_tp_size or req.tp_size
    p_pp = req.prefill_pp_size or req.pp_size
    p_bs = req.prefill_batch_size or req.batch_size
    p_workers = req.prefill_num_workers or 1
    p_moe_tp = req.prefill_moe_tp_size or req.moe_tp_size
    p_moe_ep = req.prefill_moe_ep_size or req.moe_ep_size
    d_tp = req.decode_tp_size or req.tp_size
    d_pp = req.decode_pp_size or req.pp_size
    d_bs = req.decode_batch_size or req.batch_size
    d_workers = req.decode_num_workers or 1
    d_moe_tp = req.decode_moe_tp_size or req.moe_tp_size
    d_moe_ep = req.decode_moe_ep_size or req.moe_ep_size

    if req.simulator == "aisimulate":
        # Resolve the width pair the API asks the simulator to use, so the
        # response reports exactly the layout that was evaluated.
        p_moe_tp, p_moe_ep = _resolve_moe_widths(p_tp, req.attention_dp_size, p_moe_tp, p_moe_ep)
        d_moe_tp, d_moe_ep = _resolve_moe_widths(d_tp, req.attention_dp_size, d_moe_tp, d_moe_ep)
        agg_moe_tp, agg_moe_ep = _resolve_moe_widths(
            req.tp_size, req.attention_dp_size, req.moe_tp_size, req.moe_ep_size
        )
        try:
            prediction, fell_back_from = _run_with_version_fallback(
                req, lambda: _run_aisimulate_prediction(req, _parse_include(include))
            )
        except Exception as e:
            _common_error_handler(e, "estimate", req.model_path, req.backend, req.system)
    else:
        # BLIS has no width invariant and consumes only the values it was
        # given, so report those back unchanged.
        agg_moe_tp, agg_moe_ep = req.moe_tp_size, req.moe_ep_size
        fell_back_from = None
        try:
            prediction = _blis_estimate(req)
        except (BlisUnsupportedRequest, BlisManifestError, BlisExecutionError) as e:
            _handle_blis_error(e)
        except Exception as e:
            _common_error_handler(e, "estimate", req.model_path, req.backend, req.system)

    includes = _parse_include(include)

    tpot = _metric(prediction.summary, "tpot_ms", "mean_tpot_ms", "itl_ms")
    if req.inclusive_tpot:
        tpot = _inclusive_tpot(_metric(prediction.summary, "ttft_ms", "mean_ttft_ms"), tpot, req.osl)

    warnings = list(getattr(prediction, "warnings", []) or [])
    if fell_back_from and req.backend_version:
        warnings.insert(
            0,
            f"No performance data for {req.backend} {fell_back_from} on {req.system}; "
            f"used {req.backend_version}.",
        )

    resp = EstimateResponse(
        ttft=_metric(prediction.summary, "ttft_ms", "mean_ttft_ms") or 0.0,
        tpot=tpot,
        request_latency=_metric(prediction.summary, "e2e_latency_ms", "mean_e2e_latency_ms"),
        tokens_per_second=_metric(prediction.summary, "output_throughput_tok_s", "tokens_per_second"),
        tokens_per_second_per_gpu=_per_gpu_throughput(
            prediction.summary,
            _prediction_total_gpus(prediction.summary, req),
        ),
        tokens_per_second_per_user=_per_user_throughput(prediction.summary, req.batch_size),
        memory=None,
        concurrency=req.batch_size,
        tp=None if is_disagg else req.tp_size,
        pp=None if is_disagg else req.pp_size,
        dp=None if is_disagg else req.attention_dp_size,
        moe_tp=None if is_disagg else agg_moe_tp,
        moe_ep=None if is_disagg else agg_moe_ep,
        system=req.system,
        backend=req.backend,
        backend_version=req.backend_version,
        gemm=req.gemm_quant_mode,
        kvcache=req.kvcache_quant_mode,
        power_w=_metric(prediction.summary, "power_w"),
        mode=req.mode,
        simulator=req.simulator,
        provenance=(
            SimulatorProvenance.model_validate(prediction.provenance)
            if getattr(prediction, "provenance", None)
            else SimulatorProvenance(
                simulator=req.simulator,
                requested_model=req.model_path,
                system=req.system,
                engine_version=req.backend_version,
            )
        ),
        warnings=warnings,
    )

    if req.simulator == "blis":
        if "config" in includes and getattr(prediction, "serving_config", None):
            resp.serving_config = ServingConfig.model_validate(prediction.serving_config)
        if is_disagg:
            resp.prefill_config = WorkerConfig.model_validate(prediction.prefill_config)
            resp.decode_config = WorkerConfig.model_validate(prediction.decode_config)
        if "memory" in includes:
            resp.memory_unavailable_reason = "The selected simulator does not provide a KV-cache memory breakdown."
        return resp

    if is_disagg:
        # (p)/(d)memory are per-GPU peaks; the single figure is the worst case.
        p_mem = None
        d_mem = None
        phase_mems = [m for m in (p_mem, d_mem) if m is not None]
        if phase_mems:
            resp.memory = max(phase_mems)

        gemm_q = req.gemm_quant_mode if req.gemm_quant_mode and req.gemm_quant_mode != "half" else None
        kv_q = req.kvcache_quant_mode if req.kvcache_quant_mode and req.kvcache_quant_mode != "half" else None
        want_memory = "memory" in includes

        resp.prefill_config = WorkerConfig(
            tp=p_tp, pp=p_pp, moe_tp=p_moe_tp, moe_ep=p_moe_ep,
            num_workers=p_workers, batch_size=p_bs, memory_gb=p_mem,
            backend_version=resp.backend_version,
        )
        resp.decode_config = WorkerConfig(
            tp=d_tp, pp=d_pp, moe_tp=d_moe_tp, moe_ep=d_moe_ep,
            num_workers=d_workers, batch_size=d_bs, memory_gb=d_mem,
            backend_version=resp.backend_version,
        )

        if want_memory:
            # The decode pool may run on a different GPU system than prefill.
            decode_sys = req.decode_system or req.system
            with _with_model_config(req.model_path, req.model_config_data) as effective_path:
                for worker, worker_sys in (
                    (resp.prefill_config, req.system),
                    (resp.decode_config, decode_sys),
                ):
                    if worker and worker.tp:
                        worker.memory_breakdown = _build_memory_breakdown(
                            effective_path, worker_sys, req.backend, req.backend_version,
                            worker.tp, worker.pp or 1, req.isl, req.osl,
                            worker.batch_size or req.batch_size,
                            gemm_q, kv_q, worker.moe_tp, worker.moe_ep,
                            max_seq_len=(req.prefill_max_seq_len if worker is resp.prefill_config else req.decode_max_seq_len)
                            or req.max_seq_len,
                            memory_fraction=(req.prefill_gpu_memory_utilization if worker is resp.prefill_config else req.decode_gpu_memory_utilization)
                            or req.gpu_memory_utilization,
                            attention_dp=req.attention_dp_size,
                        )
        return resp

    if "config" in includes:
        resp.serving_config = _build_serving_config(
            resp.backend or req.backend, req.tp_size, req.isl, req.osl,
            req.batch_size, resp.gemm, req.prefix, max_seq_len=req.max_seq_len,
            gpu_memory_utilization=req.gpu_memory_utilization, backend_version=req.backend_version,
            max_num_seqs=req.max_num_seqs, enable_chunked_prefill=req.enable_chunked_prefill,
            prefix_caching=req.prefix_caching,
        )

    if "memory" in includes:
        with _with_model_config(req.model_path, req.model_config_data) as effective_path:
            resp.memory_breakdown = _build_memory_breakdown(
                effective_path, req.system, req.backend, req.backend_version,
                req.tp_size, req.pp_size, req.isl, req.osl, req.batch_size,
                req.gemm_quant_mode if req.gemm_quant_mode and req.gemm_quant_mode != "half" else None,
                req.kvcache_quant_mode if req.kvcache_quant_mode and req.kvcache_quant_mode != "half" else None,
                agg_moe_tp, agg_moe_ep, max_seq_len=req.max_seq_len,
                memory_fraction=req.gpu_memory_utilization,
                attention_dp=req.attention_dp_size,
            )

    return resp


app.post("/predict", response_model=EstimateResponse)(post_predict)
app.post("/estimate", response_model=EstimateResponse, deprecated=True)(post_predict)


@app.post("/memory", response_model=MemoryResponse)
def post_memory(req: MemoryRequest):
    """Estimate GPU memory breakdown for a model configuration."""
    try:
        with _with_model_config(req.model_path, req.model_config_data) as effective_path:
            raw = estimate_kv_cache(
                model_path=effective_path,
                system=req.system,
                backend=_sdk_backend_id(req.backend),
                backend_version=req.backend_version,
                max_num_tokens=req.max_num_tokens,
                max_batch_size=req.max_batch_size,
                memory_fraction_kind=req.memory_fraction_kind,
                memory_fraction_value=req.memory_fraction_value,
                tp_size=req.tp_size,
                pp_size=req.pp_size,
                attention_dp_size=req.attention_dp_size,
                moe_tp_size=req.moe_tp_size,
                moe_ep_size=req.moe_ep_size,
                gemm_quant_mode=req.gemm_quant_mode,
                moe_quant_mode=req.moe_quant_mode,
                kvcache_quant_mode=req.kvcache_quant_mode,
                fmha_quant_mode=req.fmha_quant_mode,
                comm_quant_mode=req.comm_quant_mode,
                tolerance_fraction=req.tolerance_fraction,
            )
    except Exception as e:
        _common_error_handler(e, "memory", req.model_path, req.backend, req.system)

    breakdown = raw.get("memory_breakdown") or {}
    return MemoryResponse(
        total_gpu_capacity_bytes=raw["total_gpu_capacity_bytes"],
        total_kv_size_bytes=raw["total_kv_size_bytes"],
        kv_size_per_token_bytes=raw["kv_size_per_token_bytes"],
        total_kv_size_tokens=raw["total_kv_size_tokens"],
        source=raw.get("source", "unknown"),
        memory_breakdown=MemoryBreakdown(
            weights_bytes=int(breakdown.get("weights_bytes", 0)),
            activations_bytes=int(breakdown.get("activations_bytes", 0)),
            runtime_overhead_bytes=int(breakdown.get("runtime_overhead_bytes", 0)),
            comm_overhead_bytes=int(breakdown.get("comm_overhead_bytes", 0)),
            kv_cache_bytes=raw["total_kv_size_bytes"],
        ),
        tolerance_adjusted=raw.get("tolerance_adjusted"),
        simulator="aisimulate",
    )


def _model_specs(model_id: str) -> dict:
    """Return metadata for a single catalog model."""
    try:
        cfg = get_model_config_from_model_path(model_id)
    except Exception:
        cfg = {}
    cfg = cfg if isinstance(cfg, dict) else {}

    def _pos_int(val) -> int | None:
        return int(val) if val and int(val) > 0 else None

    return {
        "id": model_id,
        "num_experts": _pos_int(cfg.get("num_experts")),
        "num_experts_per_tok": _pos_int(cfg.get("topk")),
        "context_length": _pos_int(cfg.get("context")),
        "num_attn_heads": _pos_int(cfg.get("n")),
        "num_kv_heads": _pos_int(cfg.get("n_kv")),
        "architecture": cfg.get("architecture") or None,
    }


@app.get("/models")
def get_models(
    include: str | None = Query(default=None, examples=["specs"], description="Comma-separated extras: specs."),
    simulator: Literal["aisimulate", "blis"] = Query(
        default="aisimulate",
        description="Simulator whose model support is reported.",
    ),
):
    """List supported models."""
    includes = _parse_include(include)
    if simulator == "blis":
        # Only the AISimulate IDs BLIS can price. The picker offers AISimulate IDs, so
        # a catalog-only model would be a dead option here.
        try:
            models = supported_model_ids()
        except BlisManifestError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    else:
        models = sorted(get_default_models())
    if "specs" in includes:
        return {"simulator": simulator, "models": [_model_specs(m) for m in models]}
    return {"simulator": simulator, "models": models}


def _system_entries(simulator: Literal["aisimulate", "blis"], *, want_specs: bool) -> list[dict[str, Any]]:
    """Catalog entries for the requested simulator, in the /systems shape."""
    system_ids = sorted(supported_systems())
    if simulator == "blis":
        try:
            system_ids = supported_system_ids(system_ids)
        except BlisManifestError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    systems = []
    for sys_id in system_ids:
        device_name = _DEVICE_DISPLAY_NAMES.get(sys_id)
        if device_name is None:
            # No perf-data display name -> no benchmark data; hide it.
            continue
        entry: dict[str, Any] = {
            "id": sys_id,
            "name": device_name,
        }
        if want_specs:
            try:
                spec = load_system_spec(sys_id)
                gpu = spec.get("gpu", {})
                node = spec.get("node", {})
                sm = int(gpu.get("sm_version", 0))
                sm_arch = _architecture_from_sm(sm)
                if sm_arch == "Other":
                    vendor_name = ""
                else:
                    vendor_name = "nvidia"
                entry.update({
                    "vendor": vendor_name,
                    "architecture": sm_arch,
                    "memory_bytes": int(gpu.get("mem_capacity", 0)),
                    "memory_bandwidth_bytes": int(gpu.get("mem_bw", 0)),
                    "bf16_tflops": float(gpu.get("bfloat16_tc_flops", 0)) / 1e12,
                    "tdp_watts": float(gpu.get("power", 0)),
                    "gpus_per_node": int(node.get("num_gpus_per_node", 0)),
                })
            except Exception:
                logger.warning("failed to load spec for %s", sys_id)
        systems.append(entry)
    return systems


@app.get("/systems")
def get_systems(
    include: str | None = Query(default=None, examples=["specs"], description="Comma-separated extras: specs."),
    simulator: Literal["aisimulate", "blis"] = Query(
        default="aisimulate",
        description="Simulator whose GPU system support is reported.",
    ),
):
    """List supported GPU systems."""
    includes = _parse_include(include)
    return {
        "simulator": simulator,
        "systems": _system_entries(simulator, want_specs="specs" in includes),
    }


@app.get("/networks")
def get_networks():
    """List network fabrics available to network-aware simulators."""
    try:
        return {"networks": sorted(load_support_manifest().networks), "simulator": "blis"}
    except BlisManifestError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metrics")
def get_metrics(request: Request):
    """Expose metrics in Prometheus or OTLP JSON format via content negotiation.

    - Accept: application/json → OTLP JSON (for pmdaopentelemetry and similar)
    - anything else (default)  → Prometheus/OpenMetrics text
    """
    if not _OBS:
        raise HTTPException(
            status_code=503,
            detail="Metrics unavailable (install with: pip install '.[otel]')",
        )
    return observability.metrics_response(request.headers.get("accept", "text/plain"))


# ─── MCP ─────────────────────────────────────────────────────────────────────

# Expose the API as MCP tools if the optional extra is present. Mounted after
# every route: fastapi-mcp snapshots the routes when the server is built, so a
# route declared below this block would not become a tool.
_MCP_SERVER = None
if _MCP:
    _MCP_SERVER = mcp_support.mount(app, name="aisimulators",
                                    description="GPU recommendation and performance estimation for LLM inference")
else:
    logger.info("MCP server unavailable (install with: pip install '.[mcp]')")


# ─── Entrypoint ──────────────────────────────────────────────────────────────

def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    return parser.parse_args(args=args)


if __name__ == "__main__":
    args = parse(sys.argv[1:])
    uvicorn.run(app, host=args.server_name, port=args.server_port)

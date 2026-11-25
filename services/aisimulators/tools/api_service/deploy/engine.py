# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lower an evaluated engine section into the generator's candidate contract.

The generator consumes a Sweeper candidate: a flat mapping of the topology and
serving limits a run was evaluated with. This module builds that mapping from
:class:`EngineSection` without re-deriving anything, and refuses a deployment
when a limit the target must render cannot be stated.
"""

from __future__ import annotations

from typing import Any

from ..engine_models import EngineField, EngineRole, EngineSection
from .catalog import DeployTarget


class DeployUnsupportedRequest(ValueError):
    """The request cannot produce an artifact that matches the evaluated run."""


#: Limits every v1 target renders into the model server's startup arguments.
#: A value nobody can state would be replaced by a backend default at startup,
#: so the artifact would run a different configuration than the one predicted.
REQUIRED_ROLE_FIELDS: tuple[str, ...] = (
    "replicas",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "moe_tensor_parallel_size",
    "moe_expert_parallel_size",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "block_size",
    "gpu_memory_utilization",
    "enable_prefix_caching",
)

_TOPOLOGY_KEYS = {
    "tensor_parallel_size": "tp",
    "pipeline_parallel_size": "pp",
    "data_parallel_size": "attention_dp",
    "moe_tensor_parallel_size": "moe_tp",
    "moe_expert_parallel_size": "moe_ep",
    "context_parallel_size": "cp",
    "decode_context_parallel_size": "dcp",
}

_SERVING_KEYS = {
    "max_num_seqs": "max_num_seqs",
    "max_num_batched_tokens": "max_num_batched_tokens",
    "block_size": "block_size",
    "gpu_memory_utilization": "gpu_memory_utilization",
    "enable_prefix_caching": "enable_prefix_caching",
}


def _stated(field: EngineField) -> bool:
    return field.source != "not-modeled" and field.value is not None


def _missing(role_name: str, role: EngineRole) -> list[str]:
    return [
        f"{role_name}.{name}"
        for name in REQUIRED_ROLE_FIELDS
        if not _stated(getattr(role, name))
    ]


def roles_for_mode(engine: EngineSection) -> dict[str, EngineRole]:
    if engine.mode == "agg":
        if engine.agg is None:
            raise DeployUnsupportedRequest(
                "the engine section reports an aggregated deployment but carries no 'agg' limits"
            )
        return {"agg": engine.agg}
    missing = [name for name in ("prefill", "decode") if getattr(engine, name) is None]
    if missing:
        raise DeployUnsupportedRequest(
            "the engine section reports a disaggregated deployment but carries no "
            + " or ".join(f"'{name}'" for name in missing)
            + " limits"
        )
    return {"prefill": engine.prefill, "decode": engine.decode}  # type: ignore[dict-item]


def validate(engine: EngineSection, target: DeployTarget, *, allow_version_mismatch: bool) -> None:
    """Refuse an engine section this target cannot reproduce."""
    if engine.mode not in target.modes:
        raise DeployUnsupportedRequest(
            f"target {target.id!r} does not render {engine.mode!r} deployments; "
            f"it renders {', '.join(target.modes)}"
        )
    if engine.backend not in target.backends:
        raise DeployUnsupportedRequest(
            f"target {target.id!r} renders {', '.join(target.backends)} deployments, "
            f"not {engine.backend!r}"
        )
    missing = [
        item
        for role_name, role in roles_for_mode(engine).items()
        for item in _missing(role_name, role)
    ]
    if missing:
        raise DeployUnsupportedRequest(
            "the evaluated run does not state every limit this target renders: "
            + ", ".join(missing)
            + ". Re-run the prediction with a simulator that evaluates these limits "
            "(AISim reports all of them) before deploying."
        )
    if (
        engine.backend_version is not None
        and engine.backend_version not in target.engine_versions
        and not allow_version_mismatch
    ):
        raise DeployUnsupportedRequest(
            f"the evaluated run used {engine.backend} {engine.backend_version}, but target "
            f"{target.id!r} pins {engine.backend} {', '.join(target.engine_versions)}. "
            "Rendering evaluated limits against a different engine release can produce "
            "flags that release does not accept. Predict on the pinned version, pass "
            "`image` for a runtime you have verified, and set "
            "`allow_engine_version_mismatch=true` to render anyway."
        )


def candidate_config(
    engine: EngineSection,
    *,
    system: str,
    concurrency: int | None,
) -> dict[str, Any]:
    """The Sweeper candidate config the evaluated section describes.

    Every value below is copied from the engine section. ``from_sweeper_candidate``
    treats this mapping as the evaluated candidate, so an invented value here
    would be rendered as if the simulator had measured it.
    """
    if engine.backend_version is None:
        raise DeployUnsupportedRequest(
            "the engine section does not state a backend version, so the startup "
            "arguments cannot be pinned to a release"
        )
    roles = roles_for_mode(engine)
    config: dict[str, Any] = {
        "deployment_mode": engine.mode,
        "model_name": engine.model,
        "backend": engine.backend,
        "backend_version": engine.backend_version,
        "hardware_sku": system,
    }
    for role_name, role in roles.items():
        prefix = "" if role_name == "agg" else f"{role_name}_"
        for field_name, key in _TOPOLOGY_KEYS.items():
            field = getattr(role, field_name)
            if _stated(field):
                config[f"{prefix}{key}"] = field.value
        for field_name, key in _SERVING_KEYS.items():
            config[f"{role_name}_{key}"] = getattr(role, field_name).value
        config[f"{role_name}_max_num_seqs"] = role.max_num_seqs.value
        config[f"{role_name}_max_num_batched_tokens"] = role.max_num_batched_tokens.value
        config["replicas" if role_name == "agg" else f"{role_name}_replicas"] = role.replicas.value
        context_key = "context_length" if role_name == "agg" else f"{role_name}_context_length"
        config[context_key] = role.max_model_len.value
        config[f"{prefix}cp"] = role.context_parallel_size.value
        config[f"{prefix}dcp"] = role.decode_context_parallel_size.value
    if engine.mode == "disagg":
        config["context_length"] = max(int(role.max_model_len.value) for role in roles.values())
    if concurrency is not None:
        config["concurrency"] = concurrency
    else:
        # The run's own batch ceiling is the concurrency it can hold; a
        # disaggregated deployment is bound by the decode pool.
        limiting = roles["decode"] if "decode" in roles else roles["agg"]
        config["concurrency"] = int(limiting.max_num_seqs.value) * int(limiting.replicas.value)
    return config

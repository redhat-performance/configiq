# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-limit contract shared by the API routes and the deployment service.

``include=engine`` reports the limits a run actually evaluated, together with
how each value is known. The deployment service renders those values verbatim
and refuses a target that would have to render a limit nobody can state, so the
two sides share one definition instead of restating the shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

#: How a value in :class:`EngineField` is known.
EngineSource = Literal["evaluated", "requested", "backend-default", "not-modeled"]


class EngineField(BaseModel):
    """One resolved engine limit, and how the service knows it.

    A deployable artifact has to state every limit it renders. ``source`` says
    whether the value came from the run the simulator evaluated, from the
    caller without being evaluated, from the pinned backend's documented
    default, or whether no value can be stated at all:

    - ``evaluated``: the simulator's evaluated configuration holds this value,
      so the reported metrics depend on it.
    - ``requested``: the caller stated it and it is carried into an artifact,
      but the simulator's configuration does not hold it, so the metrics do not
      reflect it.
    - ``backend-default``: nothing stated it; the pinned backend version applies
      its documented default, reported in ``value`` when it is a value rather
      than "unset".
    - ``not-modeled``: the simulator reports no value and no pinned default
      exists. An artifact that must render this field cannot be produced.
    """

    value: bool | int | float | str | None = None
    source: EngineSource


class EngineRole(BaseModel):
    """Resolved engine limits for one worker role.

    Fields are reported even when they are unset, so a reader can tell "the
    simulator used 1" from "the simulator has no pipeline axis".
    """

    replicas: EngineField
    tensor_parallel_size: EngineField
    pipeline_parallel_size: EngineField
    data_parallel_size: EngineField
    context_parallel_size: EngineField
    decode_context_parallel_size: EngineField
    moe_tensor_parallel_size: EngineField
    moe_expert_parallel_size: EngineField
    max_model_len: EngineField
    max_num_seqs: EngineField
    max_num_batched_tokens: EngineField
    block_size: EngineField
    gpu_memory_utilization: EngineField
    enable_prefix_caching: EngineField
    enable_chunked_prefill: EngineField
    kv_cache_dtype: EngineField
    quantization: EngineField


class EngineSection(BaseModel):
    """The exact engine limits the simulator evaluated (``include=engine``).

    This is the deploy contract: a deployment artifact renders these values
    verbatim instead of re-deriving them, and a field marked ``not-modeled``
    blocks a target that has to render it.
    """

    model: str
    backend: str
    backend_version: str | None = None
    mode: Literal["agg", "disagg"] = "agg"
    agg: EngineRole | None = None
    prefill: EngineRole | None = None
    decode: EngineRole | None = None

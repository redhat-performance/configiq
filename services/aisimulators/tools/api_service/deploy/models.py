# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request and response contract for the deployment endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..engine_models import EngineSection


class DeployRequest(BaseModel):
    """One deployment, rendered from limits the simulator already evaluated.

    The request carries no sizing intent: the caller passes the ``engine``
    section a prediction returned with ``include=engine``, and the service
    renders exactly those values. Anything the target must render but the
    engine section cannot state is refused instead of defaulted.

    ``image`` and ``allow_engine_version_mismatch`` are the only ways to depart
    from the catalog's pinned image. A mismatch is recorded in the artifact
    provenance and printed by ``preflight.sh``, so a deployment that runs a
    different engine release than the one that was predicted is visible rather
    than silent.
    """

    target: str = Field(description="Deployment target id from GET /deploy/targets.")
    engine: EngineSection
    system: str = Field(description="Hardware SKU the evaluated run used, e.g. h200_sxm.")
    served_model_name: str | None = Field(
        default=None,
        description="Name the OpenAI-compatible endpoint serves. Defaults to the model path.",
    )
    model_config_data: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The model's config.json, when the model is not resolvable from this "
            "deployment. The same value the prediction was given."
        ),
    )
    isl: int = Field(
        ge=1,
        description="Input tokens per request the evaluation used; the generator states it in the artifact.",
    )
    osl: int = Field(
        ge=1,
        description="Output tokens per request the evaluation used.",
    )
    concurrency: int | None = Field(default=None, ge=1, description="Serving concurrency the run targeted.")
    namespace: str = Field(default="configiq", min_length=1, max_length=63)
    name_prefix: str | None = Field(default=None, max_length=40)
    image: str | None = Field(
        default=None,
        description="Model-server image override. Defaults to the target's pinned image.",
    )
    allow_engine_version_mismatch: bool = Field(
        default=False,
        description=(
            "Render even when the target's pinned image runs a different engine "
            "release than the evaluated one. The mismatch is recorded in the "
            "artifact provenance and checked by preflight.sh."
        ),
    )


class DeployTargetInfo(BaseModel):
    """One entry in the target catalog."""

    id: str
    name: str
    description: str
    kind: str
    simulator: str
    modes: list[str]
    backends: list[str]
    engine_versions: list[str]
    pins: dict[str, str]
    available: bool
    unavailable_reason: str | None = None
    artifact_schema_version: str
    experimental: bool = True


class DeployTargetsResponse(BaseModel):
    targets: list[DeployTargetInfo]


class DeployFile(BaseModel):
    path: str
    contents: str
    #: POSIX mode; scripts are 0o755 so a zip extract keeps them runnable.
    mode: int = 0o644


class DeployCheck(BaseModel):
    """One assertion the render gate ran against the assembled artifact."""

    id: str
    status: Literal["passed", "skipped", "failed"]
    detail: str


class DeployArtifact(BaseModel):
    target: str
    artifact_schema_version: str
    engine: EngineSection
    pins: dict[str, str]
    checks: list[DeployCheck]
    warnings: list[str] = Field(default_factory=list)
    files: list[DeployFile]

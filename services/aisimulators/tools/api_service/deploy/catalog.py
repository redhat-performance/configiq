# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned inputs for the deployment targets this service can render.

Every value here is a release pin, not a default: the bundle records the same
values in ``PROVENANCE.md`` so a rendered artifact can be traced back to the
exact upstream revision, chart, and image it was produced from. Nothing in this
module is derived from a request.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

#: llm-d release the model-server overlays are rendered against, and the exact
#: commit that release tag points at. The overlay is only valid against this
#: tree, so the bundle carries the referenced subtrees from it.
LLM_D_VERSION = "v1.0.0"
LLM_D_REVISION = "378b42d18dabe0b9c4ced4e6f3670d93b52db08e"

#: Versions the llm-d v1.0.0 guides pin in ``guides/env.sh``.
GATEWAY_API_VERSION = "v1.6.2"
GAIE_VERSION = "v1.6.2"
LWS_VERSION = "1.0.0"
ROUTER_CHART = "oci://ghcr.io/llm-d/charts/llm-d-router-standalone"
ROUTER_CHART_VERSION = "v1.0.0"

#: The model-server images llm-d v1.0.0 ships. The sidecar comes from the
#: ``nightly`` component in the P/D base; a deployable artifact replaces it with
#: the release tag so the bundle cannot drift between installs.
LLM_D_SIDECAR_IMAGE = "ghcr.io/llm-d/llm-d-router-disagg-sidecar:v1.0.0"
LLM_D_SIDECAR_IMAGE_NAME = "ghcr.io/llm-d/llm-d-router-disagg-sidecar"
LLM_D_VLLM_IMAGE = "docker.io/vllm/vllm-openai:v0.31.0"

#: Engine release each target's default image runs. A deployment re-uses the
#: limits a prediction evaluated, so the renderer refuses a target whose image
#: runs a different engine release unless the caller states an image of their own.
LLM_D_ENGINE_VERSION = "0.31.0"

#: Dynamo release the DGD target renders. The generator's version matrix maps
#: this release to its runtime image, so the image is not repeated here.
DYNAMO_VERSION = "1.3.0"
DYNAMO_ENGINE_VERSION = "0.24.0"

#: Guide subtrees the overlays reference, relative to the llm-d repository root.
LLM_D_VENDOR_PATHS = (
    "guides/recipes/modelserver",
    "guides/recipes/router",
    "guides/pd-disaggregation/modelserver",
    "guides/pd-disaggregation/router",
    "guides/optimized-baseline/modelserver",
    "guides/optimized-baseline/router",
)

#: Model-server base each llm-d layout patches, relative to the llm-d root.
LLM_D_BASE_BY_MODE = {
    "agg": "guides/optimized-baseline/modelserver/gpu/vllm/base",
    "disagg": "guides/pd-disaggregation/modelserver/gpu/vllm/base",
}

#: Router values layered on the shared base, per layout.
LLM_D_ROUTER_VALUES_BY_MODE = {
    "agg": "guides/optimized-baseline/router/optimized-baseline.values.yaml",
    "disagg": "guides/pd-disaggregation/router/pd-disaggregation.values.yaml",
}

LLM_D_ROUTER_BASE_VALUES = "guides/recipes/router/base.values.yaml"

ARTIFACT_SCHEMA_VERSION = "1"


def _service_root() -> Path:
    # deploy/catalog.py -> deploy -> api_service -> tools -> aisimulators
    return Path(__file__).resolve().parents[3]


def llm_d_vendor_dir() -> Path:
    """Where the pinned llm-d subtrees were vendored at image build time."""
    override = os.environ.get("AISIMULATORS_LLM_D_VENDOR_DIR", "").strip()
    base = Path(override) if override else _service_root() / "vendor" / "llm-d"
    return base / LLM_D_REVISION


def kustomize_command() -> list[str] | None:
    """The pinned-capable kustomize entry point, or None when absent.

    ``kubectl kustomize`` is a fallback for a host that has no standalone
    binary; both accept the same overlay. The render gate fails closed rather
    than shipping unverified manifests, so callers must treat None as
    unavailable.
    """
    override = os.environ.get("AISIMULATORS_KUSTOMIZE_BIN", "").strip()
    if override:
        return [override] if shutil.which(override) else None
    if shutil.which("kustomize"):
        return ["kustomize", "build"]
    if shutil.which("kubectl"):
        return ["kubectl", "kustomize"]
    return None


@dataclass(frozen=True)
class DeployTarget:
    id: str
    name: str
    description: str
    kind: str
    modes: tuple[str, ...]
    backends: tuple[str, ...]
    engine_versions: tuple[str, ...]
    pins: dict[str, str] = field(default_factory=dict)

    def unavailable_reason(self) -> str | None:
        if self.kind == "llm-d":
            if not llm_d_vendor_dir().is_dir():
                return (
                    "The pinned llm-d source tree is not present in this deployment "
                    f"({llm_d_vendor_dir()})."
                )
            if kustomize_command() is None:
                return "The render gate needs kustomize, which is not installed in this deployment."
        return None

    def as_dict(self) -> dict[str, object]:
        reason = self.unavailable_reason()
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "simulator": "aisim",
            "modes": list(self.modes),
            "backends": list(self.backends),
            "engine_versions": list(self.engine_versions),
            "pins": dict(self.pins),
            "available": reason is None,
            "unavailable_reason": reason,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "experimental": True,
        }


TARGETS: dict[str, DeployTarget] = {
    "dynamo-dgd": DeployTarget(
        id="dynamo-dgd",
        name="Dynamo DynamoGraphDeployment",
        description=(
            "One DynamoGraphDeployment for the evaluated topology, ready for "
            "`kubectl apply` on a cluster running the Dynamo platform."
        ),
        kind="dynamo-dgd",
        modes=("agg", "disagg"),
        backends=("vllm",),
        engine_versions=(DYNAMO_ENGINE_VERSION,),
        pins={
            "dynamo": DYNAMO_VERSION,
            "engine": f"vllm {DYNAMO_ENGINE_VERSION}",
            "deployment_target": "dynamo-j2",
        },
    ),
    "llm-d": DeployTarget(
        id="llm-d",
        name="llm-d well-lit path",
        description=(
            "An llm-d v1.0.0 model server (Kustomize overlay over the pinned "
            "guide base) plus the pinned router Helm release. Aggregated mode "
            "renders a Deployment; disaggregated mode renders a DisaggregatedSet."
        ),
        kind="llm-d",
        modes=("agg", "disagg"),
        backends=("vllm",),
        engine_versions=(LLM_D_ENGINE_VERSION,),
        pins={
            "llm-d": f"{LLM_D_VERSION} ({LLM_D_REVISION})",
            "engine": f"vllm {LLM_D_ENGINE_VERSION}",
            "deployment_target": "llm-d-kustomize",
            "router_chart": f"{ROUTER_CHART}:{ROUTER_CHART_VERSION}",
            "gateway_api": GATEWAY_API_VERSION,
            "gateway_api_inference_extension": GAIE_VERSION,
            "leader_worker_set": LWS_VERSION,
        },
    ),
}

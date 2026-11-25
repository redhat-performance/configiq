# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployment artifacts for an already-evaluated engine configuration.

The service renders a deployment only from limits the simulator evaluated. This
package turns one ``include=engine`` section into a pinned, render-checked
bundle for a deployment target, and refuses any target that cannot reproduce
the evaluated configuration.
"""

from __future__ import annotations

from .catalog import TARGETS, DeployTarget
from .engine import DeployUnsupportedRequest
from .models import (
    DeployArtifact,
    DeployCheck,
    DeployFile,
    DeployRequest,
    DeployTargetInfo,
    DeployTargetsResponse,
)

# The entry point is re-exported under a distinct name: binding it to `render`
# here would shadow the `render` submodule and make `deploy.render` ambiguous.
from .render import DeployRenderError
from .render import render as render_artifact

__all__ = [
    "TARGETS",
    "DeployArtifact",
    "DeployCheck",
    "DeployFile",
    "DeployRenderError",
    "DeployRequest",
    "DeployTarget",
    "DeployTargetInfo",
    "DeployTargetsResponse",
    "DeployUnsupportedRequest",
    "render_artifact",
]

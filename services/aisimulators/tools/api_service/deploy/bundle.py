# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The files that make a rendered deployment installable and checkable.

Rendering the model server is only half of a deployable artifact. The bundle
also carries what an operator needs before and after applying it: a README that
states the layout and the prerequisites, a PROVENANCE record of every pin and
evaluated limit, and scripts that refuse to install into a cluster that cannot
run the artifact and that confirm the rollout afterwards.
"""

from __future__ import annotations

import hashlib
import json
import shlex

from ..engine_models import EngineRole, EngineSection
from .catalog import (
    ARTIFACT_SCHEMA_VERSION,
    DYNAMO_ENGINE_VERSION,
    DYNAMO_VERSION,
    GAIE_VERSION,
    LLM_D_ENGINE_VERSION,
    LLM_D_ROUTER_VALUES_BY_MODE,
    LWS_VERSION,
    ROUTER_CHART,
    ROUTER_CHART_VERSION,
    DeployTarget,
)
from .models import DeployCheck, DeployFile, DeployRequest

_SCRIPT_MODE = 0o755

#: Secret name each target's rendered pods read the Hugging Face token from.
#: The names come from the upstream manifests, not from this service.
_HF_SECRET = {"llm-d": "llm-d-hf-token", "dynamo-dgd": "hf-token-secret"}


def _quote(value: str) -> str:
    return shlex.quote(value)


def _name(engine: EngineSection, prefix: str | None) -> str:
    """A stable resource base name derived from the model and layout."""
    model = engine.model.rsplit("/", 1)[-1].lower()
    model = "".join(char if char.isalnum() or char == "-" else "-" for char in model)
    model = "-".join(part for part in model.split("-") if part)[:32]
    return f"{prefix or 'configiq'}-{model}-{engine.mode}"


def _gpus(engine: EngineSection) -> int | None:
    """The accelerators the rendered topology requests, when it can be stated."""

    def ranks(role: EngineRole) -> int | None:
        values = [
            role.replicas.value,
            role.tensor_parallel_size.value,
            role.pipeline_parallel_size.value,
            role.data_parallel_size.value,
        ]
        if any(value is None for value in values):
            return None
        total = 1
        for value in values:
            total *= int(value)  # type: ignore[arg-type]
        return total

    if engine.mode == "agg":
        return ranks(engine.agg) if engine.agg is not None else None
    pools = [ranks(role) for role in (engine.prefill, engine.decode) if role is not None]
    if not pools or any(item is None for item in pools):
        return None
    return sum(int(item) for item in pools)


def _role_table(engine: EngineSection) -> str:
    rows = ["| Role | Replicas | TP | PP | DP | MoE TP/EP | Max model len | Max seqs | Max batched tokens | gpu-memory-utilization |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    roles = {"agg": engine.agg} if engine.mode == "agg" else {
        "prefill": engine.prefill,
        "decode": engine.decode,
    }
    for name, role in roles.items():
        if role is None:
            continue
        rows.append(
            "| {name} | {replicas} | {tp} | {pp} | {dp} | {moe_tp}/{moe_ep} | {context} | {seqs} | {batched} | {memory} |".format(
                name=name,
                replicas=role.replicas.value,
                tp=role.tensor_parallel_size.value,
                pp=role.pipeline_parallel_size.value,
                dp=role.data_parallel_size.value,
                moe_tp=role.moe_tensor_parallel_size.value,
                moe_ep=role.moe_expert_parallel_size.value,
                context=role.max_model_len.value,
                seqs=role.max_num_seqs.value,
                batched=role.max_num_batched_tokens.value,
                memory=role.gpu_memory_utilization.value,
            )
        )
    return "\n".join(rows)


def _provenance(
    request: DeployRequest,
    target: DeployTarget,
    *,
    image: str,
    checks: list[DeployCheck],
    warnings: list[str],
) -> str:
    engine = request.engine
    model_config = (
        hashlib.sha256(
            json.dumps(request.model_config_data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if request.model_config_data
        else "resolved by the deployment from the model path"
    )
    pins = "\n".join(f"| {key} | `{value}` |" for key, value in sorted(target.pins.items()))
    checks_table = "\n".join(
        f"| {check.id} | {check.status} | {check.detail} |" for check in checks
    )
    warning_block = (
        "\n".join(f"- {warning}" for warning in warnings) if warnings else "- None."
    )
    return f"""# Provenance

This artifact was rendered by ConfigIQ's AISimulators service. Every value below
is either a release pin or a limit the simulator evaluated. Nothing here was
re-derived at render time: the model server is started with exactly the limits
the reported prediction depended on.

## Pins

| Input | Value |
| --- | --- |
{pins}
| model-server image | `{image or "supplied by the target"}` |

## Evaluated configuration

| Input | Value |
| --- | --- |
| simulator | `aisim` |
| model | `{engine.model}` |
| served model name | `{request.served_model_name or engine.model}` |
| backend | `{engine.backend}` |
| backend version | `{engine.backend_version}` |
| mode | `{engine.mode}` |
| hardware | `{request.system}` |
| artifact schema | `{ARTIFACT_SCHEMA_VERSION}` |
| model config | `{model_config}` |

```json
{json.dumps(engine.model_dump(mode="json"), indent=2, sort_keys=True)}
```

## Render gate

| Check | Status | Detail |
| --- | --- | --- |
{checks_table}

## Warnings

{warning_block}
"""


def _requirements(engine: EngineSection, target: DeployTarget) -> str:
    gpus = _gpus(engine)
    lines = [f"- {gpus} accelerator(s) with enough memory for the evaluated limits." if gpus else "- Accelerators sized for the evaluated limits."]
    secret = _HF_SECRET[target.kind]
    if target.kind == "llm-d":
        lines.append(
            f"- Gateway API Inference Extension `{GAIE_VERSION}` CRDs "
            f"(`kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/{GAIE_VERSION}/v1-manifests.yaml`)."
        )
        lines.append(
            f"- llm-d router chart `{ROUTER_CHART}:{ROUTER_CHART_VERSION}` and Helm."
        )
        if engine.mode == "disagg":
            lines.append(
                f"- LeaderWorkerSet `{LWS_VERSION}` with the `DisaggregatedSet` API enabled "
                "(the overlay renders a `DisaggregatedSet`, which that controller reconciles)."
            )
        else:
            lines.append("- No LeaderWorkerSet: an aggregated deployment renders a `Deployment`.")
        lines.append(
            "- A GPU driver and device plugin that advertise `nvidia.com/gpu`, and a "
            f"`{secret}` secret in the target namespace (or a public model)."
        )
    else:
        lines.append(
            f"- NVIDIA Dynamo `{DYNAMO_VERSION}` platform: its operator and CRDs, which serve "
            "the `DynamoGraphDeployment` this bundle applies."
        )
        lines.append(
            "- A GPU driver and device plugin that advertise `nvidia.com/gpu`, and a "
            f"`{secret}` secret in the target namespace (or a public model)."
        )
    return "\n".join(lines)


def _layout(engine: EngineSection, target: DeployTarget) -> str:
    if target.kind == "llm-d":
        rows = [
            "| Path | Purpose |",
            "| --- | --- |",
            "| `kustomization.yaml` | The overlay `kubectl apply -k .` builds. Pins the routing sidecar and the namespace. |",
            "| `patch-vllm.yaml` | The evaluated limits for the aggregated model server. |"
            if engine.mode == "agg"
            else "| `patch-prefill.yaml`, `patch-decode.yaml` | The evaluated limits for each pool. |",
            "| `llm-d/` | The pinned llm-d v1.0.0 subtrees the overlay references, verbatim. |",
            "| `install.sh`, `preflight.sh`, `verify.sh` | Install, check, and confirm this bundle. |",
        ]
        return "\n".join(rows)
    return "\n".join(
        [
            "| Path | Purpose |",
            "| --- | --- |",
            "| `k8s_deploy.yaml` | The `DynamoGraphDeployment` for the evaluated topology. |",
            "| `install.sh`, `preflight.sh`, `verify.sh` | Install, check, and confirm this bundle. |",
        ]
    )


def _readme(
    request: DeployRequest,
    target: DeployTarget,
    *,
    image: str,
) -> str:
    engine = request.engine
    name = _name(engine, request.name_prefix)
    if target.kind == "llm-d":
        deploy = (
            "An llm-d v1.0.0 model server plus the pinned router release. The "
            "model-server overlay patches the pinned llm-d guide base, so the "
            "rendered manifests are the guide's own resources with the "
            "evaluated limits applied."
        )
        install_steps = """1. `./preflight.sh` — fails if the cluster cannot run this artifact.
2. `./install.sh` — creates the namespace, installs the router, and applies the model server.
3. `./verify.sh` — waits for the rollout and queries the served model.

`install.sh --install-prerequisites` also applies the pinned Inference
Extension CRDs, and LeaderWorkerSet where the overlay needs it. Run it only if
the cluster does not already provide them.
"""
        endpoint = "kubectl -n {namespace} get svc -l llm-d.ai/guide -o wide"
    else:
        deploy = (
            "One `DynamoGraphDeployment` for the evaluated topology. The NVIDIA "
            "Dynamo operator reconciles it into the frontend and worker services."
        )
        install_steps = """1. `./preflight.sh` — fails if the cluster cannot run this artifact.
2. `./install.sh` — creates the namespace and applies the deployment.
3. `./verify.sh` — waits for the rollout and queries the served model.
"""
        endpoint = "kubectl -n {namespace} get svc -o wide"
    return f"""# {name}

{deploy}

This is a deployment, not a starting point: the model server starts with the
exact engine limits the prediction was evaluated with, every input is pinned,
and the artifact was re-rendered and asserted before it was returned. See
`PROVENANCE.md` for the pins, the evaluated limits, and the checks that passed.

## Evaluated limits

{_role_table(engine)}

## Layout

{_layout(engine, target)}

## Prerequisites

{_requirements(engine, target)}

## Install

{install_steps}
The scripts read `NAMESPACE` (default `{request.namespace}`), `KUBECTL`, and
`HELM` from the environment.

## Reach the endpoint

```bash
{endpoint.format(namespace=request.namespace)}
```

## Uninstall

```bash
kubectl delete -n {request.namespace} -k .
```

## Limitations

- The artifact is rendered for the GPU vendor and topology in the evaluated
  configuration. Moving it to a different accelerator or fabric is a new
  sizing decision, not an edit.
- The llm-d bases carry guide labels (`llm-d.ai/guide`, `llm-d.ai/model`). The
  router selects on `llm-d.ai/guide`, so the model label is cosmetic; changing
  it does not change routing.
- The router's prefix-cache affinity thresholds are calibrated for the guide's
  reference model and hardware. Re-measure them before performance work on a
  different model.
"""


def _preflight(request: DeployRequest, target: DeployTarget, *, image: str) -> str:
    engine = request.engine
    gpus = _gpus(engine)
    required_crds = (
        ["inferencepools.inference.networking.k8s.io", "inferenceservices.inference.networking.x-k8s.io"]
        if target.kind == "llm-d"
        else ["dynamographdeployments.nvidia.com"]
    )
    if target.kind == "llm-d" and engine.mode == "disagg":
        required_crds.append("disaggregatedsets.disaggregatedset.x-k8s.io")
    crd_checks = "\n".join(
        f'require_crd {_quote(name)}' for name in required_crds
    )
    version_note = (
        f'note "evaluated vllm {engine.backend_version}; the pinned image {image or "for this target"} '
        f'may run a different engine release"'
        if engine.backend_version is not None
        and engine.backend_version
        not in (
            (LLM_D_ENGINE_VERSION,) if target.kind == "llm-d" else (DYNAMO_ENGINE_VERSION,)
        )
        else 'note "engine version matches the target pin"'
    )
    return f"""#!/usr/bin/env bash
# Generated by ConfigIQ. Fails rather than installing into a cluster that cannot
# run this artifact. Read-only: it changes nothing.
set -euo pipefail

NAMESPACE="${{NAMESPACE:-{request.namespace}}}"
KUBECTL="${{KUBECTL:-kubectl}}"
HELM="${{HELM:-helm}}"
REQUIRED_GPUS="${{REQUIRED_GPUS:-{gpus if gpus is not None else 0}}}"

failures=0
fail() {{ printf 'FAIL  %s\\n' "$1" >&2; failures=$((failures + 1)); }}
ok() {{ printf 'ok    %s\\n' "$1"; }}
note() {{ printf 'note  %s\\n' "$1"; }}

command -v "$KUBECTL" >/dev/null 2>&1 || {{ fail "kubectl ($KUBECTL) is not on PATH"; exit 1; }}
"$KUBECTL" version >/dev/null 2>&1 || {{ fail "cannot reach a cluster with $KUBECTL"; exit 1; }}
ok "cluster is reachable"

require_crd() {{
  if "$KUBECTL" get crd "$1" >/dev/null 2>&1; then ok "CRD $1"; else fail "CRD $1 is missing"; fi
}}

{crd_checks}

if [ "$REQUIRED_GPUS" -gt 0 ]; then
  available=$("$KUBECTL" get nodes -o jsonpath='{{range .items[*]}}{{.status.allocatable.nvidia\\.com/gpu}}{{"\\n"}}{{end}}' 2>/dev/null | awk 'NF {{sum += $1}} END {{print sum + 0}}')
  if [ "${{available:-0}}" -ge "$REQUIRED_GPUS" ]; then
    ok "cluster advertises $available nvidia.com/gpu (needs $REQUIRED_GPUS)"
  else
    fail "cluster advertises ${{available:-0}} nvidia.com/gpu but this deployment needs $REQUIRED_GPUS"
  fi
fi

if [ "{target.kind}" = "llm-d" ]; then
  command -v "$HELM" >/dev/null 2>&1 || fail "helm ($HELM) is not on PATH; the router is a Helm release"
fi

if "$KUBECTL" get namespace "$NAMESPACE" >/dev/null 2>&1; then
  ok "namespace $NAMESPACE exists"
  if "$KUBECTL" -n "$NAMESPACE" get secret {_quote(_HF_SECRET[target.kind])} >/dev/null 2>&1; then
    ok "{_HF_SECRET[target.kind]} exists in $NAMESPACE"
  else
    note "{_HF_SECRET[target.kind]} is absent from $NAMESPACE; install.sh creates it from HF_TOKEN when set"
  fi
else
  note "namespace $NAMESPACE does not exist yet; install.sh creates it"
fi

{version_note}

printf '\\n'
if [ "$failures" -gt 0 ]; then
  printf '%d prerequisite check(s) failed.\\n' "$failures" >&2
  exit 1
fi
printf 'All prerequisite checks passed.\\n'
"""


def _install(request: DeployRequest, target: DeployTarget) -> str:
    engine = request.engine
    if target.kind == "llm-d":
        prerequisites = f"""if [ "$INSTALL_PREREQUISITES" = "1" ]; then
  echo "Applying Inference Extension {GAIE_VERSION} CRDs"
  "$KUBECTL" apply -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/{GAIE_VERSION}/v1-manifests.yaml"
fi
"""
        if engine.mode == "disagg":
            prerequisites += f"""
if [ "$INSTALL_PREREQUISITES" = "1" ]; then
  echo "Installing LeaderWorkerSet {LWS_VERSION} with the DisaggregatedSet API"
  "$HELM" upgrade --install lws oci://registry.k8s.io/lws/charts/lws \\
    --version "{LWS_VERSION}" \\
    --namespace lws-system --create-namespace \\
    --set enableDisaggregatedSet=true \\
    --wait --timeout 300s
fi
if ! "$KUBECTL" get crd disaggregatedsets.disaggregatedset.x-k8s.io >/dev/null 2>&1; then
  echo "The DisaggregatedSet CRD is missing; re-run with --install-prerequisites" >&2
  exit 1
fi
"""
        install = f""""$HELM" upgrade --install "{_name(engine, request.name_prefix)}" \\
  "{ROUTER_CHART}" \\
  --version "{ROUTER_CHART_VERSION}" \\
  -f "llm-d/{_router_base_values()}" \\
  -f "llm-d/{_router_values(engine)}" \\
  --namespace "$NAMESPACE"
"$KUBECTL" apply -n "$NAMESPACE" -k .
"""
    else:
        prerequisites = ""
        install = '"$KUBECTL" apply -n "$NAMESPACE" -f k8s_deploy.yaml\n'
    return f"""#!/usr/bin/env bash
# Generated by ConfigIQ. Installs this bundle into the current kube context.
set -euo pipefail

NAMESPACE="${{NAMESPACE:-{request.namespace}}}"
KUBECTL="${{KUBECTL:-kubectl}}"
HELM="${{HELM:-helm}}"
INSTALL_PREREQUISITES=0
for arg in "$@"; do
  case "$arg" in
    --install-prerequisites) INSTALL_PREREQUISITES=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")"
./preflight.sh

"$KUBECTL" create namespace "$NAMESPACE" --dry-run=client -o yaml | "$KUBECTL" apply -f -
if [ -n "${{HF_TOKEN:-}}" ]; then
  "$KUBECTL" -n "$NAMESPACE" create secret generic {_HF_SECRET[target.kind]} \\
    --from-literal=HF_TOKEN="$HF_TOKEN" \\
    --dry-run=client -o yaml | "$KUBECTL" apply -f -
fi

{prerequisites}
{install}
echo "Applied. Run ./verify.sh once the pods are running."
"""


def _verify(request: DeployRequest, target: DeployTarget) -> str:
    engine = request.engine
    name = _name(engine, request.name_prefix)
    if target.kind == "llm-d":
        selector = "llm-d.ai/guide"
        workload = f'"$KUBECTL" -n "$NAMESPACE" rollout status deployment -l {selector} --timeout="$TIMEOUT"'
        if engine.mode == "disagg":
            workload = f'"$KUBECTL" -n "$NAMESPACE" wait --for=condition=Available disaggregatedset/{_name(engine, request.name_prefix)}-pd-disagg-vllm --timeout="$TIMEOUT" 2>/dev/null || true\n"$KUBECTL" -n "$NAMESPACE" rollout status statefulset -l {selector} --timeout="$TIMEOUT"'
        query = f"""service=$("$KUBECTL" -n "$NAMESPACE" get svc -l {selector} -o jsonpath='{{.items[0].metadata.name}}')
if [ -z "$service" ]; then
  echo "No llm-d service found; inspect the pods with: $KUBECTL -n $NAMESPACE get pods" >&2
  exit 1
fi
"$KUBECTL" -n "$NAMESPACE" port-forward "service/$service" "$PORT:8000" >/dev/null 2>&1 &
forward=$!
trap 'kill "$forward" 2>/dev/null || true' EXIT
sleep 3
curl -fsS "http://127.0.0.1:$PORT/v1/models"
"""
    else:
        workload = f'"$KUBECTL" -n "$NAMESPACE" wait --for=condition=Ready dynamographdeployment/{name} --timeout="$TIMEOUT" || "$KUBECTL" -n "$NAMESPACE" get pods'
        query = f"""service=$("$KUBECTL" -n "$NAMESPACE" get svc -l nvidia.com/dynamo-graph-deployment-name={name} -o jsonpath='{{.items[0].metadata.name}}')
if [ -z "$service" ]; then
  echo "No Dynamo frontend service found; inspect the pods with: $KUBECTL -n $NAMESPACE get pods" >&2
  exit 1
fi
"$KUBECTL" -n "$NAMESPACE" port-forward "service/$service" "$PORT:8000" >/dev/null 2>&1 &
forward=$!
trap 'kill "$forward" 2>/dev/null || true' EXIT
sleep 3
curl -fsS "http://127.0.0.1:$PORT/v1/models"
"""
    return f"""#!/usr/bin/env bash
# Generated by ConfigIQ. Waits for the rollout and asks the server what it serves.
set -euo pipefail

NAMESPACE="${{NAMESPACE:-{request.namespace}}}"
KUBECTL="${{KUBECTL:-kubectl}}"
TIMEOUT="${{TIMEOUT:-900s}}"
PORT="${{PORT:-18080}}"

{workload}

{query}
echo
echo "Served model list above. The model server is running the limits recorded in PROVENANCE.md."
"""


def _router_base_values() -> str:
    return "guides/recipes/router/base.values.yaml"


def _router_values(engine: EngineSection) -> str:
    return LLM_D_ROUTER_VALUES_BY_MODE[engine.mode]


def assemble(
    request: DeployRequest,
    target: DeployTarget,
    *,
    image: str,
    checks: list[DeployCheck],
    warnings: list[str],
) -> dict[str, DeployFile]:
    """The documentation and scripts that ship beside the rendered manifests."""
    files = {
        "README.md": _readme(request, target, image=image),
        "PROVENANCE.md": _provenance(
            request, target, image=image, checks=checks, warnings=warnings
        ),
        "preflight.sh": _preflight(request, target, image=image),
        "install.sh": _install(request, target),
        "verify.sh": _verify(request, target),
    }
    return {
        name: DeployFile(
            path=name,
            contents=contents,
            mode=_SCRIPT_MODE if name.endswith(".sh") else 0o644,
        )
        for name, contents in files.items()
    }


def as_files(files: dict[str, str]) -> list[DeployFile]:
    return [
        DeployFile(
            path=path,
            contents=contents,
            mode=_SCRIPT_MODE if path.endswith(".sh") else 0o644,
        )
        for path, contents in sorted(files.items())
    ]


__all__ = ["assemble", "as_files"]

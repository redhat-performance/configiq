# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render one deployment artifact and prove it assembles.

Rendering is not "write the files and hope": every artifact passes a render gate
before the service returns it. For a Kustomize target that gate runs the pinned
overlay (exactly what an installer runs) and asserts on the manifests it
produced; for a single-manifest target the assertions run on the manifest
directly. A target that cannot pass the gate is an explicit error, not a bundle
the caller discovers is broken after applying it.
"""

from __future__ import annotations

import functools
import json
import re
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from ..engine_models import EngineSection
from . import bundle
from . import engine as engine_lowering
from .catalog import (
    LLM_D_BASE_BY_MODE,
    LLM_D_SIDECAR_IMAGE,
    LLM_D_SIDECAR_IMAGE_NAME,
    LLM_D_VLLM_IMAGE,
    DeployTarget,
    kustomize_command,
    llm_d_vendor_dir,
)
from .models import DeployCheck, DeployFile, DeployRequest

#: Generous enough for a cold kustomize build, short enough that a wedged child
#: process cannot hold a worker thread until the request times out.
_RENDER_TIMEOUT_SECONDS = 120

#: A placeholder left in a rendered manifest means the upstream base expected an
#: overlay to replace it. Shipping one would deploy an image that does not exist.
_PLACEHOLDER = re.compile(r"REPLACE_[A-Z0-9_]+")

#: Flags that pin behaviour the prediction depended on. Each is asserted to
#: carry the evaluated value in the rendered argv.
_LIMIT_FLAGS = (
    ("--max-model-len", "max_model_len"),
    ("--max-num-seqs", "max_num_seqs"),
    ("--max-num-batched-tokens", "max_num_batched_tokens"),
    ("--tensor-parallel-size", "tensor_parallel_size"),
    ("--pipeline-parallel-size", "pipeline_parallel_size"),
)


class DeployRenderError(ValueError):
    """The artifact could not be rendered and verified."""


def _as_dicts(text: str) -> list[dict[str, Any]]:
    return [doc for doc in yaml.safe_load_all(text) if isinstance(doc, dict)]


def _containers_from_pod(pod: dict[str, Any]) -> list[dict[str, Any]]:
    """Every container in a pod, including the init containers.

    The routing sidecar runs as an init container with ``restartPolicy: Always``,
    so leaving init containers out would hide the image the bundle pins.
    """
    containers = list(pod.get("initContainers") or []) + list(pod.get("containers") or [])
    return [item for item in containers if isinstance(item, dict)]


def _roles(docs: list[dict[str, Any]]) -> list[tuple[str, int | None, list[dict[str, Any]]]]:
    """Read a rendered manifest as (role, replicas, containers) triples.

    DGD services, a Deployment, and a DisaggregatedSet all describe roles
    differently, so each shape is read explicitly rather than by guessing at
    nested structures.
    """
    roles: list[tuple[str, int | None, list[dict[str, Any]]]] = []
    for doc in docs:
        kind = doc.get("kind")
        if kind == "DynamoGraphDeployment":
            for name, service in ((doc.get("spec") or {}).get("services") or {}).items():
                if not isinstance(service, dict):
                    continue
                container = (service.get("extraPodSpec") or {}).get("mainContainer")
                roles.append(
                    (str(name), service.get("replicas"), [container] if isinstance(container, dict) else [])
                )
        elif kind == "Deployment":
            spec = doc.get("spec") or {}
            pod = ((spec.get("template") or {}).get("spec")) or {}
            roles.append(("agg", spec.get("replicas"), _containers_from_pod(pod)))
        elif kind == "DisaggregatedSet":
            for role in (doc.get("spec") or {}).get("roles") or []:
                if not isinstance(role, dict):
                    continue
                spec = role.get("spec") or {}
                worker = ((spec.get("leaderWorkerTemplate") or {}).get("workerTemplate")) or {}
                roles.append(
                    (str(role.get("name")), spec.get("replicas"), _containers_from_pod(worker.get("spec") or {}))
                )
    return roles


def _flag(args: list[Any], name: str) -> str | None:
    """The value of ``--name`` in an argv list, or None when absent."""
    for index, token in enumerate(args):
        if str(token) != name:
            continue
        return str(args[index + 1]) if index + 1 < len(args) else None
    return None


#: Tokens only an inference engine's argv carries. A frontend (``--http-port``)
#: and the routing sidecar (``--kv-connector``) carry neither.
_ENGINE_TOKENS = ("--model", "--tensor-parallel-size", "--max-model-len")


def _is_model_server(container: dict[str, Any]) -> bool:
    if str(container.get("name", "")) == "modelserver":
        return True
    args = container.get("args")
    return isinstance(args, list) and any(str(token) in _ENGINE_TOKENS for token in args)


def _model_server(containers: list[dict[str, Any]]) -> dict[str, Any] | None:
    for container in containers:
        if _is_model_server(container):
            return container
    return None


def _match_role(
    roles: list[tuple[str, int | None, list[dict[str, Any]]]],
    role_name: str,
) -> tuple[str, int | None, list[dict[str, Any]]] | None:
    """Find the rendered role that serves ``role_name``.

    DGD prefixes its worker service names (``VllmPrefillWorker``), so a role is
    matched by name first and then by the engine containers it runs. An
    aggregated deployment has exactly one engine role, so it needs no naming
    convention at all.
    """
    for role in roles:
        if role[0].lower() == role_name:
            return role
    for role in roles:
        if role_name in role[0].lower():
            return role
    engine_roles = [role for role in roles if _model_server(role[2]) is not None]
    if role_name == "agg" and len(engine_roles) == 1:
        return engine_roles[0]
    return None


def _assert_engine_limits(
    roles: list[tuple[str, int | None, list[dict[str, Any]]]],
    engine: EngineSection,
) -> list[str]:
    """Every evaluated limit that changes behaviour must appear in the argv."""
    failures: list[str] = []
    for role_name, expected in engine_lowering.roles_for_mode(engine).items():
        actual = _match_role(roles, role_name)
        if actual is None:
            failures.append(f"{role_name}: no rendered role")
            continue
        container = _model_server(actual[2])
        if container is None:
            failures.append(f"{role_name}: no model-server container")
            continue
        args = container.get("args") or []
        for flag, field_name in _LIMIT_FLAGS:
            value = getattr(expected, field_name).value
            if value is None:
                continue
            rendered = _flag(args, flag)
            if rendered != str(value):
                failures.append(f"{role_name}: {flag} is {rendered!r}, evaluated {value!r}")
        if expected.replicas.value is not None and actual[1] != expected.replicas.value:
            failures.append(
                f"{role_name}: replicas is {actual[1]!r}, evaluated {expected.replicas.value!r}"
            )
    return failures


def _assert_images(
    roles: list[tuple[str, int | None, list[dict[str, Any]]]],
    *,
    expected_model_server: str,
    expect_sidecar: bool,
) -> list[str]:
    images = [
        str(container.get("image"))
        for _, _, containers in roles
        for container in containers
        if container.get("image")
    ]
    failures: list[str] = []
    if not images:
        failures.append("no container images rendered")
    if expected_model_server not in images:
        failures.append(
            f"model-server image {expected_model_server!r} is not among {sorted(set(images))}"
        )
    if expect_sidecar and LLM_D_SIDECAR_IMAGE not in images:
        failures.append(
            f"routing sidecar {LLM_D_SIDECAR_IMAGE!r} is not among {sorted(set(images))}"
        )
    return failures


def _assert_no_placeholders(rendered: str) -> list[str]:
    found = sorted(set(_PLACEHOLDER.findall(rendered)))
    return [f"unreplaced placeholder(s) {', '.join(found)}"] if found else []


def _run_kustomize(directory: Path) -> tuple[str | None, str | None]:
    command = kustomize_command()
    if command is None:
        return None, "no kustomize binary is available to run the render gate"
    try:
        completed = subprocess.run(
            command + [str(directory)],
            capture_output=True,
            text=True,
            timeout=_RENDER_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"kustomize build timed out after {_RENDER_TIMEOUT_SECONDS}s"
    except OSError as exc:
        return None, f"kustomize build could not start: {exc}"
    if completed.returncode != 0:
        lines = (completed.stderr or completed.stdout or "").strip().splitlines()
        detail = lines[-1] if lines else f"exit status {completed.returncode}"
        return None, f"kustomize build failed: {detail}"
    return completed.stdout, None


def _kustomization_closure(start: Path, root: Path) -> set[str]:
    """Every file Kustomize reads when it builds ``start``.

    The bundle carries only what it references, so the vendored tree stays small
    and a file the overlay needs but nobody copied fails the render gate instead
    of silently shipping.
    """
    seen: set[str] = set()
    visited: set[Path] = set()

    def walk(directory: Path) -> None:
        directory = directory.resolve()
        if directory in visited or not directory.is_dir():
            return
        visited.add(directory)
        manifest = next(
            (
                directory / candidate
                for candidate in ("kustomization.yaml", "kustomization.yml", "Kustomization")
                if (directory / candidate).is_file()
            ),
            None,
        )
        if manifest is None:
            return
        document = yaml.safe_load(manifest.read_text()) or {}
        references: list[str] = []
        for key in (
            "resources",
            "components",
            "bases",
            "crds",
            "configurations",
            "transformers",
            "generators",
        ):
            references.extend(
                item
                for item in document.get(key) or []
                if isinstance(item, str) and not re.match(r"^(https?:|git@|github\.com)", item)
            )
        for key in ("patchesStrategicMerge", "patchesJson6902"):
            for patch in document.get(key) or []:
                if isinstance(patch, str):
                    references.append(patch)
                elif isinstance(patch, dict) and patch.get("path"):
                    references.append(str(patch["path"]))
        for patch in document.get("patches") or []:
            if isinstance(patch, dict) and patch.get("path"):
                references.append(str(patch["path"]))
            elif isinstance(patch, str):
                references.append(patch)
        for generator in ("configMapGenerator", "secretGenerator"):
            for item in document.get(generator) or []:
                for entry in (item.get("files") or []) + (item.get("envs") or []):
                    references.append(str(entry).split("=", 1)[-1])
        for reference in references:
            target = (directory / reference).resolve()
            try:
                relative = target.relative_to(root)
            except ValueError as exc:
                raise DeployRenderError(
                    f"{manifest.relative_to(root)} references {reference!r} outside the pinned tree"
                ) from exc
            if target.is_dir():
                walk(target)
                continue
            if not target.exists():
                raise DeployRenderError(
                    f"{manifest.relative_to(root)} references missing file {reference!r}"
                )
            seen.add(relative.as_posix())
        seen.add(manifest.relative_to(root).as_posix())

    walk(start)
    return seen


@functools.lru_cache(maxsize=4)
def vendored_closure(root: str, mode: str) -> dict[str, str]:
    """The readable files the overlay for ``mode`` needs, keyed by bundle path.

    Only the requested layout's base is carried, so the bundle holds what it
    references and nothing else. The render gate builds the bundle from exactly
    these files, so a missing dependency fails here rather than at install time.
    """
    base = Path(root)
    if not base.is_dir():
        raise DeployRenderError(
            f"the pinned llm-d source tree is missing ({base}); this deployment was "
            "built without it and cannot render an llm-d target"
        )
    files: dict[str, str] = {}
    for relative in sorted(_kustomization_closure(base / LLM_D_BASE_BY_MODE[mode], base)):
        try:
            files[f"llm-d/{relative}"] = (base / relative).read_text()
        except (OSError, UnicodeDecodeError) as exc:
            raise DeployRenderError(f"could not read vendored {relative}: {exc}") from exc
    return files


def _post_process_kustomization(contents: str, namespace: str, mode: str) -> str:
    """Pin the sidecar tag and the namespace the bundle installs into."""
    kustomization = yaml.safe_load(contents) or {}
    kustomization.setdefault("namespace", namespace)
    if mode == "disagg":
        images = list(kustomization.get("images") or [])
        if not any(
            isinstance(item, dict) and item.get("name") == LLM_D_SIDECAR_IMAGE_NAME
            for item in images
        ):
            # The P/D base pulls the sidecar from the ``nightly`` component,
            # which tracks a moving tag. A deployable artifact names the release.
            images.append(
                {
                    "name": LLM_D_SIDECAR_IMAGE_NAME,
                    "newName": LLM_D_SIDECAR_IMAGE_NAME,
                    "newTag": LLM_D_SIDECAR_IMAGE.rpartition(":")[2],
                }
            )
        kustomization["images"] = images
    return yaml.safe_dump(kustomization, sort_keys=False)


def _generator_files(request: DeployRequest, target: DeployTarget, model_path: str) -> dict[str, str]:
    """Render the model-server artifacts for one evaluated engine section."""
    from aisimulate.generator.api import generate_from_request
    from aisimulate.generator.request import from_sweeper_candidate

    engine = request.engine
    config = engine_lowering.candidate_config(
        engine,
        system=request.system,
        concurrency=request.concurrency,
    )
    overrides: dict[str, Any] = {}
    if target.kind == "llm-d":
        overrides = {
            "LlmdConfig.vllm_image": request.image or LLM_D_VLLM_IMAGE,
            "LlmdConfig.kustomize_base_path": f"llm-d/{LLM_D_BASE_BY_MODE[engine.mode]}",
            "LlmdConfig.kustomize_layout": (
                "disaggregatedset" if engine.mode == "disagg" else "deployment"
            ),
            "LlmdConfig.kustomize_disaggregatedset_name": "pd-disagg-vllm",
            "LlmdConfig.kustomize_name_prefix": request.name_prefix or "configiq-",
        }
    elif request.image is not None:
        overrides["K8sConfig.k8s_image"] = request.image

    generator_request = from_sweeper_candidate(
        {"config": config},
        workload=(
            {"isl": request.isl, "osl": request.osl} if request.isl is not None else None
        ),
        deployment_target=str(target.pins["deployment_target"]),
        generator_overrides=overrides,
        model_facts=None,
    )
    if request.served_model_name:
        generator_request = replace(
            generator_request,
            model=replace(generator_request.model, served_model_name=request.served_model_name),
        )
    return generate_from_request(generator_request)


def _deployable_subset(rendered: dict[str, str], target: DeployTarget) -> dict[str, str]:
    keep = (
        {"kustomization.yaml", "patch-vllm.yaml", "patch-decode.yaml", "patch-prefill.yaml"}
        if target.kind == "llm-d"
        else {"k8s_deploy.yaml"}
    )
    return {name: contents for name, contents in rendered.items() if name in keep}


def _resolved_image(
    request: DeployRequest,
    target: DeployTarget,
    generated: dict[str, str],
) -> str:
    """The model-server image the artifact actually runs."""
    if request.image:
        return request.image
    if target.kind == "llm-d":
        return LLM_D_VLLM_IMAGE
    for container in _model_servers_from_text(next(iter(generated.values()))):
        image = container.get("image")
        if image:
            return str(image)
    return ""


def _model_servers_from_text(contents: str) -> list[dict[str, Any]]:
    return [
        container
        for _, _, containers in _roles(_as_dicts(contents))
        for container in containers
        if _is_model_server(container) and container.get("image")
    ]


def _write(files: dict[str, str], root: Path) -> None:
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)


class _model_context:
    """Resolve the model the way the evaluation did, including a supplied config."""

    def __init__(self, request: DeployRequest):
        self._request = request
        self._temporary: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> str:
        config = self._request.model_config_data
        if not config:
            return self._request.engine.model
        self._temporary = tempfile.TemporaryDirectory(prefix="configiq-deploy-model-")
        (Path(self._temporary.name) / "config.json").write_text(
            json.dumps(config, sort_keys=True)
        )
        return self._temporary.name

    def __exit__(self, *_: object) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()


def _check(check_id: str, failures: list[str], success: str, *, assertions: int) -> DeployCheck:
    if failures:
        return DeployCheck(
            id=check_id,
            status="failed",
            detail="; ".join(failures),
        )
    return DeployCheck(
        id=check_id,
        status="passed",
        detail=f"{success} ({assertions} assertion(s))",
    )


def render(
    request: DeployRequest,
    target: DeployTarget,
) -> tuple[list[DeployFile], list[DeployCheck], list[str]]:
    """Render, gate, and assemble one deployment artifact.

    Returns the bundle files, the checks the gate ran, and any warnings worth
    carrying into the artifact's provenance.
    """
    engine = request.engine
    # Refuse a target that cannot reproduce the evaluated configuration before
    # any work is done, so the error names the gap instead of a render symptom.
    engine_lowering.validate(
        engine, target, allow_version_mismatch=request.allow_engine_version_mismatch
    )
    warnings: list[str] = []
    is_kustomize = target.kind == "llm-d"

    with _model_context(request) as model_path:
        rendered = _generator_files(request, target, model_path)
    generated = _deployable_subset(rendered, target)
    if not generated:
        raise DeployRenderError(
            f"the generator produced no deployable file for target {target.id!r}; "
            f"it produced {sorted(rendered)}"
        )
    omitted = sorted(set(rendered) - set(generated))
    if omitted:
        warnings.append(
            "The generator also produced " + ", ".join(omitted) + ", which reproduce the "
            "measurement rather than deploy it; they are not part of this bundle."
        )
    image = _resolved_image(request, target, generated)

    files = dict(generated)
    if is_kustomize:
        files.update(vendored_closure(str(llm_d_vendor_dir()), engine.mode))
        files["kustomization.yaml"] = _post_process_kustomization(
            files["kustomization.yaml"], request.namespace, engine.mode
        )

    checks: list[DeployCheck] = []
    with tempfile.TemporaryDirectory(prefix="configiq-deploy-") as directory:
        root = Path(directory)
        _write(files, root)
        if is_kustomize:
            output, error = _run_kustomize(root)
            if error is not None:
                raise DeployRenderError(error)
            checks.append(
                DeployCheck(
                    id="kustomize-build",
                    status="passed",
                    detail="kustomize build assembled the overlay against the pinned tree",
                )
            )
        else:
            output = files["k8s_deploy.yaml"]
            checks.append(
                DeployCheck(
                    id="kustomize-build",
                    status="skipped",
                    detail="this target is a single manifest; the checks below read it directly",
                )
            )
    assert output is not None
    try:
        docs = _as_dicts(output)
    except yaml.YAMLError as exc:
        raise DeployRenderError(f"the render gate produced unparsable YAML: {exc}") from exc
    if not docs:
        raise DeployRenderError("the render gate produced no Kubernetes resources")

    roles = _roles(docs)
    kinds = sorted({str(doc.get("kind")) for doc in docs})
    required = (
        ["DisaggregatedSet"]
        if is_kustomize and engine.mode == "disagg"
        else ["Deployment"]
        if is_kustomize
        else ["DynamoGraphDeployment"]
    )
    checks.append(
        _check(
            "resources",
            [f"expected {kind}, rendered {kinds}" for kind in required if kind not in kinds],
            f"rendered {', '.join(kinds)}",
            assertions=len(required),
        )
    )

    checks.append(
        _check(
            "engine-limits",
            _assert_engine_limits(roles, engine),
            "the rendered argv and replica counts carry the evaluated limits",
            assertions=len(engine_lowering.REQUIRED_ROLE_FIELDS)
            * len(engine_lowering.roles_for_mode(engine)),
        )
    )

    checks.append(
        _check(
            "images",
            _assert_images(
                roles,
                expected_model_server=image,
                expect_sidecar=is_kustomize and engine.mode == "disagg",
            ),
            f"the model server runs {image}",
            assertions=2,
        )
    )

    checks.append(
        _check(
            "placeholders",
            _assert_no_placeholders(output),
            "no unreplaced upstream placeholder survives",
            assertions=1,
        )
    )

    failed = [check for check in checks if check.status == "failed"]
    if failed:
        raise DeployRenderError(
            "the render gate rejected the artifact: "
            + "; ".join(f"{check.id}: {check.detail}" for check in failed)
        )

    if (
        request.image is not None
        and engine.backend_version is not None
        and engine.backend_version not in target.engine_versions
        and request.allow_engine_version_mismatch
    ):
        warnings.append(
            f"The evaluated run used {engine.backend} {engine.backend_version}, and the "
            f"requested image {request.image!r} replaces the target's pinned "
            f"{', '.join(target.engine_versions)}. preflight.sh prints this mismatch; "
            "the rendered flags are the evaluated ones."
        )

    documentation = bundle.assemble(
        request, target, image=image, checks=checks, warnings=warnings
    )
    all_files = bundle.as_files(files) + list(documentation.values())
    return sorted(all_files, key=lambda item: item.path), checks, warnings


__all__ = ["DeployRenderError", "render", "vendored_closure"]

"""Turn a resolved configuration into runnable artifacts.

The artifacts are the point of the replay record: a reader should be able to
copy one command, paste it on a machine with the simulator installed, and get
the same run. That means emitting the input file the simulator actually reads,
not a summary of the request ConfigIQ received.

Nothing here imports the simulators. Callers pass plain dictionaries, so the
artifact builder can be tested without an installed SDK.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass

import yaml

from .record import ReplayFile, ReplayInvocation

#: Where a supplied model config is written inside the bundle. The YAML points
#: at it relatively, so the directory can move without editing the file.
MODEL_DIRECTORY = "model"
MODEL_CONFIG_PATH = f"{MODEL_DIRECTORY}/config.json"


@dataclass(frozen=True)
class ModelInput:
    """How the emitted YAML should name the model."""

    #: Value written to ``engine.model``.
    reference: str
    files: tuple[ReplayFile, ...]

    @property
    def relative_to_bundle(self) -> bool:
        return self.reference.startswith("./")


def model_input(model_path: str, model_config: dict | None) -> ModelInput:
    """Point the YAML at a shipped config.json when the caller supplied one.

    The service validates predictions against a temporary directory holding
    that file, so the recorded config would otherwise name a path that no
    longer exists. Shipping the file and referring to it relatively is what
    makes the command reproducible after the temp directory is gone.
    """
    if not model_config:
        return ModelInput(reference=model_path, files=())
    return ModelInput(
        reference=f"./{MODEL_DIRECTORY}",
        files=(ReplayFile(MODEL_CONFIG_PATH, json.dumps(model_config, indent=2)),),
    )


def slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
    return slug or "run"


def _dump(document: dict) -> str:
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=False)


def _with_model(document: dict, model: ModelInput) -> dict:
    """Return a copy of the config document naming the model the bundle ships."""
    copied = json.loads(json.dumps(document))
    engine = copied.get("engine")
    if isinstance(engine, dict) and model.relative_to_bundle:
        engine["model"] = model.reference
    return copied


def _filename(invocation_id: str) -> str:
    return f"{invocation_id}.yaml"


def prediction_invocation(
    document: dict,
    *,
    id: str,
    label: str,
    model: ModelInput,
    include_memory_detail: bool = False,
    group: str | None = None,
    candidate_id: str | None = None,
    notes: tuple[str, ...] = (),
) -> ReplayInvocation:
    """Artifacts for one single-point prediction."""
    filename = _filename(id)
    command = ["aisimulate", "predict", "-c", filename, "--format", "json"]
    if include_memory_detail:
        command.extend(["--detail", "memory"])

    invocation_notes = list(notes)
    invocation_notes.append(
        "Equivalent CLI invocation: the service calls the aisimulate Python API "
        "directly, so this command repeats the same work rather than the same "
        "process."
    )
    return ReplayInvocation(
        id=id,
        label=label,
        command=shlex.join(command),
        execution="in_process",
        files=[ReplayFile(filename, _dump(_with_model(document, model))), *model.files],
        group=group,
        candidate_id=candidate_id,
        notes=invocation_notes,
    )


def recommendation_invocation(
    document: dict,
    *,
    id: str,
    label: str,
    model: ModelInput,
    group: str | None = None,
    notes: tuple[str, ...] = (),
) -> ReplayInvocation:
    """Artifacts for one recommendation search window."""
    filename = _filename(id)
    invocation_notes = list(notes)
    invocation_notes.append(
        "Equivalent CLI invocation: the service calls the aisimulate Python API "
        "directly. This config carries the window's own trial budget, so a "
        "re-run explores the same range with the same sample size."
    )
    return ReplayInvocation(
        id=id,
        label=label,
        command=shlex.join(
            ["aisimulate", "recommend", "-c", filename, "--format", "json"]
        ),
        execution="in_process",
        files=[ReplayFile(filename, _dump(_with_model(document, model))), *model.files],
        group=group,
        notes=invocation_notes,
    )


def _render_command(
    argv: list[str],
    *,
    variables: dict[str, str] | None = None,
    literals: dict[str, str] | None = None,
) -> str:
    rendered: list[str] = []
    for token in argv:
        if variables and token in variables:
            rendered.append(f'"${variables[token]}"')
        elif literals and token in literals:
            rendered.append(shlex.quote(literals[token]))
        else:
            rendered.append(shlex.quote(token))
    return " ".join(rendered)


def blis_run_invocation(
    *,
    id: str,
    label: str,
    argv: list[str],
    files: dict[str, str],
    binary: str,
    scenario_directory: str,
    group: str | None = None,
    candidate_id: str | None = None,
    notes: tuple[str, ...] = (),
) -> ReplayInvocation:
    """Artifacts for one BLIS subprocess, captured while its files exist.

    The command is rewritten to read its binary from ``BLIS_BINARY`` and its
    scenarios from the bundle directory, so it does not name this container's
    layout. BLIS reads its catalog and coefficient registry from the binary
    itself, so no data path is rewritten.
    """
    command = _render_command(
        argv,
        variables={binary: "BLIS_BINARY"},
        literals={scenario_directory: "."},
    )
    invocation_notes = list(notes)
    invocation_notes.append(
        "Exact subprocess invocation: these are the scenario files ConfigIQ "
        "wrote and the command it ran."
    )
    return ReplayInvocation(
        id=id,
        label=label,
        command=command,
        execution="subprocess",
        files=[
            ReplayFile(path, contents)
            for path, contents in sorted(files.items())
        ],
        group=group,
        candidate_id=candidate_id,
        notes=invocation_notes,
    )


def memory_snippet_invocation(
    *,
    id: str,
    label: str,
    kwargs: dict,
    model: ModelInput,
    group: str | None = None,
    notes: tuple[str, ...] = (),
) -> ReplayInvocation:
    """Artifacts for the memory estimate, which has no CLI verb."""
    arguments = ",\n".join(
        f"    {name}={value!r}" for name, value in sorted(kwargs.items())
    )
    script = (
        "from aisimulate_core.sdk.memory import estimate_kv_cache\n\n"
        "result = estimate_kv_cache(\n"
        f"{arguments},\n"
        ")\n"
        "print(result)\n"
    )
    filename = f"{id}.py"
    invocation_notes = list(notes)
    invocation_notes.append(
        "Computed in process by the memory model; aisimulate has no CLI verb "
        "for it, so this snippet is the replay."
    )
    return ReplayInvocation(
        id=id,
        label=label,
        command=f"python {shlex.quote(filename)}",
        execution="in_process",
        files=[ReplayFile(filename, script), *model.files],
        group=group,
        notes=invocation_notes,
    )


def run_script(
    invocations: list[ReplayInvocation],
    *,
    header: tuple[str, ...] = (),
    environment: tuple[str, ...] = (),
) -> str | None:
    """Assemble one script running every invocation in order.

    Candidate artifacts are left out: a Pareto sweep records one prediction per
    candidate, and running all of them is not the sweep. Each candidate's
    command is copied from the candidate itself.
    """
    scripted = [invocation for invocation in invocations if invocation.candidate_id is None]
    if not scripted:
        return None
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
    ]
    for name in environment:
        lines.append(f': "${{{name}:?set {name} to the matching runtime path}}"')
    for line in header:
        lines.append(f"# {line}")
    for invocation in scripted:
        lines.append("")
        lines.append(f"# {invocation.label}")
        lines.append(invocation.command)
    return "\n".join(lines) + "\n"

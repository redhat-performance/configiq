"""Run BLIS and parse its Result document."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

from .runtime import BlisRuntime


class BlisExecutionError(RuntimeError):
    """BLIS failed, timed out, or returned output outside its API contract."""


class BlisTimeoutError(BlisExecutionError):
    """A BLIS run exceeded its configured timeout."""


@dataclass(frozen=True)
class RunRequest:
    scenario: str
    scenario_directory: str
    isl: int
    osl: int
    # Closed-loop load. Mutually exclusive with `rate`; BLIS refuses a command
    # that sets both.
    concurrency: int | None
    num_requests: int
    # Open-loop arrival rate (requests/second). When set, BLIS generates the
    # `num_requests` slice at a constant interval of `1 / rate`.
    rate: float | None = None
    prefix_tokens: int = 0
    prefill_instances: int = 0
    decode_instances: int = 0


def build_run_command(runtime: BlisRuntime, request: RunRequest) -> list[str]:
    """Build argv only from resolved image paths and validated numeric inputs."""
    if (request.rate is None) == (request.concurrency is None):
        raise ValueError("BLIS runs require exactly one of a request rate or a concurrency")
    command = [
        runtime.binary,
        "run",
        "--catalog",
        runtime.catalog,
        "--registry",
        runtime.registry,
        "--scenarios",
        request.scenario_directory,
        "--scenario",
        request.scenario,
        "--prompt-tokens",
        str(request.isl),
        "--output-tokens",
        str(request.osl),
        "--num-requests",
        str(request.num_requests),
    ]
    if request.rate is not None:
        # BLIS rejects --concurrency alongside --rate, so open-loop runs send
        # only the arrival rate.
        command.extend(["--rate", f"{request.rate:g}"])
    elif request.concurrency is not None:
        command.extend(["--concurrency", str(request.concurrency)])
    if request.prefix_tokens > 0:
        command.extend(["--prefix-tokens", str(request.prefix_tokens)])
    if request.prefill_instances or request.decode_instances:
        if request.prefill_instances < 1 or request.decode_instances < 1:
            raise ValueError("P/D runs require at least one prefill and one decode instance")
        command.extend([
            "--num-instances", str(request.prefill_instances + request.decode_instances),
            "--prefill-instances", str(request.prefill_instances),
            "--decode-instances", str(request.decode_instances),
        ])
    return command


def parse_result(stdout: str) -> dict[str, Any]:
    """Return the last `kind: Result` document BLIS wrote to stdout.

    BLIS v0.15 replaced the human-readable metrics block with a single
    schematized Result document. Diagnostics go to stderr, but scan the stream
    for a Result rather than assuming stdout contains nothing else, so a tool
    that prints a banner cannot break parsing.
    """
    decoder = json.JSONDecoder()
    results: list[dict[str, Any]] = []
    position = 0
    while True:
        start = stdout.find("{", position)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(stdout, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        position = end
        if isinstance(value, dict) and value.get("kind") == "Result":
            results.append(value)
    if not results:
        raise BlisExecutionError("BLIS output did not contain a Result document")
    return results[-1]


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten a Result into the metric keys the service reasons about.

    BLIS reports rates over its measured window. Fall back to the simulated
    duration only when the window is absent, matching the previous contract.
    """
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    requests = result.get("requests") if isinstance(result.get("requests"), dict) else {}
    engine = result.get("engine") if isinstance(result.get("engine"), dict) else {}
    cluster = engine.get("cluster") if isinstance(engine.get("cluster"), dict) else {}
    window = result.get("window") if isinstance(result.get("window"), dict) else {}

    duration = window.get("duration_s")
    if not isinstance(duration, (int, float)) or duration <= 0:
        duration = cluster.get("simulated_duration_s")

    def number(source: dict[str, Any], key: str) -> float | int | None:
        value = source.get(key)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    return {
        "ttft_mean_ms": number(metrics, "ttft_mean_ms"),
        "itl_mean_ms": number(metrics, "itl_mean_ms"),
        "e2e_mean_ms": number(metrics, "e2e_mean_ms"),
        "tokens_per_sec": number(metrics, "output_tok_s"),
        "responses_per_sec": number(metrics, "request_throughput"),
        "completed_requests": number(requests, "completed"),
        "dropped_unservable": number(requests, "dropped_unservable") or 0,
        "vllm_estimated_duration_s": duration,
        # The smallest instance's GPU KV pool in tokens. BLIS reports this in
        # the engine cluster section; capacity sizing reads it instead of
        # re-deriving KV bytes from weights and memory fractions.
        "kv_capacity_tokens_per_replica": number(cluster, "kv_capacity_tokens_per_replica"),
    }


def run_blis(
    runtime: BlisRuntime,
    request: RunRequest,
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    command = build_run_command(runtime, request)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=os.environ.get("BLIS_HOME") or None,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BlisTimeoutError(f"BLIS simulation timed out after {timeout_seconds:g}s") from exc
    except OSError as exc:
        raise BlisExecutionError(f"could not start BLIS: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        raise BlisExecutionError(f"BLIS exited with status {completed.returncode}: {detail or 'no diagnostics'}")
    return parse_result(completed.stdout)

"""Read the engines a BLIS build models, as the build itself reports them.

The build's own engine snapshot (for example ``0.34.0+0fda86a35419``) is the
upstream tree the rules pack was generated from. It is not the release whose
serving behaviour the kernel prices, and it is not a release a user can match
benchmarked results against. ``blis version --json`` reports the modelled release
beside the build snapshot; a build that predates that flag reports nothing here, and
callers must show the version as unreported rather than substitute the snapshot.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlisEngine:
    """One engine kind a BLIS build can run."""

    kind: str
    #: The upstream release whose serving behaviour the kernel prices.
    modeled_version: str | None
    #: The upstream commit of ``modeled_version``, when the build names it.
    modeled_commit: str | None
    #: The release the rules pack was generated from. Provenance only.
    build_version: str | None
    #: Serving roles the build models for this kind.
    roles: tuple[str, ...]


def _string(source: dict[str, Any], key: str) -> str | None:
    value = source.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def parse_engines(document: Any) -> tuple[BlisEngine, ...]:
    """Parse an ``Engines`` document into the engine kinds it lists."""
    if not isinstance(document, dict):
        return ()
    entries = document.get("engines")
    if not isinstance(entries, list):
        return ()
    engines: list[BlisEngine] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        kind = _string(entry, "kind")
        if not kind:
            continue
        roles = entry.get("roles")
        engines.append(
            BlisEngine(
                kind=kind,
                modeled_version=_string(entry, "modeled_version"),
                modeled_commit=_string(entry, "modeled_commit"),
                build_version=_string(entry, "build_version"),
                roles=tuple(
                    role for role in roles if isinstance(role, str)
                ) if isinstance(roles, list) else (),
            )
        )
    return tuple(engines)


def read_engines(binary: str, *, timeout_seconds: float = 10.0) -> tuple[BlisEngine, ...]:
    """Ask the pinned build which engines it models.

    ``version`` exists on every supported build; one that predates ``--json``
    refuses the flag. Either refusal is reported as no engines, not as an error, so
    the caller can say the version is unreported instead of failing the catalog
    endpoint.
    """
    try:
        completed = subprocess.run(
            [binary, "version", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("could not ask BLIS for its engines: %s", exc)
        return ()
    if completed.returncode != 0:
        logger.info(
            "BLIS did not report its engines (exit %s): %s",
            completed.returncode,
            (completed.stderr or "").strip()[:400],
        )
        return ()
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError:
        logger.warning("BLIS version output was not JSON")
        return ()
    return parse_engines(document)


#: How long a failed probe is remembered before it is retried, in seconds.
#:
#: A probe fails for reasons that pass: a timeout while the host is loaded, an
#: exec that runs out of memory across ten replicas starting at once. Remembering
#: that failure would turn it into a permanent claim that the build models no
#: engine, so failures expire and the next request asks again.
PROBE_RETRY_SECONDS = 30.0

_ENGINES_BY_BINARY: dict[str, tuple[BlisEngine, ...]] = {}
_RETRY_AFTER: dict[str, float] = {}


def cached_engines(binary: str) -> tuple[BlisEngine, ...]:
    """The engines the build reports, cached once a probe succeeds.

    ``read_engines`` answers the same empty tuple for a probe that failed and for
    a build that cannot report at all, so caching its result as-is would make one
    transient failure permanent for the process. Only a successful report is
    kept, and it is kept for good: the build cannot change while the service
    runs. A failed probe is retried after ``PROBE_RETRY_SECONDS``, which also
    bounds how often a failing build is spawned.
    """
    reported = _ENGINES_BY_BINARY.get(binary)
    if reported is not None:
        return reported
    now = time.monotonic()
    retry_at = _RETRY_AFTER.get(binary)
    if retry_at is not None and now < retry_at:
        return ()
    engines = read_engines(binary)
    if not engines:
        _RETRY_AFTER[binary] = now + PROBE_RETRY_SECONDS
        return ()
    _ENGINES_BY_BINARY[binary] = engines
    _RETRY_AFTER.pop(binary, None)
    return engines


def clear_engine_cache() -> None:
    """Forget every cached report and retry deadline (tests and reloads)."""
    _ENGINES_BY_BINARY.clear()
    _RETRY_AFTER.clear()

"""Collect the artifacts of one request while it is executing."""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import threading
import time
from collections.abc import Iterator

from . import artifacts
from .record import ReplayInvocation, ReplayRecord

#: How many invocations one record may hold. A Pareto sweep records one
#: prediction per evaluated candidate so each row can reproduce itself, and a
#: wide sweep can evaluate several hundred. The byte cap below is the bound that
#: matters; this only stops pathological growth.
DEFAULT_MAX_INVOCATIONS = 512

#: What one response may carry as replay artifacts. A sweep's per-candidate
#: commands are the part that can grow without bound, and they are the part
#: dropped first; the run that produced the answer is a handful of invocations
#: and stays far below this.
DEFAULT_MAX_RECORD_BYTES = 2 * 1024 * 1024


def _max_record_bytes() -> int:
    raw = os.environ.get("AISIMULATORS_REPLAY_MAX_BYTES", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_RECORD_BYTES
    return value if value > 0 else DEFAULT_MAX_RECORD_BYTES


#: Runtime paths a BLIS command reads from the environment. A record only
#: asks for the ones its own commands use. BLIS's catalog, coefficient registry,
#: and engine facts are embedded in its binary, so the binary is the only one.
_BLIS_ENVIRONMENT = ("BLIS_BINARY",)


class ReplayRecorder:
    """Build one record from the calls a request makes.

    A recorder is created only when the caller asked for ``debug``, so the
    normal path pays nothing. Recording never raises into the handler: a
    failure to describe a run must not fail a run that succeeded.

    Recording is locked even though one request normally runs on one thread:
    the capture hook is reachable from any code the handler calls, and a
    simulator is free to fan out internally.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        simulator: str,
        execution: str,
        versions: dict[str, str | None],
        model_path: str,
        model_config: dict | None = None,
        notes: tuple[str, ...] = (),
        max_invocations: int = DEFAULT_MAX_INVOCATIONS,
    ) -> None:
        self.endpoint = endpoint
        self.simulator = simulator
        self.execution = execution
        self.versions = versions
        self.max_invocations = max_invocations
        self._lock = threading.Lock()
        self._notes = list(notes)
        self._selection: str | None = None
        self._model = artifacts.model_input(model_path, model_config)
        self._invocations: list[ReplayInvocation] = []
        self._ids: set[str] = set()
        self._truncated = False

    # ── recording ──────────────────────────────────────────────────────────

    def prediction(
        self,
        document: dict,
        *,
        label: str,
        group: str | None = None,
        candidate_id: str | None = None,
        include_memory_detail: bool = False,
        notes: tuple[str, ...] = (),
    ) -> ReplayInvocation | None:
        invocation_id = self._unique(artifacts.slugify(label))
        if invocation_id is None:
            return None
        invocation = artifacts.prediction_invocation(
            document,
            id=invocation_id,
            label=label,
            model=self._model,
            include_memory_detail=include_memory_detail,
            group=group,
            candidate_id=candidate_id,
            notes=notes,
        )
        return self._add(invocation)

    def recommendation(
        self,
        document: dict,
        *,
        label: str,
        group: str | None = None,
        notes: tuple[str, ...] = (),
    ) -> ReplayInvocation | None:
        invocation_id = self._unique(artifacts.slugify(label))
        if invocation_id is None:
            return None
        invocation = artifacts.recommendation_invocation(
            document,
            id=invocation_id,
            label=label,
            model=self._model,
            group=group,
            notes=notes,
        )
        return self._add(invocation)

    def memory(
        self,
        kwargs: dict,
        *,
        label: str,
        group: str | None = None,
        notes: tuple[str, ...] = (),
    ) -> ReplayInvocation | None:
        invocation_id = self._unique(artifacts.slugify(label))
        if invocation_id is None:
            return None
        resolved = dict(kwargs)
        # Point the snippet at the shipped model directory, not the temp path
        # the service used.
        resolved["model_path"] = self._model.reference
        invocation = artifacts.memory_snippet_invocation(
            id=invocation_id,
            label=label,
            kwargs=resolved,
            model=self._model,
            group=group,
            notes=notes,
        )
        return self._add(invocation)

    def blis_run(
        self,
        *,
        label: str,
        argv: list[str],
        files: dict[str, str],
        binary: str,
        scenario_directory: str,
        group: str | None = None,
        candidate_id: str | None = None,
        notes: tuple[str, ...] = (),
    ) -> ReplayInvocation | None:
        invocation_id = self._unique(artifacts.slugify(label))
        if invocation_id is None:
            return None
        invocation = artifacts.blis_run_invocation(
            id=invocation_id,
            label=label,
            argv=argv,
            files=files,
            binary=binary,
            scenario_directory=scenario_directory,
            group=group,
            candidate_id=candidate_id,
            notes=notes,
        )
        return self._add(invocation)

    def note(self, text: str) -> None:
        with self._lock:
            if text not in self._notes:
                self._notes.append(text)

    def select(self, summary: str) -> None:
        with self._lock:
            self._selection = summary

    def checkpoint(self) -> int:
        with self._lock:
            return len(self._invocations)

    def rollback(self, mark: int) -> None:
        """Discard everything recorded after ``mark``.

        A version fallback that fails leaves a command that fails the same
        way. The record describes the run that produced the answer, so the
        failed attempt is dropped instead of shipped alongside it.
        """
        with self._lock:
            for invocation in self._invocations[mark:]:
                self._ids.discard(invocation.id)
            del self._invocations[mark:]
            self._truncated = False

    # ── assembly ───────────────────────────────────────────────────────────

    def _unique(self, base: str) -> str | None:
        with self._lock:
            if len(self._invocations) >= self.max_invocations:
                self._truncated = True
                return None
            candidate = base
            suffix = 2
            while candidate in self._ids:
                candidate = f"{base}-{suffix}"
                suffix += 1
            self._ids.add(candidate)
            return candidate

    def _add(self, invocation: ReplayInvocation) -> ReplayInvocation:
        with self._lock:
            self._invocations.append(invocation)
        return invocation

    def build(self) -> ReplayRecord | None:
        with self._lock:
            if not self._invocations:
                return None
            invocations = list(self._invocations)
            notes = list(self._notes)
            if self._truncated:
                notes.append(
                    f"Only the first {self.max_invocations} invocations were recorded; "
                    "the rest of the run is represented by the selected result."
                )
            environment = tuple(
                name
                for name in _BLIS_ENVIRONMENT
                if any(name in invocation.command for invocation in invocations)
            )
            header = tuple(
                f"{name}: {value}" for name, value in self.versions.items() if value
            )
            return self._bounded(ReplayRecord(
                endpoint=self.endpoint,
                simulator=self.simulator,
                execution=self.execution,
                created_at=time.time(),
                versions=dict(self.versions),
                selection=self._selection,
                notes=notes,
                invocations=invocations,
                script=artifacts.run_script(
                    invocations,
                    header=header,
                    environment=environment,
                ),
            ))

    def _bounded(self, record: ReplayRecord) -> ReplayRecord | None:
        """Keep a record small enough to travel in the response.

        The per-candidate commands are the only part that grows with the size of
        a sweep, so they go first and the record says so. If the answer's own
        command is still over the cap there is nothing left to shed, and an
        incomplete record would be worse than none: the caller asked to see how
        the answer was made, not a fragment of it.
        """
        limit = _max_record_bytes()
        if record.size_bytes() <= limit:
            return record
        primary = [item for item in record.invocations if item.candidate_id is None]
        if len(primary) == len(record.invocations):
            logging.getLogger(__name__).warning(
                "Replay record for %s is above the %d-byte cap; not returning it",
                record.endpoint,
                limit,
            )
            return None
        omitted = len(record.invocations) - len(primary)
        record.invocations = primary
        record.notes.append(
            f"{omitted} per-candidate commands were omitted to keep this record "
            f"under {limit} bytes. Each candidate's own figures are unaffected; "
            "run the request again to capture them."
        )
        return record if record.size_bytes() <= limit else None


#: The recorder for the request running in this context. Sync handlers run on
#: one thread, and the BLIS capture hook is called from that same thread, so a
#: context variable is enough to connect them without threading a parameter
#: through every simulator call.
_ACTIVE: contextvars.ContextVar[ReplayRecorder | None] = contextvars.ContextVar(
    "aisimulators_replay_recorder", default=None
)


def active_recorder() -> ReplayRecorder | None:
    """The recorder for the current request, or None when debug was not asked."""
    return _ACTIVE.get()


@contextlib.contextmanager
def capture_into(recorder: ReplayRecorder | None) -> Iterator[None]:
    if recorder is None:
        yield
        return
    token = _ACTIVE.set(recorder)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def capture_blis_run(
    *,
    label: str,
    argv: list[str],
    files: dict[str, str],
    binary: str,
    scenario_directory: str,
    candidate_id: str | None = None,
) -> None:
    """Record one BLIS run while its scenario directory still exists."""
    recorder = _ACTIVE.get()
    if recorder is None:
        return
    try:
        recorder.blis_run(
            label=label,
            argv=argv,
            files=files,
            binary=binary,
            scenario_directory=scenario_directory,
            candidate_id=candidate_id,
        )
    except Exception:
        logging.getLogger(__name__).exception("could not record a BLIS replay artifact")

"""The shape of what a request actually executed.

A replay record is written once, at execution time, because the inputs that
produced a result are gone by the time the response is read: BLIS builds its
scenario in a temporary directory and the SDK's resolved config is not part of
the response. The record travels in the response itself, so retrieving it does
not depend on which worker or replica answered the next request.

Nothing here imports the simulators, and nothing here keeps state.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReplayFile:
    """One input file, relative to the bundle root."""

    path: str
    contents: str


@dataclass
class ReplayInvocation:
    """One executable unit: a command plus the files it reads."""

    id: str
    label: str
    #: Shell command, already quoted. Never contains a temporary path.
    command: str
    #: ``subprocess`` when ConfigIQ ran exactly this command; ``in_process``
    #: when the command is the equivalent CLI invocation for code the service
    #: called directly.
    execution: str
    files: list[ReplayFile] = field(default_factory=list)
    #: Optional UI sub-tab: a pool, a search window, a candidate.
    group: str | None = None
    #: Set when this invocation reproduces one Pareto candidate.
    candidate_id: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ReplayRecord:
    endpoint: str
    simulator: str
    execution: str
    created_at: float
    versions: dict[str, str | None] = field(default_factory=dict)
    #: What the run chose, in one line, when the endpoint had a choice to make.
    selection: str | None = None
    notes: list[str] = field(default_factory=list)
    invocations: list[ReplayInvocation] = field(default_factory=list)
    #: ``run.sh`` for the whole record: every invocation in execution order.
    script: str | None = None

    def size_bytes(self) -> int:
        """The wire size of the artifacts, to bound what one response carries."""
        total = len(self.script.encode("utf-8")) if self.script else 0
        for invocation in self.invocations:
            total += len(invocation.command.encode("utf-8"))
            for item in invocation.files:
                total += len(item.path.encode("utf-8"))
                total += len(item.contents.encode("utf-8"))
        return total

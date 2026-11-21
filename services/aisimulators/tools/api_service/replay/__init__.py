"""Replay records: what a request executed, served by id.

The service records artifacts only when the caller asks for ``debug``. A
record is best-effort: if it cannot be built, the response omits ``replay_id``
rather than failing a simulation that succeeded.
"""

from .capture import ReplayRecorder, capture_into
from .record import ReplayFile, ReplayInvocation, ReplayRecord

__all__ = [
    "ReplayFile",
    "ReplayInvocation",
    "ReplayRecord",
    "ReplayRecorder",
    "capture_into",
]

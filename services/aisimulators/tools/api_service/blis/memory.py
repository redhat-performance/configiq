"""Read the memory each BLIS pool holds, from the build's own run.

A run's Result states the record per pool under
``reports.engine.<role>.memory``: the device and the budget the pool sizes
against, the static reservation taken out of it, the kernel's
request-independent occupancy decomposed, and the KV pool the remainder was cut
into. This module reads that one shape and computes no memory figure of its own,
so the endpoint and a prediction cannot disagree about a pool.

Every figure is per rank (per GPU) except the ``aggregate_*`` terms, which are
what one instance holds. A page that labels a number "per GPU" reads the
per-rank term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .runner import BlisExecutionError


@dataclass(frozen=True)
class FixedMemory:
    """The kernel's request-independent occupancy, decomposed by term."""

    total: int
    weights: int
    activation_peak: int
    cuda_graph: int
    eplb_redundant: int
    comm_buffers: int
    non_torch: int
    engine_buffers: int


@dataclass(frozen=True)
class KvPoolMemory:
    """The KV pool one rank holds, plus what one instance holds."""

    block_tokens: int
    block_bytes: int
    per_token_bytes: int
    blocks_per_rank: int
    tokens_per_rank: int
    bytes_per_rank: int
    data_parallel: int
    aggregate_blocks: int
    aggregate_tokens: int
    aggregate_bytes: int


@dataclass(frozen=True)
class PoolMemory:
    """One pool's memory as the engine that sizes it leaves it."""

    role: str
    engine: str
    device_bytes: int
    budget_bytes: int
    reserved_bytes: int
    fixed: FixedMemory
    allocatable_bytes: int
    kv: KvPoolMemory

    # The engine's own terms, grouped into the four buckets a client that has
    # only ever seen one simulator's breakdown can draw. EPLB's redundant expert
    # copy is weight memory; the CUDA graph, non-torch allocation and engine
    # buffers are runtime overhead. The buckets partition `fixed.total` exactly.
    @property
    def weights_bytes(self) -> int:
        return self.fixed.weights + self.fixed.eplb_redundant

    @property
    def activations_bytes(self) -> int:
        return self.fixed.activation_peak

    @property
    def runtime_overhead_bytes(self) -> int:
        return self.fixed.cuda_graph + self.fixed.non_torch + self.fixed.engine_buffers

    @property
    def comm_overhead_bytes(self) -> int:
        return self.fixed.comm_buffers


def _mapping(source: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = source.get(key)
    if not isinstance(value, dict):
        raise BlisExecutionError(f"BLIS memory report has no object at {where}.{key}")
    return value


def _integer(source: dict[str, Any], key: str, where: str) -> int:
    value = source.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BlisExecutionError(f"BLIS memory report has no integer at {where}.{key}")
    return value


def _text(source: dict[str, Any], key: str, where: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value:
        raise BlisExecutionError(f"BLIS memory report has no text at {where}.{key}")
    return value


def _pool_memory(entry: dict[str, Any], where: str) -> PoolMemory:
    fixed = _mapping(entry, "fixed", where)
    kv = _mapping(entry, "kv", where)
    return PoolMemory(
        role=_text(entry, "role", where),
        engine=_text(entry, "engine", where),
        device_bytes=_integer(entry, "device_bytes", where),
        budget_bytes=_integer(entry, "budget_bytes", where),
        reserved_bytes=_integer(entry, "reserved_bytes", where),
        fixed=FixedMemory(
            total=_integer(fixed, "total", f"{where}.fixed"),
            weights=_integer(fixed, "weights", f"{where}.fixed"),
            activation_peak=_integer(fixed, "activation_peak", f"{where}.fixed"),
            cuda_graph=_integer(fixed, "cuda_graph", f"{where}.fixed"),
            eplb_redundant=_integer(fixed, "eplb_redundant", f"{where}.fixed"),
            comm_buffers=_integer(fixed, "comm_buffers", f"{where}.fixed"),
            non_torch=_integer(fixed, "non_torch", f"{where}.fixed"),
            engine_buffers=_integer(fixed, "engine_buffers", f"{where}.fixed"),
        ),
        allocatable_bytes=_integer(entry, "allocatable_bytes", where),
        kv=KvPoolMemory(
            block_tokens=_integer(kv, "block_tokens", f"{where}.kv"),
            block_bytes=_integer(kv, "block_bytes", f"{where}.kv"),
            per_token_bytes=_integer(kv, "per_token_bytes", f"{where}.kv"),
            blocks_per_rank=_integer(kv, "blocks_per_rank", f"{where}.kv"),
            tokens_per_rank=_integer(kv, "tokens_per_rank", f"{where}.kv"),
            bytes_per_rank=_integer(kv, "bytes_per_rank", f"{where}.kv"),
            data_parallel=_integer(kv, "data_parallel", f"{where}.kv"),
            aggregate_blocks=_integer(kv, "aggregate_blocks", f"{where}.kv"),
            aggregate_tokens=_integer(kv, "aggregate_tokens", f"{where}.kv"),
            aggregate_bytes=_integer(kv, "aggregate_bytes", f"{where}.kv"),
        ),
    )


def pool_memory_from_result(result: dict[str, Any]) -> tuple[PoolMemory, ...]:
    """Read the pool memory a run's Result already carries.

    Result reports one engine section per pool, keyed by role, and one
    ``cluster`` section that holds no memory. A record that is absent or not an
    object is skipped: a build that does not report memory leaves the prediction
    without a breakdown rather than with an invented one.
    """
    reports = result.get("reports")
    engine = reports.get("engine") if isinstance(reports, dict) else None
    if not isinstance(engine, dict):
        return ()
    records: list[PoolMemory] = []
    for role, section in engine.items():
        if not isinstance(section, dict):
            continue
        memory = section.get("memory")
        if not isinstance(memory, dict):
            continue
        records.append(_pool_memory(memory, f"reports.engine.{role}.memory"))
    return tuple(records)

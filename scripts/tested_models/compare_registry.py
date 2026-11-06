#!/usr/bin/env python3
"""Summarize a freshly built tested-model registry against a baseline.

The build workflow appends this report to the GitHub Actions job summary so a
reviewer can see which pairs gained or lost a validated envelope before the
bundle is installed with ``configiq-deploy``. It is advisory: it never fails the
build and never changes the artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _pair_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(pair["id"]): pair
        for pair in registry.get("pairs", [])
        if isinstance(pair, dict) and isinstance(pair.get("id"), str)
    }


def _coverage(pair: dict[str, Any], key: str) -> int | None:
    coverage = pair.get("coverage")
    if not isinstance(coverage, dict):
        return None
    value = coverage.get(key)
    return value if isinstance(value, int) else None


def _test_metric(pair: dict[str, Any], key: str) -> float | None:
    metrics = pair.get("metrics")
    if not isinstance(metrics, dict):
        return None
    test = metrics.get("test")
    if not isinstance(test, dict):
        return None
    value = test.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _delta(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return ""
    change = after - before
    if abs(change) < 0.0005:
        return ""
    return f"{change:+.3f}"


_VALID_STATUSES = {"validated", "measured_no_envelope"}

# Mirrors DEFAULT_GATE in lib/tested-models/registry-types.ts. Legacy v1
# bundles stored metrics but no status, so the comparison derives it the same
# way the UI does.
_LEGACY_MIN_TEST_AUC = 0.7
_LEGACY_MIN_TEST_F1 = 0.3


def _status(pair: dict[str, Any]) -> str:
    return str(pair.get("status") or "unknown")


def _legacy_status(pair: dict[str, Any]) -> str | None:
    auc = _test_metric(pair, "roc_auc")
    f1 = _test_metric(pair, "f1")
    if auc is None or f1 is None:
        return None
    return (
        "validated"
        if auc >= _LEGACY_MIN_TEST_AUC and f1 >= _LEGACY_MIN_TEST_F1
        else "measured_no_envelope"
    )


def _resolved_status(pair: dict[str, Any], schema_version: Any) -> str | None:
    status = pair.get("status")
    if isinstance(status, str) and status in _VALID_STATUSES:
        return status
    if status is None and schema_version == 1:
        return _legacy_status(pair)
    return None


def _classify(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> str:
    if before is None:
        return "new"
    if after is None:
        return "removed"
    if _status(before) != _status(after):
        return "status"
    if _delta(_test_metric(before, "roc_auc"), _test_metric(after, "roc_auc")):
        return "metrics"
    if _delta(_test_metric(before, "f1"), _test_metric(after, "f1")):
        return "metrics"
    return "unchanged"


_CHANGE_ORDER = {"new": 0, "removed": 1, "status": 2, "metrics": 3, "unchanged": 4}
_CHANGE_LABEL = {
    "new": "new pair",
    "removed": "removed",
    "status": "status changed",
    "metrics": "metrics changed",
    "unchanged": "",
}


def summarize(current: dict[str, Any], baseline: dict[str, Any] | None = None) -> str:
    # The deployed API answers 200 with status "unavailable" when it has no
    # bundle. Treat that as no baseline instead of comparing against zero pairs,
    # which would report every current pair as new and hide removals.
    note = ""
    if isinstance(baseline, dict) and baseline.get("status") == "unavailable":
        baseline = None
        note = "Baseline registry unavailable; reporting the new bundle only.\n\n"
    current_pairs = _pair_map(current)
    baseline_pairs = _pair_map(baseline) if baseline is not None else {}

    rows = []
    for pair_id in sorted(set(current_pairs) | set(baseline_pairs)):
        before = baseline_pairs.get(pair_id)
        after = current_pairs.get(pair_id)
        change = _classify(before, after) if baseline is not None else "unchanged"
        rows.append((pair_id, before, after, change))
    rows.sort(key=lambda row: (_CHANGE_ORDER[row[3]], row[0]))

    current_validated = sum(1 for pair in current_pairs.values() if _status(pair) == "validated")
    current_measured = len(current_pairs) - current_validated
    lines: list[str] = []

    if baseline is None:
        lines.append(
            f"{len(current_pairs)} pairs: {current_validated} validated, {current_measured} measured-only."
        )
    else:
        baseline_validated = sum(1 for pair in baseline_pairs.values() if _status(pair) == "validated")
        baseline_measured = len(baseline_pairs) - baseline_validated
        added = sum(1 for row in rows if row[3] == "new")
        removed = sum(1 for row in rows if row[3] == "removed")
        transitions = sum(1 for row in rows if row[3] == "status")
        lines.append(
            "Pairs: "
            f"{len(baseline_pairs)} → {len(current_pairs)} "
            f"(+{added} new, -{removed} removed). "
            "Validated: "
            f"{baseline_validated} → {current_validated}. "
            "Measured-only: "
            f"{baseline_measured} → {current_measured}. "
            f"Status changes: {transitions}."
        )
    lines.append("")
    lines.append("| Pair | Status | Records | Test AUC | Test F1 | Change |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")

    for pair_id, before, after, change in rows:
        shown = after if after is not None else before
        if shown is None:
            continue
        flags = [_CHANGE_LABEL[change]] if _CHANGE_LABEL[change] else []
        if before is not None and after is not None:
            if _status(before) != _status(after):
                flags.append(f"{_status(before)} → {_status(after)}")
            auc_delta = _delta(_test_metric(before, "roc_auc"), _test_metric(after, "roc_auc"))
            f1_delta = _delta(_test_metric(before, "f1"), _test_metric(after, "f1"))
            if auc_delta:
                flags.append(f"AUC {auc_delta}")
            if f1_delta:
                flags.append(f"F1 {f1_delta}")
        lines.append(
            "| "
            f"`{pair_id}` | {_status(shown)} | "
            f"{_number(_coverage(shown, 'records'))} | "
            f"{_number(_test_metric(shown, 'roc_auc'))} | "
            f"{_number(_test_metric(shown, 'f1'))} | "
            f"{'; '.join(flags)} |"
        )

    return note + "\n".join(lines)


def load_baseline(path: Path | None) -> dict[str, Any] | None:
    """Load a deployed registry, or None when it cannot be used as a baseline.

    The comparison is advisory, so an unreachable, HTML, or malformed response
    must fall back to the new-bundle summary instead of failing the build.
    """
    if path is None:
        return None
    try:
        baseline = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"status": "unavailable"}
    if not isinstance(baseline, dict):
        return {"status": "unavailable"}
    # The deployed registry parser only understands these versions; an
    # incompatible baseline must not be compared as if it were valid.
    schema_version = baseline.get("schemaVersion")
    if schema_version not in (1, 2):
        return {"status": "unavailable"}
    pairs = baseline.get("pairs")
    if not isinstance(pairs, list) or not all(
        isinstance(pair, dict) and isinstance(pair.get("id"), str) for pair in pairs
    ):
        return {"status": "unavailable"}
    # A malformed status would otherwise surface as an invented "unknown"
    # status or a bogus transition. Legacy v1 pairs are normalized instead.
    resolved: list[dict[str, Any]] = []
    for pair in pairs:
        status = _resolved_status(pair, schema_version)
        if status is None:
            return {"status": "unavailable"}
        resolved.append({**pair, "status": status})
    return {**baseline, "pairs": resolved}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True, help="Newly built registry.json.")
    parser.add_argument("--baseline", type=Path, default=None, help="Previously deployed registry.json.")
    args = parser.parse_args()

    current = json.loads(args.current.read_text())
    print(summarize(current, load_baseline(args.baseline)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

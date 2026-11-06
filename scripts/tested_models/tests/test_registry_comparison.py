from __future__ import annotations

import json
from pathlib import Path

import compare_registry


def _pair(
    pair_id: str,
    *,
    status: str = "validated",
    records: int = 200,
    auc: float | None = 0.85,
    f1: float | None = 0.6,
) -> dict:
    return {
        "id": pair_id,
        "status": status,
        "coverage": {"records": records, "concurrencyLevelCount": 6},
        "metrics": {"test": {"roc_auc": auc, "f1": f1}},
    }


def test_compare_reports_added_removed_and_status_changes() -> None:
    baseline = {
        "pairs": [
            _pair("kept__h200"),
            _pair("dropped__h200"),
            _pair("regressed__h200"),
        ]
    }
    current = {
        "pairs": [
            _pair("kept__h200", auc=0.90),
            _pair("regressed__h200", status="measured_no_envelope", auc=0.62, f1=0.2),
            _pair("added__b200"),
        ]
    }

    report = compare_registry.summarize(current, baseline)

    assert "Pairs: 3 → 3 (+1 new, -1 removed)" in report
    assert "Validated: 3 → 2" in report
    assert "| `added__b200` | validated | 200 | 0.850 | 0.600 | new pair |" in report
    assert "| `dropped__h200` | validated | 200 | 0.850 | 0.600 | removed |" in report
    assert "status changed; validated → measured_no_envelope" in report
    assert "AUC +0.050" in report


def test_compare_without_baseline_summarizes_the_new_bundle() -> None:
    report = compare_registry.summarize({"pairs": [_pair("only__h200")]})

    assert report.startswith("1 pairs: 1 validated, 0 measured-only.")
    assert "| `only__h200` | validated | 200 | 0.850 | 0.600 |  |" in report


def test_compare_treats_an_unavailable_api_response_as_no_baseline() -> None:
    report = compare_registry.summarize(
        {"pairs": [_pair("only__h200")]},
        {"status": "unavailable", "pairs": []},
    )

    assert report.startswith("Baseline registry unavailable; reporting the new bundle only.")
    assert "1 pairs: 1 validated, 0 measured-only." in report
    assert "new pair" not in report


def test_compare_handles_measured_only_pairs_without_metrics() -> None:
    report = compare_registry.summarize({
        "pairs": [_pair("weak__h200", status="measured_no_envelope", records=23, auc=None, f1=None)],
    })

    assert "| `weak__h200` | measured_no_envelope | 23 | — | — |  |" in report


def test_load_baseline_rejects_malformed_or_non_registry_payloads(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("<html>not json</html>")
    assert compare_registry.load_baseline(malformed) == {"status": "unavailable"}

    missing_pairs = tmp_path / "missing-pairs.json"
    missing_pairs.write_text('{"status": "available"}')
    assert compare_registry.load_baseline(missing_pairs) == {"status": "unavailable"}

    missing = tmp_path / "absent.json"
    assert compare_registry.load_baseline(missing) == {"status": "unavailable"}

    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe\x00")
    assert compare_registry.load_baseline(binary) == {"status": "unavailable"}

    bad_pair = tmp_path / "bad-pair.json"
    bad_pair.write_text('{"pairs": ["not-an-object"]}')
    assert compare_registry.load_baseline(bad_pair) == {"status": "unavailable"}

    missing_id = tmp_path / "missing-id.json"
    missing_id.write_text('{"pairs": [{"status": "validated"}]}')
    assert compare_registry.load_baseline(missing_id) == {"status": "unavailable"}


def test_load_baseline_accepts_a_registry_with_pairs(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text('{"schemaVersion": 2, "status": "available", "pairs": []}')

    assert compare_registry.load_baseline(path) == {
        "schemaVersion": 2,
        "status": "available",
        "pairs": [],
    }
    assert compare_registry.load_baseline(None) is None


def test_load_baseline_rejects_missing_or_unsupported_schema_versions(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-version.json"
    missing.write_text('{"pairs": [{"id": "pair__h200", "status": "validated"}]}')
    assert compare_registry.load_baseline(missing) == {"status": "unavailable"}

    unsupported = tmp_path / "unsupported-version.json"
    unsupported.write_text(
        '{"schemaVersion": 99, "pairs": [{"id": "pair__h200", "status": "validated"}]}'
    )
    assert compare_registry.load_baseline(unsupported) == {"status": "unavailable"}

    boolean = tmp_path / "boolean-version.json"
    boolean.write_text(
        '{"schemaVersion": true, "pairs": [{"id": "pair__h200", "status": "validated"}]}'
    )
    assert compare_registry.load_baseline(boolean) == {"status": "unavailable"}


def test_load_baseline_rejects_missing_or_unknown_statuses(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown-status.json"
    unknown.write_text('{"schemaVersion": 2, "pairs": [{"id": "pair__h200", "status": "bogus"}]}')
    assert compare_registry.load_baseline(unknown) == {"status": "unavailable"}

    missing = tmp_path / "missing-status.json"
    missing.write_text('{"schemaVersion": 2, "pairs": [{"id": "pair__h200"}]}')
    assert compare_registry.load_baseline(missing) == {"status": "unavailable"}


def test_load_baseline_derives_legacy_v1_statuses_from_metrics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "schemaVersion": 1,
        "pairs": [
            {"id": "strong__h200", "metrics": {"test": {"roc_auc": 0.9, "f1": 0.6}}},
            {"id": "weak__h200", "metrics": {"test": {"roc_auc": 0.9, "f1": 0.1}}},
        ],
    }))

    baseline = compare_registry.load_baseline(path)

    assert baseline is not None
    assert {pair["id"]: pair["status"] for pair in baseline["pairs"]} == {
        "strong__h200": "validated",
        "weak__h200": "measured_no_envelope",
    }


def test_load_baseline_rejects_legacy_v1_pairs_without_metrics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-missing-metrics.json"
    path.write_text('{"schemaVersion": 1, "pairs": [{"id": "pair__h200"}]}')

    assert compare_registry.load_baseline(path) == {"status": "unavailable"}

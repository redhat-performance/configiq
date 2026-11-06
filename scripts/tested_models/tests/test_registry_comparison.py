from __future__ import annotations

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

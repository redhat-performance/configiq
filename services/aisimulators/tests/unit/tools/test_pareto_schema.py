# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pareto objective schema additions to RecommendRequest/Response."""

from __future__ import annotations

import pytest

from tools.api_service.app import (
    CandidateResponse,
    CandidateViews,
    RecommendRequest,
    RecommendResponse,
)

VALID_BASE = {
    "model_path": "Qwen/Qwen3-32B",
    "system": "h200_sxm",
    "target_concurrency": 32,
    "isl": 4000,
    "osl": 1000,
    "ttft": 2000.0,
    "tpot": 30.0,
}


def test_recommend_request_default_objective():
    req = RecommendRequest(**VALID_BASE)
    assert req.objective == "min_gpus"


def test_recommend_request_pareto_objective():
    req = RecommendRequest(**VALID_BASE, objective="pareto")
    assert req.objective == "pareto"


def test_recommend_request_kv_transfer_bandwidth_default():
    req = RecommendRequest(**VALID_BASE, objective="pareto")
    assert req.kv_transfer_bandwidth_gb_per_second == 25.0


def test_recommend_request_kv_transfer_bandwidth_custom():
    req = RecommendRequest(**VALID_BASE, objective="pareto", kv_transfer_bandwidth_gb_per_second=50.0)
    assert req.kv_transfer_bandwidth_gb_per_second == 50.0


def test_candidate_response_requires_candidate_id():
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        CandidateResponse(status="feasible", mode="agg", topology={}, metrics={})


def test_candidate_response_all_fields():
    c = CandidateResponse(
        candidate_id="candidate-000001",
        status="feasible",
        mode="agg",
        topology={"deployment_mode": "agg", "tp": 8},
        used_gpus=8,
        metrics={"ttft_ms": 1200.0, "tpot_ms": 25.0},
        objectives={"throughput_per_gpu": 42.0, "throughput_per_user": 12.0},
        transfer_delay_modeled=False,
        provenance_sources=["fpm_interpolation"],
    )
    assert c.candidate_id == "candidate-000001"
    assert c.objectives["throughput_per_gpu"] == 42.0


def test_candidate_views_fields():
    v = CandidateViews(pareto_front=["candidate-000001"], top_n=[])
    assert v.pareto_front == ["candidate-000001"]
    assert v.top_n == []


def test_recommend_response_preserves_configs():
    resp = RecommendResponse(
        configs=[],
        chosen_mode="agg",
        candidates=[],
        views=CandidateViews(pareto_front=[], top_n=[]),
    )
    assert resp.configs == []
    assert resp.candidates == []
    assert resp.views is not None


def test_recommend_response_candidates_default_empty():
    resp = RecommendResponse(configs=[], chosen_mode="agg")
    assert resp.candidates == []
    assert resp.views is None

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the REST API service (tools/api_service/app.py).

These tests mock the SDK functions to test API layer logic without
requiring the Rust native extension or performance databases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from configiq.systems import load_device_names_from_perf_data, supported_systems
from fastapi.testclient import TestClient

from tools.api_service import app as app_module
from tools.api_service.app import MemoryBreakdown, app

client = TestClient(app)


def _sdk_available() -> bool:
    try:
        from aisimulate_core.sdk.common import SupportedSystems
        return len(SupportedSystems) > 0
    except Exception:
        return False


# ─── Fixtures ────────────────────────────────────────────────────────────────

VALID_RECOMMEND_BODY = {
    "model_path": "Qwen/Qwen3-32B",
    "system": "h200_sxm",
    "target_concurrency": 32,
    "isl": 4000,
    "osl": 1000,
    "ttft": 2000.0,
    "tpot": 30.0,
    "backend": "vllm",
    "database_mode": "HYBRID",
    "top_n": 2,
}

VALID_MEMORY_BODY = {
    "model_path": "Qwen/Qwen3-32B",
    "system": "h200_sxm",
    "backend": "vllm",
    "backend_version": "0.24.0",
    "max_num_tokens": 8192,
    "max_batch_size": 128,
    "tp_size": 2,
    "pp_size": 1,
    "memory_fraction_kind": "of_total",
    "memory_fraction_value": 0.9,
}

MOCK_KV_CACHE_RESULT = {
    "total_gpu_capacity_bytes": 151397597184,
    "total_kv_size_bytes": 98507266457,
    "kv_size_per_token_bytes": 131072,
    "total_kv_size_tokens": 751550,
    "source": "native",
    "memory_breakdown": {
        "weights_bytes": 32761446400,
        "activations_bytes": 532480000,
        "runtime_overhead_bytes": 3758096384,
        "comm_overhead_bytes": 358612992,
    },
    "tolerance_adjusted": None,
}

MOCK_SYSTEM_SPEC = {
    "gpu": {
        "mem_bw": 4800000000000,
        "mem_capacity": 151397597184,
        "bfloat16_tc_flops": 989000000000000,
        "fp8_tc_flops": 1978000000000000,
        "power": 700,
        "sm_version": 90,
    },
    "node": {
        "num_gpus_per_node": 8,
    },
}


@dataclass
class MockCandidate:
    used_gpus: int = 2
    metrics: dict = field(default_factory=lambda: {
        "ttft_ms": 471.378,
        "tpot_ms": 28.118,
        "output_throughput_tok_s": 1678.925,
        "output_throughput_tok_s_per_gpu": 839.462,
    })
    prediction_config: dict = field(default_factory=lambda: {
        "engine": {
            "model": "Qwen/Qwen3-32B", "hardware": "h200_sxm",
            "backend": "vllm", "backend_version": "0.24.0", "mode": "aggregated",
            "workers": {"aggregated": {
                "parallelism": {"tensor": 2, "pipeline": 1, "attention_data": 1, "replicas": 1},
                "scheduler": {"max_sequences": 48},
            }},
        },
    })


@dataclass
class MockRecommendationResult:
    selected_candidates: list[MockCandidate] = field(default_factory=lambda: [MockCandidate()])


def make_mock_recommendation_result(candidates=None):
    return MockRecommendationResult(selected_candidates=candidates or [MockCandidate()])


#: Request shape for a throughput-demand run: no latency bound, an offered
#: arrival rate instead of closed-loop concurrency.
VALID_DEMAND_BODY = {
    "model_path": "Qwen/Qwen3-32B",
    "system": "h200_sxm",
    "isl": 2048,
    "osl": 128,
    "objective": "throughput_demand",
    "sla_mode": "none",
    "target_request_rate": 10.0,
    "backend": "vllm",
}


@dataclass
class MockLedgerProvenance:
    topology: dict = field(default_factory=lambda: {
        "deployment_mode": "agg",
        "tp": 2,
        "pp": 1,
        "attention_dp": 1,
        "replicas": 1,
    })


@dataclass
class MockLedgerCandidate:
    """A candidate as it appears in the SDK result ledger (not `selected_candidates`).

    The demand path reads `status`, `used_gpus`, `metrics`, `candidate_id`, and
    `provenance.topology` from every evaluated record, then converts only the
    winner into a RecommendConfig.
    """

    candidate_id: str = "candidate-000001"
    used_gpus: int = 2
    status: str = "feasible"
    metrics: dict = field(default_factory=dict)
    provenance: MockLedgerProvenance = field(default_factory=MockLedgerProvenance)


@dataclass
class MockRecommendationCounts:
    evaluated: int = 2
    feasible: int = 2
    timed_out: int = 0


@dataclass
class MockRecommendationProvenance:
    run_id: str = "run-test-1"


@dataclass
class MockLedgerRecommendationResult:
    candidates: list[MockLedgerCandidate] = field(default_factory=list)
    counts: MockRecommendationCounts | None = None
    provenance: MockRecommendationProvenance | None = None

    @property
    def selected_candidates(self) -> list[MockLedgerCandidate]:
        return self.candidates[:1]


@dataclass
class MockPredictionResult:
    summary: dict = field(default_factory=lambda: {
        "ttft_ms": 471.378, "tpot_ms": 28.118,
        "output_throughput_tok_s": 1678.925,
        "output_throughput_tok_s_per_gpu": 839.462,
    })
    native: dict = field(default_factory=lambda: {"summary": {}})


# ─── /recommend tests ────────────────────────────────────────────────────────


class TestRecommend:

    def test_recommendation_search_is_bounded(self, monkeypatch):
        monkeypatch.delenv("AISIMULATORS_MAX_CANDIDATE_GPUS", raising=False)
        request = app_module.RecommendRequest.model_validate(VALID_RECOMMEND_BODY)
        config = app_module._aisimulate_recommendation_config(request)

        assert config.optimization.constraints.max_candidate_gpus == 1024
        assert config.optimizer.max_trials == 8

    def test_recommendation_search_budget_is_configurable(self, monkeypatch):
        monkeypatch.setenv("AISIMULATORS_MAX_CANDIDATE_GPUS", "4096")
        request = app_module.RecommendRequest.model_validate(VALID_RECOMMEND_BODY)
        config = app_module._aisimulate_recommendation_config(request)

        assert config.optimization.constraints.max_candidate_gpus == 4096

    def test_requested_gpu_window_is_clamped_to_server_budget(self, monkeypatch):
        monkeypatch.setenv("AISIMULATORS_MAX_CANDIDATE_GPUS", "8")
        body = {**VALID_RECOMMEND_BODY, "max_candidate_gpus": 64}
        request = app_module.RecommendRequest.model_validate(body)
        config = app_module._aisimulate_recommendation_config(request)

        assert config.optimization.constraints.max_candidate_gpus == 8

    def test_rejects_window_lower_bound_above_effective_server_budget(self, monkeypatch):
        monkeypatch.setenv("AISIMULATORS_MAX_CANDIDATE_GPUS", "8")
        body = {**VALID_RECOMMEND_BODY, "min_candidate_gpus": 9, "max_candidate_gpus": 64}
        request = app_module.RecommendRequest.model_validate(body)

        with pytest.raises(ValueError, match="effective max_candidate_gpus"):
            app_module._aisimulate_recommendation_config(request)

    def test_recommendation_window_bounds_are_forwarded(self):
        body = {**VALID_RECOMMEND_BODY, "min_candidate_gpus": 2, "max_candidate_gpus": 4}
        request = app_module.RecommendRequest.model_validate(body)
        config = app_module._aisimulate_recommendation_config(request)

        assert config.optimization.constraints.min_candidate_gpus == 2
        assert config.optimization.constraints.max_candidate_gpus == 4

    def test_one_gpu_window_avoids_disaggregated_trials(self):
        body = {**VALID_RECOMMEND_BODY, "target_concurrency": 1, "min_candidate_gpus": 1, "max_candidate_gpus": 1}
        request = app_module.RecommendRequest.model_validate(body)
        config = app_module._aisimulate_recommendation_config(request)

        assert config.engine.mode.choices == ["aggregated"]
        assert config.optimizer.max_trials == 1
        assert config.optimizer.parallelism == 1
        assert config.optimizer.candidate_timeout_seconds == 15

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_success(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result()
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 200
        data = resp.json()
        assert "configs" in data
        assert "chosen_mode" in data
        assert data["chosen_mode"] == "agg"
        assert len(data["configs"]) >= 1

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_response_fields(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result()
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        cfg = resp.json()["configs"][0]
        assert cfg["tp"] == 2
        assert cfg["pp"] == 1
        assert cfg["dp"] == 1
        assert cfg["total_gpus_needed"] == 2
        assert cfg["replicas_needed"] == 1
        assert cfg["num_total_gpus"] == 2
        assert cfg["ttft"] == 471.378
        assert cfg["tpot"] == 28.118
        assert cfg["tokens_per_second"] == 1678.925
        assert cfg["tokens_per_second_per_gpu"] == 839.462
        assert cfg["memory"] is None
        assert cfg["model"] == "Qwen/Qwen3-32B"
        assert cfg["system"] == "h200_sxm"
        assert cfg["backend"] == "vllm"
        assert cfg["backend_version"] == "0.24.0"

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_response_derives_compatibility_metrics_from_engine_report(self, mock_recommend):
        # The engine runner reports canonical aggregate metrics rather than the
        # legacy REST names for concurrency and per-unit throughput.
        candidate = MockCandidate(
            used_gpus=1,
            metrics={
                "ttft_ms": 192.178766,
                "mean_tpot_ms": 8.88037,
                "mean_e2e_latency_ms": 1319.985776,
                "request_throughput_rps": 0.757584,
                "output_throughput_tok_s": 96.970742,
                "mean_output_token_throughput_per_user": 112.608,
            },
        )
        mock_recommend.return_value = make_mock_recommendation_result([candidate])

        body = {**VALID_RECOMMEND_BODY, "target_concurrency": 1, "isl": 2048, "osl": 128}
        resp = client.post("/recommend", json=body)

        assert resp.status_code == 200
        cfg = resp.json()["configs"][0]
        assert cfg["concurrency"] == 1
        assert cfg["request_rate"] == pytest.approx(0.757584)
        assert cfg["tokens_per_second_per_gpu"] == pytest.approx(96.970742)
        assert cfg["tokens_per_second_per_user"] == pytest.approx(112.608)

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_cluster_gpu_count_is_distinct_from_per_replica_count(self, mock_recommend):
        candidate = MockCandidate(
            used_gpus=8,
            metrics={
                "ttft_ms": 900.0,
                "tpot_ms": 25.0,
                "output_throughput_tok_s": 4000.0,
                "output_throughput_tok_s_per_gpu": 500.0,
            },
            prediction_config={"engine": {
                "model": "Qwen/Qwen3-8B", "hardware": "a100_sxm",
                "backend": "vllm", "backend_version": "0.24.0", "mode": "aggregated",
                "workers": {"aggregated": {
                    "parallelism": {
                        "tensor": 2, "pipeline": 1, "attention_data": 1,
                        "context": 1, "replicas": 4,
                    },
                    "scheduler": {"max_sequences": 256},
                }},
            }},
        )
        mock_recommend.return_value = make_mock_recommendation_result([candidate])

        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)

        assert resp.status_code == 200
        cfg = resp.json()["configs"][0]
        assert cfg["total_gpus_needed"] == 8
        assert cfg["replicas_needed"] == 4
        assert cfg["num_total_gpus"] == 2
        assert cfg["tokens_per_second"] == 4000.0

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_inclusive_tpot(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result()
        body = {**VALID_RECOMMEND_BODY, "inclusive_tpot": True}
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 200
        cfg = resp.json()["configs"][0]
        expected = (471.378 + 28.118 * (1000 - 1)) / 1000
        assert cfg["tpot"] == pytest.approx(expected)
        assert cfg["ttft"] == pytest.approx(471.378)

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_inclusive_tpot_default_false(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result()
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.json()["configs"][0]["tpot"] == pytest.approx(28.118)

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_inclusive_tpot_preserves_missing_ttft(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result([
            MockCandidate(metrics={"tpot_ms": 28.118}),
        ])
        body = {**VALID_RECOMMEND_BODY, "inclusive_tpot": True}
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 200
        cfg = resp.json()["configs"][0]
        assert cfg["ttft"] is None
        assert cfg["tpot"] == pytest.approx(28.118)

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_inclusive_tpot_preserves_missing_tpot(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result([
            MockCandidate(metrics={"ttft_ms": 471.378}),
        ])
        body = {**VALID_RECOMMEND_BODY, "inclusive_tpot": True}
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 200
        assert resp.json()["configs"][0]["tpot"] is None

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_no_serving_config_by_default(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result()
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        cfg = resp.json()["configs"][0]
        assert cfg["serving_config"] is None
        assert cfg["memory_breakdown"] is None

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_include_config(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result()
        resp = client.post("/recommend?include=config", json=VALID_RECOMMEND_BODY)
        cfg = resp.json()["configs"][0]
        assert cfg["serving_config"] is not None
        assert cfg["serving_config"]["backend"] == "vllm"
        assert cfg["serving_config"]["tensor_parallel_size"] == 2
        assert cfg["serving_config"]["max_model_len"] == 5000
        assert cfg["memory_breakdown"] is None

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_include_memory(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result()
        resp = client.post("/recommend?include=memory", json=VALID_RECOMMEND_BODY)
        cfg = resp.json()["configs"][0]
        assert cfg["memory_breakdown"] is not None
        assert cfg["serving_config"] is None

    @patch("tools.api_service.app._build_memory_breakdown")
    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_include_memory_populates_peak_memory(self, mock_recommend, mock_breakdown):
        mock_recommend.return_value = make_mock_recommendation_result()
        mock_breakdown.return_value = MemoryBreakdown(
            weights_bytes=4 * 1024 ** 3,
            activations_bytes=1 * 1024 ** 3,
            runtime_overhead_bytes=512 * 1024 ** 2,
            comm_overhead_bytes=0,
            kv_cache_bytes=2 * 1024 ** 3,
        )

        resp = client.post("/recommend?include=memory", json=VALID_RECOMMEND_BODY)

        assert resp.status_code == 200
        assert resp.json()["configs"][0]["memory"] == pytest.approx(7.5)

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_include_config_and_memory(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result()
        resp = client.post("/recommend?include=config,memory", json=VALID_RECOMMEND_BODY)
        cfg = resp.json()["configs"][0]
        assert cfg["serving_config"] is not None
        assert cfg["memory_breakdown"] is not None

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_top_n_limits_results(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result(
            [MockCandidate() for _ in range(5)]
        )
        body = {**VALID_RECOMMEND_BODY, "top_n": 2}
        resp = client.post("/recommend", json=body)
        assert len(resp.json()["configs"]) == 2

    def test_requires_exactly_one_target(self):
        body = {**VALID_RECOMMEND_BODY}
        del body["target_concurrency"]
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 422

    def test_rejects_both_targets(self):
        body = {**VALID_RECOMMEND_BODY, "target_request_rate": 10}
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 422

    def test_accepts_target_request_rate(self):
        body = {**VALID_RECOMMEND_BODY}
        del body["target_concurrency"]
        body["target_request_rate"] = 10.0
        with patch("tools.api_service.app._run_aisimulate_recommendation") as mock:
            mock.return_value = make_mock_recommendation_result()
            resp = client.post("/recommend", json=body)
            assert resp.status_code == 200
            request = mock.call_args.args[0]
            assert request.target_request_rate == 10.0
            assert request.target_concurrency is None

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_passes_context_window_limits(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result()
        body = {
            **VALID_RECOMMEND_BODY,
            "max_seq_len": 64000,
            "prefill_max_seq_len": 128000,
            "decode_max_seq_len": 64000,
        }
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 200
        request = mock_recommend.call_args.args[0]
        assert request.max_seq_len == 64000
        assert request.prefill_max_seq_len == 128000
        assert request.decode_max_seq_len == 64000

    def test_requires_model_path(self):
        body = {**VALID_RECOMMEND_BODY}
        del body["model_path"]
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 422

    def test_requires_system(self):
        body = {**VALID_RECOMMEND_BODY}
        del body["system"]
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 422

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_no_config_found_returns_422(self, mock_recommend):
        mock_recommend.return_value = MockRecommendationResult(selected_candidates=[])
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 422
        assert "No configuration" in resp.json()["detail"]

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_value_error_returns_422(self, mock_recommend):
        mock_recommend.side_effect = ValueError("bad input")
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 422
        assert "bad input" in resp.json()["detail"]

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_wrapped_no_viable_parallel_config_returns_422(self, mock_recommend):
        mock_recommend.side_effect = RuntimeError(
            "NoViableParallelConfig: no deployment_mode has a viable parallel config"
        )
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 422

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_no_viable_parallel_config_exception_returns_422(self, mock_recommend):
        class NoViableParallelConfig(Exception):
            pass

        mock_recommend.side_effect = NoViableParallelConfig(
            "no deployment_mode has a viable parallel config"
        )
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 422
        assert "no deployment_mode" in resp.json()["detail"]

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_unexpected_error_returns_500(self, mock_recommend):
        mock_recommend.side_effect = RuntimeError("internal failure")
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 500

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_defaults_applied(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result()
        body = {
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "target_concurrency": 32,
            "ttft": 2000.0,
            "tpot": 30.0,
        }
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 200
        request = mock_recommend.call_args.args[0]
        assert request.backend == "vllm"
        assert request.isl == 4000
        assert request.osl == 1000
        assert request.ttft == 2000.0
        assert request.tpot == 30.0
        assert request.database_mode == "HYBRID"
        assert request.top_n == 5

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_null_fields_get_request_defaults(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result()
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        cfg = resp.json()["configs"][0]
        assert cfg["system"] == "h200_sxm"
        assert cfg["backend"] == "vllm"

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_recommend_builds_memory_breakdown(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result()
        resp = client.post("/recommend?include=memory", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 200
        assert resp.json()["configs"][0]["memory_breakdown"] is not None

    @patch("tools.api_service.app._blis_recommend")
    def test_blis_recommendation_reports_an_enumerated_sweep(self, mock_blis):
        mock_blis.return_value = MagicMock(
            configs=[{"mode": "agg", "total_gpus_needed": 1}],
            search_summary={
                "candidate_configurations": 4,
                "evaluated_configurations": 3,
                "budget_exhausted": False,
                "effective_max_gpus": 4,
            },
            demand=None,
        )

        resp = client.post(
            "/recommend", json={**VALID_RECOMMEND_BODY, "simulator": "blis"}
        )

        assert resp.status_code == 200
        sweep = resp.json()["sweep"]
        assert sweep["algorithm"] == "enumerate"
        assert sweep["max_trials"] == 4
        assert sweep["evaluated"] == 3
        assert sweep["window_max_gpus"] == 4
        assert sweep["strict_sla"] is True


# ─── throughput-demand tests ─────────────────────────────────────────────────


def _demand_metrics(
    offered_rps: float,
    *,
    gpus: int = 2,
    e2e_ms: float = 2000.0,
    window_s: float = 10.0,
    completed: float | None = None,
) -> dict:
    """Metrics for a run that offered `offered_rps` for `window_s` seconds.

    A replay is timed from the first arrival to the last completion, so the
    recorded throughput is `completed / (window + e2e)` while the arrival-window
    rate is `completed / window`. Deriving both from the same numbers keeps the
    fixture physically consistent with the compliance check.
    """
    requests = completed if completed is not None else round(offered_rps * window_s)
    duration_ms = window_s * 1000.0 + e2e_ms
    return {
        "request_throughput_rps": requests / (duration_ms / 1000.0),
        "mean_e2e_latency_ms": e2e_ms,
        "duration_ms": duration_ms,
        "completed_requests": requests,
        "mean_ttft_ms": 180.0,
        "mean_tpot_ms": 18.0,
        "output_throughput_tok_s": offered_rps * 128,
    }


class TestRecommendDemand:
    def test_config_uses_open_loop_rate_and_no_sla(self):
        request = app_module.RecommendRequest.model_validate(VALID_DEMAND_BODY)
        config = app_module._aisimulate_recommendation_config(request)

        dumped = config.model_dump()
        assert dumped["traffic"]["load"]["type"] == "constant_rate"
        assert dumped["traffic"]["load"]["requests_per_second"] == 10.0
        # A demand run stops after enough requests to measure the offered rate,
        # not after a fixed handful of requests.
        assert dumped["traffic"]["stop"]["requests_per_load_unit"] == (
            app_module._demand_requests_per_load_unit()
        )
        assert (dumped.get("evaluation") or {}).get("sla") is None
        assert dumped["optimization"]["strict_sla"] is False
        assert dumped["optimization"]["target"] == "throughput_per_gpu"

    def test_demand_trial_budget_is_separate_from_sizing(self, monkeypatch):
        monkeypatch.setenv("DEMAND_MAX_TRIALS", "12")
        monkeypatch.setenv("DEMAND_PARALLELISM", "3")
        monkeypatch.setenv("DEMAND_CANDIDATE_TIMEOUT_S", "9")
        request = app_module.RecommendRequest.model_validate(VALID_DEMAND_BODY)
        config = app_module._aisimulate_recommendation_config(request)

        assert config.optimizer.max_trials == 12
        assert config.optimizer.parallelism == 3
        assert config.optimizer.candidate_timeout_seconds == 9

    def test_demand_rejects_an_enforced_sla(self):
        body = {**VALID_DEMAND_BODY, "sla_mode": "enforce"}
        with pytest.raises(ValueError, match="sla_mode='none'"):
            app_module.RecommendRequest.model_validate(body)

    def test_demand_rejects_latency_targets(self):
        for field_name, value in (("ttft", 1000.0), ("tpot", 30.0), ("request_latency", 3000.0)):
            body = {**VALID_DEMAND_BODY, field_name: value}
            with pytest.raises(ValueError, match="does not accept"):
                app_module.RecommendRequest.model_validate(body)

    def test_demand_requires_rate_and_hides_concurrency(self):
        body = {**VALID_DEMAND_BODY}
        del body["target_request_rate"]
        with pytest.raises(ValueError, match="requires target_request_rate"):
            app_module.RecommendRequest.model_validate(body)

        with pytest.raises(ValueError, match="no target_concurrency"):
            app_module.RecommendRequest.model_validate(
                {**VALID_DEMAND_BODY, "target_concurrency": 8}
            )

    def test_min_gpus_still_requires_an_enforced_sla(self):
        body = {**VALID_RECOMMEND_BODY, "sla_mode": "none"}
        with pytest.raises(ValueError, match="min_gpus"):
            app_module.RecommendRequest.model_validate(body)

    def test_transfer_bandwidth_is_rejected_outside_pareto(self):
        for body in (VALID_RECOMMEND_BODY, VALID_DEMAND_BODY):
            with pytest.raises(ValueError, match="only supported with objective='pareto'"):
                app_module.RecommendRequest.model_validate(
                    {**body, "kv_transfer_bandwidth_gb_per_second": 50.0}
                )

    def test_transfer_bandwidth_is_accepted_for_pareto(self):
        request = app_module.RecommendRequest.model_validate({
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "objective": "pareto",
            "sla_mode": "none",
            "kv_transfer_bandwidth_gb_per_second": 50.0,
        })
        assert request.kv_transfer_bandwidth_gb_per_second == 50.0

    def test_closed_loop_rate_skips_the_arrival_window_correction(self):
        # A closed-loop replay starts every request together, so its duration is
        # about one wave. Subtracting a mean latency would inflate the rate.
        metrics = _demand_metrics(10.0, e2e_ms=4000.0, window_s=10.0)
        raw = metrics["request_throughput_rps"]

        assert app_module._measured_request_rate(metrics, open_loop=False) == raw
        assert app_module._measured_request_rate(metrics, open_loop=True) > raw

    def test_closed_loop_concurrency_does_not_use_the_arrival_window(self):
        request = app_module.RecommendRequest.model_validate(VALID_RECOMMEND_BODY)
        metrics = _demand_metrics(10.0, e2e_ms=4000.0, window_s=10.0)

        concurrency = app_module._recommendation_concurrency(metrics, request)

        assert concurrency == round(metrics["request_throughput_rps"] * 4)

    def test_enforced_sla_requires_a_bound(self):
        body = {**VALID_RECOMMEND_BODY}
        del body["ttft"]
        del body["tpot"]
        with pytest.raises(ValueError, match="requires request_latency"):
            app_module.RecommendRequest.model_validate(body)

    def test_e2e_bound_replaces_decomposed_bounds(self):
        body = {**VALID_RECOMMEND_BODY, "request_latency": 2500.0}
        del body["ttft"]
        del body["tpot"]
        request = app_module.RecommendRequest.model_validate(body)
        assert request.request_latency == 2500.0

        with pytest.raises(ValueError, match="mutually exclusive"):
            app_module.RecommendRequest.model_validate(
                {**VALID_RECOMMEND_BODY, "request_latency": 2500.0, "ttft": 1000.0}
            )

    def test_sla_block_maps_each_bound_shape(self):
        token_request = app_module.RecommendRequest.model_validate(VALID_RECOMMEND_BODY)
        token_config = app_module._aisimulate_recommendation_config(token_request)
        token_sla = (token_config.model_dump().get("evaluation") or {}).get("sla") or {}
        assert token_sla.get("ttft_ms") == 2000.0
        assert token_sla.get("itl_ms") == 30.0
        # The SDK serializes an unused bound as None rather than omitting it.
        assert token_sla.get("e2e_ms") is None

        e2e_request = app_module.RecommendRequest.model_validate({
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "target_concurrency": 32,
            "request_latency": 2500.0,
        })
        e2e_config = app_module._aisimulate_recommendation_config(e2e_request)
        e2e_sla = (e2e_config.model_dump().get("evaluation") or {}).get("sla") or {}
        assert e2e_sla.get("e2e_ms") == 2500.0
        assert e2e_sla.get("ttft_ms") is None


class TestSelectDemandRecord:
    def test_prefers_fewest_gpus_that_sustains_the_rate(self):
        small = MockLedgerCandidate(
            candidate_id="small", used_gpus=1, metrics=_demand_metrics(12.0)
        )
        medium = MockLedgerCandidate(
            candidate_id="medium", used_gpus=2, metrics=_demand_metrics(40.0)
        )
        winner, met, best, measured = app_module._select_demand_record(
            [small, medium], 10.0
        )

        assert winner is small
        assert met is True
        assert best == 40.0
        assert measured == 2

    def test_accepts_legacy_throughput_metric_names(self):
        # The engine runner has used both names for measured request rate. With
        # no completion count or duration, the reported rate is used directly.
        record = MockLedgerCandidate(used_gpus=2, metrics={"requests_per_second": 12.0})

        winner, met, _, _ = app_module._select_demand_record([record], 10.0)
        assert winner is record
        assert met is True

    def test_ties_on_gpus_break_by_highest_rate_then_latency(self):
        faster = MockLedgerCandidate(
            candidate_id="faster", used_gpus=2, metrics=_demand_metrics(20.0)
        )
        slower = MockLedgerCandidate(
            candidate_id="slower", used_gpus=2, metrics=_demand_metrics(15.0)
        )
        winner, _, _, _ = app_module._select_demand_record([slower, faster], 10.0)
        assert winner is faster

        lower_latency = MockLedgerCandidate(
            candidate_id="low-latency",
            used_gpus=2,
            metrics=_demand_metrics(20.0, e2e_ms=1500.0),
        )
        same_rate = MockLedgerCandidate(
            candidate_id="same-rate", used_gpus=2, metrics=_demand_metrics(20.0, e2e_ms=2500.0)
        )
        winner, _, _, _ = app_module._select_demand_record([same_rate, lower_latency], 10.0)
        assert winner is lower_latency

    def test_unmet_target_returns_best_observed_at_200(self):
        weak = MockLedgerCandidate(
            candidate_id="weak", used_gpus=1, metrics=_demand_metrics(3.0)
        )
        better = MockLedgerCandidate(
            candidate_id="better", used_gpus=4, metrics=_demand_metrics(7.0)
        )
        winner, met, best, measured = app_module._select_demand_record(
            [weak, better], 10.0
        )

        assert winner is better
        assert met is False
        assert best == 7.0
        assert measured == 2

    def test_ignores_infeasible_and_unmeasured_records(self):
        infeasible = MockLedgerCandidate(
            candidate_id="infeasible",
            used_gpus=1,
            status="infeasible",
            metrics=_demand_metrics(99.0),
        )
        unmeasured = MockLedgerCandidate(
            candidate_id="unmeasured",
            used_gpus=1,
            status="timed_out",
            metrics={},
        )
        zero_completed = MockLedgerCandidate(
            candidate_id="zero",
            used_gpus=1,
            metrics=_demand_metrics(50.0, completed=0.0),
        )
        usable = MockLedgerCandidate(
            candidate_id="usable", used_gpus=2, metrics=_demand_metrics(11.0)
        )

        winner, met, _, measured = app_module._select_demand_record(
            [infeasible, unmeasured, zero_completed, usable], 10.0
        )
        assert winner is usable
        assert met is True
        assert measured == 1

    def test_no_measured_records_returns_nothing(self):
        winner, met, best, measured = app_module._select_demand_record(
            [MockLedgerCandidate(status="infeasible", metrics={})], 10.0
        )
        assert winner is None
        assert met is False
        assert best is None
        assert measured == 0

    def test_unknown_gpu_count_is_not_ranked_cheapest(self):
        unknown = MockLedgerCandidate(
            candidate_id="unknown", used_gpus=None, metrics=_demand_metrics(12.0)
        )
        known = MockLedgerCandidate(
            candidate_id="known", used_gpus=2, metrics=_demand_metrics(12.0)
        )

        winner, _, _, _ = app_module._select_demand_record([unknown, known], 10.0)
        assert winner is known

    def test_compliance_excludes_the_final_response_drain(self):
        # 100 requests offered over 10 s, timed to the last completion. The
        # recorded throughput is below target, but the arrival-window rate is
        # exactly the offered rate.
        record = MockLedgerCandidate(
            used_gpus=2,
            metrics=_demand_metrics(10.0, e2e_ms=4000.0, window_s=10.0),
        )
        assert record.metrics["request_throughput_rps"] < 10.0

        winner, met, _, _ = app_module._select_demand_record([record], 10.0)
        assert winner is record
        assert met is True


class TestDemandWindows:
    def test_ladder_doubles_from_the_start_window_to_the_ceiling(self, monkeypatch):
        monkeypatch.setenv("AISIMULATORS_MAX_CANDIDATE_GPUS", "64")
        monkeypatch.delenv("DEMAND_WINDOW_START_GPUS", raising=False)
        request = app_module.RecommendRequest.model_validate(VALID_DEMAND_BODY)

        assert app_module._demand_gpu_windows(request) == [
            (1, 8), (1, 16), (1, 32), (1, 64),
        ]

    def test_ceiling_below_the_start_window_yields_one_window(self, monkeypatch):
        monkeypatch.setenv("AISIMULATORS_MAX_CANDIDATE_GPUS", "4")
        request = app_module.RecommendRequest.model_validate(VALID_DEMAND_BODY)

        assert app_module._demand_gpu_windows(request) == [(1, 4)]

    def test_explicit_max_pins_a_single_window(self, monkeypatch):
        monkeypatch.setenv("AISIMULATORS_MAX_CANDIDATE_GPUS", "1024")
        request = app_module.RecommendRequest.model_validate(
            {**VALID_DEMAND_BODY, "max_candidate_gpus": 16}
        )

        assert app_module._demand_gpu_windows(request) == [(1, 16)]

    def test_minimum_gpu_count_is_kept_as_the_lower_bound(self, monkeypatch):
        monkeypatch.setenv("AISIMULATORS_MAX_CANDIDATE_GPUS", "64")
        request = app_module.RecommendRequest.model_validate(
            {**VALID_DEMAND_BODY, "min_candidate_gpus": 4}
        )

        windows = app_module._demand_gpu_windows(request)
        assert windows[0] == (4, 8)
        assert windows[1] == (4, 16)


class TestRunDemandWindows:
    def _request(self, **overrides):
        return app_module.RecommendRequest.model_validate(
            {**VALID_DEMAND_BODY, **overrides}
        )

    def test_stops_at_the_first_window_that_sustains_the_rate(self, monkeypatch):
        monkeypatch.setenv("AISIMULATORS_MAX_CANDIDATE_GPUS", "32")
        windows: list[tuple[int, int]] = []

        def fake(request, *, max_trials_override=None, candidate_timeout_override=None):
            windows.append((request.min_candidate_gpus, request.max_candidate_gpus))
            return MockLedgerRecommendationResult(
                candidates=[
                    MockLedgerCandidate(
                        candidate_id="fits", used_gpus=8, metrics=_demand_metrics(12.0)
                    )
                ],
                counts=MockRecommendationCounts(evaluated=4, feasible=1, timed_out=0),
                provenance=MockRecommendationProvenance(run_id="run-demand"),
            )

        monkeypatch.setattr(app_module, "_run_aisimulate_recommendation", fake)

        records, sweep = app_module._run_demand_windows(self._request())

        assert windows == [(1, 8)]
        assert [record.candidate_id for record in records] == ["fits"]
        assert sweep.window_min_gpus == 1
        assert sweep.window_max_gpus == 8
        assert sweep.run_id == "run-demand"
        assert sweep.strict_sla is False
        assert sweep.evaluated == 4

    def test_skips_a_window_that_cannot_hold_the_model(self, monkeypatch):
        monkeypatch.setenv("AISIMULATORS_MAX_CANDIDATE_GPUS", "32")

        class NoViableParallelConfig(Exception):
            pass

        windows: list[tuple[int, int]] = []

        def fake(request, *, max_trials_override=None, candidate_timeout_override=None):
            windows.append((request.min_candidate_gpus, request.max_candidate_gpus))
            if len(windows) == 1:
                raise NoViableParallelConfig(
                    "no deployment_mode has a viable parallel config"
                )
            return MockLedgerRecommendationResult(
                candidates=[
                    MockLedgerCandidate(
                        candidate_id="fits", used_gpus=16, metrics=_demand_metrics(12.0)
                    )
                ],
                counts=MockRecommendationCounts(evaluated=2, feasible=1, timed_out=0),
            )

        monkeypatch.setattr(app_module, "_run_aisimulate_recommendation", fake)

        records, sweep = app_module._run_demand_windows(self._request())

        assert windows == [(1, 8), (1, 16)]
        assert [record.candidate_id for record in records] == ["fits"]
        assert sweep.window_max_gpus == 16
        assert sweep.evaluated == 2
        assert sweep.feasible == 1

    def test_merges_and_deduplicates_records_across_windows(self, monkeypatch):
        monkeypatch.setenv("AISIMULATORS_MAX_CANDIDATE_GPUS", "16")
        windows: list[tuple[int, int]] = []

        def fake(request, *, max_trials_override=None, candidate_timeout_override=None):
            windows.append((request.min_candidate_gpus, request.max_candidate_gpus))
            if len(windows) == 1:
                return MockLedgerRecommendationResult(
                    candidates=[
                        MockLedgerCandidate(
                            candidate_id="small", used_gpus=2, metrics=_demand_metrics(5.0)
                        ),
                        MockLedgerCandidate(
                            candidate_id="medium", used_gpus=4, metrics=_demand_metrics(6.0)
                        ),
                    ],
                    counts=MockRecommendationCounts(evaluated=2, feasible=2, timed_out=0),
                )
            return MockLedgerRecommendationResult(
                candidates=[
                    # A re-measured shape from the previous window.
                    MockLedgerCandidate(
                        candidate_id="small-again", used_gpus=2, metrics=_demand_metrics(5.0)
                    ),
                    MockLedgerCandidate(
                        candidate_id="large", used_gpus=8, metrics=_demand_metrics(12.0)
                    ),
                ],
                counts=MockRecommendationCounts(evaluated=3, feasible=2, timed_out=1),
            )

        monkeypatch.setattr(app_module, "_run_aisimulate_recommendation", fake)

        records, sweep = app_module._run_demand_windows(self._request())

        assert windows == [(1, 8), (1, 16)]
        assert {record.candidate_id for record in records} == {"small", "medium", "large"}
        assert sweep.evaluated == 5
        assert sweep.feasible == 4
        assert sweep.timed_out == 1
        assert sweep.max_trials > 0

    def test_reports_budget_exhausted_when_a_window_spends_its_trials(self, monkeypatch):
        monkeypatch.setenv("AISIMULATORS_MAX_CANDIDATE_GPUS", "16")

        def fake(request, *, max_trials_override=None, candidate_timeout_override=None):
            # Evaluate more trials than the window offered, so the window
            # reports that its budget was spent without meeting the target.
            return MockLedgerRecommendationResult(
                candidates=[
                    MockLedgerCandidate(
                        candidate_id="weak", used_gpus=2, metrics=_demand_metrics(1.0)
                    )
                ],
                counts=MockRecommendationCounts(evaluated=12, feasible=1, timed_out=0),
            )

        monkeypatch.setattr(app_module, "_run_aisimulate_recommendation", fake)

        _, sweep = app_module._run_demand_windows(self._request())

        assert sweep.budget_exhausted is True
        assert sweep.evaluated >= 12

    def test_stops_growing_when_the_wall_clock_budget_is_spent(self, monkeypatch):
        monkeypatch.setenv("AISIMULATORS_MAX_CANDIDATE_GPUS", "64")
        monkeypatch.setattr(app_module, "_demand_search_budget_seconds", lambda: 1)
        windows: list[tuple[int, int]] = []

        def fake(request, *, max_trials_override=None, candidate_timeout_override=None):
            windows.append((request.min_candidate_gpus, request.max_candidate_gpus))
            return MockLedgerRecommendationResult(
                candidates=[
                    MockLedgerCandidate(
                        candidate_id="weak", used_gpus=2, metrics=_demand_metrics(1.0)
                    )
                ],
                counts=MockRecommendationCounts(evaluated=1, feasible=1, timed_out=0),
            )

        monkeypatch.setattr(app_module, "_run_aisimulate_recommendation", fake)

        _, sweep = app_module._run_demand_windows(self._request())

        assert windows == [(1, 8)]
        assert sweep.budget_exhausted is True
        assert sweep.window_max_gpus == 8


class TestDemandConfigFromRecord:
    def test_maps_aggregated_topology_and_measured_metrics(self):
        record = MockLedgerCandidate(
            used_gpus=4,
            metrics=_demand_metrics(12.0),
            provenance=MockLedgerProvenance(topology={
                "deployment_mode": "agg",
                "tp": 4,
                "pp": 1,
                "attention_dp": 1,
                "replicas": 1,
                "moe_tp": 2,
                "moe_ep": 2,
            }),
        )
        request = app_module.RecommendRequest.model_validate(VALID_DEMAND_BODY)

        config = app_module._demand_config_from_record(record, request)

        assert config.total_gpus_needed == 4
        assert config.num_total_gpus == 4
        assert config.replicas_needed == 1
        assert config.tp == 4
        assert config.moe_tp == 2
        assert config.mode == "agg"
        assert config.request_rate == 12.0
        # Little's law: 12 req/s x 2 s mean E2E.
        assert config.concurrency == 24
        assert config.prefill_config is None
        assert config.decode_config is None

    def test_maps_disaggregated_topology_into_role_configs(self):
        record = MockLedgerCandidate(
            used_gpus=6,
            metrics=_demand_metrics(9.0),
            provenance=MockLedgerProvenance(topology={
                "deployment_mode": "disagg",
                "prefill_tp": 4,
                "prefill_pp": 1,
                "prefill_attention_dp": 1,
                "prefill_replicas": 1,
                "decode_tp": 2,
                "decode_pp": 1,
                "decode_attention_dp": 1,
                "decode_replicas": 2,
            }),
        )
        request = app_module.RecommendRequest.model_validate(VALID_DEMAND_BODY)

        config = app_module._demand_config_from_record(record, request)

        assert config.mode == "disagg"
        # A complete P/D deployment is one scaling unit.
        assert config.num_total_gpus == 6
        assert config.prefill_config is not None
        assert config.prefill_config.tp == 4
        assert config.decode_config is not None
        assert config.decode_config.tp == 2
        assert config.decode_config.num_workers == 2


class TestDemandOutcome:
    def test_reports_replay_derived_window_and_concurrency(self):
        record = MockLedgerCandidate(
            used_gpus=2,
            metrics=_demand_metrics(12.0, e2e_ms=2000.0, window_s=10.0),
        )
        outcome = app_module._demand_outcome(
            record,
            target_rps=10.0,
            met=True,
            best_observed_rps=40.0,
            evaluated=5,
        )

        assert outcome.target_rps == 10.0
        # 120 requests over the 10 s arrival window, excluding the e2e drain.
        assert outcome.achieved_rps == 12.0
        assert outcome.met is True
        assert outcome.best_observed_rps == 40.0
        assert outcome.evaluated_configurations == 5
        # Little's law: 12 req/s x 2 s = 24 in flight.
        assert outcome.observed_concurrency == 24.0
        assert outcome.window_seconds == 10.0
        assert outcome.completed_requests == 120
        assert outcome.simulator == "aisimulate"

    def test_omits_derived_values_when_metrics_are_missing(self):
        record = MockLedgerCandidate(metrics={"request_throughput_rps": 4.0})
        outcome = app_module._demand_outcome(
            record,
            target_rps=10.0,
            met=False,
            best_observed_rps=None,
            evaluated=1,
        )

        assert outcome.achieved_rps == 4.0
        # Falling back to the selected candidate's own rate is explicit.
        assert outcome.best_observed_rps == 4.0
        assert outcome.observed_concurrency is None
        assert outcome.window_seconds is None


class TestSweepMeta:
    def test_reports_budget_exhaustion_without_claiming_exhaustive_search(self, monkeypatch):
        monkeypatch.setenv("DEMAND_MAX_TRIALS", "4")
        request = app_module.RecommendRequest.model_validate(VALID_DEMAND_BODY)
        result = MockLedgerRecommendationResult(
            candidates=[MockLedgerCandidate()],
            counts=MockRecommendationCounts(evaluated=4, feasible=1, timed_out=0),
            provenance=MockRecommendationProvenance(run_id="run-42"),
        )

        meta = app_module._sweep_meta(result, request)

        assert meta.run_id == "run-42"
        assert meta.max_trials == 4
        assert meta.evaluated == 4
        assert meta.budget_exhausted is True
        assert meta.strict_sla is False
        assert meta.window_min_gpus == 1
        assert meta.window_max_gpus == 1024
        assert meta.algorithm in {"random", "bayesian"}

    def test_not_exhausted_when_the_search_stops_early(self, monkeypatch):
        monkeypatch.setenv("DEMAND_MAX_TRIALS", "16")
        request = app_module.RecommendRequest.model_validate(VALID_DEMAND_BODY)
        result = MockLedgerRecommendationResult(
            candidates=[MockLedgerCandidate(), MockLedgerCandidate(candidate_id="c2")],
            counts=MockRecommendationCounts(evaluated=2, feasible=2, timed_out=0),
        )

        meta = app_module._sweep_meta(result, request)
        assert meta.budget_exhausted is False
        assert meta.evaluated == 2

    def test_falls_back_to_ledger_length_without_counts(self, monkeypatch):
        monkeypatch.setenv("DEMAND_MAX_TRIALS", "16")
        request = app_module.RecommendRequest.model_validate(VALID_DEMAND_BODY)
        result = MockLedgerRecommendationResult(
            candidates=[MockLedgerCandidate(), MockLedgerCandidate(candidate_id="c2")],
        )

        meta = app_module._sweep_meta(result, request)
        assert meta.evaluated == 2


class TestRecommendDemandEndpoint:
    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_met_demand_returns_200_with_measured_outcome(self, mock_recommend):
        small = MockLedgerCandidate(
            candidate_id="small", used_gpus=1, metrics=_demand_metrics(4.0)
        )
        medium = MockLedgerCandidate(
            candidate_id="medium", used_gpus=2, metrics=_demand_metrics(12.0)
        )
        mock_recommend.return_value = MockLedgerRecommendationResult(
            candidates=[small, medium],
            counts=MockRecommendationCounts(evaluated=2),
        )

        resp = client.post("/recommend", json=VALID_DEMAND_BODY)

        assert resp.status_code == 200
        data = resp.json()
        assert data["demand"] is not None
        assert data["demand"]["met"] is True
        assert data["demand"]["target_rps"] == 10.0
        assert data["demand"]["achieved_rps"] == 12.0
        assert data["demand"]["evaluated_configurations"] == 2
        assert data["demand"]["simulator"] == "aisimulate"
        # The winner, not the SDK's throughput-per-GPU pick, is the response.
        assert data["configs"][0]["total_gpus_needed"] == 2
        assert data["sweep"]["strict_sla"] is False

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_unmet_demand_returns_200_with_best_observed(self, mock_recommend):
        mock_recommend.return_value = MockLedgerRecommendationResult(
            candidates=[
                MockLedgerCandidate(candidate_id="a", used_gpus=1, metrics=_demand_metrics(3.0)),
                MockLedgerCandidate(candidate_id="b", used_gpus=4, metrics=_demand_metrics(7.0)),
            ],
            counts=MockRecommendationCounts(evaluated=2),
        )

        resp = client.post("/recommend", json=VALID_DEMAND_BODY)

        assert resp.status_code == 200
        demand = resp.json()["demand"]
        assert demand["met"] is False
        assert demand["achieved_rps"] == 7.0
        assert demand["best_observed_rps"] == 7.0
        assert demand["target_rps"] == 10.0

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_no_feasible_candidate_returns_422(self, mock_recommend):
        mock_recommend.return_value = MockLedgerRecommendationResult(
            candidates=[
                MockLedgerCandidate(status="infeasible", metrics={}),
                MockLedgerCandidate(candidate_id="c2", status="timed_out", metrics={}),
            ],
        )

        resp = client.post("/recommend", json=VALID_DEMAND_BODY)

        assert resp.status_code == 422
        assert "demand traffic" in resp.json()["detail"]

    def test_rejects_an_enforced_sla_before_running(self):
        body = {**VALID_DEMAND_BODY, "sla_mode": "enforce", "ttft": 1000.0}
        resp = client.post("/recommend", json=body)
        assert resp.status_code == 422

    @patch("tools.api_service.app._backend_memory_fraction_kind", return_value="of_total")
    @patch("tools.api_service.app._backend_memory_fraction", return_value=0.9)
    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_include_config_sizes_from_measured_concurrency(self, mock_recommend, _fraction, _kind):
        # 12 req/s at 2 s mean latency is 24 concurrent requests. A demand
        # winner has no requested `bs`, so the config must fall back to the
        # measured concurrency instead of a placeholder of 1.
        mock_recommend.return_value = MockLedgerRecommendationResult(
            candidates=[MockLedgerCandidate(used_gpus=2, metrics=_demand_metrics(12.0))],
            counts=MockRecommendationCounts(evaluated=1),
        )

        resp = client.post("/recommend?include=config", json=VALID_DEMAND_BODY)

        assert resp.status_code == 200
        serving = resp.json()["configs"][0]["serving_config"]
        assert serving["max_num_seqs"] == 24

    @patch("tools.api_service.app._build_memory_breakdown")
    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_include_memory_sizes_from_measured_concurrency(self, mock_recommend, mock_breakdown):
        mock_recommend.return_value = MockLedgerRecommendationResult(
            candidates=[MockLedgerCandidate(used_gpus=2, metrics=_demand_metrics(12.0))],
            counts=MockRecommendationCounts(evaluated=1),
        )
        mock_breakdown.return_value = None

        resp = client.post("/recommend?include=memory", json=VALID_DEMAND_BODY)

        assert resp.status_code == 200
        # Positional args: ..., isl, osl, concurrency.
        assert mock_breakdown.call_args.args[8] == 24


# ─── /memory tests ───────────────────────────────────────────────────────────


class TestMemory:

    @patch("tools.api_service.app.estimate_kv_cache")
    def test_success(self, mock_kv):
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        resp = client.post("/memory", json=VALID_MEMORY_BODY)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_gpu_capacity_bytes"] == 151397597184
        assert data["total_kv_size_bytes"] == 98507266457
        assert data["kv_size_per_token_bytes"] == 131072
        assert data["total_kv_size_tokens"] == 751550
        assert data["source"] == "native"

    @patch("tools.api_service.app.estimate_kv_cache")
    def test_memory_breakdown(self, mock_kv):
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        resp = client.post("/memory", json=VALID_MEMORY_BODY)
        mb = resp.json()["memory_breakdown"]
        assert mb["weights_bytes"] == 32761446400
        assert mb["activations_bytes"] == 532480000
        assert mb["runtime_overhead_bytes"] == 3758096384
        assert mb["comm_overhead_bytes"] == 358612992
        assert mb["kv_cache_bytes"] == 98507266457

    @patch("tools.api_service.app.estimate_kv_cache")
    def test_value_error_returns_422(self, mock_kv):
        mock_kv.side_effect = ValueError("unsupported model/backend/GPU for KV-cache estimation")
        resp = client.post("/memory", json=VALID_MEMORY_BODY)
        assert resp.status_code == 422
        assert "No performance data" in resp.json()["detail"]

    @patch("tools.api_service.app.estimate_kv_cache")
    def test_unexpected_error_returns_500(self, mock_kv):
        mock_kv.side_effect = RuntimeError("crash")
        resp = client.post("/memory", json=VALID_MEMORY_BODY)
        assert resp.status_code == 500

    def test_requires_model_path(self):
        body = {**VALID_MEMORY_BODY}
        del body["model_path"]
        resp = client.post("/memory", json=body)
        assert resp.status_code == 422

    def test_requires_system(self):
        body = {**VALID_MEMORY_BODY}
        del body["system"]
        resp = client.post("/memory", json=body)
        assert resp.status_code == 422

    @patch("tools.api_service.app.estimate_kv_cache")
    def test_defaults_applied(self, mock_kv):
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        body = {"model_path": "Qwen/Qwen3-32B", "system": "h200_sxm"}
        resp = client.post("/memory", json=body)
        assert resp.status_code == 200
        call_kwargs = mock_kv.call_args.kwargs
        assert call_kwargs["backend"] == "vllm"
        assert call_kwargs["max_num_tokens"] == 8192
        assert call_kwargs["max_batch_size"] == 128
        assert call_kwargs["tp_size"] == 1
        assert call_kwargs["memory_fraction_kind"] == "of_total"
        assert call_kwargs["memory_fraction_value"] == 1.0


# ─── /models tests ───────────────────────────────────────────────────────────


MOCK_MODEL_CFG_MOE = {
    "architecture": "Qwen3MoeForCausalLM",
    "num_experts": 128,
    "topk": 8,
    "context": 40960,
    "n": 64,
    "n_kv": 4,
}

MOCK_MODEL_CFG_DENSE = {
    "architecture": "Qwen3ForCausalLM",
    "num_experts": 0,
    "topk": 0,
    "context": 40960,
    "n": 64,
    "n_kv": 8,
}


class TestModels:

    @patch("tools.api_service.app.get_default_models")
    def test_returns_sorted_list(self, mock_models):
        mock_models.return_value = {"Zeta/Z-1B", "Alpha/A-7B", "Meta/M-70B"}
        resp = client.get("/models")
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert models == ["Alpha/A-7B", "Meta/M-70B", "Zeta/Z-1B"]

    @patch("tools.api_service.app.get_default_models")
    def test_empty_set(self, mock_models):
        mock_models.return_value = set()
        resp = client.get("/models")
        assert resp.status_code == 200
        assert resp.json()["models"] == []

    @patch("tools.api_service.app.get_model_config_from_model_path")
    @patch("tools.api_service.app.get_default_models")
    def test_include_specs_returns_objects(self, mock_models, mock_cfg):
        mock_models.return_value = {"Qwen/Qwen3-235B-A22B"}
        mock_cfg.return_value = MOCK_MODEL_CFG_MOE
        resp = client.get("/models?include=specs")
        assert resp.status_code == 200
        model = resp.json()["models"][0]
        assert model["id"] == "Qwen/Qwen3-235B-A22B"
        assert model["num_experts"] == 128
        assert model["num_experts_per_tok"] == 8
        assert model["context_length"] == 40960
        assert model["num_attn_heads"] == 64
        assert model["num_kv_heads"] == 4
        assert model["architecture"] == "Qwen3MoeForCausalLM"

    @patch("tools.api_service.app.get_model_config_from_model_path")
    @patch("tools.api_service.app.get_default_models")
    def test_include_specs_dense_model_nulls_moe_fields(self, mock_models, mock_cfg):
        mock_models.return_value = {"Qwen/Qwen3-32B"}
        mock_cfg.return_value = MOCK_MODEL_CFG_DENSE
        resp = client.get("/models?include=specs")
        model = resp.json()["models"][0]
        assert model["num_experts"] is None
        assert model["num_experts_per_tok"] is None
        assert model["architecture"] == "Qwen3ForCausalLM"
        assert model["context_length"] == 40960

    @patch("tools.api_service.app.get_model_config_from_model_path")
    @patch("tools.api_service.app.get_default_models")
    def test_include_specs_cfg_error_returns_nulls(self, mock_models, mock_cfg):
        mock_models.return_value = {"Unknown/Model"}
        mock_cfg.side_effect = Exception("not found")
        resp = client.get("/models?include=specs")
        assert resp.status_code == 200
        model = resp.json()["models"][0]
        assert model["id"] == "Unknown/Model"
        assert model["num_experts"] is None
        assert model["architecture"] is None

    @patch("tools.api_service.app.get_default_models")
    def test_no_include_returns_strings(self, mock_models):
        mock_models.return_value = {"Alpha/A-7B"}
        resp = client.get("/models")
        assert isinstance(resp.json()["models"][0], str)


# ─── /backends tests ─────────────────────────────────────────────────────────


class FakeBackend:
    """Stand-in for an SDK backend: fraction keyed by version, no SDK import."""

    def __init__(self, fractions: dict[str | None, float]):
        self.fractions = fractions

    def get_default_free_gpu_memory_fraction(self, backend_version: str | None = None) -> float:
        return self.fractions[backend_version]

    def memory_fraction_of_free(self) -> bool:
        return False


class TestBackends:

    def _patch_backends(self, monkeypatch, supported, slots, fractions):
        monkeypatch.setattr(app_module, "get_supported_databases", lambda: supported)
        monkeypatch.setattr(
            app_module, "get_version_slots",
            lambda system, backend: slots.get((system, backend)),
        )
        monkeypatch.setattr(app_module, "get_backend", lambda backend_id: FakeBackend(fractions))

    def test_default_version_is_the_resolvable_alias_not_the_newest_declared(self, monkeypatch):
        """A literal default is only ever correct on the systems that declare it."""
        self._patch_backends(
            monkeypatch,
            supported={
                "h200_sxm": {"vllm": ["0.24.0", "0.31.0"]},
                "a100_sxm": {"vllm": ["0.14.0"]},
                "b60": {"vllm": ["0.26.0", "0.28.0"]},
            },
            slots={
                ("h200_sxm", "vllm"): {"current": "0.24.0", "next": "0.31.0"},
                ("a100_sxm", "vllm"): {"current": "0.14.0"},
                ("b60", "vllm"): {"current": "0.28.0", "previous": "0.26.0"},
            },
            fractions={"0.24.0": 0.92, "0.14.0": 0.9, "0.28.0": 0.92, None: 0.92},
        )

        backend = client.get("/backends").json()["backends"][0]

        assert backend["default_version"] == "current"
        # The union still advertises every version; only the default namespace moved.
        assert backend["versions"] == ["0.14.0", "0.24.0", "0.26.0", "0.28.0", "0.31.0"]
        # AISimulate's P/D search takes no fabric, so it must not ask for one.
        assert backend["disaggregation_requires_fabric"] is False

    def test_system_defaults_follow_each_systems_current_slot(self, monkeypatch):
        """The per-system default and its memory fraction must come from the same version."""
        self._patch_backends(
            monkeypatch,
            supported={"h200_sxm": {"vllm": ["0.24.0", "0.31.0"]}, "a100_sxm": {"vllm": ["0.14.0"]}},
            slots={
                ("h200_sxm", "vllm"): {"current": "0.24.0", "next": "0.31.0"},
                ("a100_sxm", "vllm"): {"current": "0.14.0"},
            },
            fractions={"0.24.0": 0.92, "0.14.0": 0.9, None: 0.92},
        )

        backend = client.get("/backends").json()["backends"][0]

        assert backend["system_defaults"] == {
            "a100_sxm": {"version": "0.14.0", "memory_fraction": 0.9},
            "h200_sxm": {"version": "0.24.0", "memory_fraction": 0.92},
        }
        # Fleet figure follows the majority current version, not the newest declared.
        assert backend["memory_fraction"] == 0.92

    def test_numeric_version_order_is_used_when_slots_are_absent(self, monkeypatch):
        self._patch_backends(
            monkeypatch,
            supported={"h200_sxm": {"vllm": ["0.9.0", "0.10.0"]}},
            slots={},
            fractions={"0.9.0": 0.9, "0.10.0": 0.92, None: 0.92},
        )

        backend = client.get("/backends").json()["backends"][0]

        assert backend["default_version"] == "0.10.0"
        assert backend["system_defaults"]["h200_sxm"] == {
            "version": "0.10.0", "memory_fraction": 0.92,
        }

    def test_without_any_version_gate_default_is_newest_not_majority(self, monkeypatch):
        """The no-gate default is newest declared; fleet fraction remains majority-based."""
        self._patch_backends(
            monkeypatch,
            supported={
                "h200_sxm": {"vllm": ["0.14.0"]},
                "h100_sxm": {"vllm": ["0.14.0"]},
                "b60": {"vllm": ["0.31.0"]},
            },
            slots={},
            fractions={"0.14.0": 0.9, "0.31.0": 0.92, None: 0.92},
        )

        backend = client.get("/backends").json()["backends"][0]

        assert backend["default_version"] == "0.31.0"
        assert backend["memory_fraction"] == 0.9

    def test_current_alias_requires_slots_for_every_system(self, monkeypatch):
        """A mixed gated/ungated fleet must not advertise an alias the latter rejects."""
        self._patch_backends(
            monkeypatch,
            supported={
                "h200_sxm": {"vllm": ["0.24.0", "0.31.0"]},
                "ungated_gpu": {"vllm": ["0.26.0"]},
            },
            slots={("h200_sxm", "vllm"): {"current": "0.24.0", "next": "0.31.0"}},
            fractions={"0.24.0": 0.92, "0.26.0": 0.9, None: 0.92},
        )

        backend = client.get("/backends").json()["backends"][0]

        assert backend["default_version"] == "0.26.0"
        assert backend["system_defaults"]["h200_sxm"]["version"] == "0.24.0"
        assert backend["system_defaults"]["ungated_gpu"]["version"] == "0.26.0"


# ─── /systems tests ──────────────────────────────────────────────────────────


class TestSystems:

    @patch.dict("tools.api_service.app._DEVICE_DISPLAY_NAMES", {"h200_sxm": "NVIDIA H200 SXM", "a100_sxm": "NVIDIA A100-SXM4-80GB"})
    @patch("tools.api_service.app.supported_systems", lambda: {"h200_sxm", "a100_sxm"})
    def test_returns_sorted_objects(self):
        resp = client.get("/systems")
        assert resp.status_code == 200
        systems = resp.json()["systems"]
        assert len(systems) == 2
        assert systems[0]["id"] == "a100_sxm"
        assert systems[1]["id"] == "h200_sxm"
        assert "name" in systems[0]
        assert "vendor" not in systems[0]

    @patch.dict("tools.api_service.app._DEVICE_DISPLAY_NAMES", {"h200_sxm": "NVIDIA H200 SXM"})
    @patch("tools.api_service.app.load_system_spec")
    @patch("tools.api_service.app.supported_systems", lambda: {"h200_sxm"})
    def test_include_specs(self, mock_spec):
        mock_spec.return_value = MOCK_SYSTEM_SPEC
        resp = client.get("/systems?include=specs")
        assert resp.status_code == 200
        sys = resp.json()["systems"][0]
        assert sys["id"] == "h200_sxm"
        assert sys["name"] == "NVIDIA H200 SXM"
        assert sys["vendor"] == "nvidia"
        assert sys["architecture"] == "hopper"
        assert sys["memory_bytes"] == 151397597184
        assert sys["tdp_watts"] == 700.0
        assert sys["gpus_per_node"] == 8

    @patch.dict("tools.api_service.app._DEVICE_DISPLAY_NAMES", {"h200_sxm": "NVIDIA H200 SXM"})
    @patch("tools.api_service.app.load_system_spec")
    @patch("tools.api_service.app.supported_systems", lambda: {"h200_sxm"})
    def test_include_specs_bandwidth_and_tflops(self, mock_spec):
        mock_spec.return_value = MOCK_SYSTEM_SPEC
        resp = client.get("/systems?include=specs")
        sys = resp.json()["systems"][0]
        assert sys["memory_bandwidth_bytes"] == 4800000000000
        assert abs(sys["bf16_tflops"] - 989.0) < 0.1

    @patch.dict("tools.api_service.app._DEVICE_DISPLAY_NAMES", {"h200_sxm": "NVIDIA H200 SXM"})
    @patch("tools.api_service.app.supported_systems", lambda: {"h200_sxm"})
    def test_no_include_omits_specs(self):
        resp = client.get("/systems")
        sys = resp.json()["systems"][0]
        assert "vendor" not in sys
        assert "memory_bytes" not in sys

    @patch.dict("tools.api_service.app._DEVICE_DISPLAY_NAMES", {"h200_sxm": "NVIDIA H200 SXM"})
    @patch("tools.api_service.app.load_system_spec")
    @patch("tools.api_service.app.supported_systems", lambda: {"h200_sxm"})
    def test_spec_failure_still_returns_entry(self, mock_spec):
        mock_spec.side_effect = FileNotFoundError("missing yaml")
        resp = client.get("/systems?include=specs")
        assert resp.status_code == 200
        sys = resp.json()["systems"][0]
        assert sys["id"] == "h200_sxm"
        assert "vendor" not in sys

    @patch.dict(
        "tools.api_service.app._DEVICE_DISPLAY_NAMES",
        {"h200_sxm": "NVIDIA H200 SXM"},
        clear=True,
    )
    @patch("tools.api_service.app.supported_systems", lambda: {"h200_sxm", "l4"})
    def test_hides_systems_without_display_name(self):
        # l4 is supported but has no perf-data display name -> excluded.
        resp = client.get("/systems")
        assert resp.status_code == 200
        ids = [s["id"] for s in resp.json()["systems"]]
        assert ids == ["h200_sxm"]

    def test_startup_fails_without_display_names(self):
        with (
            patch.object(app_module, "load_device_names_from_perf_data", return_value={}),
            pytest.raises(RuntimeError, match="valid performance data"),
        ):
            app_module.startup_event()


# ─── Integration tests (require SDK) ─────────────────────────────────────────


# A 422 from these endpoints has three sources, only one of which is a benign
# "skip in CI" signal:
#   * genuine no-perf-data / no-feasible-config — the fallback perf database has
#     no data for this request. These use endpoint-controlled, stable strings:
#     "No performance data available for model=..." (_common_error_handler) and
#     "No configuration meets the specified requirements." (/recommend fallback),
#     plus NoFeasibleConfigError. Safe to skip.
#   * any other endpoint error — e.g. an unexpected AttributeError (like the
#     plotext-v6 incompatibility now pinned out via plotext<6), or an
#     unsupported/absent model/backend/system. A real failure that MUST fail.
#   * FastAPI request-validation — a list detail; the bodies below are all
#     hardcoded-valid, so this never happens here.
# Match the known no-data messages by text, not "any string detail": a spurious
# 422 (e.g. the next dependency-drift bug) then surfaces as a failure instead of
# a silent skip. Safe against CI flakiness because CI's fallback DB has data for
# these requests (all integration tests return 200), so this predicate is only
# ever reached in an environment that genuinely lacks perf data.
_NO_PERF_DATA_MARKERS = (
    "no performance data available",   # _common_error_handler perf-data path
    "no empirical performance data",   # _PerfDataGap (version/shape coverage)
    "no configuration meets",          # /recommend fallback (no config found)
    "no feasible",                     # NoFeasibleConfigError
)


def _missing_perf_data(resp) -> bool:
    if resp.status_code != 422:
        return False
    detail = resp.json().get("detail")
    if not isinstance(detail, str):
        return False
    lowered = detail.lower()
    return any(marker in lowered for marker in _NO_PERF_DATA_MARKERS)


def _skip_if_missing_perf_data(resp) -> None:
    """Skip (not fail) when the fallback perf database has no data for a request,
    embedding the exact 422 body in the skip reason. Combined with pytest's
    `-ra` summary (see pyproject addopts), this makes CI logs always show the
    precise detail string the SDK returned for a no-data 422 — so if the
    fallback data or its error wording ever changes, we can read the real
    message straight from the CI summary instead of adding a throwaway probe.
    """
    if _missing_perf_data(resp):
        pytest.skip(f"no perf data in fallback database; 422 detail: {resp.text}")


@pytest.mark.skipif(
    not _sdk_available(),
    reason="aisimulate SDK not installed or missing perf data",
)
class TestIntegration:

    @classmethod
    def setup_class(cls):
        device_names = load_device_names_from_perf_data()
        if not device_names:
            pytest.skip("device display names unavailable")
        # Replicate startup_event(): only genuine loaded names, no id fallbacks.
        # Systems absent from this map are hidden by /systems.
        app_module._DEVICE_DISPLAY_NAMES = device_names

    def test_recommend_real(self):
        resp = client.post("/recommend", json={
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "target_concurrency": 32,
            "ttft": 2000.0,
            "tpot": 30.0,
            "top_n": 1,
        })
        _skip_if_missing_perf_data(resp)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert len(data["configs"]) == 1
        cfg = data["configs"][0]
        assert cfg["total_gpus_needed"] >= 1
        assert cfg["tp"] >= 1
        assert cfg["ttft"] > 0
        assert cfg["tokens_per_second"] > 0

    def test_recommend_with_include(self):
        resp = client.post("/recommend?include=config,memory", json={
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "target_concurrency": 32,
            "ttft": 2000.0,
            "tpot": 30.0,
            "top_n": 1,
        })
        _skip_if_missing_perf_data(resp)
        assert resp.status_code == 200, resp.text
        cfg = resp.json()["configs"][0]
        assert cfg["serving_config"] is not None
        assert cfg["serving_config"]["tensor_parallel_size"] >= 1

    def test_memory_real(self):
        resp = client.post("/memory", json={
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.24.0",
            "tp_size": 2,
        })
        _skip_if_missing_perf_data(resp)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_kv_size_bytes"] > 0
        assert data["memory_breakdown"]["weights_bytes"] > 0

    def test_models_real(self):
        resp = client.get("/models")
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_systems_real(self):
        resp = client.get("/systems?include=specs")
        assert resp.status_code == 200
        systems = resp.json()["systems"]
        assert len(systems) > 0
        assert all(s["memory_bytes"] > 0 for s in systems)

    def test_systems_hides_systems_without_perf_data(self):
        """Regression guard for the GPU dropdown culling.

        /systems must return exactly the systems that have a genuine perf-data
        display name, and never a system whose only "name" is its raw id (e.g.
        a30, l4, a100_pcie, h100_pcie — SDK dirs with no parquet). A prior
        "startup fallback dict" change (58466b6) reintroduced id-fallbacks and
        dropped the cull, surfacing those raw ids in the UI.

        Ground truth is recomputed here from configiq.systems rather than read
        from the app's module state, so this cannot be defeated by adjusting
        startup_event() or the test setup to build a fallback dict.
        """
        named = set(load_device_names_from_perf_data())
        supported = supported_systems()

        resp = client.get("/systems")
        assert resp.status_code == 200
        systems = resp.json()["systems"]
        returned = {s["id"] for s in systems}

        # Exactly the genuinely-named systems appear — no id fallbacks.
        assert returned == named & supported
        # Supported-but-unnamed systems (no parquet) must be excluded.
        assert returned.isdisjoint(supported - named)
        # The visible symptom of the bug: a system shown with no real name.
        for s in systems:
            assert s["name"] != s["id"], f"{s['id']} surfaced without a display name"

    def test_estimate_real(self):
        resp = client.post("/estimate", json={
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "backend": "vllm",
            "tp_size": 2,
            "batch_size": 48,
        })
        _skip_if_missing_perf_data(resp)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["ttft"] > 0
        assert data["tpot"] > 0
        assert data["tokens_per_second"] > 0

    def test_estimate_with_include_real(self):
        resp = client.post("/estimate?include=config,memory", json={
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.24.0",
            "tp_size": 2,
            "batch_size": 48,
        })
        _skip_if_missing_perf_data(resp)
        assert resp.status_code == 200, resp.text
        cfg = resp.json()
        assert cfg["serving_config"] is not None
        assert cfg["serving_config"]["tensor_parallel_size"] == 2


# ─── /recommend disagg tests ──────────────────────────────────────────────────


class TestRecommendDisagg:

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_disagg_result_has_prefill_decode_configs(self, mock_recommend):
        candidate = MockCandidate(
            used_gpus=6,
            metrics={"ttft_ms": 180.157, "tpot_ms": 24.675},
            prediction_config={"engine": {
                "model": "Qwen/Qwen3-32B", "hardware": "h200_sxm",
                "backend": "vllm", "backend_version": "0.24.0", "mode": "disaggregated",
                "workers": {
                    "prefill": {"parallelism": {"tensor": 1, "pipeline": 1,
                        "attention_data": 1, "replicas": 1},
                        "scheduler": {"max_sequences": 1}},
                    "decode": {"parallelism": {"tensor": 4, "pipeline": 1,
                        "attention_data": 1, "replicas": 1},
                        "scheduler": {"max_sequences": 36}},
                },
            }},
        )
        mock_recommend.return_value = make_mock_recommendation_result([candidate])

        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 200
        data = resp.json()
        assert "disagg" in data["chosen_mode"]
        cfg = data["configs"][0]
        assert cfg["tp"] is None
        assert cfg["prefill_config"] is not None
        assert cfg["decode_config"] is not None
        assert cfg["prefill_config"]["tp"] == 1
        assert cfg["decode_config"]["tp"] == 4
        # Batch size and context parallel are surfaced per worker role.
        assert cfg["prefill_config"]["batch_size"] == 1
        assert cfg["decode_config"]["batch_size"] == 36
        assert cfg["prefill_config"]["cp"] == 1
        assert cfg["total_gpus_needed"] == 6
        # Per-GPU peak memory is the worst-case across pools, NOT the sum
        # (each (x)memory is checked against a single GPU's capacity).
        assert cfg["memory"] is None

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_agg_result_has_no_prefill_decode(self, mock_recommend):
        mock_recommend.return_value = make_mock_recommendation_result()
        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)
        assert resp.status_code == 200
        cfg = resp.json()["configs"][0]
        assert cfg["tp"] == 2
        assert cfg["prefill_config"] is None
        assert cfg["decode_config"] is None


# ─── /estimate tests ──────────────────────────────────────────────────────────

VALID_ESTIMATE_BODY = {
    "model_path": "Qwen/Qwen3-32B",
    "system": "h200_sxm",
    "backend": "vllm",
    "isl": 4000,
    "osl": 1000,
    "tp_size": 2,
    "batch_size": 128,
}


def make_mock_estimate_result():
    return MockPredictionResult(summary={
        "ttft_ms": 471.378, "tpot_ms": 28.118,
        "output_throughput_tok_s": 1678.925,
        "output_throughput_tok_s_per_gpu": 839.462,
    })


class TestEstimate:

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_predict_endpoint(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        resp = client.post("/predict", json=VALID_ESTIMATE_BODY)

        assert resp.status_code == 200
        assert resp.json()["ttft"] == pytest.approx(471.378)
        # VALID_ESTIMATE_BODY uses TP=2 with no widths: the API reports the
        # resolved default layout it sent.
        assert resp.json()["moe_tp"] == 2
        assert resp.json()["moe_ep"] == 1
        assert resp.json()["provenance"]["simulator"] == "aisimulate"
        assert resp.json()["provenance"]["requested_model"] == VALID_ESTIMATE_BODY["model_path"]

    @patch("tools.api_service.app._blis_estimate")
    def test_predict_provenance_fallback_matches_requested_simulator(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        body = {**VALID_ESTIMATE_BODY, "simulator": "blis"}

        resp = client.post("/predict", json=body)

        assert resp.status_code == 200
        assert resp.json()["simulator"] == "blis"
        assert resp.json()["provenance"]["simulator"] == "blis"
        assert resp.json()["provenance"]["requested_model"] == body["model_path"]

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_predict_derives_per_gpu_throughput_from_requested_topology(self, mock_estimate):
        mock_estimate.return_value = MockPredictionResult(summary={
            "ttft_ms": 471.378,
            "tpot_ms": 28.118,
            "output_throughput_tok_s": 1678.925,
        })
        resp = client.post("/predict", json=VALID_ESTIMATE_BODY)

        assert resp.status_code == 200
        assert resp.json()["tokens_per_second_per_gpu"] == pytest.approx(839.462)

    def test_estimate_is_marked_deprecated(self):
        schema = client.get("/openapi.json").json()
        assert schema["paths"]["/estimate"]["post"]["deprecated"] is True

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_unqueryable_version_is_a_client_error(self, mock_estimate):
        """A version the system does not offer is the caller's to fix, not a 500."""
        mock_estimate.side_effect = RuntimeError(
            "ValueError: no timing data for vllm 0.31.0 on a100_sxm; "
            "available: 0.14.0 (current)"
        )

        resp = client.post("/predict", json={**VALID_ESTIMATE_BODY, "backend_version": "0.31.0"})

        assert resp.status_code == 422
        assert resp.json()["detail"] == (
            "no timing data for vllm 0.31.0 on a100_sxm; available: 0.14.0 (current)"
        )

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_unqueryable_version_keeps_the_sdk_guidance(self, mock_estimate):
        mock_estimate.side_effect = ValueError(
            "vllm version '0.31.0' is not a queryable version on a100_sxm. "
            "Accepted versions for a100_sxm/vllm: current = 0.14.0."
        )

        resp = client.post("/predict", json={**VALID_ESTIMATE_BODY, "backend_version": "0.31.0"})

        assert resp.status_code == 422
        assert "Accepted versions for a100_sxm/vllm" in resp.json()["detail"]

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_unexpected_failure_is_still_a_server_error(self, mock_estimate):
        mock_estimate.side_effect = RuntimeError("disk on fire")

        resp = client.post("/predict", json=VALID_ESTIMATE_BODY)

        assert resp.status_code == 500
        assert resp.json()["detail"] == "disk on fire"

    def test_gpu_memory_default_matches_backend(self):
        assert app_module._backend_memory_fraction("vllm") == pytest.approx(0.92)
        assert app_module._backend_memory_fraction("sglang") == pytest.approx(0.88)

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_success(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        resp = client.post("/estimate", json=VALID_ESTIMATE_BODY)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ttft"] == pytest.approx(471.378)
        assert data["tpot"] == pytest.approx(28.118)
        assert data["tokens_per_second"] == pytest.approx(1678.925)
        assert data["tp"] == 2
        assert data["serving_config"] is None
        assert data["memory_breakdown"] is None

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_include_config(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        resp = client.post("/estimate?include=config", json=VALID_ESTIMATE_BODY)
        assert resp.status_code == 200
        sc = resp.json()["serving_config"]
        assert sc is not None
        assert sc["tensor_parallel_size"] == 2

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_include_config_preserves_serving_controls(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        body = {
            **VALID_ESTIMATE_BODY,
            "prefix": 512,
            "max_num_seqs": 64,
            "enable_chunked_prefill": True,
            "gpu_memory_utilization": 0.97,
        }
        resp = client.post("/estimate?include=config", json=body)

        assert resp.status_code == 200
        assert resp.json()["serving_config"] == {
            "backend": "vllm",
            "tensor_parallel_size": 2,
            "max_model_len": 5000,
            "max_num_seqs": 64,
            "max_num_batched_tokens": None,
            "gpu_memory_utilization": 0.97,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "quantization": "auto",
            "memory_fraction": 0.97,
            "memory_fraction_kind": "of_total",
            "runtime_memory_field": "gpu_memory_utilization",
        }

    @patch("tools.api_service.app.estimate_kv_cache")
    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_include_memory(self, mock_estimate, mock_kv):
        mock_estimate.return_value = make_mock_estimate_result()
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        resp = client.post("/estimate?include=memory", json=VALID_ESTIMATE_BODY)
        assert resp.status_code == 200
        mb = resp.json()["memory_breakdown"]
        assert mb is not None
        assert mb["weights_bytes"] == 32761446400

    @patch("tools.api_service.app.estimate_kv_cache")
    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_include_config_and_memory(self, mock_estimate, mock_kv):
        mock_estimate.return_value = make_mock_estimate_result()
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        resp = client.post("/estimate?include=config,memory", json=VALID_ESTIMATE_BODY)
        assert resp.status_code == 200
        assert resp.json()["serving_config"] is not None
        assert resp.json()["memory_breakdown"] is not None

    def test_zero_moe_width_is_rejected(self):
        for name in ("moe_tp_size", "moe_ep_size"):
            resp = client.post("/estimate", json={**VALID_ESTIMATE_BODY, name: 0})
            assert resp.status_code == 422

    def test_zero_pool_moe_width_is_rejected(self):
        for name in (
            "prefill_moe_tp_size",
            "prefill_moe_ep_size",
            "decode_moe_tp_size",
            "decode_moe_ep_size",
        ):
            resp = client.post("/estimate", json={**VALID_ESTIMATE_BODY, name: 0})
            assert resp.status_code == 422

    @patch("tools.api_service.app.estimate_kv_cache")
    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_include_memory_uses_resolved_attention_width(self, mock_estimate, mock_kv):
        mock_estimate.return_value = make_mock_estimate_result()
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        resp = client.post(
            "/estimate?include=memory",
            json={**VALID_ESTIMATE_BODY, "tp_size": 2, "attention_dp_size": 2},
        )
        assert resp.status_code == 200
        kwargs = mock_kv.call_args.kwargs
        assert kwargs["tp_size"] == 2
        assert kwargs["attention_dp_size"] == 2
        assert kwargs["moe_tp_size"] == 4
        assert kwargs["moe_ep_size"] == 1

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_calls_sdk_with_correct_params(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        client.post("/estimate", json=VALID_ESTIMATE_BODY)
        request, include = mock_estimate.call_args.args
        assert request.system == "h200_sxm"
        assert request.backend == "vllm"
        assert request.tp_size == 2
        assert request.batch_size == 128
        assert include == set()

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_passes_gpu_memory_utilization_to_sdk(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        body = {**VALID_ESTIMATE_BODY, "gpu_memory_utilization": 0.97}
        client.post("/estimate", json=body)
        request, _ = mock_estimate.call_args.args
        assert request.gpu_memory_utilization == pytest.approx(0.97)

    def test_prediction_config_uses_gpu_memory_utilization(self):
        body = {**VALID_ESTIMATE_BODY, "gpu_memory_utilization": 0.97}
        request = app_module.EstimateRequest.model_validate(body)
        config = app_module._aisimulate_prediction_config(request)

        assert config.engine.workers.aggregated.kv_cache.capacity.memory_fraction == pytest.approx(0.97)

    def test_prediction_config_uses_serving_controls(self):
        body = {
            **VALID_ESTIMATE_BODY,
            "prefix": 512,
            "max_num_seqs": 64,
            "enable_chunked_prefill": True,
            "gemm_quant_mode": "fp8",
            "kvcache_quant_mode": "fp8",
        }
        request = app_module.EstimateRequest.model_validate(body)
        config = app_module._aisimulate_prediction_config(request)

        assert config.traffic.source.cached_prefix_tokens == 512
        assert config.engine.workers.aggregated.scheduler.max_sequences == 64
        assert config.engine.enable_chunked_prefill is True
        assert config.engine.gemm_quant_mode == "fp8"
        assert config.engine.kvcache_quant_mode == "fp8"

    def test_prediction_config_defaults_moe_widths_to_attention_group(self):
        body = {**VALID_ESTIMATE_BODY, "tp_size": 4}
        request = app_module.EstimateRequest.model_validate(body)
        config = app_module._aisimulate_prediction_config(request)
        parallel = config.engine.workers.aggregated.parallelism

        assert parallel.moe_tensor == 4
        assert parallel.moe_expert == 1

    def test_prediction_config_preserves_explicit_moe_widths(self):
        body = {
            **VALID_ESTIMATE_BODY,
            "tp_size": 4,
            "moe_tp_size": 2,
            "moe_ep_size": 2,
        }
        request = app_module.EstimateRequest.model_validate(body)
        config = app_module._aisimulate_prediction_config(request)
        parallel = config.engine.workers.aggregated.parallelism

        assert parallel.moe_tensor == 2
        assert parallel.moe_expert == 2

    def test_prediction_config_infers_the_missing_moe_width(self):
        body = {**VALID_ESTIMATE_BODY, "tp_size": 4, "moe_ep_size": 4}
        request = app_module.EstimateRequest.model_validate(body)
        config = app_module._aisimulate_prediction_config(request)
        parallel = config.engine.workers.aggregated.parallelism

        assert parallel.moe_tensor == 1
        assert parallel.moe_expert == 4

    def test_prediction_config_resolves_each_disagg_pool_width(self):
        body = {
            **VALID_ESTIMATE_BODY,
            "mode": "disagg",
            "prefill_tp_size": 8,
            "decode_tp_size": 2,
        }
        request = app_module.EstimateRequest.model_validate(body)
        config = app_module._aisimulate_prediction_config(request)

        prefill = config.engine.workers.prefill.parallelism
        decode = config.engine.workers.decode.parallelism
        assert prefill.moe_tensor == 8
        assert prefill.moe_expert == 1
        assert decode.moe_tensor == 2
        assert decode.moe_expert == 1

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_disagg_mode(self, mock_estimate):
        mock_estimate.return_value = MockPredictionResult(
            summary={"ttft_ms": 500.0, "tpot_ms": 30.0,
                     "output_throughput_tok_s": 1000.0}
        )
        body = {
            **VALID_ESTIMATE_BODY,
            "mode": "disagg",
            "prefill_tp_size": 1, "prefill_pp_size": 1,
            "prefill_num_workers": 4, "prefill_batch_size": 1,
            "decode_tp_size": 4, "decode_pp_size": 1,
            "decode_num_workers": 1, "decode_batch_size": 64,
        }
        resp = client.post("/estimate", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "disagg"
        # Top-level parallelism is None for disagg; see prefill/decode_config.
        assert data["tp"] is None
        assert data["prefill_config"]["tp"] == 1
        assert data["prefill_config"]["moe_tp"] == 1
        assert data["prefill_config"]["moe_ep"] == 1
        assert data["prefill_config"]["num_workers"] == 4
        assert data["prefill_config"]["batch_size"] == 1
        assert data["decode_config"]["tp"] == 4
        assert data["decode_config"]["moe_tp"] == 4
        assert data["decode_config"]["moe_ep"] == 1
        assert data["decode_config"]["batch_size"] == 64
        assert data["tokens_per_second_per_gpu"] == pytest.approx(125.0)
        # Per-GPU peak is the worst case across pools, not summed.
        assert data["memory"] is None
        # SDK invoked in disagg mode with the per-role params.
        request, include = mock_estimate.call_args.args
        assert request.mode == "disagg"
        assert request.prefill_num_workers == 4
        assert request.decode_tp_size == 4
        assert include == set()

    def test_invalid_mode_rejected(self):
        body = {**VALID_ESTIMATE_BODY, "mode": "bogus"}
        resp = client.post("/estimate", json=body)
        assert resp.status_code == 422

    def test_requires_model_path(self):
        body = {**VALID_ESTIMATE_BODY}
        del body["model_path"]
        resp = client.post("/estimate", json=body)
        assert resp.status_code == 422

    def test_requires_system(self):
        body = {**VALID_ESTIMATE_BODY}
        del body["system"]
        resp = client.post("/estimate", json=body)
        assert resp.status_code == 422

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_value_error_returns_422(self, mock_estimate):
        mock_estimate.side_effect = ValueError("unsupported model/backend/GPU for estimation")
        resp = client.post("/estimate", json=VALID_ESTIMATE_BODY)
        assert resp.status_code == 422

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_unexpected_error_returns_500(self, mock_estimate):
        mock_estimate.side_effect = RuntimeError("crash")
        resp = client.post("/estimate", json=VALID_ESTIMATE_BODY)
        assert resp.status_code == 500

    @patch("tools.api_service.app._aisimulate_runner_factory")
    @patch("aisimulate.compiler.prediction_to_replay_spec")
    def test_model_config_is_forwarded_to_prediction(self, mock_compile, mock_factory):
        mock_report = MagicMock()
        mock_report.metrics = {
            "ttft_ms": 471.378, "tpot_ms": 28.118,
            "e2e_latency_ms": 28589.5, "output_throughput_tok_s": 4479.8,
            "num_total_gpus": 1.0, "power_w": None,
        }
        mock_report.metadata = {}
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_report
        mock_factory.return_value.create.return_value = mock_runner

        def check_config(config):
            assert Path(config.engine.model, "config.json").is_file()
            return MagicMock()

        mock_compile.side_effect = check_config
        body = {**VALID_ESTIMATE_BODY, "model_config": {"hidden_size": 8192, "architectures": ["LlamaForCausalLM"]}}
        request = app_module.EstimateRequest.model_validate(body)
        app_module._run_aisimulate_prediction(request, set())

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_inclusive_tpot(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        body = {**VALID_ESTIMATE_BODY, "inclusive_tpot": True}
        resp = client.post("/estimate", json=body)
        assert resp.status_code == 200
        data = resp.json()
        # inclusive = (ttft + tpot * (osl - 1)) / osl
        expected = (471.378 + 28.118 * (1000 - 1)) / 1000
        assert data["tpot"] == pytest.approx(expected)
        assert data["ttft"] == pytest.approx(471.378)

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_inclusive_tpot_default_false(self, mock_estimate):
        mock_estimate.return_value = make_mock_estimate_result()
        resp = client.post("/estimate", json=VALID_ESTIMATE_BODY)
        assert resp.json()["tpot"] == pytest.approx(28.118)


# ─── model-aware backend_version resolution ──────────────────────────────────

# The real replay error for a model the selected data never measured. The runner
# wraps it, so the service has to classify it by text, not by exception type.
EMPIRICAL_GAP = (
    "AISimulate replay failed: engine error: generalized engine worker 0 is now poisoned "
    "because a rank may have been partially mutated: executing attention-DP rank 0: external "
    "prefill prediction failed: AIC static_ctx evidence prediction failed: "
    "EmpiricalNotImplementedError: empirical estimation not implemented: No empirical "
    "utilisation data to estimate this op at query=[32.0, 2048.0, 1.0]: no own-shape, "
    "cross-shape, or sibling transfer reference available."
)


class TestVersionFallback:
    """A blank backend_version must find a version that can serve the model."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        app_module._clear_version_fallback_cache()
        yield
        app_module._clear_version_fallback_cache()

    def _patch_versions(self, monkeypatch, supported, slots):
        monkeypatch.setattr(app_module, "get_supported_databases", lambda: supported)
        monkeypatch.setattr(
            app_module, "get_version_slots",
            lambda system, backend: slots.get((system, backend)),
        )

    def _patch_versions_h200(self, monkeypatch):
        self._patch_versions(
            monkeypatch,
            supported={"h200_sxm": {"vllm": ["0.24.0", "0.31.0"]}},
            slots={("h200_sxm", "vllm"): {"current": "0.24.0", "next": "0.31.0"}},
        )

    def test_system_default_follows_the_current_slot(self, monkeypatch):
        self._patch_versions_h200(monkeypatch)
        assert app_module._system_default_version("h200_sxm", "vllm") == "0.24.0"

    def test_system_default_is_newest_without_a_version_gate(self, monkeypatch):
        self._patch_versions(
            monkeypatch,
            supported={"h200_sxm": {"vllm": ["0.9.0", "0.10.0"]}},
            slots={},
        )
        assert app_module._system_default_version("h200_sxm", "vllm") == "0.10.0"

    def test_blank_version_uses_a_version_that_can_serve_the_model(self, monkeypatch):
        self._patch_versions_h200(monkeypatch)
        calls: list[str | None] = []

        def fake_prediction(req, include):
            calls.append(req.backend_version)
            if req.backend_version == "0.24.0":
                raise RuntimeError(EMPIRICAL_GAP)
            return make_mock_estimate_result()

        monkeypatch.setattr(app_module, "_run_aisimulate_prediction", fake_prediction)

        resp = client.post("/estimate", json=VALID_ESTIMATE_BODY)

        assert resp.status_code == 200
        data = resp.json()
        # Report the version that produced the numbers, not the caller's blank.
        assert data["backend_version"] == "0.31.0"
        assert calls == ["0.24.0", "0.31.0"]
        assert data["provenance"]["engine_version"] == "0.31.0"
        assert "used 0.31.0" in data["warnings"][0]

    def test_the_fallback_is_cached_per_model_system_and_backend(self, monkeypatch):
        self._patch_versions_h200(monkeypatch)
        calls: list[str | None] = []

        def fake_prediction(req, include):
            calls.append(req.backend_version)
            if req.backend_version == "0.24.0":
                raise RuntimeError(EMPIRICAL_GAP)
            return make_mock_estimate_result()

        monkeypatch.setattr(app_module, "_run_aisimulate_prediction", fake_prediction)
        client.post("/estimate", json=VALID_ESTIMATE_BODY)

        calls.clear()
        resp = client.post("/estimate", json=VALID_ESTIMATE_BODY)

        assert resp.status_code == 200
        # The failed default is not retried on every request.
        assert calls == ["0.31.0"]

    def test_explicit_version_is_never_swapped_and_names_the_alternatives(self, monkeypatch):
        self._patch_versions_h200(monkeypatch)
        calls: list[str | None] = []

        def fake_prediction(req, include):
            calls.append(req.backend_version)
            raise RuntimeError(EMPIRICAL_GAP)

        monkeypatch.setattr(app_module, "_run_aisimulate_prediction", fake_prediction)

        resp = client.post("/estimate", json={**VALID_ESTIMATE_BODY, "backend_version": "0.24.0"})

        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "No empirical performance data" in detail
        assert "0.24.0" in detail
        # The client is told which versions the SDK ships so it can retry.
        assert "0.31.0" in detail
        assert calls == ["0.24.0"]

    def test_blank_version_reports_every_version_it_tried(self, monkeypatch):
        self._patch_versions_h200(monkeypatch)

        def fake_prediction(req, include):
            raise RuntimeError(EMPIRICAL_GAP)

        monkeypatch.setattr(app_module, "_run_aisimulate_prediction", fake_prediction)

        resp = client.post("/estimate", json=VALID_ESTIMATE_BODY)

        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "Tried 0.24.0, 0.31.0" in detail
        assert "compute shape [32, 2048, 1]" in detail

    def test_other_failures_are_not_retried(self, monkeypatch):
        self._patch_versions_h200(monkeypatch)
        calls: list[str | None] = []

        def fake_prediction(req, include):
            calls.append(req.backend_version)
            raise RuntimeError("crash")

        monkeypatch.setattr(app_module, "_run_aisimulate_prediction", fake_prediction)

        resp = client.post("/estimate", json=VALID_ESTIMATE_BODY)

        assert resp.status_code == 500
        assert calls == ["0.24.0"]

    def test_recommend_resolves_a_blank_version_the_same_way(self, monkeypatch):
        self._patch_versions_h200(monkeypatch)
        calls: list[str | None] = []

        def fake_recommendation(req):
            calls.append(req.backend_version)
            if req.backend_version == "0.24.0":
                raise RuntimeError(EMPIRICAL_GAP)
            # Report the version the SDK would have used for this attempt.
            candidate = MockCandidate()
            candidate.prediction_config["engine"]["backend_version"] = req.backend_version
            return make_mock_recommendation_result([candidate])

        monkeypatch.setattr(
            app_module, "_run_aisimulate_recommendation", fake_recommendation
        )

        resp = client.post("/recommend", json=VALID_RECOMMEND_BODY)

        assert resp.status_code == 200
        # The recommendation searched the version that has the model's data.
        assert calls == ["0.24.0", "0.31.0"]
        # Every returned config reports the version that served the request and
        # says the default could not.
        configs = resp.json()["configs"]
        assert configs
        assert all(cfg["backend_version"] == "0.31.0" for cfg in configs)
        assert all("used 0.31.0" in cfg["warnings"][0] for cfg in configs)

    def test_the_current_alias_falls_back_like_a_blank_version(self, monkeypatch):
        """``current`` means "the system default", not a pin on a literal version."""
        self._patch_versions_h200(monkeypatch)
        calls: list[str | None] = []

        def fake_prediction(req, include):
            calls.append(req.backend_version)
            if req.backend_version == "0.24.0":
                raise RuntimeError(EMPIRICAL_GAP)
            return make_mock_estimate_result()

        monkeypatch.setattr(app_module, "_run_aisimulate_prediction", fake_prediction)

        resp = client.post(
            "/estimate", json={**VALID_ESTIMATE_BODY, "backend_version": "current"}
        )

        assert resp.status_code == 200
        assert calls == ["0.24.0", "0.31.0"]
        assert resp.json()["backend_version"] == "0.31.0"

    def test_other_slot_aliases_are_pins(self, monkeypatch):
        """``previous`` and ``next`` name a slot; never swap them silently."""
        self._patch_versions_h200(monkeypatch)
        calls: list[str | None] = []

        def fake_prediction(req, include):
            calls.append(req.backend_version)
            raise RuntimeError(EMPIRICAL_GAP)

        monkeypatch.setattr(app_module, "_run_aisimulate_prediction", fake_prediction)

        resp = client.post(
            "/estimate", json={**VALID_ESTIMATE_BODY, "backend_version": "previous"}
        )

        assert resp.status_code == 422
        assert calls == ["previous"]

    def test_the_fallback_cache_is_scoped_to_the_request_shape(self, monkeypatch):
        self._patch_versions_h200(monkeypatch)
        calls: list[tuple[int, str | None]] = []

        def fake_prediction(req, include):
            calls.append((req.isl, req.backend_version))
            if req.isl == 8192 and req.backend_version == "0.24.0":
                raise RuntimeError(EMPIRICAL_GAP)
            return make_mock_estimate_result()

        monkeypatch.setattr(app_module, "_run_aisimulate_prediction", fake_prediction)

        long_context = client.post("/estimate", json={**VALID_ESTIMATE_BODY, "isl": 8192})
        assert long_context.status_code == 200
        assert calls == [(8192, "0.24.0"), (8192, "0.31.0")]

        calls.clear()
        # A different shape still gets the default, which covers it.
        short_context = client.post("/estimate", json=VALID_ESTIMATE_BODY)
        assert short_context.status_code == 200
        assert calls == [(4000, "0.24.0")]

    def test_the_fallback_cache_evicts_least_recently_used_shapes(self, monkeypatch):
        self._patch_versions_h200(monkeypatch)
        monkeypatch.setattr(app_module, "_VERSION_FALLBACK_CACHE_MAX", 2)

        def fake_prediction(req, include):
            if req.backend_version == "0.24.0":
                raise RuntimeError(EMPIRICAL_GAP)
            return make_mock_estimate_result()

        monkeypatch.setattr(app_module, "_run_aisimulate_prediction", fake_prediction)
        for isl in (1024, 2048, 4096):
            response = client.post("/estimate", json={**VALID_ESTIMATE_BODY, "isl": isl})
            assert response.status_code == 200

        assert len(app_module._VERSION_FALLBACK_CACHE) == 2

    def test_the_cache_is_invalidated_when_the_system_default_changes(self, monkeypatch):
        versions = {"h200_sxm": {"vllm": ["0.24.0", "0.28.0", "0.31.0"]}}
        self._patch_versions(
            monkeypatch,
            supported=versions,
            slots={("h200_sxm", "vllm"): {"current": "0.24.0", "next": "0.31.0"}},
        )
        calls: list[str | None] = []

        def fake_prediction(req, include):
            calls.append(req.backend_version)
            if req.backend_version == "0.24.0":
                raise RuntimeError(EMPIRICAL_GAP)
            return make_mock_estimate_result()

        monkeypatch.setattr(app_module, "_run_aisimulate_prediction", fake_prediction)
        client.post("/estimate", json=VALID_ESTIMATE_BODY)
        assert calls == ["0.24.0", "0.31.0"]

        # The SDK moves `current` to 0.28.0; the cached 0.31.0 must not bypass it.
        self._patch_versions(
            monkeypatch,
            supported=versions,
            slots={("h200_sxm", "vllm"): {"current": "0.28.0", "next": "0.31.0"}},
        )
        calls.clear()
        resp = client.post("/estimate", json=VALID_ESTIMATE_BODY)

        assert resp.status_code == 200
        assert calls == ["0.28.0"]

    def test_a_cached_fallback_that_is_no_longer_declared_is_ignored(self, monkeypatch):
        self._patch_versions_h200(monkeypatch)
        request = app_module.EstimateRequest.model_validate(VALID_ESTIMATE_BODY)
        key = app_module._fallback_cache_key(
            request, app_module._system_default_version("h200_sxm", "vllm")
        )
        app_module._VERSION_FALLBACK_CACHE[key] = "0.99.0"
        calls: list[str | None] = []

        def fake_prediction(req, include):
            calls.append(req.backend_version)
            return make_mock_estimate_result()

        monkeypatch.setattr(app_module, "_run_aisimulate_prediction", fake_prediction)

        resp = client.post("/estimate", json=VALID_ESTIMATE_BODY)

        assert resp.status_code == 200
        # A data update can retire a version; the stale entry is not replayed.
        assert calls == ["0.24.0"]

    def test_unwrapped_coverage_error_is_mapped_to_422(self, monkeypatch):
        self._patch_versions_h200(monkeypatch)
        error = RuntimeError(EMPIRICAL_GAP)
        with pytest.raises(app_module.HTTPException) as raised:
            app_module._common_error_handler(error, "memory", "org/model", "vllm", "h200_sxm")
        assert raised.value.status_code == 422
        assert "No empirical performance data" in raised.value.detail
        # /memory does not pick a version, so it names the ones the SDK ships.
        assert "Versions the SDK ships for h200_sxm: 0.24.0, 0.31.0" in raised.value.detail


# ─── model_config passthrough tests ──────────────────────────────────────────

class TestModelConfigPassthrough:

    @patch("aisimulate.recommend.run_recommendation")
    @patch("tools.api_service.app._aisimulate_runner_factory")
    def test_recommend_model_config_is_forwarded(self, mock_factory, mock_run):
        def check(config, **_kwargs):
            assert Path(config.engine.model, "config.json").is_file()
            return make_mock_recommendation_result()

        mock_run.side_effect = check
        body = {**VALID_RECOMMEND_BODY, "model_config": {"hidden_size": 8192, "architectures": ["LlamaForCausalLM"]}}
        request = app_module.RecommendRequest.model_validate(body)
        app_module._run_aisimulate_recommendation(request)

    @patch("tools.api_service.app.estimate_kv_cache")
    def test_memory_accepts_model_config(self, mock_kv):
        mock_kv.return_value = MOCK_KV_CACHE_RESULT
        body = {**VALID_MEMORY_BODY, "model_config": {"hidden_size": 8192, "architectures": ["LlamaForCausalLM"]}}
        resp = client.post("/memory", json=body)
        assert resp.status_code == 200

    @patch("aisimulate.recommend.run_recommendation")
    @patch("tools.api_service.app._aisimulate_runner_factory")
    def test_empty_model_config_is_ignored(self, mock_factory, mock_run):
        def check(config, **_kwargs):
            assert config.engine.model == "Qwen/Qwen3-32B"
            return make_mock_recommendation_result()

        mock_run.side_effect = check
        body = {**VALID_RECOMMEND_BODY, "model_config": {}}
        request = app_module.RecommendRequest.model_validate(body)
        app_module._run_aisimulate_recommendation(request)

    def test_invalid_model_config_reaches_aisimulate(self):
        body = {**VALID_RECOMMEND_BODY, "model_config": {"additionalProp1": {}}}
        request = app_module.RecommendRequest.model_validate(body)
        with patch("aisimulate.recommend.run_recommendation", side_effect=ValueError("invalid model config")), pytest.raises(
            ValueError, match="invalid model config"
        ):
            app_module._run_aisimulate_recommendation(request)


class TestOpenTelemetry:
    """Tests for OpenTelemetry instrumentation integration."""

    def test_otel_gracefully_degrades_when_unavailable(self):
        """Service starts successfully even if OpenTelemetry deps aren't installed."""
        # The app already initialized; verify it responds
        resp = client.get("/systems")
        assert resp.status_code == 200

    def test_obs_flag_set_correctly(self):
        """Verify the _OBS flag is set based on configiq[otel] import success."""
        # The flag should be True if the otel extra is installed, False otherwise
        assert isinstance(app_module._OBS, bool)

        # If available, the shared observability module should have been imported
        if app_module._OBS:
            assert hasattr(app_module, "observability")


class TestMCPServer:
    """Tests for Model Context Protocol (MCP) server integration."""

    def test_mcp_gracefully_degrades_when_unavailable(self):
        """Service starts successfully even if fastapi-mcp isn't installed."""
        resp = client.get("/systems")
        assert resp.status_code == 200

    # Every REST operation the README promises as an MCP tool, by the name
    # fastapi-mcp gives it (the FastAPI operation id).
    EXPECTED_MCP_TOOLS = frozenset({
        "post_recommend_recommend_post",
        "post_predict_predict_post",
        "post_memory_memory_post",
        "get_models_models_get",
        "get_systems_systems_get",
        "get_backends_backends_get",
    })

    def test_mcp_exposes_every_api_route(self):
        """MCP is mounted after the last route, so every endpoint is a tool.

        fastapi-mcp snapshots the routes when the server is built; mounting
        earlier silently exposed only /backends.
        """
        if not app_module._MCP:
            pytest.skip("fastapi-mcp not installed")
        tools = {t.name for t in app_module._MCP_SERVER.tools}
        missing = self.EXPECTED_MCP_TOOLS - tools
        assert not missing, f"not exposed as MCP tools: {sorted(missing)}"

    def test_mcp_serves_sse_and_streamable_http(self):
        """SSE stays at /mcp; streamable HTTP is served at /mcp/http."""
        if not app_module._MCP:
            pytest.skip("fastapi-mcp not installed")
        routes = {(getattr(r, "path", ""), m) for r in app.routes for m in (getattr(r, "methods", None) or ())}
        assert ("/mcp", "GET") in routes
        assert ("/mcp/messages/", "POST") in routes
        assert any(getattr(r, "path", "") == "/mcp/http" for r in app.routes)

    def test_streamable_http_lists_the_tools(self):
        """An MCP client speaking streamable HTTP sees the API's tools."""
        if not app_module._MCP:
            pytest.skip("fastapi-mcp not installed")
        with TestClient(app, follow_redirects=False) as c:
            names = _mcp_http_tool_names(c, "/mcp/http")
        assert self.EXPECTED_MCP_TOOLS <= names


def _mcp_http_tool_names(c: TestClient, path: str) -> set[str]:
    """initialize -> notifications/initialized -> tools/list over streamable HTTP."""
    import json as _json

    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}

    def rpc(body: dict, session: str | None = None):
        h = dict(headers)
        if session:
            h["mcp-session-id"] = session
        resp = c.post(path, json=body, headers=h)
        assert resp.status_code in (200, 202), resp.text
        return resp

    def payload(resp) -> dict:
        if resp.headers.get("content-type", "").startswith("text/event-stream"):
            data = [ln[5:].strip() for ln in resp.text.splitlines() if ln.startswith("data:")]
            return _json.loads(data[-1])
        return resp.json()

    init = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-03-26", "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"}}})
    session = init.headers.get("mcp-session-id")
    assert payload(init)["result"]["serverInfo"]["name"]
    rpc({"jsonrpc": "2.0", "method": "notifications/initialized"}, session)
    listed = payload(rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, session))
    return {t["name"] for t in listed["result"]["tools"]}


class TestMetrics:
    """Tests for /metrics endpoint with content negotiation."""

    def test_metrics_unavailable_without_otel(self):
        """Metrics endpoint returns 503 when OpenTelemetry not installed."""
        if app_module._OBS:
            pytest.skip("the otel extra is installed")

        resp = client.get("/metrics")
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()

    def test_metrics_prometheus_format_default(self):
        """Metrics endpoint returns Prometheus text format by default."""
        if not app_module._OBS:
            pytest.skip("the otel extra is not installed")

        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        # Prometheus format should contain metric names and HELP/TYPE comments
        assert b"# HELP" in resp.content or b"# TYPE" in resp.content

    def test_metrics_prometheus_format_explicit(self):
        """Metrics endpoint returns Prometheus format with text/plain Accept header."""
        if not app_module._OBS:
            pytest.skip("the otel extra is not installed")

        resp = client.get("/metrics", headers={"Accept": "text/plain"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")

    def test_metrics_otlp_json_format(self):
        """Metrics endpoint returns OTLP JSON format with worker identification."""
        if not app_module._OBS:
            pytest.skip("the otel extra is not installed")

        # Make a request to generate some metrics
        client.get("/systems")

        resp = client.get("/metrics", headers={"Accept": "application/json"})
        assert resp.status_code == 200

        # Verify it's valid JSON
        data = resp.json()
        assert "resourceMetrics" in data

        # Verify resource attributes include worker identification
        resource_attrs = data["resourceMetrics"][0]["resource"]["attributes"]
        attr_keys = {attr["key"] for attr in resource_attrs}
        assert "service.instance.id" in attr_keys
        assert "process.pid" in attr_keys

        # Verify metrics are present
        scope_metrics = data["resourceMetrics"][0]["scopeMetrics"]
        assert len(scope_metrics) > 0
        assert len(scope_metrics[0]["metrics"]) > 0

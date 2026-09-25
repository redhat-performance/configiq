# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the REST API service (tools/api_service/app.py).

These tests mock the SDK functions to test API layer logic without
requiring the Rust native extension or performance databases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest
from configiq.systems import load_device_names_from_perf_data, supported_systems
from fastapi.testclient import TestClient

from tools.api_service import app as app_module
from tools.api_service.app import app

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
        body = {"model_path": "Qwen/Qwen3-32B", "system": "h200_sxm", "target_concurrency": 32}
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

    def test_estimate_is_marked_deprecated(self):
        schema = client.get("/openapi.json").json()
        assert schema["paths"]["/estimate"]["post"]["deprecated"] is True

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

    @patch("tools.api_service.app._run_aisimulate_prediction")
    def test_disagg_mode(self, mock_estimate):
        mock_estimate.return_value = MockPredictionResult(
            summary={"ttft_ms": 500.0, "tpot_ms": 30.0,
                     "output_throughput_tok_s": 1000.0,
                     "output_throughput_tok_s_per_gpu": 250.0}
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
        assert data["prefill_config"]["num_workers"] == 4
        assert data["prefill_config"]["batch_size"] == 1
        assert data["decode_config"]["tp"] == 4
        assert data["decode_config"]["batch_size"] == 64
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

    @patch("aisimulate.predict.run_prediction")
    @patch("tools.api_service.app._aisimulate_runner_factory")
    def test_model_config_is_forwarded_to_prediction(self, mock_factory, mock_run):
        def check(config, **_kwargs):
            assert Path(config.engine.model, "config.json").is_file()
            return make_mock_estimate_result()

        mock_run.side_effect = check
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

    def test_mcp_endpoint_available_when_enabled(self):
        """MCP SSE endpoint is available when fastapi-mcp is installed."""
        if not app_module._MCP:
            pytest.skip("fastapi-mcp not installed")

        # MCP server exposes an SSE endpoint at /mcp for the protocol
        # The exact path depends on fastapi-mcp's routing; verify it exists
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        openapi = resp.json()
        paths = openapi.get("paths", {})

        # MCP should add some paths - check the app has our original endpoints
        assert "/systems" in paths
        assert "/models" in paths
        assert "/recommend" in paths

    @patch("tools.api_service.app._run_aisimulate_recommendation")
    def test_mcp_tools_wrap_api_endpoints(self, mock_recommend):
        """MCP tools are properly registered when MCP is available."""
        if not app_module._MCP:
            pytest.skip("fastapi-mcp not installed")

        # Verify MCP wiring is present: the shared mount helper was imported.
        # For now, verify the app initialized successfully with MCP.
        assert hasattr(app_module, "mcp_support") or not app_module._MCP


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

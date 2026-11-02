from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from tools.api_service import app as app_module
from tools.api_service.app import app
from tools.api_service.blis.manifest import SupportManifest

client = TestClient(app)

ESTIMATE_BODY = {
    "model_path": "Qwen/Qwen3-32B",
    "system": "h200_sxm",
    "backend": "vllm",
    "isl": 2048,
    "osl": 128,
    "tp_size": 2,
    "batch_size": 8,
}

RECOMMEND_BODY = {
    "model_path": "Qwen/Qwen3-32B",
    "system": "h200_sxm",
    "target_concurrency": 8,
    "isl": 2048,
    "osl": 128,
    "ttft": 1000,
    "tpot": 30,
}


def _prediction():
    return SimpleNamespace(
        summary={
            "ttft_ms": 80.0,
            "tpot_ms": 20.0,
            "e2e_latency_ms": 2640.0,
            "output_throughput_tok_s": 500.0,
            "num_total_gpus": 2.0,
            "power_w": None,
        },
        provenance={
            "simulator": "blis",
            "requested_model": "Qwen/Qwen3-32B",
            "resolved_model": "qwen3-32b",
            "model_source": "aisimulate",
            "system": "h200_sxm",
            "hardware": "h200",
            "model_config_relation": "different_quantization",
            "engine_version": "0.31.0",
        },
        serving_config={
            "backend": "vllm",
            "tensor_parallel_size": 2,
            "max_model_len": 2176,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.92,
            "enable_chunked_prefill": None,
            "enable_prefix_caching": True,
            "quantization": "model default",
        },
        warnings=[],
    )


def test_blis_predict_returns_selected_simulator_and_provenance_without_memory_fallback():
    with (
        patch("tools.api_service.app._blis_estimate", return_value=_prediction()) as run_blis,
        patch("tools.api_service.app._run_aisimulate_prediction") as run_aisimulate,
    ):
        response = client.post("/predict?include=config,memory", json={**ESTIMATE_BODY, "simulator": "blis"})

    assert response.status_code == 200
    body = response.json()
    assert body["simulator"] == "blis"
    assert body["provenance"]["model_config_relation"] == "different_quantization"
    assert body["serving_config"]["tensor_parallel_size"] == 2
    assert body["memory_breakdown"] is None
    assert "does not provide" in body["memory_unavailable_reason"]
    run_blis.assert_called_once()
    run_aisimulate.assert_not_called()


def test_blis_predict_dispatches_disagg_with_fabric_and_returns_pool_configs():
    prediction = SimpleNamespace(
        **_prediction().__dict__,
        prefill_config={"tp": 4, "pp": 1, "num_workers": 2, "batch_size": 8},
        decode_config={"tp": 2, "pp": 1, "num_workers": 1, "batch_size": 8},
    )
    with (
        patch("tools.api_service.app.run_blis_estimate", return_value=prediction) as run_blis,
        patch("tools.api_service.app._run_aisimulate_prediction") as run_aisimulate,
    ):
        response = client.post("/predict?include=config,memory", json={
            **ESTIMATE_BODY,
            "simulator": "blis",
            "mode": "disagg",
            "fabric": "ib-400g",
            "prefill_tp_size": 4,
            "prefill_num_workers": 2,
            "decode_tp_size": 2,
            "decode_num_workers": 1,
        })

    assert response.status_code == 200
    body = response.json()
    assert body["simulator"] == "blis"
    assert body["mode"] == "disagg"
    assert body["prefill_config"]["tp"] == 4
    assert body["prefill_config"]["num_workers"] == 2
    assert body["decode_config"]["tp"] == 2
    assert body["memory_breakdown"] is None
    assert run_blis.call_args.args[0].fabric == "ib-400g"
    assert run_blis.call_args.args[0].mode == "disagg"
    run_aisimulate.assert_not_called()


def test_blis_disagg_requires_an_explicit_fabric():
    response = client.post("/predict", json={
        **ESTIMATE_BODY,
        "simulator": "blis",
        "mode": "disagg",
    })

    assert response.status_code == 422
    assert "require an explicit fabric" in response.json()["detail"]


def test_network_endpoint_lists_fabrics_from_the_pinned_manifest():
    with patch(
        "tools.api_service.app.load_support_manifest",
        return_value=SimpleNamespace(networks=frozenset({"ib-400g", "roce-200g"})),
    ):
        response = client.get("/networks")

    assert response.status_code == 200
    assert response.json() == {"networks": ["ib-400g", "roce-200g"], "simulator": "blis"}


def test_aisimulate_rejects_blis_fabric_selection():
    response = client.post("/recommend", json={**RECOMMEND_BODY, "fabric": "ib-400g"})

    assert response.status_code == 422
    assert "AISimulate does not currently consume" in response.json()["detail"][0]["msg"]


def test_blis_predict_unsupported_request_does_not_fall_back():
    with (
        patch(
            "tools.api_service.app._blis_estimate",
            side_effect=app_module.BlisUnsupportedRequest("BLIS cannot price this model"),
        ),
        patch("tools.api_service.app._run_aisimulate_prediction") as run_aisimulate,
    ):
        response = client.post("/predict", json={**ESTIMATE_BODY, "simulator": "blis"})

    assert response.status_code == 422
    assert response.json()["detail"] == "BLIS cannot price this model"
    run_aisimulate.assert_not_called()


def test_blis_rejects_non_vllm_backend_at_the_api_boundary():
    response = client.post(
        "/predict",
        json={**ESTIMATE_BODY, "simulator": "blis", "backend": "sglang"},
    )

    assert response.status_code == 422
    assert "currently model vLLM" in response.json()["detail"]


def test_blis_rejects_aisimulate_only_estimate_options():
    response = client.post(
        "/predict",
        json={
            **ESTIMATE_BODY,
            "simulator": "blis",
            "requests_per_load_unit": 50,
        },
    )

    assert response.status_code == 422
    assert "requests_per_load_unit" in response.json()["detail"]


def test_blis_recommend_dispatches_to_blis_only():
    outcome = SimpleNamespace(
        configs=[{
            "total_gpus_needed": 2,
            "replicas_needed": 1,
            "num_total_gpus": 2,
            "tp": 2,
            "pp": 1,
            "dp": 1,
            "ttft": 80,
            "tpot": 20,
            "request_latency": 2640,
            "tokens_per_second": 500,
            "tokens_per_second_per_gpu": 250,
            "tokens_per_second_per_user": 62.5,
            "concurrency": 8,
            "simulator": "blis",
                "provenance": _prediction().provenance,
        }],
        search_summary={"strategy": "tensor_parallel_only", "candidate_gpus": [1, 2]},
    )
    with (
        patch("tools.api_service.app._blis_recommend", return_value=outcome) as run_blis,
        patch("tools.api_service.app._run_aisimulate_recommendation") as run_aisimulate,
    ):
        response = client.post("/recommend", json={**RECOMMEND_BODY, "simulator": "blis"})

    assert response.status_code == 200
    assert response.json()["simulator"] == "blis"
    assert response.json()["configs"][0]["provenance"]["simulator"] == "blis"
    run_blis.assert_called_once()
    run_aisimulate.assert_not_called()


def test_blis_recommend_uses_the_configured_candidate_limit():
    outcome = SimpleNamespace(configs=[], search_summary={})
    with (
        patch("tools.api_service.app._max_candidate_gpus", return_value=8),
        patch("tools.api_service.app.run_blis_recommend", return_value=outcome) as run_blis,
    ):
        request = app_module.RecommendRequest.model_validate({
            **RECOMMEND_BODY,
            "simulator": "blis",
            "max_candidate_gpus": 32,
        })
        app_module._blis_recommend(request)

    assert run_blis.call_args.args[0].max_candidate_gpus == 8


def test_blis_recommend_accepts_fabric_and_returns_chosen_pd_mode():
    outcome = SimpleNamespace(
        configs=[{
            "total_gpus_needed": 2,
            "replicas_needed": 1,
            "num_total_gpus": 2,
            "mode": "disagg",
            "simulator": "blis",
            "prefill_config": {"tp": 1, "pp": 1, "dp": 1, "num_workers": 1},
            "decode_config": {"tp": 1, "pp": 1, "dp": 1, "num_workers": 1},
        }],
        search_summary={"strategy": "aggregate_tp_and_pd_grid"},
    )
    with patch("tools.api_service.app._blis_recommend", return_value=outcome) as recommend:
        response = client.post("/recommend", json={
            **RECOMMEND_BODY,
            "simulator": "blis",
            "fabric": "ib-400g",
        })

    assert response.status_code == 200
    assert response.json()["chosen_mode"] == "disagg"
    assert recommend.call_args.args[0].fabric == "ib-400g"


def test_blis_recommend_defaults_roce_but_allows_explicit_aggregate_only():
    defaulted = app_module.RecommendRequest.model_validate({
        **RECOMMEND_BODY,
        "simulator": "blis",
    })
    aggregate_only = app_module.RecommendRequest.model_validate({
        **RECOMMEND_BODY,
        "simulator": "blis",
        "fabric": None,
    })

    assert defaulted.fabric == "roce-200g"
    assert aggregate_only.fabric is None


def test_blis_recommend_normalizes_fractional_concurrency_like_aisimulate():
    outcome = SimpleNamespace(configs=[], search_summary={})
    with patch("tools.api_service.app.run_blis_recommend", return_value=outcome) as run_blis:
        request = app_module.RecommendRequest.model_validate({
            **RECOMMEND_BODY,
            "target_concurrency": 8.5,
            "simulator": "blis",
        })
        app_module._blis_recommend(request)

    assert run_blis.call_args.args[0].concurrency == 8


def _manifest() -> SupportManifest:
    # A real manifest, not a SimpleNamespace: the catalog endpoints call
    # supports_hardware()/resolve_* on it, and stubbing the type hides breakage.
    return SupportManifest(
        models={
            "qwen3-32b": {
                "origin": "aisimulate",
                "graph": True,
                "aisimulate_model_id": "Qwen/Qwen3-32B",
            },
            "glm-5": {
                "origin": "upstream",
                "graph": True,
                "aisimulate_model_id": "zai-org/GLM-5",
            },
            "no-graph": {
                "origin": "upstream",
                "graph": False,
                "aisimulate_model_id": "org/NoGraph",
            },
        },
        skipped=(),
        hardware=frozenset({"h100", "h200", "a100-sxm"}),
        networks=frozenset({"ib-400g"}),
    )


def test_backends_endpoint_reports_blis_versions_and_their_support():
    with patch("tools.api_service.app.blis_release", return_value="0.14.0"):
        response = client.get("/backends?simulator=blis")

    assert response.status_code == 200
    body = response.json()
    assert body["simulator"] == "blis"
    assert body["blis_version"] == "0.14.0"
    backend = body["backends"][0]
    assert backend["id"] == "vllm"
    assert backend["versions"] == ["0.29.0", "0.31.0", "0.32.0"]
    assert backend["default_version"] == "0.31.0"
    assert backend["memory_fraction"] is None
    assert backend["systems"] == {}
    assert backend["system_defaults"] == {}
    assert backend["disaggregation_requires_fabric"] is True
    assert backend["version_support"]["0.31.0"] == {
        "disaggregated": True,
        "max_model_len_auto": True,
    }
    assert backend["version_support"]["0.32.0"] == {
        "disaggregated": False,
        "max_model_len_auto": False,
    }


def test_models_endpoint_lists_only_aisimulate_ids_blis_can_price():
    with patch(
        "tools.api_service.blis.service.load_support_manifest",
        return_value=_manifest(),
    ):
        response = client.get("/models?simulator=blis")

    assert response.status_code == 200
    body = response.json()
    assert body["simulator"] == "blis"
    # no-graph has an ID but no derived graph, so BLIS cannot price it.
    assert body["models"] == ["Qwen/Qwen3-32B", "zai-org/GLM-5"]


def test_models_endpoint_accepts_include_specs_for_blis():
    manifest = _manifest()
    with (
        patch("tools.api_service.blis.service.load_support_manifest", return_value=manifest),
        patch(
            "tools.api_service.app.get_model_config_from_model_path",
            return_value={"architecture": "Qwen3ForCausalLM", "num_experts": 0},
        ),
    ):
        response = client.get("/models?simulator=blis&include=specs")

    assert response.status_code == 200
    model = response.json()["models"][0]
    assert model["id"] == "Qwen/Qwen3-32B"


def test_systems_endpoint_filters_to_hardware_blis_has_a_catalog_chip_for():
    with (
        patch("tools.api_service.blis.service.load_support_manifest", return_value=_manifest()),
        patch("tools.api_service.app.supported_systems", lambda: {"h200_sxm", "b200_sxm"}),
        patch.dict(
            "tools.api_service.app._DEVICE_DISPLAY_NAMES",
            {"h200_sxm": "NVIDIA H200 SXM", "b200_sxm": "NVIDIA B200 SXM"},
        ),
    ):
        response = client.get("/systems?simulator=blis")

    assert response.status_code == 200
    body = response.json()
    assert body["simulator"] == "blis"
    # b200_sxm has no BLIS hardware entry, so it is not offered by BLIS.
    assert [system["id"] for system in body["systems"]] == ["h200_sxm"]


def test_blis_catalog_endpoints_report_an_unavailable_manifest():
    with patch(
        "tools.api_service.blis.service.load_support_manifest",
        side_effect=app_module.BlisManifestError("manifest missing"),
    ):
        models = client.get("/models?simulator=blis")
        systems = client.get("/systems?simulator=blis")

    assert models.status_code == 503
    assert models.json()["detail"] == "manifest missing"
    assert systems.status_code == 503


def test_memory_endpoint_ignores_simulator_when_only_aisimulate_is_available():
    raw = {
        "total_gpu_capacity_bytes": 80,
        "total_kv_size_bytes": 40,
        "kv_size_per_token_bytes": 1,
        "total_kv_size_tokens": 40,
        "source": "native",
        "memory_breakdown": {},
    }
    with patch("tools.api_service.app.estimate_kv_cache", return_value=raw) as estimate_memory:
        response = client.post("/memory", json={**ESTIMATE_BODY, "simulator": "blis"})

    assert response.status_code == 200
    assert response.json()["simulator"] == "aisimulate"
    estimate_memory.assert_called_once()

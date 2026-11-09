from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from tools.api_service import app as app_module
from tools.api_service.app import app
from tools.api_service.blis.manifest import BlisCatalog, CatalogModel

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


def _catalog_model(slug: str, repo: str, config: dict | None = None) -> CatalogModel:
    return CatalogModel(
        slug=slug,
        hf_repo=repo,
        config=config or {
            "architectures": ["Qwen3ForCausalLM"],
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "max_position_embeddings": 40960,
        },
    )


def _catalog() -> BlisCatalog:
    # A real catalog, not a SimpleNamespace: the endpoints call resolve_* and
    # specs_for on it, and stubbing the type hides contract breakage.
    return BlisCatalog(
        models={
            "qwen3-32b": _catalog_model("qwen3-32b", "Qwen/Qwen3-32B"),
            "glm-5": _catalog_model("glm-5", "zai-org/GLM-5"),
        },
        hardware=frozenset({"h100", "h200", "a100-sxm"}),
        networks=frozenset({"ib-400g", "roce-200g"}),
    )


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
            "simulator_version": "0.15.0",
            "requested_model": "Qwen/Qwen3-32B",
            "resolved_model": "qwen3-32b",
            "model_source": "blis-catalog",
            "system": "h200_sxm",
            "hardware": "h200",
            "engine_version": "0.31.0",
        },
        serving_config={
            "backend": "vllm",
            "backend_version": "0.31.0",
            "tensor_parallel_size": 2,
            # BLIS resolved the window by policy, so the report carries the
            # policy rather than an invented token count.
            "max_model_len": None,
            "max_model_len_policy": "auto",
            "max_num_seqs": 8,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.92,
            "enable_chunked_prefill": None,
            "enable_prefix_caching": True,
            "quantization": "model default",
        },
        warnings=[],
    )


def test_simulators_endpoint_lists_identity_label_description_and_version():
    response = client.get("/simulators")

    assert response.status_code == 200
    body = response.json()
    ids = [entry["id"] for entry in body["simulators"]]
    assert ids == ["aisimulate", "blis"]
    assert body["default_simulator"] == "aisimulate"
    assert {entry["label"] for entry in body["simulators"]} == {"AISimulate", "BLIS"}
    # The endpoint reports the same installed/built versions the service uses
    # elsewhere, not a literal that can drift from the artifact.
    versions = {entry["id"]: entry["version"] for entry in body["simulators"]}
    assert versions == {
        "aisimulate": app_module._AISIMULATE_VERSION,
        "blis": app_module.blis_release(),
    }
    assert all(isinstance(version, str) and version for version in versions.values())
    # Settings explains each simulator from service text, so every entry must
    # carry its own non-empty description rather than a frontend name table.
    descriptions = {entry["id"]: entry["description"] for entry in body["simulators"]}
    assert all(isinstance(description, str) and description for description in descriptions.values())
    assert len(set(descriptions.values())) == len(descriptions)


def test_blis_predict_returns_selected_simulator_and_provenance_without_memory_fallback():
    with (
        patch("tools.api_service.app._blis_estimate", return_value=_prediction()) as run_blis,
        patch("tools.api_service.app._run_aisimulate_prediction") as run_aisimulate,
    ):
        response = client.post("/predict?include=config,memory", json={**ESTIMATE_BODY, "simulator": "blis"})

    assert response.status_code == 200
    body = response.json()
    assert body["simulator"] == "blis"
    assert body["provenance"]["simulator_version"] == "0.15.0"
    assert body["serving_config"]["tensor_parallel_size"] == 2
    assert body["serving_config"]["max_model_len"] is None
    assert body["serving_config"]["max_model_len_policy"] == "auto"
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


def test_network_endpoint_lists_the_requested_simulators_fabrics_and_default():
    with patch("tools.api_service.app.load_catalog", return_value=_catalog()):
        blis = client.get("/networks?simulator=blis")
        aisimulate = client.get("/networks?simulator=aisimulate")

    assert blis.status_code == 200
    assert blis.json() == {
        "simulator": "blis",
        "networks": ["ib-400g", "roce-200g"],
        "default_network": "roce-200g",
    }
    assert aisimulate.status_code == 200
    assert aisimulate.json() == {"simulator": "aisimulate", "networks": [], "default_network": None}


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


def test_unknown_simulator_is_an_explicit_error_not_a_substitution():
    unknown = "not-a-simulator"
    for path in (
        f"/models?simulator={unknown}",
        f"/systems?simulator={unknown}",
        f"/networks?simulator={unknown}",
        f"/backends?simulator={unknown}",
    ):
        response = client.get(path)
        assert response.status_code == 422, path
        assert "Unknown simulator" in response.json()["detail"]

    recommend = client.post("/recommend", json={**RECOMMEND_BODY, "simulator": unknown})
    predict = client.post("/predict", json={**ESTIMATE_BODY, "simulator": unknown})
    memory = client.post("/memory", json={**ESTIMATE_BODY, "simulator": unknown})
    assert recommend.status_code == 422
    assert predict.status_code == 422
    assert memory.status_code == 422


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


def test_blis_recommend_pareto_returns_candidates_views_and_enumerated_sweep():
    candidate = {
        "candidate_id": "agg-tp1-c1",
        "status": "feasible",
        "mode": "agg",
        "topology": {"deployment_mode": "agg", "tp": 1, "pp": 1, "dp": 1},
        "used_gpus": 1,
        "metrics": {
            "ttft_ms": 50.0,
            "tpot_ms": 10.0,
            "request_latency_ms": 500.0,
            "tokens_per_second": 100.0,
            "concurrency": 1.0,
        },
        "objectives": {"throughput_per_user": 100.0, "throughput_per_gpu": 100.0},
        "transfer_delay_modeled": False,
        "provenance_sources": ["blis"],
        "simulator": "blis",
    }
    outcome = SimpleNamespace(
        configs=[{
            "total_gpus_needed": 1,
            "replicas_needed": 1,
            "num_total_gpus": 1,
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "ttft": 50.0,
            "tpot": 10.0,
            "request_latency": 500.0,
            "tokens_per_second": 100.0,
            "concurrency": 1,
            "mode": "agg",
            "simulator": "blis",
        }],
        search_summary={
            "strategy": "enumerated_topology_and_concurrency_grid",
            "candidate_configurations": 1,
            "evaluated_configurations": 1,
            "effective_max_gpus": 1,
            "budget_exhausted": False,
        },
        candidates=[candidate],
        views={"pareto_front": ["agg-tp1-c1"], "top_n": ["agg-tp1-c1"]},
    )
    body = {
        "model_path": RECOMMEND_BODY["model_path"],
        "system": RECOMMEND_BODY["system"],
        "isl": RECOMMEND_BODY["isl"],
        "osl": RECOMMEND_BODY["osl"],
        "ttft": 1000,
        "tpot": 30,
        "objective": "pareto",
        "simulator": "blis",
    }
    with (
        patch("tools.api_service.app._blis_recommend", return_value=outcome) as run_blis,
        patch("tools.api_service.app._run_aisimulate_recommendation") as run_aisimulate,
    ):
        response = client.post("/recommend", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["simulator"] == "blis"
    assert payload["candidates"][0]["simulator"] == "blis"
    assert payload["views"]["pareto_front"] == ["agg-tp1-c1"]
    assert payload["sweep"]["algorithm"] == "enumerate"
    assert payload["sweep"]["strict_sla"] is True
    assert payload["capacity"] is None
    run_blis.assert_called_once()
    run_aisimulate.assert_not_called()


def test_blis_recommend_capacity_returns_the_capacity_block():
    outcome = SimpleNamespace(
        configs=[{
            "total_gpus_needed": 2,
            "replicas_needed": 2,
            "num_total_gpus": 1,
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "concurrency": 8,
            "mode": "agg",
            "simulator": "blis",
        }],
        search_summary={"strategy": "aggregate_tp_capacity"},
        capacity=SimpleNamespace(
            target_concurrency=8,
            context_per_sequence_tokens=2176,
            capacity_tokens=20_000,
            capacity_sequences=8,
            fits=True,
            shortfall_sequences=None,
            shortfall_tokens=None,
            minimum_gpus=1,
            per_rank_tokens=10_000,
            replicas=2,
            attention_dp=1,
            evaluated_shapes=2,
            feasible_shapes=2,
            exhaustive=True,
            performance_available=False,
            performance_unavailable_reason="BLIS capacity sizing reads the engine's KV pool.",
            mode="agg",
            simulator="blis",
        ),
    )
    body = {
        "model_path": RECOMMEND_BODY["model_path"],
        "system": RECOMMEND_BODY["system"],
        "isl": RECOMMEND_BODY["isl"],
        "osl": RECOMMEND_BODY["osl"],
        "target_concurrency": 8,
        "objective": "concurrency_capacity",
        "sla_mode": "none",
        "simulator": "blis",
    }
    with (
        patch("tools.api_service.app._blis_recommend", return_value=outcome) as run_blis,
        patch("tools.api_service.app._run_aisimulate_recommendation") as run_aisimulate,
    ):
        response = client.post("/recommend", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["simulator"] == "blis"
    assert payload["capacity"]["simulator"] == "blis"
    assert payload["capacity"]["fits"] is True
    assert payload["capacity"]["performance_available"] is False
    assert payload["capacity_options"] == []
    run_blis.assert_called_once()
    run_aisimulate.assert_not_called()


def test_blis_capacity_rejects_an_explicit_fabric():
    body = {
        "model_path": RECOMMEND_BODY["model_path"],
        "system": RECOMMEND_BODY["system"],
        "isl": RECOMMEND_BODY["isl"],
        "osl": RECOMMEND_BODY["osl"],
        "target_concurrency": 8,
        "objective": "concurrency_capacity",
        "sla_mode": "none",
        "simulator": "blis",
        "fabric": "ib-400g",
    }

    response = client.post("/recommend", json=body)

    assert response.status_code == 422
    assert "aggregate-only" in response.json()["detail"][0]["msg"]


def test_blis_rejects_an_explicit_kv_transfer_bandwidth():
    body = {
        "model_path": RECOMMEND_BODY["model_path"],
        "system": RECOMMEND_BODY["system"],
        "isl": RECOMMEND_BODY["isl"],
        "osl": RECOMMEND_BODY["osl"],
        "ttft": 1000,
        "tpot": 30,
        "objective": "pareto",
        "simulator": "blis",
        "kv_transfer_bandwidth_gb_per_second": 400,
    }

    response = client.post("/recommend", json=body)

    assert response.status_code == 422
    assert "kv_transfer_bandwidth_gb_per_second" in response.json()["detail"][0]["msg"]


def test_blis_recommend_unsupported_request_does_not_fall_back():
    with (
        patch(
            "tools.api_service.app._blis_recommend",
            side_effect=app_module.BlisUnsupportedRequest("BLIS cannot sweep this workload"),
        ),
        patch("tools.api_service.app._run_aisimulate_recommendation") as run_aisimulate,
    ):
        response = client.post("/recommend", json={**RECOMMEND_BODY, "simulator": "blis"})

    assert response.status_code == 422
    assert response.json()["detail"] == "BLIS cannot sweep this workload"
    run_aisimulate.assert_not_called()


def test_blis_recommend_uses_the_service_candidate_ceiling():
    outcome = SimpleNamespace(configs=[], search_summary={})
    with (
        patch("tools.api_service.app._max_candidate_gpus", return_value=8),
        patch("tools.api_service.app.run_blis_recommend", return_value=outcome) as run_blis,
    ):
        request = app_module.RecommendRequest.model_validate({
            **RECOMMEND_BODY,
            "simulator": "blis",
        })
        app_module._blis_recommend(request)

    sent = run_blis.call_args.args[0]
    assert sent.max_candidate_gpus == 8
    assert sent.min_candidate_gpus is None


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


def test_blis_recommend_defaults_fabric_from_the_simulator_registry():
    defaulted = app_module.RecommendRequest.model_validate({
        **RECOMMEND_BODY,
        "simulator": "blis",
    })
    aggregate_only = app_module.RecommendRequest.model_validate({
        **RECOMMEND_BODY,
        "simulator": "blis",
        "fabric": None,
    })

    assert defaulted.fabric == app_module._simulator_entry("blis")["default_network"]
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


def test_backends_endpoint_reports_simulator_version_and_search_shape():
    with patch("tools.api_service.app.blis_release", return_value="0.15.0"):
        response = client.get("/backends?simulator=blis")

    assert response.status_code == 200
    body = response.json()
    assert body["simulator"] == "blis"
    assert body["simulator_version"] == "0.15.0"
    assert body["recommend_search"] == "single"
    assert set(body["objectives"]) == {
        "min_gpus", "throughput_demand", "pareto", "concurrency_capacity",
    }
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
        "disaggregated": True,
        "max_model_len_auto": False,
    }


def test_aisimulate_backends_reports_simulator_version_and_windowed_search():
    body = client.get("/backends?simulator=aisimulate").json()

    assert body["simulator"] == "aisimulate"
    assert body["simulator_version"]
    assert body["recommend_search"] == "windowed"
    assert set(body["objectives"]) == {
        "min_gpus", "throughput_demand", "pareto", "concurrency_capacity",
    }


def test_models_endpoint_lists_blis_catalog_hugging_face_ids():
    with patch("tools.api_service.app.load_catalog", return_value=_catalog()):
        response = client.get("/models?simulator=blis")

    assert response.status_code == 200
    body = response.json()
    assert body["simulator"] == "blis"
    assert body["models"] == ["Qwen/Qwen3-32B", "zai-org/GLM-5"]


def test_models_endpoint_accepts_include_specs_for_blis():
    with patch("tools.api_service.app.load_catalog", return_value=_catalog()):
        response = client.get("/models?simulator=blis&include=specs")

    assert response.status_code == 200
    model = response.json()["models"][0]
    assert model["id"] == "Qwen/Qwen3-32B"
    assert model["context_length"] == 40960
    assert model["num_experts"] is None


def test_systems_endpoint_filters_to_hardware_blis_has_a_catalog_chip_for():
    with (
        patch("tools.api_service.app.load_catalog", return_value=_catalog()),
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


def test_blis_catalog_endpoints_report_an_unavailable_catalog():
    with patch(
        "tools.api_service.app.load_catalog",
        side_effect=app_module.BlisManifestError("catalog missing"),
    ):
        models = client.get("/models?simulator=blis")
        systems = client.get("/systems?simulator=blis")
        networks = client.get("/networks?simulator=blis")

    assert models.status_code == 503
    assert models.json()["detail"] == "catalog missing"
    assert systems.status_code == 503
    assert networks.status_code == 503


def test_memory_endpoint_refuses_a_simulator_without_the_memory_capability():
    raw = {
        "total_gpu_capacity_bytes": 80,
        "total_kv_size_bytes": 40,
        "kv_size_per_token_bytes": 1,
        "total_kv_size_tokens": 40,
        "source": "native",
        "memory_breakdown": {},
    }
    with patch("tools.api_service.app.estimate_kv_cache", return_value=raw) as estimate_memory:
        blis = client.post("/memory", json={**ESTIMATE_BODY, "simulator": "blis"})
        aisimulate = client.post("/memory", json={**ESTIMATE_BODY, "simulator": "aisimulate"})

    assert blis.status_code == 422
    assert "does not provide a KV-cache memory breakdown" in blis.json()["detail"]
    assert aisimulate.status_code == 200
    assert aisimulate.json()["simulator"] == "aisimulate"
    estimate_memory.assert_called_once()

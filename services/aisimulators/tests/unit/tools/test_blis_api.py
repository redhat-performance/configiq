from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tools.api_service import app as app_module
from tools.api_service.app import app
from tools.api_service.blis.catalog import BlisCatalog, CatalogChip, CatalogModel
from tools.api_service.blis.engines import BlisEngine

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


def _catalog_model(name: str, repo: str) -> CatalogModel:
    return CatalogModel(
        name=name,
        hf_repo=repo,
        architecture="Qwen3ForCausalLM",
        context_window=40960,
        attention_kind="gqa",
        attention_heads=40,
        attention_kv_heads=8,
    )


def _catalog() -> BlisCatalog:
    # A real catalog, not a SimpleNamespace: the endpoints call resolve_* and
    # specs_for on it, and stubbing the type hides contract breakage.
    return BlisCatalog(
        models={
            "qwen3-32b": _catalog_model("qwen3-32b", "Qwen/Qwen3-32B"),
            "glm-5": _catalog_model("glm-5", "zai-org/GLM-5"),
        },
        chips={
            "h100": CatalogChip(name="h100", memory_gib=80, memory_bandwidth_tbs=3.35),
            "h200": CatalogChip(
                name="h200",
                memory_gib=141,
                memory_bandwidth_tbs=4.8,
                bf16_tflops=989,
                compute_capability="9.0",
                gpus_per_node=8,
            ),
            "a100-sxm": CatalogChip(name="a100-sxm", memory_gib=80, compute_capability="8.0"),
        },
        fabrics=("ib-400g", "roce-200g"),
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
            "simulator_version": "0.19.0",
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
    assert ids == ["aisim", "blis"]
    assert body["default_simulator"] == "aisim"
    assert {entry["label"] for entry in body["simulators"]} == {"AISim", "BLIS"}
    # The endpoint reports the same installed/built versions the service uses
    # elsewhere, not a literal that can drift from the artifact.
    versions = {entry["id"]: entry["version"] for entry in body["simulators"]}
    assert versions == {
        "aisim": app_module._AISIMULATE_VERSION,
        "blis": app_module.blis_release(),
    }
    assert all(isinstance(version, str) and version for version in versions.values())
    # Each simulator also reports the git commit of the artifact it runs, so
    # the footer can identify the build rather than only the release.
    revisions = {entry["id"]: entry["revision"] for entry in body["simulators"]}
    assert revisions == {
        "aisim": app_module._AISIMULATE_COMMIT,
        "blis": app_module.blis_commit(),
    }
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
    assert body["provenance"]["simulator_version"] == "0.19.0"
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
        aisim = client.get("/networks?simulator=aisim")

    assert blis.status_code == 200
    assert blis.json() == {
        "simulator": "blis",
        "networks": ["ib-400g", "roce-200g"],
        "default_network": "roce-200g",
    }
    assert aisim.status_code == 200
    assert aisim.json() == {"simulator": "aisim", "networks": [], "default_network": None}


def test_aisim_rejects_blis_fabric_selection():
    response = client.post("/recommend", json={**RECOMMEND_BODY, "fabric": "ib-400g"})

    assert response.status_code == 422
    assert "AISim does not currently consume" in response.json()["detail"][0]["msg"]


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


def test_blis_refuses_a_backend_kind_the_build_does_not_report(tmp_path, monkeypatch):
    binary = tmp_path / "blis"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("BLIS_BINARY", str(binary))
    monkeypatch.setattr(
        "tools.api_service.blis.service.cached_engines",
        lambda binary, **kwargs: (
            BlisEngine(
                kind="vllm",
                modeled_version="0.31.0",
                modeled_commit=None,
                build_version="0.34.0+0fda86a35419",
                roles=("colocated", "prefill", "decode"),
            ),
        ),
    )

    response = client.post(
        "/predict",
        json={**ESTIMATE_BODY, "simulator": "blis", "backend": "trtllm"},
    )

    assert response.status_code == 422
    assert "models vllm" in response.json()["detail"]


def test_blis_accepts_requests_per_load_unit():
    with patch(
        "tools.api_service.app._blis_estimate",
        return_value=_prediction(),
    ) as blis_estimate, patch("tools.api_service.app._run_aisimulate_prediction") as run_aisimulate:
        response = client.post(
            "/predict",
            json={
                **ESTIMATE_BODY,
                "simulator": "blis",
                "requests_per_load_unit": 50,
            },
        )

    assert response.status_code == 200
    assert blis_estimate.call_args.args[0].requests_per_load_unit == 50
    run_aisimulate.assert_not_called()


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
            "modes_searched": ["agg", "disagg"],
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
    # BLIS reports its search modes in its own summary; the shared sweep facts
    # carry them too so a client reads one field for both simulators.
    assert payload["sweep"]["modes_searched"] == ["agg", "disagg"]
    assert payload["capacity"] is None
    run_blis.assert_called_once()
    run_aisimulate.assert_not_called()


CAPACITY_BODY = {
    "model_path": RECOMMEND_BODY["model_path"],
    "system": RECOMMEND_BODY["system"],
    "isl": RECOMMEND_BODY["isl"],
    "osl": RECOMMEND_BODY["osl"],
    "target_concurrency": 8,
    "objective": "concurrency_capacity",
    "sla_mode": "none",
    "simulator": "blis",
}


def _capacity_outcome() -> SimpleNamespace:
    return SimpleNamespace(
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
            shared_prefix_tokens=0,
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


def test_blis_recommend_capacity_returns_the_capacity_block():
    outcome = _capacity_outcome()
    body = CAPACITY_BODY
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


def test_blis_capacity_records_no_replay_even_when_debug_is_asked_for():
    # A capacity search is an enumeration of probes, not one invocation to
    # rerun, so there is nothing to replay and the recorder stays uninstalled.
    observed: dict[str, object] = {}

    def fake(request):
        observed["recorder"] = app_module.active_recorder()
        return _capacity_outcome()

    with patch("tools.api_service.app._blis_recommend", side_effect=fake):
        response = client.post("/recommend?include=debug", json=CAPACITY_BODY)

    assert response.status_code == 200
    assert observed["recorder"] is None
    assert response.json()["debug"] is None


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


def test_blis_recommend_normalizes_fractional_concurrency_like_aisim():
    outcome = SimpleNamespace(configs=[], search_summary={})
    with patch("tools.api_service.app.run_blis_recommend", return_value=outcome) as run_blis:
        request = app_module.RecommendRequest.model_validate({
            **RECOMMEND_BODY,
            "target_concurrency": 8.5,
            "simulator": "blis",
        })
        app_module._blis_recommend(request)

    assert run_blis.call_args.args[0].concurrency == 8


def test_backends_endpoint_reports_the_engine_release_the_build_models(monkeypatch):
    monkeypatch.setattr(app_module, "blis_release", lambda: "0.19.0")
    monkeypatch.setattr(
        app_module,
        "_blis_engines",
        lambda: (
            BlisEngine(
                kind="vllm",
                modeled_version="0.31.0",
                modeled_commit=None,
                # The build's own snapshot is provenance, never the served
                # version.
                build_version="0.34.0+0fda86a35419",
                roles=("colocated", "prefill", "decode"),
            ),
        ),
    )

    response = client.get("/backends?simulator=blis")

    assert response.status_code == 200
    body = response.json()
    assert body["simulator"] == "blis"
    assert body["simulator_version"] == "0.19.0"
    assert body["recommend_search"] == "single"
    assert set(body["objectives"]) == {
        "min_gpus", "throughput_demand", "pareto", "concurrency_capacity",
    }
    backend = body["backends"][0]
    assert backend["id"] == "vllm"
    assert backend["versions"] == ["0.31.0"]
    assert backend["default_version"] == "0.31.0"
    assert backend["memory_fraction"] is None
    assert backend["systems"] == {}
    assert backend["system_defaults"] == {}
    assert backend["disaggregation_requires_fabric"] is True
    assert backend["runtime_field"] == "gpu_memory_utilization"
    # The roles the build reports are what a client gates the P/D control on.
    assert backend["version_support"] == {
        "0.31.0": {"disaggregated": True, "max_model_len_auto": False},
    }
    assert "0.34.0" not in str(body)


def test_backends_endpoint_reports_every_engine_kind_and_its_roles(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "_blis_engines",
        lambda: (
            BlisEngine(
                kind="sglang",
                modeled_version="0.5.23",
                modeled_commit=None,
                build_version="0.5.23",
                roles=("colocated",),
            ),
            BlisEngine(
                kind="vllm",
                modeled_version="0.31.0",
                modeled_commit=None,
                build_version="0.34.0+0fda86a35419",
                roles=("colocated", "prefill", "decode"),
            ),
        ),
    )

    body = client.get("/backends?simulator=blis").json()

    by_id = {backend["id"]: backend for backend in body["backends"]}
    assert sorted(by_id) == ["sglang", "vllm"]
    # An engine whose reported roles cover colocated serving only is not offered
    # a disaggregated topology by the client.
    assert by_id["sglang"]["version_support"] == {
        "0.5.23": {"disaggregated": False, "max_model_len_auto": False},
    }
    assert by_id["vllm"]["version_support"] == {
        "0.31.0": {"disaggregated": True, "max_model_len_auto": False},
    }
    assert by_id["sglang"]["runtime_field"] == "mem_fraction_static"


def test_backends_endpoint_reports_an_unreported_version_rather_than_the_snapshot(monkeypatch):
    # A build that predates the engine report lists no version; the client shows
    # it as unreported instead of substituting the build's engine snapshot.
    monkeypatch.setattr(app_module, "_blis_engines", lambda: ())

    body = client.get("/backends?simulator=blis").json()

    # No engine kinds are reported, so there is no backend to offer a version
    # for. The client shows the catalogue as unavailable rather than guessing.
    assert body["backends"] == []
    assert "0.34.0" not in str(body)


def test_aisim_backends_reports_simulator_version_and_windowed_search():
    body = client.get("/backends?simulator=aisim").json()

    assert body["simulator"] == "aisim"
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


def test_systems_endpoint_reports_the_specs_the_blis_catalog_states():
    with (
        patch("tools.api_service.app.load_catalog", return_value=_catalog()),
        patch("tools.api_service.app.supported_systems", lambda: {"h200_sxm", "a100_sxm"}),
        patch.dict(
            "tools.api_service.app._DEVICE_DISPLAY_NAMES",
            {"h200_sxm": "NVIDIA H200 SXM", "a100_sxm": "NVIDIA A100 SXM"},
        ),
    ):
        response = client.get("/systems?simulator=blis&include=specs")

    assert response.status_code == 200
    systems = {system["id"]: system for system in response.json()["systems"]}
    h200 = systems["h200_sxm"]
    assert h200["name"] == "NVIDIA H200 SXM"
    assert h200["memory_bytes"] == 141 * 1024 ** 3
    assert h200["memory_bandwidth_bytes"] == 4_800_000_000_000
    assert h200["bf16_tflops"] == 989
    assert h200["architecture"] == "hopper"
    assert h200["vendor"] == "nvidia"
    # The chip states its own packaging, so that wins.
    assert h200["gpus_per_node"] == 8
    # The build states no TDP, so the client shows it as unavailable rather
    # than borrowing another simulator's figure.
    assert "tdp_watts" not in h200
    assert systems["a100_sxm"]["architecture"] == "ampere"


def test_systems_endpoint_uses_the_system_catalogue_node_width_when_the_chip_states_none():
    with (
        patch("tools.api_service.app.load_catalog", return_value=_catalog()),
        patch("tools.api_service.app.supported_systems", lambda: {"h100_sxm"}),
        patch.dict(
            "tools.api_service.app._DEVICE_DISPLAY_NAMES",
            {"h100_sxm": "NVIDIA H100 SXM"},
        ),
        patch(
            "tools.api_service.app.load_system_spec",
            lambda system: {"node": {"num_gpus_per_node": 8}},
        ),
    ):
        response = client.get("/systems?simulator=blis&include=specs")

    assert response.status_code == 200
    system = response.json()["systems"][0]
    assert system["gpus_per_node"] == 8
    # The chip states no compute capability here, so no architecture or vendor
    # is invented for it.
    assert "architecture" not in system
    assert "vendor" not in system


def test_blis_catalog_endpoints_report_an_unavailable_catalog():
    with patch(
        "tools.api_service.app.load_catalog",
        side_effect=app_module.BlisCatalogError("catalog missing"),
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
        aisim = client.post("/memory", json={**ESTIMATE_BODY, "simulator": "aisim"})

    assert blis.status_code == 422
    assert "does not provide a KV-cache memory breakdown" in blis.json()["detail"]
    assert aisim.status_code == 200
    assert aisim.json()["simulator"] == "aisim"
    estimate_memory.assert_called_once()


def test_blis_predict_reports_the_limits_its_scenario_held():
    with (
        patch("tools.api_service.app._blis_estimate", return_value=_prediction()),
        patch("tools.api_service.app._run_aisimulate_prediction") as run_aisimulate,
    ):
        response = client.post(
            "/predict?include=engine", json={**ESTIMATE_BODY, "simulator": "blis"}
        )

    assert response.status_code == 200
    engine = response.json()["engine"]
    assert engine["mode"] == "agg"
    assert engine["backend"] == "vllm"
    assert engine["backend_version"] == "0.31.0"
    assert engine["agg"]["tensor_parallel_size"] == {"value": 2, "source": "evaluated"}
    assert engine["agg"]["max_num_seqs"]["value"] == 8
    assert engine["agg"]["max_num_batched_tokens"]["value"] == 8192
    assert engine["agg"]["gpu_memory_utilization"]["value"] == pytest.approx(0.92)
    assert engine["agg"]["enable_prefix_caching"]["value"] is True
    # BLIS reports a context-window policy, not a token count, and states no
    # block size: those stay unmodeled instead of borrowing a backend default.
    assert engine["agg"]["max_model_len"] == {"value": None, "source": "not-modeled"}
    assert engine["agg"]["block_size"] == {"value": None, "source": "not-modeled"}
    assert engine["agg"]["enable_chunked_prefill"]["source"] == "not-modeled"
    run_aisimulate.assert_not_called()


def test_blis_disagg_reports_each_pool_and_no_aggregate_role():
    prediction = SimpleNamespace(
        **_prediction().__dict__,
        prefill_config={"tp": 4, "pp": 1, "num_workers": 2, "batch_size": 8},
        decode_config={"tp": 2, "pp": 1, "num_workers": 1, "batch_size": 8},
    )
    with (
        patch("tools.api_service.app.run_blis_estimate", return_value=prediction),
        patch("tools.api_service.app._run_aisimulate_prediction") as run_aisimulate,
    ):
        response = client.post(
            "/predict?include=engine",
            json={
                **ESTIMATE_BODY,
                "simulator": "blis",
                "mode": "disagg",
                "fabric": "ib-400g",
                "prefill_tp_size": 4,
                "prefill_num_workers": 2,
                "decode_tp_size": 2,
                "decode_num_workers": 1,
            },
        )

    assert response.status_code == 200
    engine = response.json()["engine"]
    assert engine["mode"] == "disagg"
    assert engine["prefill"] is not None
    assert engine["decode"] is not None
    assert engine["prefill"]["tensor_parallel_size"]["value"] == 4
    assert engine["prefill"]["replicas"]["value"] == 2
    assert engine["prefill"]["max_num_seqs"]["value"] == 8
    assert engine["decode"]["tensor_parallel_size"]["value"] == 2
    assert engine["decode"]["replicas"]["value"] == 1
    # A pool config holds no context window of its own.
    assert engine["prefill"]["max_model_len"]["source"] == "not-modeled"
    run_aisimulate.assert_not_called()


def test_blis_predict_omits_engine_unless_asked():
    with patch("tools.api_service.app._blis_estimate", return_value=_prediction()):
        response = client.post("/predict", json={**ESTIMATE_BODY, "simulator": "blis"})

    assert response.status_code == 200
    assert response.json()["engine"] is None

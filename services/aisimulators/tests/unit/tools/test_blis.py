from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tools.api_service.blis import service as blis_service
from tools.api_service.blis.manifest import BlisUnsupportedRequest, ModelResolution, SupportManifest
from tools.api_service.blis.runner import (
    BlisExecutionError,
    RunRequest,
    build_run_command,
    parse_metrics,
)
from tools.api_service.blis.runtime import BlisRuntime
from tools.api_service.blis.scenario import (
    ScenarioPoolRequest,
    ScenarioRequest,
    build_documents,
    engine_version_support,
    resolve_engine_version,
    write_scenario,
)


@pytest.fixture
def manifest() -> SupportManifest:
    return SupportManifest.from_dict({
        "schema_version": 1,
        "models": {
            "qwen3-32b": {
                "origin": "aisimulate",
                "graph": True,
                "aisimulate_model_id": "Qwen/Qwen3-32B",
                "config_relation": "identical",
                "equivalent_catalog_name": None,
            },
            "glm-5": {
                "origin": "upstream",
                "graph": True,
                "aisimulate_model_id": "zai-org/GLM-5",
                "config_relation": "different_quantization",
                "equivalent_catalog_name": None,
            },
        },
        "skipped": [{
            "aisimulate_model_id": "org/Unsupported",
            "catalog_name": "unsupported",
            "reason": "graph_derivation_failed",
            "detail": "missing attention metadata",
        }],
        "hardware": ["h100", "h200", "a100-sxm"],
        "networks": ["ib-400g"],
    })


def test_manifest_resolves_catalog_and_aisimulate_model_ids(manifest):
    direct = manifest.resolve_model("qwen3-32b")
    hf_id = manifest.resolve_model("Qwen/Qwen3-32B")

    assert direct.catalog_name == "qwen3-32b"
    assert hf_id.catalog_name == "qwen3-32b"
    assert hf_id.config_relation == "identical"
    assert manifest.resolve_hardware("h100_sxm") == "h100"
    assert manifest.resolve_network("ib-400g") == "ib-400g"
    with pytest.raises(BlisUnsupportedRequest, match="available fabric keys: ib-400g"):
        manifest.resolve_network("unknown-fabric")


def test_manifest_preserves_config_relation_and_reports_skips(manifest):
    resolution = manifest.resolve_model("zai-org/GLM-5")

    assert resolution.config_relation == "different_quantization"
    with pytest.raises(BlisUnsupportedRequest, match="graph_derivation_failed.*missing attention metadata"):
        manifest.resolve_model("org/Unsupported")


def test_manifest_does_not_accept_unlisted_model_or_hardware(manifest):
    with pytest.raises(BlisUnsupportedRequest, match="does not support model"):
        manifest.resolve_model("Unknown/Model")
    with pytest.raises(BlisUnsupportedRequest, match="available BLIS hardware keys: a100-sxm, h100, h200"):
        manifest.resolve_hardware("b60")


def test_blis_rejects_non_vllm_before_loading_runtime():
    with pytest.raises(BlisUnsupportedRequest, match="model vLLM"):
        blis_service._resolve_request("Qwen/Qwen3-32B", "h200_sxm", "sglang", None)


def _scenario_request(**overrides) -> ScenarioRequest:
    values = {
        "model": ModelResolution(
            catalog_name="qwen3-32b",
            origin="aisimulate",
            aisimulate_model_id="Qwen/Qwen3-32B",
            config_relation="identical",
            equivalent_catalog_name=None,
        ),
        "hardware": "h100",
        "system": "h100_sxm",
        "gpus_per_node": 8,
        "engine_version": "0.31.0",
        "tp": 2,
        "pp": 1,
        "dp": 1,
        "moe_tp_size": None,
        "moe_ep_size": None,
        "enable_expert_parallel": False,
        "isl": 2048,
        "osl": 128,
        "concurrency": 8,
        "max_num_seqs": 8,
        "max_model_len": None,
        "max_num_batched_tokens": None,
        "gpu_memory_utilization": 0.9,
        "weight_quantization": "NVFP4",
        "cache_dtype": "FP8",
        "prefix_caching": True,
    }
    values.update(overrides)
    return ScenarioRequest(**values)


def test_generated_documents_use_catalog_id_and_requested_deployment():
    scenario, deployment = build_documents(_scenario_request())
    pool = deployment["pools"][0]

    assert scenario["kind"] == "Scenario"
    assert scenario["model"] == "qwen3-32b"
    assert scenario["cluster"] == {"hardware": "h100", "nodes": 1, "gpus_per_node": 8}
    assert pool["parallel"] == {
        "tp": 2, "pp": 1, "dp": 1, "enable_expert_parallel": False,
    }
    assert pool["engine"]["quantization"] == "nvfp4"
    assert pool["engine"]["cache_dtype"] == "fp8"
    assert pool["engine"]["enable_prefix_caching"] is True
    assert pool["engine"]["max_model_len"] == "auto"


def test_generated_scenario_is_two_yaml_documents(tmp_path: Path):
    filename = write_scenario(tmp_path, _scenario_request())
    docs = list(yaml.safe_load_all((tmp_path / filename).read_text()))

    assert filename.endswith(".yaml")
    assert [doc["kind"] for doc in docs] == ["Scenario", "Deployment"]


def test_different_engine_settings_get_distinct_scenario_files(tmp_path: Path):
    first = write_scenario(tmp_path, _scenario_request(cache_dtype="FP8", gpu_memory_utilization=0.9))
    second = write_scenario(tmp_path, _scenario_request(cache_dtype="FP16", gpu_memory_utilization=0.92))

    assert first != second
    assert len(list(tmp_path.glob("*.yaml"))) == 2


def test_serving_config_reports_only_fields_stated_by_scenario():
    scenario, _deployment = build_documents(_scenario_request(prefix_caching=None))
    scenario_request = _scenario_request(prefix_caching=None)
    serving_config = blis_service._serving_config(scenario_request)

    assert scenario["cluster"]["hardware"] == "h100"
    assert serving_config["max_num_batched_tokens"] == 8192
    assert serving_config["enable_prefix_caching"] is True
    assert serving_config["enable_chunked_prefill"] is None


def test_half_cache_dtype_leaves_the_blis_model_default_unmodified():
    _scenario, deployment = build_documents(_scenario_request(cache_dtype="half"))

    assert "cache_dtype" not in deployment["pools"][0]["engine"]


def test_scenario_rejects_cross_node_and_unrepresentable_parallelism():
    with pytest.raises(BlisUnsupportedRequest, match="single-node"):
        build_documents(_scenario_request(tp=16))
    with pytest.raises(BlisUnsupportedRequest, match="derives expert parallel width"):
        build_documents(_scenario_request(enable_expert_parallel=True, moe_ep_size=4))
    with pytest.raises(BlisUnsupportedRequest, match="derives expert tensor parallelism"):
        build_documents(_scenario_request(tp=2, moe_tp_size=1))
    with pytest.raises(BlisUnsupportedRequest, match="derives expert parallel width"):
        build_documents(_scenario_request(
            tp=2,
            enable_expert_parallel=True,
            moe_ep_size=1,
        ))
    with pytest.raises(BlisUnsupportedRequest, match="cannot represent weight quantization"):
        build_documents(_scenario_request(weight_quantization="unsupported"))
    with pytest.raises(BlisUnsupportedRequest, match="cannot separate MoE and dense"):
        build_documents(_scenario_request(weight_quantization="bf16", moe_quantization="fp8"))
    with pytest.raises(BlisUnsupportedRequest, match="cannot separate MoE and dense"):
        build_documents(_scenario_request(weight_quantization=None, moe_quantization="fp8"))
    with pytest.raises(BlisUnsupportedRequest, match="cannot separate MoE and dense"):
        build_documents(_scenario_request(weight_quantization="half", moe_quantization="fp8"))


def _pool_request(role: str, *, nodes: int, tp: int = 4) -> ScenarioPoolRequest:
    return ScenarioPoolRequest(
        role=role,
        nodes=nodes,
        tp=tp,
        pp=1,
        dp=1,
        moe_tp_size=None,
        moe_ep_size=None,
        enable_expert_parallel=False,
        max_model_len=2176,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.9,
        weight_quantization=None,
        moe_quantization=None,
        cache_dtype="FP8",
        prefix_caching=True,
    )


def test_disagg_documents_define_two_pools_transfer_and_router(tmp_path: Path):
    request = _scenario_request(
        fabric="ib-400g",
        prefill_pool=_pool_request("prefill", nodes=2),
        decode_pool=_pool_request("decode", nodes=1),
    )

    scenario, deployment = build_documents(request)
    filename = write_scenario(tmp_path, request)
    router_config = yaml.safe_load((tmp_path / "router/pd-always.yaml").read_text())

    assert scenario["cluster"] == {
        "hardware": "h100",
        "fabric": "ib-400g",
        "nodes": 3,
        "gpus_per_node": 8,
    }
    assert [pool["role"] for pool in deployment["pools"]] == ["prefill", "decode"]
    assert [pool["engine"]["block_size"] for pool in deployment["pools"]] == [64, 64]
    assert deployment["pd_transfer"] == {"connector": "NixlConnector"}
    assert deployment["router"]["endpoint_picker_config"] == "router/pd-always.yaml"
    assert router_config["kind"] == "EndpointPickerConfig"
    assert {plugin["type"] for plugin in router_config["plugins"]} >= {
        "always-disagg-pd-decider",
        "disagg-profile-handler",
        "prefill-filter",
        "decode-filter",
    }
    assert filename.endswith(".yaml")


def test_disagg_documents_reject_missing_fabric():
    request = _scenario_request(
        prefill_pool=_pool_request("prefill", nodes=1),
        decode_pool=_pool_request("decode", nodes=1),
    )
    with pytest.raises(BlisUnsupportedRequest, match="require an explicit inter-node fabric"):
        build_documents(request)


def test_blis_requires_exact_version_instead_of_resolving_aisimulate_aliases():
    with pytest.raises(BlisUnsupportedRequest, match="exact backend_version"):
        resolve_engine_version("current")


def test_engine_version_defaults_to_the_pd_capable_release():
    assert resolve_engine_version(None) == "0.31.0"
    assert resolve_engine_version("") == "0.31.0"
    assert resolve_engine_version("0.29.0") == "0.29.0"


def test_unmodeled_engine_version_names_the_modeled_releases():
    with pytest.raises(BlisUnsupportedRequest) as error:
        resolve_engine_version("0.24.0")

    message = str(error.value)
    assert "0.29.0, 0.31.0, 0.32.0" in message
    assert "blank" in message


def test_v032_scenario_states_an_explicit_context_window():
    _scenario, deployment = build_documents(_scenario_request(engine_version="0.32.0"))

    assert deployment["pools"][0]["engine"]["max_model_len"] == 2048 + 128


def test_v032_refuses_a_disaggregated_topology():
    request = _scenario_request(
        engine_version="0.32.0",
        fabric="ib-400g",
        prefill_pool=_pool_request("prefill", nodes=1),
        decode_pool=_pool_request("decode", nodes=1),
    )
    with pytest.raises(BlisUnsupportedRequest, match="does not model prefill/decode disaggregation"):
        build_documents(request)


def test_engine_version_support_reports_what_each_release_models():
    assert engine_version_support("0.31.0") == {"disaggregated": True, "max_model_len_auto": True}
    assert engine_version_support("0.32.0") == {"disaggregated": False, "max_model_len_auto": False}
    assert engine_version_support("0.99.0") == {"disaggregated": False, "max_model_len_auto": False}


def test_run_command_passes_catalog_registry_and_scenario_directory():
    runtime = BlisRuntime(
        binary="/opt/venv/bin/blis",
        catalog="/opt/blis/catalog",
        registry="/opt/blis/registry",
    )

    command = build_run_command(runtime, RunRequest(
        scenario="generated.yaml",
        scenario_directory="/tmp/scenarios",
        isl=2048,
        osl=128,
        concurrency=8,
        num_requests=8,
        prefix_tokens=16,
    ))

    assert command == [
        "/opt/venv/bin/blis", "run",
        "--catalog", "/opt/blis/catalog",
        "--registry", "/opt/blis/registry",
        "--scenarios", "/tmp/scenarios",
        "--scenario", "generated.yaml",
        "--prompt-tokens", "2048",
        "--output-tokens", "128",
        "--concurrency", "8",
        "--num-requests", "8",
        "--prefix-tokens", "16",
    ]


def test_run_command_passes_p_and_d_instance_counts():
    runtime = BlisRuntime("/opt/venv/bin/blis", "/catalog", "/registry")
    command = build_run_command(runtime, RunRequest(
        scenario="generated.yaml",
        scenario_directory="/tmp/scenarios",
        isl=2048,
        osl=128,
        concurrency=8,
        num_requests=8,
        prefill_instances=2,
        decode_instances=1,
    ))

    assert command[-6:] == [
        "--num-instances", "3",
        "--prefill-instances", "2",
        "--decode-instances", "1",
    ]


def test_parse_metrics_requires_marker_and_json_payload():
    assert parse_metrics("prelude\n=== Simulation Metrics ===\n{\"ttft_mean_ms\": 1.25}\n") == {
        "ttft_mean_ms": 1.25,
    }
    with pytest.raises(BlisExecutionError, match="missing"):
        parse_metrics("no marker")
    with pytest.raises(BlisExecutionError, match="does not contain"):
        parse_metrics("=== Simulation Metrics ===\nno json")


def test_parse_metrics_prefers_cluster_aggregate_for_disagg_runs():
    output = (
        '=== Simulation Metrics ===\n{"instance_id":"instance_0","ttft_mean_ms":100,"itl_mean_ms":0}\n'
        '=== Simulation Metrics ===\n{"instance_id":"instance_1","ttft_mean_ms":0,"itl_mean_ms":20}\n'
        '=== Simulation Metrics ===\n{"instance_id":"cluster","ttft_mean_ms":110,"itl_mean_ms":20}'
    )

    assert parse_metrics(output)["instance_id"] == "cluster"
    assert parse_metrics(output)["ttft_mean_ms"] == 110


def test_recommend_candidate_ceiling_uses_configured_limit_and_hardware(monkeypatch, manifest):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_support_manifest", lambda: manifest)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 128}},
    )
    calls = []

    def mock_run(_runtime, run_request, *, timeout_seconds):
        calls.append(run_request)
        return {
            "completed_requests": 4,
            "dropped_unservable": 0,
            "ttft_mean_ms": 80.0,
            "itl_mean_ms": 20.0,
            "e2e_mean_ms": 2640.0,
            "tokens_per_sec": 500.0,
        }

    monkeypatch.setattr(blis_service, "run_blis", mock_run)

    result = blis_service.recommend(blis_service.RecommendInput(
        model_path="Qwen/Qwen3-32B",
        system="h100_sxm",
        backend="vllm",
        backend_version=None,
        isl=2048,
        osl=128,
        concurrency=1,
        request_rate=None,
        ttft=1000,
        tpot=30,
        request_latency=None,
        prefix=0,
        min_candidate_gpus=1,
        max_candidate_gpus=128,
        top_n=20,
        search_budget_seconds=60,
    ))

    assert result.search_summary["candidate_gpus"] == [1, 2, 4, 8, 16, 32, 64, 128]
    assert result.search_summary["effective_max_gpus"] == 128
    assert result.search_summary["hardware_max_gpus"] == 128
    assert len(calls) == 8


def test_recommend_includes_bounded_pd_topologies_when_fabric_is_selected(monkeypatch, manifest):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_support_manifest", lambda: manifest)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 4}},
    )
    calls = []

    def mock_run(_runtime, run_request, *, timeout_seconds):
        documents = list((Path(run_request.scenario_directory) / run_request.scenario).read_text().split("---"))
        calls.append((run_request, documents))
        return {
            "instance_id": "cluster",
            "completed_requests": 8,
            "dropped_unservable": 0,
            "ttft_mean_ms": 80.0,
            "itl_mean_ms": 20.0,
            "e2e_mean_ms": 2640.0,
            "tokens_per_sec": 500.0,
        }

    monkeypatch.setattr(blis_service, "run_blis", mock_run)
    result = blis_service.recommend(blis_service.RecommendInput(
        model_path="Qwen/Qwen3-32B",
        system="h100_sxm",
        backend="vllm",
        backend_version=None,
        isl=2048,
        osl=128,
        concurrency=8,
        request_rate=None,
        ttft=1000,
        tpot=30,
        request_latency=None,
        prefix=0,
        min_candidate_gpus=1,
        max_candidate_gpus=8,
        top_n=20,
        search_budget_seconds=60,
        fabric="ib-400g",
    ))

    disagg_configs = [config for config in result.configs if config["mode"] == "disagg"]
    assert result.search_summary["modes_searched"] == ["agg", "disagg"]
    assert result.search_summary["hardware_max_gpus"] == 8
    assert result.search_summary["evaluated_configurations"] == len(calls)
    assert disagg_configs
    first_pd = disagg_configs[0]
    assert first_pd["total_gpus_needed"] == (
        first_pd["prefill_config"]["tp"] * first_pd["prefill_config"]["num_workers"]
        + first_pd["decode_config"]["tp"] * first_pd["decode_config"]["num_workers"]
    )
    assert any(call.prefill_instances == 1 and call.decode_instances == 1 for call, _docs in calls)


def test_recommend_skips_pd_candidates_without_a_fabric(monkeypatch, manifest):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_support_manifest", lambda: manifest)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(blis_service, "load_system_spec", lambda _system: {"node": {"num_gpus_per_node": 4}})
    monkeypatch.setattr(blis_service, "run_blis", lambda *_args, **_kwargs: {
        "completed_requests": 8,
        "dropped_unservable": 0,
        "ttft_mean_ms": 80.0,
        "itl_mean_ms": 20.0,
        "e2e_mean_ms": 2640.0,
        "tokens_per_sec": 500.0,
    })

    result = blis_service.recommend(blis_service.RecommendInput(
        model_path="Qwen/Qwen3-32B", system="h100_sxm", backend="vllm", backend_version=None,
        isl=2048, osl=128, concurrency=8, request_rate=None, ttft=1000, tpot=30,
        request_latency=None, prefix=0, min_candidate_gpus=1, max_candidate_gpus=4,
        top_n=5, search_budget_seconds=60,
    ))

    assert result.search_summary["modes_searched"] == ["agg"]
    assert result.configs
    assert all(config["mode"] == "agg" for config in result.configs)
    assert all(
        any("P/D candidates were not searched" in warning for warning in config["warnings"])
        for config in result.configs
    )


def test_disagg_estimate_builds_p_and_d_pools_and_uses_requested_fabric(monkeypatch, manifest):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_support_manifest", lambda: manifest)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 8}},
    )
    calls = []

    def mock_run(_runtime, run_request, *, timeout_seconds):
        scenario_path = Path(run_request.scenario_directory) / run_request.scenario
        documents = list(yaml.safe_load_all(scenario_path.read_text()))
        calls.append((run_request, timeout_seconds, documents))
        return {
            "completed_requests": 8,
            "dropped_unservable": 0,
            "ttft_mean_ms": 80.0,
            "itl_mean_ms": 20.0,
            "e2e_mean_ms": 2640.0,
            "tokens_per_sec": 500.0,
        }

    monkeypatch.setattr(blis_service, "run_blis", mock_run)
    result = blis_service.estimate(blis_service.EstimateInput(
        model_path="Qwen/Qwen3-32B",
        system="h100_sxm",
        backend="vllm",
        backend_version=None,
        isl=2048,
        osl=128,
        concurrency=8,
        tp=1,
        pp=1,
        dp=1,
        mode="disagg",
        fabric="ib-400g",
        prefill_tp_size=4,
        prefill_num_workers=2,
        prefill_batch_size=8,
        decode_tp_size=2,
        decode_num_workers=1,
        decode_batch_size=8,
    ))

    run_request, _timeout, documents = calls[0]
    assert documents[0]["cluster"]["fabric"] == "ib-400g"
    assert [pool["role"] for pool in documents[1]["pools"]] == ["prefill", "decode"]
    assert run_request.prefill_instances == 2
    assert run_request.decode_instances == 1
    assert result.summary["num_total_gpus"] == 10.0
    assert result.prefill_config["tp"] == 4
    assert result.decode_config["tp"] == 2
    assert result.serving_config is None


def test_disagg_estimate_rejects_heterogeneous_systems_before_runtime_load():
    request = blis_service.EstimateInput(
        model_path="Qwen/Qwen3-32B",
        system="h100_sxm",
        backend="vllm",
        backend_version=None,
        isl=2048,
        osl=128,
        concurrency=8,
        tp=1,
        pp=1,
        dp=1,
        mode="disagg",
        decode_system="h200_sxm",
        fabric="ib-400g",
    )
    with pytest.raises(BlisUnsupportedRequest, match="decode_system must match system"):
        blis_service.estimate(request)

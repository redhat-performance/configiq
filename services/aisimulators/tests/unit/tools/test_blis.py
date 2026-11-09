from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools.api_service.blis import service as blis_service
from tools.api_service.blis.manifest import (
    BlisCatalog,
    BlisUnsupportedRequest,
    CatalogModel,
    ModelResolution,
    load_catalog,
    model_specs,
)
from tools.api_service.blis.runner import (
    BlisExecutionError,
    RunRequest,
    build_run_command,
    normalize_result,
    parse_result,
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


def _catalog_model(slug: str, repo: str | None, config: dict | None = None) -> CatalogModel:
    return CatalogModel(
        slug=slug,
        hf_repo=repo,
        config=config or {
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
        },
    )


@pytest.fixture
def catalog() -> BlisCatalog:
    return BlisCatalog(
        models={
            "qwen3-32b": _catalog_model("qwen3-32b", "Qwen/Qwen3-32B"),
            "glm-5": _catalog_model("glm-5", "zai-org/GLM-5"),
        },
        hardware=frozenset({"h100", "h200", "a100-sxm", "gb200-nvl72"}),
        networks=frozenset({"roce-200g", "ib-400g"}),
    )


def _write_catalog(root: Path) -> Path:
    """A catalog shaped exactly like blis-catalog's on-disk layout."""
    def write_model(slug: str, repo: str, config: dict, *, graph: bool = True) -> None:
        directory = root / "models" / slug
        directory.mkdir(parents=True)
        (directory / "model.yaml").write_text(yaml.safe_dump({
            "name": slug,
            "source": {"provider": "huggingface", "repo": repo},
        }))
        (directory / "config.json").write_text(json.dumps(config))
        if graph:
            (directory / "graph.yaml").write_text("kind: Graph\n")

    write_model("qwen3-32b", "Qwen/Qwen3-32B", {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
    })
    write_model("qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B", {
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "max_position_embeddings": 40960,
    })
    write_model("vision-model", "org/Vision", {
        "architectures": ["VisionForConditionalGeneration"],
        "text_config": {
            "num_experts": 64,
            "num_experts_per_tok": 4,
            "max_position_embeddings": 131072,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        },
    })
    # Declared but not priceable: blis-catalog ships no derived graph.
    write_model("no-graph", "org/NoGraph", {}, graph=False)

    (root / "hardware").mkdir()
    for name in ("h100", "h200", "a100-sxm", "gb200-nvl72"):
        (root / "hardware" / f"{name}.yaml").write_text("MemoryGiB: 80\n")
    (root / "networks").mkdir()
    for name in ("roce-200g", "ib-400g"):
        (root / "networks" / f"{name}.yaml").write_text("kind: Network\n")
    return root


def test_catalog_reads_hugging_face_ids_and_skips_unpriceable_models(tmp_path: Path):
    catalog = load_catalog(str(_write_catalog(tmp_path)))

    assert catalog.model_ids() == ["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-32B", "org/Vision"]
    assert "no-graph" not in catalog.models


def test_catalog_resolves_slug_and_hugging_face_id_to_one_entry(tmp_path: Path):
    catalog = load_catalog(str(_write_catalog(tmp_path)))

    by_slug = catalog.resolve_model("qwen3-32b")
    by_hf_id = catalog.resolve_model("Qwen/Qwen3-32B")

    assert by_slug.catalog_name == "qwen3-32b"
    assert by_hf_id.catalog_name == "qwen3-32b"
    assert by_slug.hf_repo == "Qwen/Qwen3-32B"
    assert by_slug.origin == "blis-catalog"


def test_catalog_refuses_models_it_cannot_price(tmp_path: Path):
    catalog = load_catalog(str(_write_catalog(tmp_path)))

    with pytest.raises(BlisUnsupportedRequest, match="does not support model.*org/Vision"):
        catalog.resolve_model("Unknown/Model")


def test_catalog_translates_system_aliases_but_not_pcie_parts(tmp_path: Path):
    catalog = load_catalog(str(_write_catalog(tmp_path)))

    assert catalog.resolve_hardware("h100_sxm") == "h100"
    assert catalog.resolve_hardware("a100_sxm") == "a100-sxm"
    assert catalog.resolve_hardware("gb200") == "gb200-nvl72"
    assert catalog.supports_hardware("h100_sxm") is True
    with pytest.raises(BlisUnsupportedRequest, match="available BLIS hardware keys"):
        catalog.resolve_hardware("h100_pcie")


def test_catalog_reports_only_fabrics_it_declares(tmp_path: Path):
    catalog = load_catalog(str(_write_catalog(tmp_path)))

    assert catalog.resolve_network("roce-200g") == "roce-200g"
    with pytest.raises(BlisUnsupportedRequest, match="available fabric keys: ib-400g, roce-200g"):
        catalog.resolve_network("unknown-fabric")


def test_catalog_specs_read_moe_and_text_config_fields(tmp_path: Path):
    catalog = load_catalog(str(_write_catalog(tmp_path)))

    dense = catalog.specs_for("Qwen/Qwen3-32B")
    moe = catalog.specs_for("qwen3-30b-a3b")
    multimodal = catalog.specs_for("org/Vision")

    assert dense["num_experts"] is None
    assert dense["context_length"] == 40960
    assert moe["num_experts"] == 128
    assert moe["num_experts_per_tok"] == 8
    assert multimodal["num_experts"] == 64
    assert multimodal["context_length"] == 131072
    assert multimodal["architecture"] == "VisionForConditionalGeneration"


def test_model_specs_prefers_architecture_list_and_ignores_zero_experts():
    specs = model_specs("org/Model", {
        "architectures": ["SomeForCausalLM"],
        "model_type": "some",
        "num_experts": 0,
        "max_position_embeddings": 0,
    })

    assert specs["architecture"] == "SomeForCausalLM"
    assert specs["num_experts"] is None
    assert specs["context_length"] is None


def test_catalog_missing_models_directory_is_an_explicit_error(tmp_path: Path):
    from tools.api_service.blis.manifest import BlisManifestError

    with pytest.raises(BlisManifestError, match="unavailable"):
        load_catalog(str(tmp_path / "missing"))


def test_blis_rejects_non_vllm_before_loading_the_catalog():
    with pytest.raises(BlisUnsupportedRequest, match="model vLLM"):
        blis_service._resolve_request("Qwen/Qwen3-32B", "h200_sxm", "sglang", None)


def _scenario_request(**overrides) -> ScenarioRequest:
    values = {
        "model": ModelResolution(
            catalog_name="qwen3-32b",
            hf_repo="Qwen/Qwen3-32B",
            origin="blis-catalog",
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


def test_serving_config_reports_the_settings_blis_resolved():
    result = {
        "input": {
            "deployment": {
                "pools": [{
                    "parallel": {"tp": 4},
                    "engine": {
                        "max_model_len": "auto",
                        "max_num_seqs": 8,
                        "max_num_batched_tokens": 8192,
                        "gpu_memory_utilization": 0.92,
                        "enable_prefix_caching": True,
                        "quantization": "nvfp4",
                    },
                }],
            },
        },
        "revisions": {"engine": {"upstream_version": "0.31.0"}},
    }

    config = blis_service._serving_config(result)

    assert config["tensor_parallel_size"] == 4
    assert config["max_model_len"] is None
    assert config["max_model_len_policy"] == "auto"
    assert config["enable_prefix_caching"] is True
    assert config["enable_chunked_prefill"] is None
    assert config["backend_version"] == "0.31.0"


def test_serving_config_reports_an_explicit_context_window():
    result = {
        "input": {"deployment": {"pools": [{
            "parallel": {"tp": 2},
            "engine": {"max_model_len": 2176},
        }]}},
    }

    config = blis_service._serving_config(result)

    assert config["max_model_len"] == 2176
    assert config["max_model_len_policy"] is None


def test_serving_config_is_absent_when_the_result_has_no_deployment():
    assert blis_service._serving_config({"metrics": {}}) is None


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


def test_v032_can_model_a_disaggregated_topology():
    request = _scenario_request(
        engine_version="0.32.0",
        fabric="ib-400g",
        prefill_pool=_pool_request("prefill", nodes=1),
        decode_pool=_pool_request("decode", nodes=1),
    )

    _scenario, deployment = build_documents(request)

    assert [pool["role"] for pool in deployment["pools"]] == ["prefill", "decode"]


def test_engine_version_support_reports_what_each_release_models():
    assert engine_version_support("0.31.0") == {"disaggregated": True, "max_model_len_auto": True}
    assert engine_version_support("0.32.0") == {"disaggregated": True, "max_model_len_auto": False}
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
        "--num-requests", "8",
        "--concurrency", "8",
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


def test_parse_result_returns_the_last_result_document():
    output = (
        'banner\n{"kind":"Progress","step":1}\n'
        '{"kind":"Result","metrics":{"ttft_mean_ms":100}}\n'
        '{"kind":"Result","metrics":{"ttft_mean_ms":110}}\n'
    )

    result = parse_result(output)

    assert result["metrics"]["ttft_mean_ms"] == 110


def test_parse_result_requires_a_result_document():
    with pytest.raises(BlisExecutionError, match="did not contain a Result"):
        parse_result("no json here")
    with pytest.raises(BlisExecutionError, match="did not contain a Result"):
        parse_result('{"kind":"Progress","step":1}')


def test_normalize_result_maps_metrics_requests_and_window():
    result = {
        "metrics": {
            "ttft_mean_ms": 80.0,
            "itl_mean_ms": 20.0,
            "e2e_mean_ms": 2640.0,
            "output_tok_s": 500.0,
            "request_throughput": 2.5,
        },
        "requests": {"completed": 8, "dropped_unservable": 1},
        "window": {"duration_s": 3.2},
    }

    metrics = normalize_result(result)

    assert metrics["ttft_mean_ms"] == 80.0
    assert metrics["tokens_per_sec"] == 500.0
    assert metrics["responses_per_sec"] == 2.5
    assert metrics["completed_requests"] == 8
    assert metrics["dropped_unservable"] == 1
    assert metrics["vllm_estimated_duration_s"] == 3.2


def test_normalize_result_falls_back_to_the_simulated_duration():
    result = {
        "metrics": {"ttft_mean_ms": 80.0, "itl_mean_ms": 20.0},
        "engine": {"cluster": {"simulated_duration_s": 9.5}},
    }

    metrics = normalize_result(result)

    assert metrics["vllm_estimated_duration_s"] == 9.5
    assert metrics["completed_requests"] is None
    assert metrics["dropped_unservable"] == 0


def _result(
    *,
    completed: int = 8,
    dropped: int = 0,
    ttft: float = 80.0,
    itl: float = 20.0,
    e2e: float = 2640.0,
    tok_s: float = 500.0,
    duration: float = 3.2,
    throughput: float | None = None,
    kv_capacity: float | None = None,
) -> dict:
    """A minimal BLIS v0.15 Result document for service-level tests."""
    result = {
        "kind": "Result",
        "metrics": {
            "ttft_mean_ms": ttft,
            "itl_mean_ms": itl,
            "e2e_mean_ms": e2e,
            "output_tok_s": tok_s,
            "request_throughput": throughput if throughput is not None else completed / duration,
        },
        "requests": {"completed": completed, "dropped_unservable": dropped},
        "window": {"duration_s": duration},
        "input": {"deployment": {"pools": [{
            "parallel": {"tp": 1},
            "engine": {"max_model_len": "auto"},
        }]}},
        "revisions": {"engine": {"upstream_version": "0.31.0"}},
    }
    if kv_capacity is not None:
        result["engine"] = {"cluster": {"kv_capacity_tokens_per_replica": kv_capacity}}
    return result


def test_recommend_candidate_ceiling_uses_configured_limit_and_hardware(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 128}},
    )
    calls = []

    def mock_run(_runtime, run_request, *, timeout_seconds):
        calls.append(run_request)
        return _result(completed=4)

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


def test_recommend_includes_bounded_pd_topologies_when_fabric_is_selected(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)
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
        return _result()

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


def test_run_command_uses_rate_and_omits_concurrency():
    runtime = BlisRuntime("/opt/venv/bin/blis", "/catalog", "/registry")
    command = build_run_command(runtime, RunRequest(
        scenario="generated.yaml",
        scenario_directory="/tmp/scenarios",
        isl=2048,
        osl=128,
        concurrency=None,
        num_requests=50,
        rate=10.0,
    ))

    assert "--rate" in command
    assert command[command.index("--rate") + 1] == "10"
    assert "--concurrency" not in command
    assert command[command.index("--num-requests") + 1] == "50"


def test_run_command_requires_exactly_one_load_control():
    runtime = BlisRuntime("/opt/venv/bin/blis", "/catalog", "/registry")
    with pytest.raises(ValueError, match="exactly one"):
        build_run_command(runtime, RunRequest(
            scenario="generated.yaml",
            scenario_directory="/tmp/scenarios",
            isl=2048,
            osl=128,
            concurrency=8,
            num_requests=8,
            rate=10.0,
        ))
    with pytest.raises(ValueError, match="exactly one"):
        build_run_command(runtime, RunRequest(
            scenario="generated.yaml",
            scenario_directory="/tmp/scenarios",
            isl=2048,
            osl=128,
            concurrency=None,
            num_requests=8,
        ))


def _demand_replay(completed: int, arrival_rps: float, *, e2e_ms: float = 2640.0) -> dict:
    """A Result for a run that offered `arrival_rps` over its arrivals.

    BLIS times a run to its last completion, so the window is the arrival
    window plus one mean latency. Request throughput is measured over that
    longer window, which is exactly what the drain-aware compliance check
    corrects for.
    """
    arrival_s = completed / arrival_rps
    duration_s = arrival_s + e2e_ms / 1000.0
    return _result(
        completed=completed,
        e2e=e2e_ms,
        duration=duration_s,
        tok_s=arrival_rps * 128,
        throughput=completed / duration_s,
    )


def _demand_recommend_input() -> blis_service.RecommendInput:
    return blis_service.RecommendInput(
        model_path="Qwen/Qwen3-32B",
        system="h100_sxm",
        backend="vllm",
        backend_version=None,
        isl=2048,
        osl=128,
        concurrency=None,
        request_rate=10.0,
        ttft=None,
        tpot=None,
        request_latency=None,
        prefix=0,
        min_candidate_gpus=1,
        max_candidate_gpus=4,
        top_n=5,
        search_budget_seconds=60,
        demand=True,
    )


def test_recommend_demand_selects_fewest_gpus_that_sustains_rate(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(blis_service, "load_system_spec", lambda _system: {"node": {"num_gpus_per_node": 4}})
    calls = []

    def mock_run(_runtime, run_request, *, timeout_seconds):
        calls.append(run_request)
        # Candidate order is ascending: 1 GPU, then 2, then 4.
        achieved = 4.0 if len(calls) == 1 else 12.0
        return _demand_replay(run_request.num_requests, achieved)

    monkeypatch.setattr(blis_service, "run_blis", mock_run)

    result = blis_service.recommend(_demand_recommend_input())

    assert len(calls) == 3
    assert all(call.rate == 10.0 for call in calls)
    assert all(call.concurrency is None for call in calls)
    # max(4, round(10 req/s * 5 s)) = 50 requests per candidate.
    assert all(call.num_requests == 50 for call in calls)
    assert result.demand is not None
    assert result.demand.met is True
    assert result.demand.achieved_rps == pytest.approx(12.0)
    assert result.demand.evaluated_configurations == 3
    assert result.configs[0]["total_gpus_needed"] == 2
    assert result.search_summary["modes_searched"] == ["agg"]


def test_recommend_demand_reports_best_observed_when_rate_is_unmet(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(blis_service, "load_system_spec", lambda _system: {"node": {"num_gpus_per_node": 4}})

    def mock_run(_runtime, run_request, *, timeout_seconds):
        return _demand_replay(run_request.num_requests, 4.0)

    monkeypatch.setattr(blis_service, "run_blis", mock_run)

    result = blis_service.recommend(_demand_recommend_input())

    assert result.demand is not None
    assert result.demand.met is False
    assert result.demand.best_observed_rps == pytest.approx(4.0)
    assert result.demand.target_rps == 10.0
    assert result.configs[0]["total_gpus_needed"] == 1


def test_recommend_demand_rejects_latency_targets(monkeypatch, catalog):
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)
    request = _demand_recommend_input()
    with pytest.raises(blis_service.BlisUnsupportedRequest, match="does not apply latency targets"):
        blis_service.recommend(blis_service.RecommendInput(
            **{**request.__dict__, "ttft": 1000.0}
        ))


def test_recommend_skips_pd_candidates_without_a_fabric(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(blis_service, "load_system_spec", lambda _system: {"node": {"num_gpus_per_node": 4}})
    monkeypatch.setattr(blis_service, "run_blis", lambda *_args, **_kwargs: _result())

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
        any("No network fabric was selected" in warning for warning in config["warnings"])
        for config in result.configs
    )


def test_disagg_estimate_builds_p_and_d_pools_and_uses_requested_fabric(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)
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
        return _result()

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


def test_normalize_result_maps_the_engine_kv_pool():
    metrics = normalize_result({
        "metrics": {},
        "engine": {"cluster": {"kv_capacity_tokens_per_replica": 123456}},
    })

    assert metrics["kv_capacity_tokens_per_replica"] == 123456


def _capacity_input(**overrides) -> blis_service.RecommendInput:
    values = {
        "model_path": "Qwen/Qwen3-32B",
        "system": "h100_sxm",
        "backend": "vllm",
        "backend_version": None,
        "isl": 2048,
        "osl": 128,
        "concurrency": 8,
        "request_rate": None,
        "ttft": None,
        "tpot": None,
        "request_latency": None,
        "prefix": 0,
        "min_candidate_gpus": 1,
        "max_candidate_gpus": 4,
        "top_n": 5,
        "search_budget_seconds": 60,
        "capacity": True,
    }
    values.update(overrides)
    return blis_service.RecommendInput(**values)


def test_recommend_capacity_reads_the_blis_kv_pool_and_sizes_replicas(
    monkeypatch, catalog
):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 4}},
    )
    calls = []
    monkeypatch.setattr(
        blis_service,
        "run_blis",
        lambda _runtime, run_request, *, timeout_seconds: (
            calls.append(run_request) or _result(kv_capacity=10_000)
        ),
    )

    result = blis_service.recommend(_capacity_input())

    # Every shape is probed at concurrency 1; the pool is a shape property,
    # not a load property.
    assert len(calls) == 3
    assert all(call.concurrency == 1 for call in calls)
    capacity = result.capacity
    assert capacity is not None
    assert capacity.target_concurrency == 8
    assert capacity.context_per_sequence_tokens == 2176
    assert capacity.capacity_sequences == 9
    assert capacity.fits is True
    assert capacity.minimum_gpus == 1
    assert capacity.replicas == 2
    assert capacity.per_rank_tokens == 10_000
    assert capacity.performance_available is False
    assert result.configs[0]["total_gpus_needed"] == 2
    assert result.configs[0]["replicas_needed"] == 2
    assert result.configs[0]["mode"] == "agg"
    assert "does not simulate the target concurrency" in " ".join(
        result.configs[0]["warnings"]
    )
    assert result.search_summary["strategy"] == "aggregate_tp_capacity"


def test_recommend_capacity_rejects_a_request_rate_before_running(monkeypatch, catalog):
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)

    with pytest.raises(BlisUnsupportedRequest, match="does not accept target_request_rate"):
        blis_service.recommend(_capacity_input(request_rate=10.0, concurrency=None))


def test_recommend_capacity_reports_the_shortfall_when_the_window_is_too_small(
    monkeypatch, catalog
):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 2}},
    )
    monkeypatch.setattr(
        blis_service,
        "run_blis",
        lambda _runtime, _run_request, *, timeout_seconds: _result(kv_capacity=5_000),
    )

    result = blis_service.recommend(_capacity_input(
        max_candidate_gpus=2,
        concurrency=16,
    ))

    capacity = result.capacity
    assert capacity is not None
    assert capacity.fits is False
    assert capacity.capacity_sequences == 4
    assert capacity.shortfall_sequences == 12
    assert capacity.shortfall_tokens == 16 * 2176 - 10_000
    assert result.configs[0]["total_gpus_needed"] == 2


def test_recommend_capacity_sizes_replicas_beyond_one_node(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 2}},
    )
    monkeypatch.setattr(
        blis_service,
        "run_blis",
        lambda _runtime, _run_request, *, timeout_seconds: _result(kv_capacity=10_000),
    )

    # The window totals GPUs across replicas: 4 replicas of one GPU exceed the
    # two-GPU node but still fit the requested eight-GPU window.
    result = blis_service.recommend(_capacity_input(
        max_candidate_gpus=8,
        concurrency=16,
    ))

    capacity = result.capacity
    assert capacity is not None
    assert capacity.fits is True
    assert capacity.replicas == 4
    assert capacity.per_rank_tokens == 10_000
    assert result.configs[0]["total_gpus_needed"] == 4
    assert result.configs[0]["replicas_needed"] == 4
    assert result.search_summary["effective_max_gpus"] == 8
    assert result.search_summary["hardware_max_gpus"] == 2


def test_recommend_capacity_fails_when_blis_reports_no_kv_pool(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 2}},
    )
    monkeypatch.setattr(
        blis_service,
        "run_blis",
        lambda _runtime, _run_request, *, timeout_seconds: _result(),
    )

    with pytest.raises(BlisExecutionError, match="kv_capacity_tokens_per_replica"):
        blis_service.recommend(_capacity_input(max_candidate_gpus=2))


def test_recommend_pareto_enumerates_topologies_and_concurrency(
    monkeypatch, catalog
):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 2}},
    )
    monkeypatch.setattr(blis_service, "_pareto_concurrency_levels", lambda: [1, 8])
    calls = []

    def mock_run(_runtime, run_request, *, timeout_seconds):
        scenario_path = Path(run_request.scenario_directory) / run_request.scenario
        documents = list(yaml.safe_load_all(scenario_path.read_text()))
        tp = documents[1]["pools"][0]["parallel"]["tp"]
        concurrency = run_request.concurrency
        calls.append((tp, concurrency))
        # One GPU wins throughput per GPU; two GPUs win throughput per user.
        tokens_per_second = concurrency * (100 if tp == 1 else 150)
        return _result(
            completed=concurrency,
            tok_s=tokens_per_second,
            ttft=50.0,
            itl=10.0,
        )

    monkeypatch.setattr(blis_service, "run_blis", mock_run)

    result = blis_service.recommend(_capacity_input(
        capacity=False,
        concurrency=None,
        max_candidate_gpus=2,
        pareto=True,
        ttft=1000.0,
        tpot=30.0,
        strict_sla=True,
    ))

    assert calls == [(1, 1), (2, 1), (1, 8), (2, 8)]
    assert result.search_summary["strategy"] == "enumerated_topology_and_concurrency_grid"
    assert result.search_summary["candidate_configurations"] == 4
    assert result.search_summary["evaluated_configurations"] == 4
    assert result.search_summary["modes_searched"] == ["agg"]
    assert len(result.candidates) == 4
    assert all(candidate["status"] == "feasible" for candidate in result.candidates)
    assert all(candidate["simulator"] == "blis" for candidate in result.candidates)
    # Both shapes are non-dominated at each load level.
    assert set(result.views["pareto_front"]) == {
        "agg-tp1-c1", "agg-tp2-c1", "agg-tp1-c8", "agg-tp2-c8",
    }
    assert len(result.configs) == 4
    assert all(config["simulator"] == "blis" for config in result.configs)


def test_recommend_pareto_marks_sla_violations_infeasible(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis", "/catalog", "/registry")
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 2}},
    )
    monkeypatch.setattr(blis_service, "_pareto_concurrency_levels", lambda: [1])
    monkeypatch.setattr(
        blis_service,
        "run_blis",
        lambda _runtime, _run_request, *, timeout_seconds: _result(ttft=500.0),
    )

    result = blis_service.recommend(_capacity_input(
        capacity=False,
        concurrency=None,
        max_candidate_gpus=1,
        pareto=True,
        ttft=100.0,
        tpot=30.0,
        strict_sla=True,
    ))

    assert result.configs == []
    assert result.views == {"pareto_front": [], "top_n": []}
    assert all(
        candidate["status"] == "infeasible"
        and candidate["reason_category"] == "latency_sla"
        for candidate in result.candidates
    )


def test_recommend_pareto_rejects_a_load_target(monkeypatch, catalog):
    monkeypatch.setattr(blis_service, "load_catalog", lambda: catalog)

    with pytest.raises(BlisUnsupportedRequest, match="does not accept target_concurrency"):
        blis_service.recommend(_capacity_input(
            capacity=False,
            concurrency=8,
            pareto=True,
        ))


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

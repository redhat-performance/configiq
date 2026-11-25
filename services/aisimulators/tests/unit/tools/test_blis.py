from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools.api_service.blis import engines as blis_engines
from tools.api_service.blis import service as blis_service
from tools.api_service.blis.catalog import (
    BlisCatalog,
    CatalogChip,
    CatalogModel,
    ModelResolution,
    parse_catalog,
    read_catalog,
)
from tools.api_service.blis.engines import BlisEngine
from tools.api_service.blis.errors import BlisCatalogError, BlisUnsupportedRequest
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
    write_scenario,
)


def _catalog_model(
    name: str,
    repo: str | None,
    *,
    architecture: str = "Qwen3ForCausalLM",
    context_window: int | None = 40960,
    experts: int | None = None,
    experts_top_k: int | None = None,
    attention_heads: int = 40,
    attention_kv_heads: int = 8,
) -> CatalogModel:
    return CatalogModel(
        name=name,
        hf_repo=repo,
        architecture=architecture,
        context_window=context_window,
        experts=experts,
        experts_top_k=experts_top_k,
        attention_kind="gqa",
        attention_heads=attention_heads,
        attention_kv_heads=attention_kv_heads,
    )


@pytest.fixture
def catalog() -> BlisCatalog:
    return BlisCatalog(
        models={
            "qwen3-32b": _catalog_model("qwen3-32b", "Qwen/Qwen3-32B"),
            "glm-5": _catalog_model("glm-5", "zai-org/GLM-5"),
        },
        chips={
            "h100": CatalogChip(name="h100", memory_gib=80),
            "h200": CatalogChip(name="h200", memory_gib=141),
            "a100-sxm": CatalogChip(name="a100-sxm", memory_gib=80),
            "gb200-nvl72": CatalogChip(name="gb200-nvl72", memory_gib=192, gpus_per_node=4),
        },
        fabrics=("ib-400g", "roce-200g"),
    )


def _catalog_document() -> dict:
    """A report shaped exactly like `blis catalog list --json`."""
    return {
        "models": [
            {
                "name": "qwen3-32b",
                "source": {
                    "provider": "huggingface",
                    "repo": "Qwen/Qwen3-32B",
                    "revision": "abc123",
                },
                "architecture": "Qwen3ForCausalLM",
                "context_window": 40960,
                "layers": 64,
                "hidden_size": 5120,
                "vocab_size": 151936,
                "weight_dtype": "bf16",
                "attention": {"kind": "gqa", "heads": 40, "kv_heads": 8, "head_dim": 128},
            },
            {
                "name": "qwen3-30b-a3b",
                "source": {"provider": "huggingface", "repo": "Qwen/Qwen3-30B-A3B"},
                "architecture": "Qwen3MoeForCausalLM",
                "layers": 48,
                "hidden_size": 2048,
                "vocab_size": 151936,
                "weight_dtype": "bf16",
                "attention": {"kind": "gqa", "heads": 32, "kv_heads": 4, "head_dim": 128},
                "experts": {"count": 128, "top_k": 8},
                "drafters": [{"name": "speculator", "method": "mtp", "num_spec_tokens": 1}],
            },
            {
                # A draft graph can be published under the served model's repo.
                "name": "speculator",
                "source": {"provider": "huggingface", "repo": "Qwen/Qwen3-30B-A3B"},
                "architecture": "Qwen3MoeForCausalLM",
                "layers": 1,
                "hidden_size": 2048,
                "vocab_size": 151936,
                "weight_dtype": "bf16",
                "attention": {"kind": "gqa", "heads": 32, "kv_heads": 4, "head_dim": 128},
            },
            {
                # A graph that states no window: the report must carry that gap
                # rather than a value derived from a file the build never read.
                "name": "no-window",
                "source": {"provider": "huggingface", "repo": "org/NoWindow"},
                "architecture": "Qwen3ForCausalLM",
                "attention": {"kind": "gqa", "heads": 32, "kv_heads": 8, "head_dim": 128},
            },
            {
                "name": "mla-model",
                "source": {"provider": "huggingface", "repo": "org/Mla"},
                "architecture": "DeepseekV3ForCausalLM",
                "context_window": 1048576,
                "attention": {"kind": "mla", "heads": 128, "kv_heads": 1, "head_dim": 576},
                "experts": {"count": 256, "top_k": 8},
            },
        ],
        "chips": [
            {
                "name": "h100",
                "provenance": "vendor_spec",
                "field_provenance": {"memory_gib": "vendor_spec"},
                "bf16_tflops": 989,
                "fp8_tflops": 1979,
                "nvfp4_tflops": 3958,
                "memory_gib": 80,
                "memory_bandwidth_tbs": 3.35,
                "intra_node_bw_gbps": 900,
                "sm_count": 132,
                "compute_capability": "9.0",
                "host_cpu": "x86_64",
            },
            {
                "name": "gb200-nvl72",
                "memory_gib": 192,
                "memory_bandwidth_tbs": 8,
                "compute_capability": "10.0",
                "gpus_per_node": 4,
                "gpus_per_rack": 72,
            },
            {
                "name": "rtx6000pro-server",
                "memory_gib": 96,
                "compute_capability": "12.0",
            },
            {
                # A chip whose entry states neither capability nor packaging.
                "name": "mystery",
                "memory_gib": 48,
            },
        ],
        "fabrics": [
            {"name": "roce-200g", "provenance": "vendor_spec", "inter_node_bw_gbps": 25, "rdma": True},
            {"name": "ib-400g", "provenance": "vendor_spec", "inter_node_bw_gbps": 50, "rdma": True},
        ],
    }


def _stub_blis(
    tmp_path: Path,
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
) -> Path:
    """A stand-in `blis` that prints the given report and exits."""
    script = "#!/bin/sh\n"
    if stdout:
        (tmp_path / "stdout.json").write_text(stdout)
        script += f'cat "{tmp_path / "stdout.json"}"\n'
    if stderr:
        (tmp_path / "stderr.txt").write_text(stderr)
        script += f'cat "{tmp_path / "stderr.txt"}" >&2\n'
    script += f"exit {exit_code}\n"
    binary = tmp_path / "blis"
    binary.write_text(script)
    binary.chmod(0o755)
    return binary


def test_catalog_reports_the_hugging_face_ids_the_build_can_price():
    catalog = parse_catalog(_catalog_document())

    assert catalog.model_ids() == [
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
        "org/Mla",
        "org/NoWindow",
    ]


def test_catalog_resolves_slug_and_hugging_face_id_to_one_entry():
    catalog = parse_catalog(_catalog_document())

    by_slug = catalog.resolve_model("qwen3-32b")
    by_hf_id = catalog.resolve_model("Qwen/Qwen3-32B")

    assert by_slug.catalog_name == "qwen3-32b"
    assert by_hf_id.catalog_name == "qwen3-32b"
    assert by_slug.hf_repo == "Qwen/Qwen3-32B"
    assert by_slug.context_length == 40960
    assert by_slug.origin == "blis-catalog"


def test_catalog_resolves_a_repo_shared_with_a_draft_graph_to_the_served_model():
    catalog = parse_catalog(_catalog_document())

    # A draft graph may be published under the served model's repo. An ID that
    # names both has to resolve to the entry a request can serve, or a model
    # picker would offer a graph that cannot answer a request.
    resolved = catalog.resolve_model("Qwen/Qwen3-30B-A3B")

    assert resolved.catalog_name == "qwen3-30b-a3b"
    # The draft graph is still addressable by its own catalog name.
    assert catalog.resolve_model("speculator").catalog_name == "speculator"
    # A shared repo is one public identifier, not two.
    assert catalog.model_ids().count("Qwen/Qwen3-30B-A3B") == 1


def test_catalog_reports_the_window_the_graph_states_and_null_when_it_states_none():
    catalog = parse_catalog(_catalog_document())

    assert catalog.resolve_model("org/Mla").context_length == 1048576
    # No window in the graph means no window in the report: the build reads no
    # model config file, so there is nothing to derive one from.
    assert catalog.resolve_model("org/NoWindow").context_length is None


def test_catalog_refuses_models_it_cannot_price():
    catalog = parse_catalog(_catalog_document())

    with pytest.raises(BlisUnsupportedRequest, match="does not support model.*org/NoWindow"):
        catalog.resolve_model("Unknown/Model")


def test_catalog_translates_system_aliases_but_not_pcie_parts():
    catalog = parse_catalog(_catalog_document())

    assert catalog.resolve_hardware("h100_sxm") == "h100"
    assert catalog.resolve_hardware("gb200") == "gb200-nvl72"
    assert catalog.resolve_hardware("rtx_pro_6000_server") == "rtx6000pro-server"
    assert catalog.supports_hardware("h100_sxm") is True
    with pytest.raises(BlisUnsupportedRequest, match="available BLIS hardware keys"):
        catalog.resolve_hardware("h100_pcie")


def test_catalog_returns_the_chip_facts_the_build_states():
    catalog = parse_catalog(_catalog_document())

    h100 = catalog.chip("h100_sxm")
    assert h100 is not None
    assert h100.memory_gib == 80
    assert h100.memory_bandwidth_tbs == 3.35
    assert h100.bf16_tflops == 989
    assert h100.compute_capability == "9.0"
    assert h100.gpus_per_node is None
    assert catalog.chip("gb200").gpus_per_node == 4
    assert catalog.chip("h100_pcie") is None


def test_catalog_reports_only_fabrics_it_declares():
    catalog = parse_catalog(_catalog_document())

    assert catalog.fabrics == ("ib-400g", "roce-200g")
    assert catalog.resolve_network("roce-200g") == "roce-200g"
    with pytest.raises(BlisUnsupportedRequest, match="available fabric keys: ib-400g, roce-200g"):
        catalog.resolve_network("unknown-fabric")


def test_catalog_specs_read_experts_and_attention_fields():
    catalog = parse_catalog(_catalog_document())

    dense = catalog.specs_for("Qwen/Qwen3-32B")
    moe = catalog.specs_for("qwen3-30b-a3b")
    mla = catalog.specs_for("org/Mla")

    assert dense["num_experts"] is None
    assert dense["num_experts_per_tok"] is None
    assert dense["context_length"] == 40960
    assert dense["num_attn_heads"] == 40
    assert dense["num_kv_heads"] == 8
    assert dense["architecture"] == "Qwen3ForCausalLM"
    assert moe["num_experts"] == 128
    assert moe["num_experts_per_tok"] == 8
    assert mla["num_kv_heads"] == 1
    assert catalog.specs_for("org/NoWindow")["context_length"] is None


def test_read_catalog_asks_the_build_for_the_catalog_it_embeds(tmp_path: Path):
    binary = _stub_blis(tmp_path, stdout=json.dumps(_catalog_document()))

    catalog = read_catalog(str(binary))

    assert "Qwen/Qwen3-32B" in catalog.model_ids()
    assert catalog.resolve_hardware("h100_sxm") == "h100"


def test_read_catalog_reports_a_binary_that_cannot_answer(tmp_path: Path):
    binary = _stub_blis(tmp_path, stderr="catalog: not embedded", exit_code=2)

    with pytest.raises(BlisCatalogError, match="did not report its catalog.*not embedded"):
        read_catalog(str(binary))


def test_read_catalog_rejects_output_that_is_not_json(tmp_path: Path):
    binary = _stub_blis(tmp_path, stdout="not json")

    with pytest.raises(BlisCatalogError, match="was not JSON"):
        read_catalog(str(binary))


def test_read_catalog_reports_an_unrunnable_binary(tmp_path: Path):
    with pytest.raises(BlisCatalogError, match="could not ask BLIS for its catalog"):
        read_catalog(str(tmp_path / "missing"))


def test_catalog_report_without_models_is_an_explicit_error():
    with pytest.raises(BlisCatalogError, match="no models"):
        parse_catalog({"models": [], "chips": [{"name": "h100"}], "fabrics": []})
    with pytest.raises(BlisCatalogError, match="must be a JSON object"):
        parse_catalog([])


def _scenario_request(**overrides) -> ScenarioRequest:
    values = {
        "engine_kind": "vllm",
        "model": ModelResolution(
            catalog_name="qwen3-32b",
            hf_repo="Qwen/Qwen3-32B",
            origin="blis-catalog",
        ),
        "hardware": "h100",
        "system": "h100_sxm",
        "gpus_per_node": 8,
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
    assert scenario["engine"] == {"kind": "vllm"}
    assert "engine_version" not in scenario
    assert scenario["coefficients"] == [
        "cost-model-primitives",
        "cost-model-collectives",
        "cost-model-host-overheads",
        "cost-model-attention",
        "cost-model-recurrent",
        "cost-model-memory",
        "cost-model-fabric",
        "cost-model-offload-tiers",
    ]
    assert scenario["cluster"] == {"hardware": "h100", "nodes": 1, "gpus_per_node": 8}
    assert pool["parallel"] == {
        "tp": 2, "pp": 1, "dp": 1, "enable_expert_parallel": False,
    }
    assert pool["engine"]["quantization"] == "nvfp4"
    assert pool["engine"]["cache_dtype"] == "fp8"
    assert pool["engine"]["compilation_config"] == {"cudagraph_mode": "PIECEWISE"}
    assert "cudagraph_mode" not in pool["engine"]
    assert pool["engine"]["enable_prefix_caching"] is True
    assert pool["engine"]["max_model_len"] == 2176


def test_generated_documents_state_the_catalog_context_window():
    _scenario, deployment = build_documents(_scenario_request(
        model=ModelResolution(
            catalog_name="gpt-oss-120b",
            hf_repo="openai/gpt-oss-120b",
            context_length=131072,
            origin="blis-catalog",
        ),
    ))

    assert deployment["pools"][0]["engine"]["max_model_len"] == 131072


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


def test_sglang_documents_use_that_engines_own_setting_names():
    _scenario, deployment = build_documents(_scenario_request(
        engine_kind="sglang",
        weight_quantization="NVFP4",
        cache_dtype="FP8",
        prefix_caching=True,
    ))
    engine = deployment["pools"][0]["engine"]

    assert engine["max_running_requests"] == 8
    assert engine["context_length"] == 2176
    assert engine["mem_fraction_static"] == 0.9
    assert engine["quantization"] == "modelopt_fp4"
    assert engine["kv_cache_dtype"] == "fp8_e4m3"
    # This engine states the opposite setting: whether the radix cache is off.
    assert engine["disable_radix_cache"] is False
    # Nothing from the other engine's vocabulary leaks into the block.
    for name in (
        "block_size",
        "max_num_seqs",
        "max_model_len",
        "gpu_memory_utilization",
        "enable_prefix_caching",
        "compilation_config",
    ):
        assert name not in engine


def test_sglang_refuses_a_quantization_it_cannot_name():
    with pytest.raises(BlisUnsupportedRequest, match="cannot represent weight quantization"):
        build_documents(_scenario_request(engine_kind="sglang", weight_quantization="int4"))
    with pytest.raises(BlisUnsupportedRequest, match="cannot represent KV-cache dtype"):
        build_documents(_scenario_request(engine_kind="sglang", cache_dtype="nvfp4"))
    # A value this engine does name is passed through unchanged.
    _scenario, deployment = build_documents(_scenario_request(
        engine_kind="sglang", cache_dtype="fp8_e5m2",
    ))
    assert deployment["pools"][0]["engine"]["kv_cache_dtype"] == "fp8_e5m2"


def test_sglang_disagg_pools_use_that_engines_own_setting_names():
    _scenario, deployment = build_documents(_scenario_request(
        engine_kind="sglang",
        fabric="ib-400g",
        prefill_pool=_pool_request("prefill", nodes=1),
        decode_pool=_pool_request("decode", nodes=1),
    ))

    for pool in deployment["pools"]:
        engine = pool["engine"]
        assert "max_running_requests" in engine
        assert "context_length" in engine
        assert "max_num_seqs" not in engine
        assert "max_model_len" not in engine


def test_scenario_refuses_an_engine_kind_it_does_not_generate():
    with pytest.raises(BlisUnsupportedRequest, match="does not model engine kind"):
        build_documents(_scenario_request(engine_kind="trtllm"))


def _resolved_for(engine: BlisEngine | None):
    return blis_service._ResolvedRequest(
        catalog=BlisCatalog(models={}, chips={}, fabrics=()),
        runtime=BlisRuntime("/opt/blis"),
        model=ModelResolution(catalog_name="qwen3-32b", hf_repo="Qwen/Qwen3-32B"),
        hardware="h100",
        gpus_per_node=8,
        engine_kind=engine.kind if engine is not None else "vllm",
        engine=engine,
    )


def _kernel_result(engine_version: str) -> dict:
    return {
        "reports": {
            "kernel": {
                "colocated": {
                    "assumptions": [{
                        "Name": "engine_version",
                        "Scope": f"{engine_version}: the kernel prices its own tuning",
                    }],
                },
            },
        },
    }


def test_reported_engine_version_is_the_selected_kinds_own_release():
    # The run's kernel assumptions name the latency kernel's shared behaviour,
    # which is not this kind's release; the build's per-kind report is.
    resolved = _resolved_for(BlisEngine(
        kind="sglang",
        modeled_version="0.5.23",
        modeled_commit=None,
        build_version="0.5.23",
        roles=("colocated",),
    ))

    version = blis_service._reported_engine_version(_kernel_result("0.31.0"), resolved)

    assert version == "0.5.23"


def test_reported_engine_version_falls_back_to_the_runs_disclosure():
    # A build that cannot report its engines leaves the run's disclosure as the
    # only statement of the release it priced.
    version = blis_service._reported_engine_version(_kernel_result("0.31.0"), _resolved_for(None))

    assert version == "0.31.0"


def test_reported_engine_version_is_null_when_nothing_reports_one():
    assert blis_service._reported_engine_version({}, _resolved_for(None)) is None


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
    }

    config = blis_service._serving_config(result, "0.31.0", "vllm")

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

    config = blis_service._serving_config(result, None, "vllm")

    assert config["max_model_len"] == 2176
    assert config["max_model_len_policy"] is None
    assert config["backend_version"] is None


def test_serving_config_is_absent_when_the_result_has_no_deployment():
    assert blis_service._serving_config({"metrics": {}}, None, "vllm") is None


def test_half_cache_dtype_leaves_the_blis_model_default_unmodified():
    _scenario, deployment = build_documents(_scenario_request(cache_dtype="half"))

    assert "cache_dtype" not in deployment["pools"][0]["engine"]


@pytest.mark.parametrize(
    ("requested", "emitted"),
    [
        ("bf16", "bfloat16"),
        ("fp16", "float16"),
        ("BF16", "bfloat16"),
        ("fp8", "fp8"),
        ("fp8_e4m3", "fp8_e4m3"),
        ("nvfp4", "nvfp4"),
    ],
)
def test_cache_dtype_is_emitted_under_the_name_the_engine_accepts(requested, emitted):
    """The API's aliases are resolved here; the engine names the format its own way.

    `bfloat16`/`float16` are the values the rules pack accepts for
    `engine.cache_dtype`, so `bf16`/`fp16` must not reach the deployment document.
    `nvfp4` is a cache format the pack accepts, so the adapter forwards it instead
    of refusing a value the engine supports.
    """
    _scenario, deployment = build_documents(_scenario_request(cache_dtype=requested))

    assert deployment["pools"][0]["engine"]["cache_dtype"] == emitted


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
    assert [pool["engine"]["cache_dtype"] for pool in deployment["pools"]] == ["fp8", "fp8"]
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


#: The document `blis version --json` prints for the pinned build: the build's
#: identity plus one record per engine kind it runs.
VERSION_DOCUMENT = {
    "inference_sim": "0.19.2-0.20261127005804-694802f59d85",
    "substrate": "4f9c2a0e1b",
    "engines": [
        {
            "kind": "sglang",
            "modeled_version": "0.5.23",
            "modeled_commit": "b1f2c3d4e5",
            "build_version": "0.5.23",
            "build_commit": "b1f2c3d4e5",
            "roles": ["colocated", "prefill", "decode"],
        },
        {
            "kind": "vllm",
            "modeled_version": "0.31.0",
            "modeled_commit": "0a1b2c3d4e",
            "build_version": "0.34.0+0fda86a35419",
            "build_commit": "0fda86a35419",
            "roles": ["colocated", "prefill", "decode"],
        },
    ],
}


def test_read_engines_asks_version_for_its_json_document(monkeypatch):
    captured: dict = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _completed(stdout=json.dumps(VERSION_DOCUMENT))

    monkeypatch.setattr(blis_engines.subprocess, "run", fake_run)

    engines = blis_engines.read_engines("/opt/blis")

    # `version` owns the build's identity and the engines it models, so an
    # adapter reads them from that one document instead of a second command.
    assert captured["command"] == ["/opt/blis", "version", "--json"]
    assert [engine.kind for engine in engines] == ["sglang", "vllm"]
    vllm = engines[1]
    # The modelled release is what a benchmark is comparable with; the build's
    # own snapshot is provenance beside it, not a substitute for it.
    assert (vllm.modeled_version, vllm.modeled_commit) == ("0.31.0", "0a1b2c3d4e")
    assert vllm.build_version == "0.34.0+0fda86a35419"
    assert vllm.roles == ("colocated", "prefill", "decode")


def test_read_engines_reports_nothing_for_a_build_without_the_json_flag(monkeypatch):
    # A build that predates `--json` refuses the flag and prints the text form,
    # which is not a document: report no engines rather than parse prose.
    monkeypatch.setattr(
        blis_engines.subprocess,
        "run",
        lambda *_args, **_kwargs: _completed(
            stdout="inference-sim 0.19.0\nsubstrate abc\n",
            returncode=1,
            stderr="unknown flag: --json",
        ),
    )

    assert blis_engines.read_engines("/opt/blis") == ()


def test_a_failed_probe_is_not_remembered_as_no_engines(monkeypatch):
    """One failed probe must not claim, for the process lifetime, that the build models nothing."""
    blis_engines.clear_engine_cache()
    probes: list[str] = []

    def probe(binary, **_kwargs):
        probes.append(binary)
        if len(probes) == 1:
            return ()
        return (BlisEngine("vllm", "0.31.0", None, None, ()),)

    monkeypatch.setattr(blis_engines, "read_engines", probe)
    monkeypatch.setattr(blis_engines, "PROBE_RETRY_SECONDS", 0.0)

    assert blis_engines.cached_engines("/opt/blis-retry") == ()
    assert [engine.modeled_version for engine in blis_engines.cached_engines("/opt/blis-retry")] == ["0.31.0"]
    assert probes == ["/opt/blis-retry", "/opt/blis-retry"]
    blis_engines.clear_engine_cache()


def test_a_successful_probe_is_read_once(monkeypatch):
    blis_engines.clear_engine_cache()
    probes: list[str] = []

    def probe(binary, **_kwargs):
        probes.append(binary)
        return (BlisEngine("vllm", "0.31.0", None, None, ()),)

    monkeypatch.setattr(blis_engines, "read_engines", probe)

    blis_engines.cached_engines("/opt/blis-once")
    blis_engines.cached_engines("/opt/blis-once")

    assert probes == ["/opt/blis-once"]
    blis_engines.clear_engine_cache()


def test_a_failed_probe_is_not_repeated_inside_the_retry_window(monkeypatch):
    """The retry window also bounds how often an unreadable build is spawned."""
    blis_engines.clear_engine_cache()
    probes: list[str] = []
    monkeypatch.setattr(blis_engines, "read_engines", lambda binary, **_k: probes.append(binary) or ())

    blis_engines.cached_engines("/opt/blis-window")
    blis_engines.cached_engines("/opt/blis-window")

    assert probes == ["/opt/blis-window"]
    blis_engines.clear_engine_cache()


def _completed(*, stdout: str, returncode: int = 0, stderr: str = ""):
    return blis_engines.subprocess.CompletedProcess(
        args=(), returncode=returncode, stdout=stdout, stderr=stderr
    )
def _resolving(monkeypatch, catalog, binary: str, engines: tuple[BlisEngine, ...]) -> None:
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: BlisRuntime(binary))
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 8}},
    )
    monkeypatch.setattr(blis_service, "cached_engines", lambda _binary: engines)


def test_request_for_an_engine_kind_the_build_does_not_report_is_refused(monkeypatch, catalog):
    _resolving(
        monkeypatch,
        catalog,
        "/opt/blis-only-vllm",
        (
            BlisEngine(
                kind="vllm",
                modeled_version="0.31.0",
                modeled_commit=None,
                build_version="0.34.0+0fda86a35419",
                roles=("colocated", "prefill", "decode"),
            ),
        ),
    )

    with pytest.raises(BlisUnsupportedRequest, match="no engine kind 'sglang'"):
        blis_service._resolve_request("Qwen/Qwen3-32B", "h100_sxm", "sglang", None)


def test_aggregated_request_is_refused_for_an_engine_without_that_role(monkeypatch, catalog):
    _resolving(
        monkeypatch,
        catalog,
        "/opt/blis-prefill-only",
        (
            BlisEngine(
                kind="vllm",
                modeled_version="0.31.0",
                modeled_commit=None,
                build_version=None,
                roles=("prefill", "decode"),
            ),
        ),
    )
    resolved = blis_service._resolve_request("Qwen/Qwen3-32B", "h100_sxm", "vllm", None)

    assert resolved.supports_roles(("prefill", "decode")) is True
    assert resolved.supports_roles(("colocated",)) is False
    with pytest.raises(BlisUnsupportedRequest, match="models vllm for prefill, decode serving"):
        blis_service._require_roles(resolved, ("colocated",), topology="aggregated")


def test_disaggregated_request_is_refused_for_an_engine_without_pd_roles(monkeypatch, catalog):
    _resolving(
        monkeypatch,
        catalog,
        "/opt/blis-colocated-only",
        (
            BlisEngine(
                kind="sglang",
                modeled_version="0.5.23",
                modeled_commit=None,
                build_version=None,
                roles=("colocated",),
            ),
        ),
    )
    resolved = blis_service._resolve_request("Qwen/Qwen3-32B", "h100_sxm", "sglang", None)

    assert resolved.supports_roles(("colocated",)) is True
    with pytest.raises(BlisUnsupportedRequest, match="a disaggregated request needs prefill, decode"):
        blis_service._require_roles(resolved, ("prefill", "decode"), topology="disaggregated")


def test_a_build_that_reports_no_roles_is_not_refused(monkeypatch, catalog):
    _resolving(
        monkeypatch,
        catalog,
        "/opt/blis-no-roles",
        (
            BlisEngine(
                kind="vllm",
                modeled_version="0.31.0",
                modeled_commit=None,
                build_version=None,
                roles=(),
            ),
        ),
    )
    resolved = blis_service._resolve_request("Qwen/Qwen3-32B", "h100_sxm", "vllm", None)

    assert resolved.supports_roles(("colocated",)) is True
    assert resolved.supports_roles(("prefill", "decode")) is True


def test_request_for_an_unmodeled_release_is_refused_with_the_modeled_one(
    monkeypatch, catalog
):
    runtime = BlisRuntime("/opt/blis-unmodeled")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 8}},
    )
    monkeypatch.setattr(
        blis_service,
        "cached_engines",
        lambda _binary: (
            BlisEngine(
                kind="vllm",
                modeled_version="0.31.0",
                modeled_commit=None,
                build_version="0.34.0+0fda86a35419",
                roles=("colocated", "prefill", "decode"),
            ),
        ),
    )

    with pytest.raises(BlisUnsupportedRequest, match="models vllm 0.31.0"):
        blis_service._resolve_request("Qwen/Qwen3-32B", "h100_sxm", "vllm", "0.30.0")


def test_request_naming_the_modeled_release_resolves(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis-modeled")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 8}},
    )
    monkeypatch.setattr(
        blis_service,
        "cached_engines",
        lambda _binary: (
            BlisEngine(
                kind="vllm",
                modeled_version="0.31.0",
                modeled_commit=None,
                build_version=None,
                roles=(),
            ),
        ),
    )

    resolved = blis_service._resolve_request("Qwen/Qwen3-32B", "h100_sxm", "vllm", "0.31.0")

    assert resolved.hardware == "h100"


def test_request_for_a_release_is_refused_when_the_build_cannot_report_it(monkeypatch, catalog):
    """An unreadable build must not answer a pinned release with a different one.

    The probe cannot tell a failed read from a build that never reported, so an
    empty report means "unknown", not "models nothing": answering would
    substitute whatever release the build pins for the one the caller asked for.
    """
    runtime = BlisRuntime("/opt/blis-silent")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 8}},
    )
    monkeypatch.setattr(blis_service, "cached_engines", lambda _binary: ())

    with pytest.raises(BlisUnsupportedRequest, match="did not report which vllm release"):
        blis_service._resolve_request("Qwen/Qwen3-32B", "h100_sxm", "vllm", "0.30.0")


def test_request_without_a_version_runs_on_a_build_that_cannot_report_it(monkeypatch, catalog):
    """Leaving the version blank asks for no release, so nothing is substituted."""
    runtime = BlisRuntime("/opt/blis-silent-blank")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 8}},
    )
    monkeypatch.setattr(blis_service, "cached_engines", lambda _binary: ())

    resolved = blis_service._resolve_request("Qwen/Qwen3-32B", "h100_sxm", "vllm", None)

    assert resolved.model.catalog_name == "qwen3-32b"


def test_current_alias_means_the_release_the_build_models(monkeypatch, catalog):
    """`current` is the documented spelling of "the default", not a pin."""
    runtime = BlisRuntime("/opt/blis-current")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 8}},
    )
    monkeypatch.setattr(
        blis_service,
        "cached_engines",
        lambda _binary: (
            BlisEngine(
                kind="vllm",
                modeled_version="0.31.0",
                modeled_commit=None,
                build_version=None,
                roles=(),
            ),
        ),
    )

    resolved = blis_service._resolve_request("Qwen/Qwen3-32B", "h100_sxm", "vllm", "current")

    assert resolved.hardware == "h100"


@pytest.mark.parametrize("alias", ["previous", "next"])
def test_aisim_slot_aliases_are_refused_by_name(monkeypatch, catalog, alias):
    """`previous`/`next` name SDK data slots a BLIS build has no equivalent for.

    The refusal does not depend on the probe: the value is meaningless whether or
    not the build can report its release.
    """
    runtime = BlisRuntime("/opt/blis-aliases")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 8}},
    )
    monkeypatch.setattr(blis_service, "cached_engines", lambda _binary: ())

    with pytest.raises(BlisUnsupportedRequest, match="name AISim data slots"):
        blis_service._resolve_request("Qwen/Qwen3-32B", "h100_sxm", "vllm", alias)


def test_scenario_states_an_explicit_context_window():
    _scenario, deployment = build_documents(_scenario_request())

    assert deployment["pools"][0]["engine"]["max_model_len"] == 2048 + 128


def test_scenario_can_model_a_disaggregated_topology():
    request = _scenario_request(
        fabric="ib-400g",
        prefill_pool=_pool_request("prefill", nodes=1),
        decode_pool=_pool_request("decode", nodes=1),
    )

    _scenario, deployment = build_documents(request)

    assert [pool["role"] for pool in deployment["pools"]] == ["prefill", "decode"]


def test_run_command_passes_the_scenario_directory_without_external_data_paths():
    runtime = BlisRuntime(binary="/opt/venv/bin/blis")

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
        "--scenarios", "/tmp/scenarios",
        "--scenario", "generated.yaml",
        "--prompt-tokens", "2048",
        "--prompt-tokens-min", "2048",
        "--prompt-tokens-max", "2048",
        "--prompt-tokens-stdev", "0",
        "--output-tokens", "128",
        "--output-tokens-min", "128",
        "--output-tokens-max", "128",
        "--output-tokens-stdev", "0",
        "--num-requests", "8",
        "--concurrency", "8",
        "--prefix-tokens", "16",
    ]
    assert "--catalog" not in command
    assert "--registry" not in command


def test_run_command_passes_p_and_d_instance_counts():
    runtime = BlisRuntime("/opt/venv/bin/blis")
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


def _metric(value: float) -> dict:
    """One Result v4 aggregate: the value BLIS measured, its sample count, scope."""
    return {"value": value, "samples": 1, "scope": "cluster"}


def test_normalize_result_maps_metrics_requests_and_window():
    result = {
        "schema_version": 4,
        "metrics": {
            "ttft_mean_ms": _metric(80.0),
            "itl_mean_ms": _metric(20.0),
            "e2e_mean_ms": _metric(2640.0),
            "output_tok_s": _metric(500.0),
            "request_throughput": _metric(2.5),
        },
        "requests": {"completed": 8, "dropped": {"unservable": 1}},
        "window": {"duration_s": 3.2},
    }

    metrics = normalize_result(result)

    assert metrics["ttft_mean_ms"] == 80.0
    assert metrics["tokens_per_sec"] == 500.0
    assert metrics["responses_per_sec"] == 2.5
    assert metrics["completed_requests"] == 8
    assert metrics["dropped_unservable"] == 1
    assert metrics["vllm_estimated_duration_s"] == 3.2


def test_normalize_result_reads_whole_run_rates_when_a_metric_is_absent():
    result = {
        "schema_version": 4,
        "metrics": {"ttft_mean_ms": _metric(80.0), "itl_mean_ms": _metric(20.0)},
        "reports": {
            "engine": {
                "cluster": {
                    "whole_run": {"tokens_per_sec": 480.0, "responses_per_sec": 2.4},
                    "kv_capacity_tokens_per_replica": 99_999,
                },
            },
        },
    }

    metrics = normalize_result(result)

    assert metrics["tokens_per_sec"] == 480.0
    assert metrics["responses_per_sec"] == 2.4
    assert metrics["kv_capacity_tokens_per_replica"] == 99_999
    assert metrics["completed_requests"] is None
    assert metrics["dropped_unservable"] == 0


def test_normalize_result_falls_back_to_the_simulated_duration():
    result = {
        "schema_version": 4,
        "metrics": {"ttft_mean_ms": _metric(80.0), "itl_mean_ms": _metric(20.0)},
        "reports": {"engine": {"cluster": {"simulated_duration_s": 9.5}}},
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
    modeled_engine_version: str | None = "0.31.0",
) -> dict:
    """A minimal BLIS Result v4 document for service-level tests."""
    cluster: dict = {
        "whole_run": {
            "tokens_per_sec": tok_s,
            "responses_per_sec": throughput if throughput is not None else completed / duration,
        },
    }
    if kv_capacity is not None:
        cluster["kv_capacity_tokens_per_replica"] = kv_capacity
    kernel: dict = {}
    if modeled_engine_version is not None:
        kernel["colocated"] = {
            "assumptions": [{
                "Name": "engine_version",
                "Scope": f"{modeled_engine_version}: the scenario declares the release the kernel prices",
            }],
        }
    return {
        "kind": "Result",
        "schema_version": 4,
        # The build's own engine snapshot. It is provenance, never the release
        # the service reports as modelled.
        "engine": {
            "kind": "vllm",
            "version": "0.34.0+0fda86a35419",
            "commit": "0fda86a35419c9c70dbfada224c838c10fbb62ad",
        },
        "metrics": {
            "ttft_mean_ms": _metric(ttft),
            "itl_mean_ms": _metric(itl),
            "e2e_mean_ms": _metric(e2e),
            "output_tok_s": _metric(tok_s),
            "request_throughput": _metric(
                throughput if throughput is not None else completed / duration
            ),
        },
        "requests": {
            "generated": completed + dropped,
            "completed": completed,
            "dropped": {"unservable": dropped},
        },
        "window": {"duration_s": duration},
        "input": {"deployment": {"pools": [{
            "parallel": {"tp": 1},
            "engine": {"max_model_len": 2176},
        }]}},
        "reports": {"engine": {"cluster": cluster}, "kernel": kernel},
    }


def test_recommend_candidate_ceiling_uses_configured_limit_and_hardware(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
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
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
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


def test_recommend_warns_when_a_named_fabric_reached_no_pd_candidate(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 4}},
    )
    monkeypatch.setattr(blis_service, "run_blis", lambda *_args, **_kwargs: _result())

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
        # One aggregate shape fills the requested top-N, so the grid stops at
        # the end of its group before the P/D half runs.
        top_n=1,
        search_budget_seconds=60,
        fabric="ib-400g",
    ))

    assert result.search_summary["modes_searched"] == ["agg"]
    assert all(config["mode"] == "agg" for config in result.configs)
    assert all(
        any(
            "stopped before evaluating a P/D candidate" in warning
            for warning in config["warnings"]
        )
        for config in result.configs
    )


def test_recommend_aggregated_mode_skips_pd_even_when_a_fabric_is_selected(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 4}},
    )

    def mock_run(_runtime, run_request, *, timeout_seconds):
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
        mode="agg",
    ))

    assert result.search_summary["modes_searched"] == ["agg"]
    assert all(config["mode"] == "agg" for config in result.configs)
    assert any(
        "Aggregated serving was requested" in warning
        for warning in result.configs[0]["warnings"]
    )


def test_recommend_validates_a_named_fabric_even_when_serving_is_aggregated(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 4}},
    )

    # An aggregated search does not use the fabric, but the caller still named a
    # topology assumption, so an unknown one fails instead of being ignored.
    with pytest.raises(BlisUnsupportedRequest, match="cannot price network fabric"):
        blis_service.recommend(blis_service.RecommendInput(
            model_path="Qwen/Qwen3-32B", system="h100_sxm", backend="vllm", backend_version=None,
            isl=2048, osl=128, concurrency=8, request_rate=None, ttft=1000, tpot=30,
            request_latency=None, prefix=0, min_candidate_gpus=1, max_candidate_gpus=8,
            top_n=20, search_budget_seconds=60, fabric="bogus-fabric", mode="agg",
        ))


def test_recommend_searches_pd_when_disaggregated_is_named(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 4}},
    )
    monkeypatch.setattr(blis_service, "run_blis", lambda *_args, **_kwargs: _result())

    result = blis_service.recommend(blis_service.RecommendInput(
        model_path="Qwen/Qwen3-32B", system="h100_sxm", backend="vllm", backend_version=None,
        isl=2048, osl=128, concurrency=8, request_rate=None, ttft=1000, tpot=30,
        request_latency=None, prefix=0, min_candidate_gpus=1, max_candidate_gpus=8,
        top_n=20, search_budget_seconds=60, fabric="ib-400g", mode="disagg",
    ))

    # An explicit P/D choice must match the search an omitted mode gets.
    assert result.search_summary["modes_searched"] == ["agg", "disagg"]
    assert any(config["mode"] == "disagg" for config in result.configs)


def test_recommend_demand_stays_aggregated_with_a_named_fabric(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 4}},
    )
    monkeypatch.setattr(
        blis_service,
        "run_blis",
        lambda _runtime, run_request, *, timeout_seconds: _demand_replay(
            run_request.num_requests, 12.0
        ),
    )
    request = _demand_recommend_input()
    result = blis_service.recommend(blis_service.RecommendInput(
        **{**request.__dict__, "fabric": "ib-400g", "mode": "disagg"}
    ))

    # A rate-driven replay cannot attach a P/D transfer fabric, so naming one
    # keeps the run aggregate-only rather than implying a topology it did not
    # simulate.
    assert result.search_summary["modes_searched"] == ["agg"]
    assert all(config["mode"] == "agg" for config in result.configs)


def test_run_command_uses_rate_and_omits_concurrency():
    runtime = BlisRuntime("/opt/venv/bin/blis")
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
    runtime = BlisRuntime("/opt/venv/bin/blis")
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
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
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
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
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
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    request = _demand_recommend_input()
    with pytest.raises(blis_service.BlisUnsupportedRequest, match="does not apply latency targets"):
        blis_service.recommend(blis_service.RecommendInput(
            **{**request.__dict__, "ttft": 1000.0}
        ))


def test_recommend_skips_pd_candidates_without_a_fabric(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
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
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
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


def test_predict_maps_requests_per_load_unit_to_the_simulation_slice(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 8}},
    )
    calls = []

    def mock_run(_runtime, run_request, *, timeout_seconds):
        calls.append(run_request)
        return _result()

    monkeypatch.setattr(blis_service, "run_blis", mock_run)

    def predict(**overrides):
        return blis_service.estimate(blis_service.EstimateInput(
            model_path="Qwen/Qwen3-32B",
            system="h100_sxm",
            backend="vllm",
            backend_version=None,
            isl=8192,
            osl=1024,
            concurrency=8,
            tp=1,
            pp=1,
            dp=1,
            **overrides,
        ))

    predict(requests_per_load_unit=10)
    predict()

    assert calls[0].num_requests == 80
    # Omitting the field stops after one total request, matching /predict.
    assert calls[1].num_requests == 1


def test_prediction_requests_rejects_an_oversized_slice():
    with pytest.raises(BlisUnsupportedRequest, match="supported limit"):
        blis_service._prediction_requests(8, 1e308)


def test_normalize_result_maps_the_engine_kv_pool():
    metrics = normalize_result({
        "schema_version": 4,
        "metrics": {},
        "reports": {
            "engine": {"cluster": {"kv_capacity_tokens_per_replica": 123456}},
        },
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
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
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
    # 10,000 tokens hold 4 whole 2,176-token sequences per replica; two
    # replicas hold 8, not the 9 the pooled token total would suggest.
    assert capacity.capacity_sequences == 8
    assert capacity.fits is True
    assert capacity.minimum_gpus == 1
    assert capacity.replicas == 2
    assert capacity.per_rank_tokens == 10_000
    assert capacity.performance_available is False
    assert result.configs[0]["total_gpus_needed"] == 2
    assert result.configs[0]["replicas_needed"] == 2
    # num_total_gpus is one replica's scalable unit, not the cluster total.
    assert result.configs[0]["num_total_gpus"] == 1
    assert result.configs[0]["mode"] == "agg"
    assert "does not simulate the target concurrency" in " ".join(
        result.configs[0]["warnings"]
    )
    assert result.search_summary["strategy"] == "aggregate_tp_capacity"


def test_recommend_capacity_credits_the_shared_prefix_once_per_replica(
    monkeypatch, catalog
):
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(blis_service, "resolve_runtime", lambda: runtime)
    monkeypatch.setattr(
        blis_service,
        "load_system_spec",
        lambda _system: {"node": {"num_gpus_per_node": 4}},
    )
    monkeypatch.setattr(
        blis_service,
        "run_blis",
        lambda _runtime, _run_request, *, timeout_seconds: _result(kv_capacity=10_000),
    )

    result = blis_service.recommend(_capacity_input(
        max_candidate_gpus=8,
        concurrency=80,
        prefix=2_000,
    ))

    capacity = result.capacity
    assert capacity is not None
    assert capacity.shared_prefix_tokens == 2_000
    # 8,000 free tokens hold 45 sequences of the 176-token suffix per replica,
    # so two replicas reach 90. Charging the full context would hold only 4 per
    # replica, which cannot reach 80 inside this GPU window.
    assert capacity.capacity_sequences == 90
    assert capacity.fits is True
    assert capacity.replicas == 2
    assert result.configs[0]["total_gpus_needed"] == 2


def test_recommend_capacity_rejects_a_request_rate_before_running(monkeypatch, catalog):
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)

    with pytest.raises(BlisUnsupportedRequest, match="does not accept target_request_rate"):
        blis_service.recommend(_capacity_input(request_rate=10.0, concurrency=None))


def test_recommend_capacity_reports_the_shortfall_when_the_window_is_too_small(
    monkeypatch, catalog
):
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
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
    # Two replicas of 5,000 tokens hold 2 whole sequences each.
    assert capacity.capacity_sequences == 4
    assert capacity.shortfall_sequences == 12
    assert capacity.shortfall_tokens == 12 * 2176
    assert result.configs[0]["total_gpus_needed"] == 2
    assert result.configs[0]["num_total_gpus"] == 1


def test_recommend_capacity_sizes_replicas_beyond_one_node(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
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
    assert result.configs[0]["num_total_gpus"] == 1
    assert result.search_summary["effective_max_gpus"] == 8
    assert result.search_summary["hardware_max_gpus"] == 2


def test_recommend_capacity_fails_when_blis_reports_no_kv_pool(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
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
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
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
    # Higher load wins both axes, so the c8 shapes dominate the c1 shapes.
    assert set(result.views["pareto_front"]) == {"agg-tp1-c8", "agg-tp2-c8"}
    assert len(result.configs) == 4
    assert all(config["simulator"] == "blis" for config in result.configs)


def test_recommend_pareto_marks_sla_violations_infeasible(monkeypatch, catalog):
    runtime = BlisRuntime("/opt/blis")
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)
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
    monkeypatch.setattr(blis_service, "load_catalog", lambda *_args, **_kwargs: catalog)

    with pytest.raises(BlisUnsupportedRequest, match="drive their own load"):
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

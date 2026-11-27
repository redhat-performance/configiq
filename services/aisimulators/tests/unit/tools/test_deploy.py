# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deployment endpoints and the artifact render gate.

The Dynamo target renders from the SDK generator alone, so those tests always
run. The llm-d target also builds the pinned Kustomize overlay, so its tests are
skipped when the vendored tree or kustomize is absent from the environment; the
CI job and the image both provide them.
"""

from __future__ import annotations

import dataclasses
import io
import json
import zipfile
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml
from fastapi.testclient import TestClient

# The `render` submodule, not a re-export: the tests patch its catalog binding
# and call its entry point directly.
import tools.api_service.deploy.bundle as bundle_module
import tools.api_service.deploy.render as render_module
from tools.api_service import app as app_module
from tools.api_service.deploy import catalog
from tools.api_service.deploy.engine import (
    DeployUnsupportedRequest,
    candidate_config,
)
from tools.api_service.engine_models import EngineSection

client = TestClient(app_module.app)


def _field(value: Any, source: str = "evaluated") -> dict[str, Any]:
    return {"value": value, "source": source}


def _role(**overrides: Any) -> dict[str, Any]:
    role = {
        "replicas": _field(2),
        "tensor_parallel_size": _field(2),
        "pipeline_parallel_size": _field(1),
        "data_parallel_size": _field(1),
        "context_parallel_size": _field(1),
        "decode_context_parallel_size": _field(1),
        "moe_tensor_parallel_size": _field(1),
        "moe_expert_parallel_size": _field(1),
        "max_model_len": _field(5056),
        "max_num_seqs": _field(128),
        "max_num_batched_tokens": _field(8192),
        "block_size": _field(64),
        "gpu_memory_utilization": _field(0.9),
        "enable_prefix_caching": _field(False),
        "enable_chunked_prefill": _field(None, "not-modeled"),
        "kv_cache_dtype": _field("auto", "backend-default"),
        "quantization": _field(None, "backend-default"),
    }
    role.update(overrides)
    return role


def _engine(mode: str, version: str, **overrides: Any) -> dict[str, Any]:
    engine: dict[str, Any] = {
        "model": "Qwen/Qwen3-32B",
        "backend": "vllm",
        "backend_version": version,
        "mode": mode,
    }
    if mode == "agg":
        engine["agg"] = _role()
    else:
        engine["prefill"] = _role(
            replicas=_field(1),
            tensor_parallel_size=_field(1),
            max_num_seqs=_field(32),
            gpu_memory_utilization=_field(0.85),
        )
        engine["decode"] = _role(replicas=_field(3))
    engine.update(overrides)
    return engine


def _body(target: str, engine: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    body = {"target": target, "engine": engine, "system": "h200_sxm", "isl": 4000, "osl": 1000}
    body.update(overrides)
    return body


DGD_AGG = _body("dynamo-dgd", _engine("agg", "0.24.0"))


def _section(engine: dict[str, Any]) -> EngineSection:
    return EngineSection.model_validate(engine)


def _files(payload: dict[str, Any]) -> dict[str, str]:
    return {item["path"]: item["contents"] for item in payload["files"]}


def _kustomize_ready() -> bool:
    return catalog.llm_d_vendor_dir().is_dir() and catalog.kustomize_command() is not None


requires_kustomize = pytest.mark.skipif(
    not _kustomize_ready(),
    reason="the pinned llm-d tree or kustomize is not available in this environment",
)


class TestDeployTargets:
    def test_both_targets_are_published_with_their_pins(self):
        response = client.get("/deploy/targets")

        assert response.status_code == 200
        targets = {item["id"]: item for item in response.json()["targets"]}
        assert set(targets) == {"dynamo-dgd", "llm-d"}
        assert targets["dynamo-dgd"]["pins"]["dynamo"] == catalog.DYNAMO_VERSION
        assert targets["llm-d"]["pins"]["llm-d"].startswith(catalog.LLM_D_VERSION)
        assert all(item["artifact_schema_version"] == "1" for item in targets.values())
        assert all(item["experimental"] is True for item in targets.values())

    def test_a_missing_source_tree_reports_the_target_unavailable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(catalog, "llm_d_vendor_dir", lambda: tmp_path / "absent")

        targets = {item["id"]: item for item in client.get("/deploy/targets").json()["targets"]}

        assert targets["llm-d"]["available"] is False
        assert "pinned llm-d source tree" in targets["llm-d"]["unavailable_reason"]
        assert targets["dynamo-dgd"]["available"] is True


class TestDeployValidation:
    def test_an_unknown_target_is_a_404_naming_the_choices(self):
        response = client.post("/deploy", json=_body("nope", _engine("agg", "0.31.0")))

        assert response.status_code == 404
        assert "dynamo-dgd" in response.json()["detail"]

    def test_an_unavailable_target_is_a_503_not_a_broken_bundle(self, monkeypatch, tmp_path):
        monkeypatch.setattr(catalog, "llm_d_vendor_dir", lambda: tmp_path / "absent")
        monkeypatch.setattr(render_module, "llm_d_vendor_dir", lambda: tmp_path / "absent")

        response = client.post("/deploy", json=_body("llm-d", _engine("agg", "0.31.0")))

        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]

    def test_an_unstated_limit_is_refused_rather_than_defaulted(self):
        engine = _engine("agg", "0.24.0")
        engine["agg"]["max_num_seqs"] = _field(None, "not-modeled")

        response = client.post("/deploy", json=_body("dynamo-dgd", engine))

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "agg.max_num_seqs" in detail
        assert "Re-run the prediction" in detail

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("enable_chunked_prefill", True),
            ("kv_cache_dtype", "fp8"),
            ("quantization", "fp8"),
        ],
    )
    def test_a_limit_the_generator_cannot_render_is_refused(self, field_name, value):
        engine = _engine("agg", "0.24.0")
        engine["agg"][field_name] = _field(value, "evaluated")

        response = client.post("/deploy", json=_body("dynamo-dgd", engine))

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert field_name in detail
        assert "cannot render" in detail

    def test_backend_defaults_and_not_modeled_limits_still_render(self):
        response = client.post("/deploy", json=DGD_AGG)

        assert response.status_code == 200, response.text
        assert DGD_AGG["engine"]["agg"]["kv_cache_dtype"]["source"] == "backend-default"

    @pytest.mark.parametrize(
        "field",
        [
            {"namespace": "a$(id)"},
            {"name_prefix": 'x";reboot'},
            {"image": "img;rm -rf /"},
        ],
    )
    def test_a_script_unsafe_input_is_rejected(self, field):
        response = client.post("/deploy", json=_body("dynamo-dgd", _engine("agg", "0.24.0"), **field))

        assert response.status_code == 422

    def test_dns_and_image_shaped_inputs_still_render(self):
        response = client.post(
            "/deploy",
            json=_body(
                "dynamo-dgd",
                _engine("agg", "0.24.0"),
                namespace="team-a",
                name_prefix="abc",
                image="nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0",
            ),
        )

        assert response.status_code == 200, response.text

    def test_a_different_engine_release_is_refused_with_the_way_out(self, monkeypatch, tmp_path):
        # This test only exercises validation, so make the target available even
        # where the vendored tree or kustomize is absent.
        monkeypatch.setattr(catalog, "llm_d_vendor_dir", lambda: tmp_path)
        monkeypatch.setattr(catalog, "kustomize_command", lambda: ["kustomize", "build"])

        response = client.post("/deploy", json=_body("llm-d", _engine("agg", "0.24.0")))

        # Refused before the render gate, so it does not depend on the vendor tree.
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "pins vllm 0.31.0" in detail
        assert "allow_engine_version_mismatch=true" in detail

    def test_a_mismatched_image_is_a_warning_when_the_caller_accepts_it(self):
        response = client.post(
            "/deploy",
            json=_body(
                "dynamo-dgd",
                _engine("agg", "0.31.0"),
                image="nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0",
                allow_engine_version_mismatch=True,
            ),
        )

        assert response.status_code == 200
        payload = response.json()
        assert any("replaces the target's pinned" in item for item in payload["warnings"])
        assert "0.31.0" in _files(payload)["PROVENANCE.md"]


class TestDynamoDgdArtifact:
    def test_an_aggregated_artifact_renders_and_passes_every_check(self):
        response = client.post("/deploy", json=DGD_AGG)

        assert response.status_code == 200, response.text
        payload = response.json()
        assert [check["id"] for check in payload["checks"]] == [
            "kustomize-build",
            "resources",
            "engine-limits",
            "images",
            "placeholders",
        ]
        assert all(check["status"] != "failed" for check in payload["checks"])

        files = _files(payload)
        assert set(files) >= {
            "README.md",
            "PROVENANCE.md",
            "install.sh",
            "preflight.sh",
            "verify.sh",
            "k8s_deploy.yaml",
        }

        documents = [doc for doc in yaml.safe_load_all(files["k8s_deploy.yaml"]) if doc]
        assert [doc["kind"] for doc in documents] == ["DynamoGraphDeployment"]
        worker = documents[0]["spec"]["services"]["VllmWorker"]
        assert worker["replicas"] == 2
        args = worker["extraPodSpec"]["mainContainer"]["args"]
        assert args[args.index("--max-model-len") + 1] == "5056"
        assert args[args.index("--max-num-seqs") + 1] == "128"
        assert args[args.index("--tensor-parallel-size") + 1] == "2"

    def test_the_bundle_records_the_image_and_the_pins_it_used(self):
        files = _files(client.post("/deploy", json=DGD_AGG).json())

        provenance = files["PROVENANCE.md"]
        assert "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0" in provenance
        assert catalog.DYNAMO_VERSION in provenance
        assert "kustomize-build | skipped" in provenance

    def test_a_disaggregated_artifact_renders_both_pools(self):
        response = client.post("/deploy", json=_body("dynamo-dgd", _engine("disagg", "0.24.0")))

        assert response.status_code == 200, response.text
        documents = [
            doc for doc in yaml.safe_load_all(_files(response.json())["k8s_deploy.yaml"]) if doc
        ]
        services = documents[0]["spec"]["services"]
        prefill = services["VllmPrefillWorker"]["extraPodSpec"]["mainContainer"]["args"]
        decode = services["VllmDecodeWorker"]["extraPodSpec"]["mainContainer"]["args"]
        assert prefill[prefill.index("--tensor-parallel-size") + 1] == "1"
        assert prefill[prefill.index("--max-num-seqs") + 1] == "32"
        assert decode[decode.index("--tensor-parallel-size") + 1] == "2"
        assert decode[decode.index("--max-num-seqs") + 1] == "128"
        assert services["VllmPrefillWorker"]["replicas"] == 1
        assert services["VllmDecodeWorker"]["replicas"] == 3

    def test_the_zip_carries_the_same_files_with_runnable_scripts(self):
        response = client.post("/deploy?format=zip", json=DGD_AGG)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert "attachment" in response.headers["content-disposition"]
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        assert set(archive.namelist()) >= {"k8s_deploy.yaml", "install.sh", "PROVENANCE.md"}
        modes = {info.filename: (info.external_attr >> 16) & 0o777 for info in archive.infolist()}
        assert modes["install.sh"] == 0o755
        assert modes["k8s_deploy.yaml"] == 0o644
        manifest = archive.read("k8s_deploy.yaml").decode()
        assert "kind: DynamoGraphDeployment" in manifest

    def test_the_readme_states_the_layout_and_the_gpu_request(self):
        files = _files(client.post("/deploy", json=DGD_AGG).json())

        readme = files["README.md"]
        assert "k8s_deploy.yaml" in readme
        assert "accelerator(s)" in readme
        assert "DynamoGraphDeployment" in readme

    def test_install_creates_the_namespace_before_preflight(self):
        install = _files(client.post("/deploy", json=DGD_AGG).json())["install.sh"]

        assert install.index("create namespace") < install.index("./preflight.sh")

    def test_the_uninstall_command_matches_the_single_manifest(self):
        readme = _files(client.post("/deploy", json=DGD_AGG).json())["README.md"]

        assert "kubectl delete -n configiq -f k8s_deploy.yaml" in readme
        assert "-k ." not in readme


class TestDeployableSubset:
    """The deployable subset is derived from the overlay the generator rendered.

    These run without the vendored llm-d tree or kustomize, so they are the
    guard that still fires where the render gate is skipped.
    """

    OVERLAY: ClassVar[dict[str, Any]] = {
        "apiVersion": "kustomize.config.k8s.io/v1beta1",
        "kind": "Kustomization",
        "resources": ["llm-d/guides/pd-disaggregation/modelserver/gpu/vllm/base"],
        "configurations": ["name-reference.yaml"],
        "patches": [{"path": "patch-decode.yaml"}, {"path": "patch-prefill.yaml"}],
    }

    def _rendered(self, **overrides: str) -> dict[str, str]:
        rendered = {
            "kustomization.yaml": yaml.safe_dump(self.OVERLAY),
            "name-reference.yaml": "nameReference:\n",
            "patch-decode.yaml": "patch: decode\n",
            "patch-prefill.yaml": "patch: prefill\n",
            "cli_args_decode": "--max-num-seqs 8\n",
            "run_0.sh": "#!/bin/sh\n",
            "sflow.yaml": "scenario: true\n",
        }
        rendered.update(overrides)
        return rendered

    def test_every_file_the_overlay_references_is_kept(self):
        kept = render_module._deployable_subset(self._rendered(), catalog.TARGETS["llm-d"])

        assert set(kept) == {
            "kustomization.yaml",
            "name-reference.yaml",
            "patch-decode.yaml",
            "patch-prefill.yaml",
        }

    def test_the_measurement_files_are_not_part_of_the_bundle(self):
        rendered = self._rendered()

        kept = render_module._deployable_subset(rendered, catalog.TARGETS["llm-d"])

        assert set(rendered) - set(kept) == {"cli_args_decode", "run_0.sh", "sflow.yaml"}

    def test_a_resource_is_kept_only_when_the_render_produced_it(self):
        overlay = {**self.OVERLAY, "resources": ["llm-d/base", "extra.yaml"]}
        rendered = self._rendered(
            **{"kustomization.yaml": yaml.safe_dump(overlay), "extra.yaml": "kind: ConfigMap\n"}
        )

        kept = render_module._deployable_subset(rendered, catalog.TARGETS["llm-d"])

        assert "extra.yaml" in kept
        assert "llm-d/base" not in kept

    def test_a_reference_the_generator_did_not_produce_is_an_error(self):
        rendered = self._rendered()
        del rendered["name-reference.yaml"]

        with pytest.raises(render_module.DeployRenderError, match="name-reference.yaml"):
            render_module._deployable_subset(rendered, catalog.TARGETS["llm-d"])

    def test_a_kustomization_that_is_not_a_mapping_is_an_error(self):
        rendered = self._rendered(**{"kustomization.yaml": "- not a mapping\n"})

        with pytest.raises(render_module.DeployRenderError, match="not a Kustomization"):
            render_module._deployable_subset(rendered, catalog.TARGETS["llm-d"])

    def test_a_missing_kustomization_is_an_error(self):
        rendered = self._rendered()
        del rendered["kustomization.yaml"]

        with pytest.raises(render_module.DeployRenderError, match="no deployable overlay"):
            render_module._deployable_subset(rendered, catalog.TARGETS["llm-d"])

    def test_an_unparsable_kustomization_is_an_error(self):
        rendered = self._rendered(**{"kustomization.yaml": "resources: [\n"})

        with pytest.raises(render_module.DeployRenderError, match="unparsable"):
            render_module._deployable_subset(rendered, catalog.TARGETS["llm-d"])

    def test_a_single_manifest_target_keeps_only_its_manifest(self):
        rendered = {"k8s_deploy.yaml": "kind: DynamoGraphDeployment\n", "sflow.yaml": "x\n"}

        kept = render_module._deployable_subset(rendered, catalog.TARGETS["dynamo-dgd"])

        assert set(kept) == {"k8s_deploy.yaml"}

    def test_the_layout_lists_the_name_reference_a_disaggregatedset_needs(self):
        layout = bundle_module._layout(
            _section(_engine("disagg", "0.31.0")), catalog.TARGETS["llm-d"]
        )

        assert "`name-reference.yaml`" in layout

    def test_the_layout_has_no_name_reference_for_a_deployment(self):
        layout = bundle_module._layout(_section(_engine("agg", "0.31.0")), catalog.TARGETS["llm-d"])

        assert "`name-reference.yaml`" not in layout


@requires_kustomize
class TestLlmdArtifact:
    def test_an_aggregated_overlay_renders_a_deployment(self):
        response = client.post("/deploy", json=_body("llm-d", _engine("agg", "0.31.0")))

        assert response.status_code == 200, response.text
        payload = response.json()
        assert all(check["status"] != "failed" for check in payload["checks"])

        files = _files(payload)
        assert "llm-d/guides/optimized-baseline/modelserver/gpu/vllm/base/kustomization.yaml" in files
        assert "llm-d/guides/pd-disaggregation/modelserver/gpu/vllm/base/kustomization.yaml" not in files
        assert "name-reference.yaml" not in files
        overlay = yaml.safe_load(files["kustomization.yaml"])
        assert overlay["resources"] == ["llm-d/guides/optimized-baseline/modelserver/gpu/vllm/base"]
        assert overlay["namespace"] == "configiq"
        assert "images" not in overlay

    def test_a_disaggregated_overlay_pins_the_sidecar_and_renders_a_disaggregatedset(self):
        response = client.post("/deploy", json=_body("llm-d", _engine("disagg", "0.31.0")))

        assert response.status_code == 200, response.text
        payload = response.json()
        assert all(check["status"] != "failed" for check in payload["checks"])

        files = _files(payload)
        assert "name-reference.yaml" in files
        overlay = yaml.safe_load(files["kustomization.yaml"])
        # Every bundle-root file the overlay names has to ship with it: the
        # gate builds this same tree, and kustomize fails on a missing one.
        for reference in overlay.get("configurations") or []:
            assert reference in files
        for patch in overlay.get("patches") or []:
            if isinstance(patch, dict) and patch.get("path"):
                assert patch["path"] in files
        assert overlay["images"] == [
            {
                "name": "ghcr.io/llm-d/llm-d-router-disagg-sidecar",
                "newName": "ghcr.io/llm-d/llm-d-router-disagg-sidecar",
                "newTag": "v1.0.0",
            }
        ]
        # The gate that produced these files rendered a DisaggregatedSet.
        kinds = {check["id"]: check for check in payload["checks"]}
        assert kinds["resources"]["status"] == "passed"
        assert "DisaggregatedSet" in kinds["resources"]["detail"]

    def test_the_bundle_carries_the_router_values_it_installs(self):
        payload = client.post("/deploy", json=_body("llm-d", _engine("disagg", "0.31.0"))).json()
        files = _files(payload)

        install = files["install.sh"]
        assert catalog.LLM_D_ROUTER_VALUES_BY_MODE["disagg"] in install
        assert catalog.LLM_D_ROUTER_BASE_VALUES in install
        assert catalog.ROUTER_CHART_VERSION in install
        assert "llm-d-hf-token" in files["preflight.sh"]
        for reference in (
            f"llm-d/{catalog.LLM_D_ROUTER_BASE_VALUES}",
            f"llm-d/{catalog.LLM_D_ROUTER_VALUES_BY_MODE['disagg']}",
        ):
            assert reference in files
        checks = {check["id"]: check for check in payload["checks"]}
        assert checks["install-references"]["status"] == "passed"

    def test_an_aggregated_bundle_carries_its_router_values(self):
        payload = client.post("/deploy", json=_body("llm-d", _engine("agg", "0.31.0"))).json()
        files = _files(payload)

        assert f"llm-d/{catalog.LLM_D_ROUTER_VALUES_BY_MODE['agg']}" in files
        assert f"llm-d/{catalog.LLM_D_ROUTER_BASE_VALUES}" in files

    def test_install_runs_preflight_after_prerequisites(self):
        install = _files(client.post("/deploy", json=_body("llm-d", _engine("agg", "0.31.0"))).json())[
            "install.sh"
        ]

        assert install.index("Applying Inference Extension") < install.index("./preflight.sh")

    def test_preflight_requires_only_the_crds_the_release_ships(self):
        preflight = _files(client.post("/deploy", json=_body("llm-d", _engine("agg", "0.31.0"))).json())[
            "preflight.sh"
        ]
        required = {
            line.split("require_crd ", 1)[1].strip().strip("'\"")
            for line in preflight.splitlines()
            if line.startswith("require_crd ")
        }

        assert required == {"inferencepools.inference.networking.k8s.io"}


class TestSuppliedModelConfig:
    def test_a_supplied_config_avoids_resolving_the_model_id(self):
        config = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "intermediate_size": 25600,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
        }
        engine = _engine("agg", "0.24.0", model="configiq/not-resolvable-model")

        response = client.post(
            "/deploy",
            json=_body("dynamo-dgd", engine, model_config_data=config),
        )

        assert response.status_code == 200, response.text
        documents = [
            doc for doc in yaml.safe_load_all(_files(response.json())["k8s_deploy.yaml"]) if doc
        ]
        args = documents[0]["spec"]["services"]["VllmWorker"]["extraPodSpec"]["mainContainer"]["args"]
        assert args[args.index("--model") + 1] == "configiq/not-resolvable-model"


class TestEngineLowering:
    def test_the_candidate_config_copies_every_evaluated_limit(self):
        engine = _section(_engine("disagg", "0.31.0"))

        config = candidate_config(engine, system="h200_sxm", concurrency=96)

        assert config["deployment_mode"] == "disagg"
        assert config["model_name"] == "Qwen/Qwen3-32B"
        assert config["backend_version"] == "0.31.0"
        assert config["hardware_sku"] == "h200_sxm"
        assert config["prefill_tp"] == 1
        assert config["decode_tp"] == 2
        assert config["prefill_replicas"] == 1
        assert config["decode_replicas"] == 3
        assert config["prefill_max_num_seqs"] == 32
        assert config["decode_max_num_batched_tokens"] == 8192
        assert config["decode_gpu_memory_utilization"] == 0.9
        assert config["decode_context_length"] == 5056
        assert config["concurrency"] == 96

    def test_an_omitted_concurrency_uses_the_run_own_batch_ceiling(self):
        engine = _section(_engine("agg", "0.24.0"))

        config = candidate_config(engine, system="h200_sxm", concurrency=None)

        assert config["concurrency"] == 128 * 2

    def test_a_missing_backend_version_stops_the_lowering(self):
        engine = _section(_engine("agg", "0.31.0", backend_version=None))

        with pytest.raises(DeployUnsupportedRequest, match="backend version"):
            candidate_config(engine, system="h200_sxm", concurrency=1)

    def test_a_mode_the_target_does_not_render_is_refused(self):
        target = dataclasses.replace(catalog.TARGETS["dynamo-dgd"], modes=("agg",))

        with pytest.raises(DeployUnsupportedRequest, match="does not render 'disagg'"):
            render_module.render(
                app_module.DeployRequest.model_validate(
                    _body("dynamo-dgd", _engine("disagg", "0.24.0"))
                ),
                target,
            )


class TestEngineLimitGate:
    def _role_and_args(self):
        role = _section(_engine("agg", "0.24.0")).agg
        args = ["--model", "Qwen/Qwen3-32B"]
        for flag, field_name in render_module._LIMIT_FLAGS:
            args.extend([flag, str(getattr(role, field_name).value)])
        if role.enable_prefix_caching.value is False:
            args.append("--no-enable-prefix-caching")
        return role, args

    def _failures(self, role, args):
        failures, _ = render_module._assert_engine_limits(
            [("agg", role.replicas.value, [{"name": "modelserver", "args": args}])],
            _section(_engine("agg", "0.24.0")),
        )
        return failures

    def test_every_rendered_limit_is_asserted(self):
        role, args = self._role_and_args()

        failures, assertions = render_module._assert_engine_limits(
            [("agg", role.replicas.value, [{"name": "modelserver", "args": args}])],
            _section(_engine("agg", "0.24.0")),
        )

        assert failures == []
        assert assertions == len(render_module._LIMIT_FLAGS) + len(
            render_module._NEGATED_FLAGS
        ) + 1

    @pytest.mark.parametrize("flag", ["--block-size", "--gpu-memory-utilization"])
    def test_a_tampered_value_fails(self, flag):
        role, args = self._role_and_args()
        args[args.index(flag) + 1] = "999"

        assert any(flag in failure for failure in self._failures(role, args))

    def test_prefix_caching_presence_is_asserted(self):
        role, args = self._role_and_args()
        args.remove("--no-enable-prefix-caching")

        assert any(
            "--no-enable-prefix-caching" in failure for failure in self._failures(role, args)
        )


class TestKustomizeCommand:
    def test_an_override_binary_gets_the_build_subcommand(self, monkeypatch):
        monkeypatch.setenv("AISIMULATORS_KUSTOMIZE_BIN", "/opt/kustomize")
        monkeypatch.setattr(
            catalog.shutil,
            "which",
            lambda name: "/opt/kustomize" if name == "/opt/kustomize" else None,
        )

        assert catalog.kustomize_command() == ["/opt/kustomize", "build"]

    def test_a_missing_override_binary_is_unavailable(self, monkeypatch):
        monkeypatch.setenv("AISIMULATORS_KUSTOMIZE_BIN", "/opt/absent")
        monkeypatch.setattr(catalog.shutil, "which", lambda name: None)

        assert catalog.kustomize_command() is None


class TestVendorPins:
    def test_the_fetch_script_matches_the_catalog_pins(self):
        service_root = Path(__file__).resolve().parents[3]
        text = (service_root / "scripts" / "fetch-llm-d-vendor.sh").read_text()

        assert catalog.LLM_D_REVISION in text
        for path in catalog.LLM_D_VENDOR_PATHS:
            assert path in text

    def test_every_pinned_subtree_exists_in_a_vendored_tree(self):
        if not catalog.llm_d_vendor_dir().is_dir():
            pytest.skip("the pinned llm-d tree is not vendored in this environment")
        for path in catalog.LLM_D_VENDOR_PATHS:
            assert (catalog.llm_d_vendor_dir() / path).is_dir()

    def test_the_engine_fields_serialize_with_their_source(self):
        engine = _section(_engine("agg", "0.31.0"))

        payload = json.loads(engine.model_dump_json())

        assert payload["agg"]["max_num_seqs"] == {"value": 128, "source": "evaluated"}
        assert payload["agg"]["enable_chunked_prefill"] == {"value": None, "source": "not-modeled"}

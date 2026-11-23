"""Unit tests for recorded replay artifacts.

The point of a record is that a reader can re-run the same work outside
ConfigIQ, so these tests assert the artifacts themselves: the command, the
input file it names, and the pins the record reports. They avoid the SDK where
they can; the BLIS capture test stubs the subprocess.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

from tools.api_service import app as app_module
from tools.api_service.blis import service as blis_service
from tools.api_service.blis.manifest import ModelResolution
from tools.api_service.blis.runtime import BlisRuntime
from tools.api_service.blis.scenario import ScenarioRequest
from tools.api_service.replay import (
    ReplayFile,
    ReplayInvocation,
    ReplayRecord,
    ReplayRecorder,
    artifacts,
    capture_into,
)
from tools.api_service.replay import capture as capture_module

# ── artifacts ────────────────────────────────────────────────────────────────


def test_a_supplied_model_config_is_shipped_and_named_relatively():
    model = artifacts.model_input(
        "/tmp/does-not-exist",
        {"architectures": ["Qwen2ForCausalLM"], "hidden_size": 8},
    )

    assert model.reference == "./model"
    assert [item.path for item in model.files] == ["model/config.json"]

    invocation = artifacts.prediction_invocation(
        {"engine": {"model": "/tmp/does-not-exist", "workers": {}}},
        id="prediction",
        label="Prediction",
        model=model,
    )

    assert invocation.command == "aisimulate predict -c prediction.yaml --format json"
    document = yaml.safe_load(invocation.files[0].contents)
    assert document["engine"]["model"] == "./model"
    # The command is only reproducible if the model directory ships beside it.
    assert json.loads(invocation.files[1].contents)["hidden_size"] == 8


def test_a_model_reference_without_supplied_config_is_left_alone():
    model = artifacts.model_input("Qwen/Qwen3-32B", None)

    invocation = artifacts.prediction_invocation(
        {"engine": {"model": "Qwen/Qwen3-32B"}},
        id="prediction",
        label="Prediction",
        model=model,
    )

    assert model.files == ()
    assert yaml.safe_load(invocation.files[0].contents)["engine"]["model"] == "Qwen/Qwen3-32B"


def test_memory_has_a_snippet_because_the_sdk_has_no_cli_verb():
    invocation = artifacts.memory_snippet_invocation(
        id="memory-estimate",
        label="Memory estimate",
        kwargs={"model_path": "./model", "tp_size": 2, "backend": "vllm"},
        model=artifacts.model_input("Qwen/Qwen3-32B", None),
    )

    assert invocation.command == "python memory-estimate.py"
    assert "estimate_kv_cache(" in invocation.files[0].contents
    assert "tp_size=2" in invocation.files[0].contents
    assert any("no CLI verb" in note for note in invocation.notes)


def test_the_script_skips_candidate_commands():
    invocations = [
        ReplayInvocation("search", "Search", "aisimulate recommend -c search.yaml", "in_process"),
        ReplayInvocation(
            "candidate", "Candidate c1", "aisimulate predict -c c1.yaml", "in_process",
            candidate_id="c1",
        ),
    ]

    script = artifacts.run_script(invocations, environment=("BLIS_BINARY",))

    assert script is not None
    assert "aisimulate recommend" in script
    assert "c1.yaml" not in script
    # The runtime paths are required, not defaulted: the record cannot know
    # where the reader installed them.
    assert "BLIS_BINARY" in script


def test_the_script_requires_the_runtime_paths_its_commands_use():
    invocation = artifacts.blis_run_invocation(
        id="run",
        label="Run",
        argv=[
            "/opt/blis/bin/blis",
            "run",
            "--scenarios",
            "/tmp/x",
            "--catalog",
            "/opt/blis/catalog",
            "--registry",
            "/opt/blis/registry",
        ],
        files={"scenario.yaml": "kind: Scenario\n"},
        binary="/opt/blis/bin/blis",
        catalog="/opt/blis/catalog",
        registry="/opt/blis/registry",
        scenario_directory="/tmp/x",
    )

    assert '"$BLIS_BINARY"' in invocation.command
    assert '"$BLIS_CATALOG"' in invocation.command
    assert '"$BLIS_REGISTRY"' in invocation.command
    assert "--scenarios ." in invocation.command
    assert "/tmp/x" not in invocation.command


# ── recorder ─────────────────────────────────────────────────────────────────


def _recorder(**overrides) -> ReplayRecorder:
    values = {
        "endpoint": "/predict",
        "simulator": "aisimulate",
        "execution": "in_process",
        "versions": {"aisimulate": "0.14.0"},
        "model_path": "Qwen/Qwen3-32B",
    }
    values.update(overrides)
    return ReplayRecorder(**values)


def test_no_record_when_nothing_was_executed():
    assert _recorder().build() is None


def test_rollback_drops_a_failed_attempt():
    recorder = _recorder()
    recorder.prediction({"engine": {}}, label="Prediction")
    mark = recorder.checkpoint()

    recorder.prediction({"engine": {"backend_version": "0.14.0"}}, label="Prediction")
    recorder.rollback(mark)

    record = recorder.build()
    assert record is not None
    assert [invocation.label for invocation in record.invocations] == ["Prediction"]
    # The id is reusable after a rollback, so the next attempt keeps its name.
    recorder.prediction({"engine": {"backend_version": "0.31.0"}}, label="Prediction")
    assert recorder.build().invocations[-1].id == "prediction-2"


def test_duplicate_labels_get_distinct_ids():
    recorder = _recorder()
    recorder.prediction({"engine": {"tp": 2}}, label="Candidate")
    recorder.prediction({"engine": {"tp": 4}}, label="Candidate")

    record = recorder.build()

    assert [invocation.id for invocation in record.invocations] == ["candidate", "candidate-2"]


def test_build_reports_the_versions_it_was_run_with():
    recorder = _recorder(versions={"aisimulate": "0.14.0", "blis": None})
    recorder.prediction({"engine": {}}, label="Prediction")

    record = recorder.build()

    assert record is not None
    assert record.versions["blis"] is None
    # Only the non-null pins belong in the script header.
    assert "aisimulate: 0.14.0" in record.script
    assert "blis:" not in record.script


def test_a_candidate_invocation_carries_the_id_its_row_uses():
    recorder = _recorder(endpoint="/recommend", simulator="blis", execution="subprocess")
    recorder.prediction(
        {"engine": {"tp": 2}},
        label="Candidate agg-tp2-c8",
        group="Candidates",
        candidate_id="agg-tp2-c8",
    )

    record = recorder.build()

    assert record is not None
    assert record.invocations[0].candidate_id == "agg-tp2-c8"
    # run.sh is the sweep; a candidate row copies its own command instead.
    assert record.script is None


def test_candidate_commands_are_dropped_before_the_answer(monkeypatch):
    recorder = _recorder()
    recorder.prediction({"engine": {}}, label="Prediction")
    # Calibrate against the answer's own record: the cap has to admit it
    # including the JSON structure, and still be far below 20 candidates.
    primary = recorder.build()
    monkeypatch.setattr(
        capture_module,
        "_max_record_bytes",
        lambda: primary.size_bytes() + 512,
    )
    for index in range(20):
        recorder.prediction(
            {"engine": {"tp": index}},
            label=f"Candidate {index}",
            candidate_id=f"c{index}",
        )

    record = recorder.build()

    assert record is not None
    assert [invocation.label for invocation in record.invocations] == ["Prediction"]
    assert any("20 per-candidate commands were omitted" in note for note in record.notes)


def test_a_primary_command_over_the_cap_is_not_returned(monkeypatch):
    monkeypatch.setattr(capture_module, "_max_record_bytes", lambda: 1)
    recorder = _recorder()
    recorder.prediction({"engine": {}}, label="Prediction")

    assert recorder.build() is None


# ── BLIS capture ─────────────────────────────────────────────────────────────


def _scenario(**overrides) -> ScenarioRequest:
    values = {
        "model": ModelResolution(catalog_name="qwen3-32b", hf_repo="Qwen/Qwen3-32B"),
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
    }
    values.update(overrides)
    return ScenarioRequest(**values)


def test_blis_scenario_files_are_captured_before_the_temp_directory_goes(monkeypatch):
    runtime = BlisRuntime(
        binary="/opt/blis/bin/blis",
        catalog="/opt/blis/catalog",
        registry="/opt/blis/registry",
    )
    resolved = SimpleNamespace(runtime=runtime)
    recorder = _recorder(
        endpoint="/recommend",
        simulator="blis",
        execution="subprocess",
        versions={"blis": "0.15.0"},
    )
    seen: dict[str, object] = {}

    def fake_run(runtime, request, *, timeout_seconds):
        # The capture must already hold the files; after this returns the
        # temporary directory is gone.
        seen["recorded"] = len(recorder.build().invocations)
        return {"completed_requests": 8}

    monkeypatch.setattr(blis_service, "run_blis", fake_run)

    with capture_into(recorder):
        blis_service._run_scenario(
            resolved,
            _scenario(),
            prefix=0,
            timeout_seconds=1,
            num_requests=8,
            candidate_id="agg-tp2-c8",
        )

    assert seen["recorded"] == 1
    invocation = recorder.build().invocations[0]
    assert invocation.candidate_id == "agg-tp2-c8"
    assert invocation.execution == "subprocess"
    assert any(path.endswith(".yaml") for path in (item.path for item in invocation.files))
    assert "$BLIS_BINARY" in invocation.command


def test_nothing_is_captured_without_a_recorder(monkeypatch):
    runtime = BlisRuntime(
        binary="/opt/blis/bin/blis",
        catalog="/opt/blis/catalog",
        registry="/opt/blis/registry",
    )
    calls: list[str] = []

    def fake_run(runtime, request, *, timeout_seconds):
        calls.append(request.scenario)
        return {"completed_requests": 8}

    monkeypatch.setattr(blis_service, "run_blis", fake_run)

    blis_service._run_scenario(
        SimpleNamespace(runtime=runtime),
        _scenario(),
        prefix=0,
        timeout_seconds=1,
        num_requests=8,
    )

    assert len(calls) == 1


# ── API contract ─────────────────────────────────────────────────────────────

VALID_ESTIMATE_BODY = {
    "model_path": "Qwen/Qwen3-32B",
    "system": "h200_sxm",
    "backend": "vllm",
    # An explicit pin keeps the run off the version-fallback ladder, so the
    # test exercises recording rather than SDK metadata.
    "backend_version": "0.31.0",
    "isl": 4000,
    "osl": 1000,
    "tp_size": 2,
    "batch_size": 128,
}


@pytest.fixture
def client():
    # No context manager: entering would run startup, which needs the SDK's
    # performance data. These tests only exercise the HTTP contract.
    yield TestClient(app_module.app)


def test_predict_returns_the_record_inline_only_when_debug_was_requested(
    client, monkeypatch
):
    def fake(req, include):
        # Debug requests record; the plain request must not.
        recorder = app_module.active_recorder()
        if recorder is not None:
            recorder.prediction({"engine": {"model": req.model_path}}, label="Prediction")
        return SimpleNamespace(
            summary={
                "ttft_ms": 100.0,
                "tpot_ms": 10.0,
                "output_throughput_tok_s": 1000.0,
            },
            provenance=None,
            serving_config=None,
            warnings=[],
        )

    monkeypatch.setattr(app_module, "_run_aisimulate_prediction", fake)

    quiet = client.post("/predict", json=VALID_ESTIMATE_BODY)
    assert quiet.status_code == 200
    assert quiet.json()["debug"] is None

    loud = client.post("/predict?include=debug", json=VALID_ESTIMATE_BODY)
    assert loud.status_code == 200
    debug = loud.json()["debug"]
    # The record travels in the response: with more than one worker process
    # there is no guarantee a later fetch reaches the worker that made it.
    assert debug["endpoint"] == "/predict"
    assert debug["simulator"] == "aisimulate"
    assert debug["execution"] == "in_process"
    assert "aisimulate" in debug["versions"]
    assert [invocation["label"] for invocation in debug["invocations"]] == ["Prediction"]
    assert debug["invocations"][0]["files"][0]["path"].endswith(".yaml")
    assert debug["script"].startswith("#!/usr/bin/env bash")


def test_memory_returns_the_snippet_inline(client, monkeypatch):
    def fake_estimate(**kwargs):
        return {
            "total_gpu_capacity_bytes": 1,
            "total_kv_size_bytes": 2,
            "kv_size_per_token_bytes": 3,
            "total_kv_size_tokens": 4,
            "source": "stub",
            "memory_breakdown": {},
        }

    monkeypatch.setattr(app_module, "estimate_kv_cache", fake_estimate)

    response = client.post(
        "/memory?include=debug",
        json={"model_path": "Qwen/Qwen3-32B", "system": "h200_sxm"},
    )

    assert response.status_code == 200
    debug = response.json()["debug"]
    assert debug["endpoint"] == "/memory"
    invocation = debug["invocations"][0]
    assert invocation["command"] == "python memory-estimate.py"
    assert "estimate_kv_cache(" in invocation["files"][0]["contents"]
    # The snippet names the model the caller asked for, not a temp directory.
    assert "model_path='Qwen/Qwen3-32B'" in invocation["files"][0]["contents"]


def test_a_record_reports_the_size_it_adds_to_a_response():
    record = ReplayRecord(
        endpoint="/predict",
        simulator="aisimulate",
        execution="in_process",
        created_at=1000.0,
        invocations=[
            ReplayInvocation(
                id="prediction",
                label="Prediction",
                command="aisimulate predict -c prediction.yaml",
                execution="in_process",
                files=[ReplayFile("prediction.yaml", "engine: {}\n")],
            )
        ],
        script="#!/usr/bin/env bash\n",
    )

    assert record.size_bytes() > 0


def test_the_size_cap_counts_the_escaping_json_adds():
    # A scenario file is full of quotes; the record travels as JSON, so each
    # one costs a second byte on the wire. Summing raw strings would let a
    # record just under the cap overflow it in the response.
    contents = '"' * 4000
    record = ReplayRecord(
        endpoint="/recommend",
        simulator="blis",
        execution="subprocess",
        created_at=1000.0,
        invocations=[
            ReplayInvocation(
                id="run",
                label="Run",
                command="./run.sh",
                execution="subprocess",
                files=[ReplayFile("scenario.json", contents)],
            )
        ],
    )

    assert record.size_bytes() >= 2 * len(contents)


def test_the_prediction_document_is_plain_data_for_the_recorder():
    # The recorder serializes the document before the run, so the record needs
    # a JSON-safe copy; recording the SDK model itself made every
    # `include=debug` prediction fail to serialize.
    req = app_module.EstimateRequest(
        model_path="Qwen/Qwen3-32B",
        system="h200_sxm",
        backend="vllm",
        isl=4000,
        osl=1000,
        tp_size=2,
        batch_size=128,
    )

    document = app_module._recorded_prediction_document(
        app_module._aisimulate_prediction_config(req)
    )

    assert json.loads(json.dumps(document))["engine"]["model"] == "Qwen/Qwen3-32B"


def test_a_failed_memory_estimate_leaves_no_memory_step(monkeypatch):
    # The breakdown is best-effort: a miss returns None, and the record must
    # not keep an invocation whose snippet would make `run.sh` fail.
    recorder = _recorder()
    monkeypatch.setattr(app_module, "_backend_memory_fraction_kind", lambda backend: "of_free")
    monkeypatch.setattr(
        app_module, "_backend_memory_fraction", lambda backend, version=None: 0.9
    )
    arguments = {
        "model_path": "Qwen/Qwen3-32B",
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.31.0",
        "tp": 1,
        "pp": 1,
        "isl": 1024,
        "osl": 128,
        "concurrency": 8,
        "replay_label": "Memory breakdown",
        "replay_recorder": recorder,
    }

    def boom(**_kwargs):
        raise RuntimeError("no memory model for these inputs")

    monkeypatch.setattr(app_module, "estimate_kv_cache", boom)
    with capture_into(recorder):
        assert app_module._build_memory_breakdown(**arguments) is None
    assert recorder.build() is None

    # A successful estimate still keeps its step, after the numbers exist.
    monkeypatch.setattr(
        app_module,
        "estimate_kv_cache",
        lambda **_kwargs: {"memory_breakdown": {}, "total_kv_size_bytes": 0},
    )
    with capture_into(recorder):
        assert app_module._build_memory_breakdown(**arguments) is not None
    assert [item.label for item in recorder.build().invocations] == ["Memory breakdown"]


# ── window ladders ───────────────────────────────────────────────────────────


def _skipped_window_fake(recorder):
    """A ladder step that records, then reports the window cannot hold the model."""

    def fake(req, *, window, **_kwargs):
        recorder.prediction(
            {"engine": {"window": list(window)}},
            label=f"GPU window {window[0]}-{window[1]}",
        )
        raise RuntimeError(f"NoViableParallelConfig: {window} is too small")

    return fake


def test_a_skipped_demand_window_leaves_no_command(monkeypatch):
    # `run.sh` stops at the first failure, so a window that was skipped must
    # not survive in the record ahead of the window that produced the answer.
    recorder = _recorder(endpoint="/recommend")
    req = app_module.RecommendRequest(
        model_path="Qwen/Qwen3-32B",
        system="h200_sxm",
        backend="vllm",
        objective="throughput_demand",
        sla_mode="none",
        target_request_rate=10.0,
        isl=2048,
        osl=128,
    )
    windows = [(1, 2), (3, 4)]
    skipped = _skipped_window_fake(recorder)

    def fake(req, *, window, **_kwargs):
        if window == windows[0]:
            return skipped(req, window=window)
        recorder.prediction({"engine": {"window": list(window)}}, label="GPU window 3-4")
        return SimpleNamespace(candidates=[], counts=None, provenance=None)

    monkeypatch.setattr(app_module, "_demand_gpu_windows", lambda req: windows)
    monkeypatch.setattr(app_module, "_run_aisimulate_recommendation", fake)

    with capture_into(recorder):
        app_module._run_demand_windows(req)

    record = recorder.build()
    assert [invocation.label for invocation in record.invocations] == ["GPU window 3-4"]


def test_a_skipped_min_gpu_window_leaves_no_command(monkeypatch):
    recorder = _recorder(endpoint="/recommend")
    req = app_module.RecommendRequest(
        model_path="Qwen/Qwen3-32B",
        system="h200_sxm",
        backend="vllm",
        objective="min_gpus",
        sla_mode="enforce",
        target_concurrency=8,
        ttft=1000,
        tpot=30,
    )
    windows = [(1, 16), (17, 32)]
    skipped = _skipped_window_fake(recorder)

    def fake(req, *, window, **_kwargs):
        if window == windows[0]:
            return skipped(req, window=window)
        recorder.prediction({"engine": {"window": list(window)}}, label="GPU window 17-32")
        return SimpleNamespace(
            candidates=[],
            counts=None,
            provenance=None,
            selected_candidates=[SimpleNamespace(used_gpus=17)],
        )

    monkeypatch.setattr(app_module, "_max_candidate_gpus", lambda: 32)
    shape = app_module._CapacityShape(tp=1, dp=1, moe_tp=1, moe_ep=1, per_rank_tokens=65536)
    monkeypatch.setattr(app_module, "_discover_fit_shapes", lambda *a, **k: [shape])
    monkeypatch.setattr(app_module, "_run_aisimulate_recommendation", fake)

    with capture_into(recorder):
        app_module._run_min_gpus_windows(req)

    record = recorder.build()
    assert [invocation.label for invocation in record.invocations] == ["GPU window 17-32"]


@pytest.mark.parametrize("failure", ["probe", "refinement", "resource_limit"])
def test_min_gpu_replay_keeps_only_the_successful_attempt_after_a_suppressed_error(monkeypatch, failure):
    from aisimulate.resources import ResourceLimitError

    recorder = _recorder(endpoint="/recommend")
    req = app_module.RecommendRequest(
        model_path="Qwen/Qwen3-32B", system="h200_sxm", backend="vllm",
        objective="min_gpus", sla_mode="enforce", target_concurrency=10,
        isl=45000, osl=300, ttft=3000, tpot=30,
    )
    shape = app_module._CapacityShape(tp=8, dp=1, moe_tp=8, moe_ep=1, per_rank_tokens=65536)
    monkeypatch.setattr(app_module, "_max_candidate_gpus", lambda: 128)
    monkeypatch.setattr(app_module, "_discover_fit_shapes", lambda *a, **k: [shape])
    monkeypatch.setattr(app_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setenv("MIN_GPUS_CANDIDATE_TIMEOUT_S", "30")
    attempts = []

    def fake(req, *, window, replicated_probe=None, **kwargs):
        label = "Replicated probe" if replicated_probe else "Ordinary GPU window"
        attempts.append(label)
        recorder.prediction({"engine": {"window": list(window)}}, label=label)
        if replicated_probe and failure == "probe" or not replicated_probe and failure == "refinement":
            raise RuntimeError("missing performance data")
        if not replicated_probe and failure == "resource_limit":
            raise ResourceLimitError("not enough host memory")
        return SimpleNamespace(
            candidates=[], counts=None, provenance=None,
            selected_candidates=[SimpleNamespace(used_gpus=80 if replicated_probe else 8)],
        )

    monkeypatch.setattr(app_module, "_run_aisimulate_recommendation", fake)

    with capture_into(recorder):
        result, _, _ = app_module._run_min_gpus_windows(req, deadline=90)

    assert attempts == ["Replicated probe", "Ordinary GPU window"]
    winning_label = "Ordinary GPU window" if failure == "probe" else "Replicated probe"
    assert result.selected_candidates[0].used_gpus == (8 if failure == "probe" else 80)
    assert [item.label for item in recorder.build().invocations] == [winning_label]

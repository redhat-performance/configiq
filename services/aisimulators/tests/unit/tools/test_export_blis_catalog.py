from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools import export_blis_catalog
from tools.export_blis_catalog import (
    DERIVE_TIMEOUT_SECONDS,
    SKIP_CATALOG_NAME_COLLISION,
    SKIP_GRAPH_DERIVATION_FAILED,
    SKIP_GRAPH_DERIVER_MISSING,
    SKIP_NOT_HF_CONFIG,
    export_catalog,
    safe_name,
)


@pytest.fixture(autouse=True)
def pinned_wheel_metadata(monkeypatch) -> None:
    """Pin the wheel metadata the exporter reads.

    ``export_catalog`` consults the installed distribution for the release build it
    records as each derived ``model.yaml`` revision. Without this the assertions
    below would depend on how (or whether) ``aisimulate`` is installed in the
    environment running the tests — CI installs it from the pinned wheel URL, so
    ``direct_url.json`` is present there and absent on a bare checkout.
    """
    monkeypatch.setattr(
        export_blis_catalog,
        "installed_wheel",
        lambda: {"version": "1.2.3", "build": None, "url": None},
    )


def hf_config(hidden_size: int = 1024) -> dict:
    return {"hidden_size": hidden_size, "num_hidden_layers": 2}


def write_derive_stub(catalog: Path, failing: set[str] | None = None) -> Path:
    """Write a stand-in for blis-catalog's scripts/derive_graph.py.

    The real deriver needs the whole catalog and PyYAML; the exporter's contract is
    narrower — "run the deriver, keep the model only if it produced graph.yaml" — so the
    tests pin that contract with a stub that succeeds for every model except the ones a
    test asks it to reject.
    """
    failing = failing or set()
    script = catalog / "scripts" / "derive_graph.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "root = Path(args[0])\n"
        "model = args[args.index('--model') + 1]\n"
        f"if model in {sorted(failing)!r}:\n"
        "    print(f'{model}: unsupported architecture', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "(root / 'models' / model / 'graph.yaml').write_text('kind: ModelGraph\\n')\n"
    )
    return script


def write_source(source: Path, files: dict[str, dict]) -> Path:
    source.mkdir(parents=True, exist_ok=True)
    for name, config in files.items():
        (source / f"{name}_config.json").write_text(json.dumps(config))
    return source


def write_upstream(catalog: Path, name: str, config: dict, model_yaml: str = "upstream\n") -> Path:
    entry = catalog / "models" / name
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "config.json").write_text(json.dumps(config))
    (entry / "model.yaml").write_text(model_yaml)
    (entry / "graph.yaml").write_text("kind: ModelGraph\n")
    return entry


def test_export_adds_models_with_a_derived_graph_and_skips_quant_configs(tmp_path) -> None:
    catalog = tmp_path / "catalog"
    source = write_source(
        tmp_path / "source",
        {"Org--Model": hf_config()},
    )
    (source / "Org--Model_hf_quant_config.json").write_text(json.dumps({"quant_method": "fp8"}))
    write_derive_stub(catalog)

    result = export_catalog(catalog, source, "1.2.3")

    assert (result.added, result.shared, result.skipped_count) == (1, 0, 0)
    assert json.loads((catalog / "models/model/config.json").read_text()) == hf_config()
    assert (catalog / "models/model/graph.yaml").is_file()
    assert "aisimulate-1.2.3" in (catalog / "models/model/model.yaml").read_text()
    assert not (catalog / "models/model_hf_quant").exists()
    assert result.manifest["models"]["model"]["origin"] == "aisimulate"
    assert result.manifest["models"]["model"]["aisimulate_model_id"] == "Org/Model"
    assert result.manifest["models"]["model"]["quantization"]["aisimulate"] is None


def test_export_records_the_installed_release_build_as_the_model_revision(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        export_blis_catalog,
        "installed_wheel",
        lambda: {
            "version": "0.13.0",
            "build": "deploy-api-v0.13.0+dd7494b9",
            "url": "https://example.invalid/aisimulate-0.13.0.whl",
        },
    )
    catalog = tmp_path / "catalog"
    source = write_source(tmp_path / "source", {"Org--Model": hf_config()})
    write_derive_stub(catalog)

    result = export_catalog(catalog, source)

    model_yaml = (catalog / "models/model/model.yaml").read_text()
    assert 'revision: "deploy-api-v0.13.0+dd7494b9"' in model_yaml
    assert result.manifest["aisimulate_wheel"]["build"] == "deploy-api-v0.13.0+dd7494b9"


def test_explicit_package_version_wins_over_the_installed_build(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        export_blis_catalog,
        "installed_wheel",
        lambda: {
            "version": "0.13.0",
            "build": "deploy-api-v0.13.0+dd7494b9",
            "url": "https://example.invalid/aisimulate-0.13.0.whl",
        },
    )
    catalog = tmp_path / "catalog"
    source = write_source(tmp_path / "source", {"Org--Model": hf_config()})
    write_derive_stub(catalog)

    result = export_catalog(catalog, source, "1.2.3")

    assert 'revision: "aisimulate-1.2.3"' in (catalog / "models/model/model.yaml").read_text()
    assert result.manifest["aisimulate_version"] == "1.2.3"


def test_export_preserves_identical_upstream_entry_and_reports_it_identical(tmp_path) -> None:
    catalog = tmp_path / "catalog"
    entry = write_upstream(catalog, "model", hf_config())
    source = write_source(tmp_path / "source", {"Org--Model": hf_config()})
    write_derive_stub(catalog)

    result = export_catalog(catalog, source, "1.2.3")

    assert (result.added, result.shared, result.skipped_count) == (0, 1, 0)
    assert (entry / "model.yaml").read_text() == "upstream\n"
    manifest = result.manifest["models"]["model"]
    assert manifest["origin"] == "upstream"
    assert manifest["config_relation"] == "identical"
    assert manifest["aisimulate_model_id"] == "Org/Model"
    assert manifest["differences"] == []


def test_upstream_wins_a_quantization_conflict_and_reports_both_variants(tmp_path) -> None:
    catalog = tmp_path / "catalog"
    upstream = {**hf_config(), "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"}}
    entry = write_upstream(catalog, "model", upstream)
    source = write_source(tmp_path / "source", {"Org--Model": hf_config()})
    write_derive_stub(catalog)

    result = export_catalog(catalog, source, "1.2.3")

    assert (result.added, result.shared, result.skipped_count) == (0, 1, 0)
    # The reviewed upstream config is never replaced.
    assert json.loads((entry / "config.json").read_text()) == upstream
    manifest = result.manifest["models"]["model"]
    assert manifest["config_relation"] == "different_quantization"
    assert manifest["quantization"] == {"aisimulate": None, "upstream": "fp8:e4m3"}
    assert any("quantization_config" in difference for difference in manifest["differences"])


def test_upstream_wins_a_same_quantization_config_difference(tmp_path) -> None:
    catalog = tmp_path / "catalog"
    upstream = {**hf_config(2048), "quantization_config": {"quant_method": "fp8"}}
    write_upstream(catalog, "model", upstream)
    source = write_source(
        tmp_path / "source",
        {"Org--Model": {**hf_config(1024), "quantization_config": {"quant_method": "fp8"}}},
    )
    write_derive_stub(catalog)

    result = export_catalog(catalog, source, "1.2.3")

    manifest = result.manifest["models"]["model"]
    assert manifest["config_relation"] == "different"
    assert any("hidden_size" in difference for difference in manifest["differences"])


def test_derivation_failure_removes_the_entry_and_records_the_reason(tmp_path) -> None:
    catalog = tmp_path / "catalog"
    write_upstream(catalog, "kept", hf_config())
    source = write_source(
        tmp_path / "source",
        {"Org--Kept": hf_config(), "Org--Unsupported": hf_config(2048)},
    )
    write_derive_stub(catalog, failing={"unsupported"})

    result = export_catalog(catalog, source, "1.2.3")

    assert (result.added, result.shared, result.skipped_count) == (0, 1, 1)
    assert not (catalog / "models/unsupported").exists()
    skipped = result.manifest["skipped"][0]
    assert skipped["catalog_name"] == "unsupported"
    assert skipped["reason"] == SKIP_GRAPH_DERIVATION_FAILED
    assert "unsupported architecture" in skipped["detail"]


def test_missing_deriver_skips_added_models_instead_of_shipping_unpriced_entries(tmp_path) -> None:
    catalog = tmp_path / "catalog"
    write_upstream(catalog, "kept", hf_config())
    source = write_source(tmp_path / "source", {"Org--Kept": hf_config(), "Org--New": hf_config(2048)})

    result = export_catalog(catalog, source, "1.2.3")

    assert (result.added, result.shared, result.skipped_count) == (0, 1, 1)
    assert not (catalog / "models/new").exists()
    assert result.manifest["skipped"][0]["reason"] == SKIP_GRAPH_DERIVER_MISSING


def write_sleeping_derive_stub(catalog: Path) -> Path:
    """A deriver that never returns in time, to exercise the subprocess bound."""
    script = catalog / "scripts" / "derive_graph.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "import sys\n"
        "import time\n"
        "time.sleep(30)\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "root = Path(args[0])\n"
        "model = args[args.index('--model') + 1]\n"
        "(root / 'models' / model / 'graph.yaml').write_text('kind: ModelGraph\\n')\n"
    )
    return script


def test_hung_deriver_is_recorded_as_a_skip_not_a_hung_export(tmp_path, monkeypatch) -> None:
    """A deriver that never finishes must not abort the export or keep the model."""
    catalog = tmp_path / "catalog"
    write_upstream(catalog, "upstream", hf_config())
    source = write_source(
        tmp_path / "source",
        {"Org--Upstream": hf_config(), "Org--New": hf_config(2048)},
    )
    write_sleeping_derive_stub(catalog)
    # The real bound is 10s; the stub sleeps 30s, so any value below that fails the same way
    # without costing the test the wall clock.
    monkeypatch.setattr(export_blis_catalog, "DERIVE_TIMEOUT_SECONDS", 0.05)

    result = export_catalog(catalog, source, "1.2.3")

    assert (result.added, result.shared, result.skipped_count) == (0, 1, 1)
    assert not (catalog / "models" / "new").exists()
    skipped = result.manifest["skipped"][0]
    assert skipped["catalog_name"] == "new"
    assert skipped["reason"] == SKIP_GRAPH_DERIVATION_FAILED
    assert skipped["detail"] == "derive_graph.py timed out after 0.05s"


def test_graph_derivation_is_actually_bounded(tmp_path, monkeypatch) -> None:
    """The timeout argument must reach subprocess.run, so the bound cannot be dropped silently."""
    calls: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        calls.update(kwargs)
        return subprocess.CompletedProcess(cmd, 1, "", "boomed\n")

    monkeypatch.setattr(export_blis_catalog.subprocess, "run", fake_run)

    derived, detail = export_blis_catalog._derive_graph(
        tmp_path, "model", tmp_path / "scripts" / "derive_graph.py"
    )

    assert derived is False
    assert detail == "boomed"
    assert calls["timeout"] == DERIVE_TIMEOUT_SECONDS
    assert calls["check"] is False


def test_catalog_name_collision_keeps_the_first_model_and_records_the_second(tmp_path) -> None:
    catalog = tmp_path / "catalog"
    source = write_source(
        tmp_path / "source",
        {"OrgA--Model": hf_config(), "OrgB--Model": hf_config(2048)},
    )
    write_derive_stub(catalog)

    result = export_catalog(catalog, source, "1.2.3")

    assert (result.added, result.shared, result.skipped_count) == (1, 0, 1)
    assert result.manifest["models"]["model"]["aisimulate_model_id"] == "OrgA/Model"
    skipped = result.manifest["skipped"][0]
    assert skipped["reason"] == SKIP_CATALOG_NAME_COLLISION
    assert skipped["aisimulate_model_id"] == "OrgB/Model"


def test_a_skipped_model_does_not_claim_its_catalog_name(tmp_path) -> None:
    catalog = tmp_path / "catalog"
    source = write_source(
        tmp_path / "source",
        {"OrgA--Model": {"note": "not a model"}, "OrgB--Model": hf_config()},
    )
    write_derive_stub(catalog)

    result = export_catalog(catalog, source, "1.2.3")

    assert (result.added, result.shared, result.skipped_count) == (1, 0, 1)
    assert result.manifest["models"]["model"]["aisimulate_model_id"] == "OrgB/Model"
    assert result.manifest["skipped"][0]["reason"] == SKIP_NOT_HF_CONFIG


def test_non_hf_config_is_skipped_not_fatal(tmp_path) -> None:
    catalog = tmp_path / "catalog"
    write_upstream(catalog, "kept", hf_config())
    source = write_source(
        tmp_path / "source",
        {"Org--Kept": hf_config(), "Org--NotAModel": {"note": "not a model"}},
    )
    write_derive_stub(catalog)

    result = export_catalog(catalog, source, "1.2.3")

    assert (result.added, result.shared, result.skipped_count) == (0, 1, 1)
    assert result.manifest["skipped"][0]["reason"] == SKIP_NOT_HF_CONFIG


def test_added_model_records_the_equivalent_upstream_entry_under_another_name(tmp_path) -> None:
    catalog = tmp_path / "catalog"
    shared = {**hf_config(), "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"}}
    write_upstream(catalog, "glm-5", shared)
    source = write_source(tmp_path / "source", {"org--GLM-5-FP8": shared})
    write_derive_stub(catalog)

    result = export_catalog(catalog, source, "1.2.3")

    assert result.added == 1
    manifest = result.manifest["models"]["glm-5-fp8"]
    assert manifest["origin"] == "aisimulate"
    assert manifest["equivalent_catalog_name"] == "glm-5"


def test_manifest_records_hardware_and_networks(tmp_path) -> None:
    catalog = tmp_path / "catalog"
    write_upstream(catalog, "model", hf_config())
    (catalog / "hardware").mkdir()
    (catalog / "hardware/h100.yaml").write_text("kind: Chip\n")
    (catalog / "networks").mkdir()
    (catalog / "networks/ib.yaml").write_text("kind: Fabric\n")
    source = write_source(tmp_path / "source", {"Org--Model": hf_config()})
    write_derive_stub(catalog)
    manifest_path = tmp_path / "out/support.json"

    export_catalog(catalog, source, "1.2.3", manifest_path=manifest_path)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["aisimulate_version"] == "1.2.3"
    assert manifest["hardware"] == ["h100"]
    assert manifest["networks"] == ["ib"]
    assert "scenarios" not in manifest


def test_no_usable_configs_is_fatal(tmp_path) -> None:
    catalog = tmp_path / "catalog"
    source = write_source(tmp_path / "source", {"Org--NotAModel": {"note": "not a model"}})
    write_derive_stub(catalog)

    with pytest.raises(RuntimeError, match="no usable AISimulate model configs"):
        export_catalog(catalog, source, "1.2.3")


def test_safe_name_matches_blis_lowercase_catalog_resolution() -> None:
    assert safe_name("Qwen/Qwen3-32B-FP8") == "qwen3-32b-fp8"


def test_export_accepts_top_level_hf_config_with_empty_text_config(tmp_path) -> None:
    catalog = tmp_path / "catalog"
    source = write_source(tmp_path / "source", {"Org--Model": {**hf_config(), "text_config": {}}})
    write_derive_stub(catalog)

    assert export_catalog(catalog, source, "1.2.3").added == 1

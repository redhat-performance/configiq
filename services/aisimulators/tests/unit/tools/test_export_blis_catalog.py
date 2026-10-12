from __future__ import annotations

import json

import pytest

from tools.export_blis_catalog import export_catalog, safe_name


def hf_config(hidden_size: int = 1024) -> dict:
    return {"hidden_size": hidden_size, "num_hidden_layers": 2}


def test_export_adds_aisimulate_models_and_skips_quant_configs(tmp_path) -> None:
    source = tmp_path / "source"
    catalog = tmp_path / "catalog"
    source.mkdir()
    (source / "Org--Model_config.json").write_text(json.dumps(hf_config()))
    (source / "Org--Model_hf_quant_config.json").write_text(json.dumps({"quant_method": "fp8"}))

    assert export_catalog(catalog, source, "1.2.3") == 1
    assert json.loads((catalog / "models/model/config.json").read_text()) == hf_config()
    assert not (catalog / "models/model_hf_quant/config.json").exists()
    assert "aisimulate-1.2.3" in (catalog / "models/model/model.yaml").read_text()


def test_export_preserves_identical_upstream_entry(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "catalog/models/model"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    config = hf_config()
    (source / "Org--Model_config.json").write_text(json.dumps(config))
    (destination / "config.json").write_text(json.dumps(config))
    (destination / "model.yaml").write_text("upstream")

    assert export_catalog(tmp_path / "catalog", source, "1.2.3") == 1
    assert (destination / "model.yaml").read_text() == "upstream"


def test_export_rejects_conflicting_upstream_entry(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "catalog/models/model"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    (source / "Org--Model_config.json").write_text(json.dumps(hf_config(1024)))
    (destination / "config.json").write_text(json.dumps(hf_config(2048)))

    with pytest.raises(RuntimeError, match="conflicting upstream"):
        export_catalog(tmp_path / "catalog", source, "1.2.3")


def test_export_preserves_reviewed_upstream_preference(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "catalog/models/glm-5.3"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    upstream = hf_config(2048)
    (source / "Org--GLM-5.3_config.json").write_text(json.dumps(hf_config(1024)))
    (destination / "config.json").write_text(json.dumps(upstream))

    assert export_catalog(tmp_path / "catalog", source, "1.2.3") == 1
    assert json.loads((destination / "config.json").read_text()) == upstream


def test_safe_name_matches_blis_lowercase_catalog_resolution() -> None:
    assert safe_name("Qwen/Qwen3-32B-FP8") == "qwen3-32b-fp8"


def test_export_accepts_top_level_hf_config_with_empty_text_config(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = {**hf_config(), "text_config": {}}
    (source / "Org--Model_config.json").write_text(json.dumps(config))

    assert export_catalog(tmp_path / "catalog", source, "1.2.3") == 1

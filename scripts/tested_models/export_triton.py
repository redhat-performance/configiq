#!/usr/bin/env python3
"""Validate and re-export the Triton FIL configuration for XGBoost bundles."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/tested-models/stage5/models"))
    parser.add_argument("--repository-dir", type=Path, default=Path("data/tested-models/stage5/triton"))
    args = parser.parse_args()
    count = 0
    for metadata_path in sorted(args.model_dir.glob("*.metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("model_format") != "xgboost-json":
            raise ValueError(f"not an XGBoost JSON model: {metadata_path}")
        pair = metadata_path.name.removesuffix(".metadata.json")
        model_path = args.model_dir / f"{pair}.json"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        feature_count = len(metadata["feature_schema"]["feature_names"])
        model_repository = args.repository_dir / pair
        version_dir = model_repository / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, version_dir / "xgboost.json")
        config = model_repository / "config.pbtxt"
        config.write_text(f'''name: "{pair}"
backend: "fil"
max_batch_size: 32768
default_model_filename: "xgboost.json"
input [{{ name: "input__0" data_type: TYPE_FP32 dims: [ {feature_count} ] }}]
output [{{ name: "output__0" data_type: TYPE_FP32 dims: [ 1 ] }}]
instance_group [{{ kind: KIND_CPU }}]
parameters: {{ key: "model_type" value: {{ string_value: "xgboost_json" }} }}
parameters: {{ key: "is_classifier" value: {{ string_value: "true" }} }}
parameters: {{ key: "threshold" value: {{ string_value: "0.5" }} }}
dynamic_batching {{}}
''')
        count += 1
    print(f"validated and wrote {count} Triton FIL configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

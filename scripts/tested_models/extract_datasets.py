#!/usr/bin/env python3
"""Extract every available discovered pair into normalized parquet files."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
import urllib3
from dataset_common import dataframe_sha256, fetch_rows, normalize_rows, write_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=Path("data/tested-models/discovery.json"))
    parser.add_argument("--api-url", default=os.getenv("PERF_DATA_API_URL"))
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for the internal ground-truth API.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/tested-models/datasets"))
    parser.add_argument("--manifest", type=Path, default=Path("data/tested-models/dataset-manifest.json"))
    return parser.parse_args()


def extract(
    discovery_path: Path,
    api_url: str,
    output_dir: Path,
    manifest_path: Path,
    *,
    insecure: bool = False,
) -> dict:
    discovery = json.loads(discovery_path.read_text())
    pairs = discovery.get("pairs")
    if not isinstance(pairs, list):
        raise TypeError("discovery manifest has no pairs array")

    manifest_pairs: list[dict] = []
    session = requests.Session()
    for pair in pairs:
        if pair.get("status") != "available":
            continue
        model_id = pair["model_id"]
        accelerator = pair["accelerator"]
        rows = fetch_rows(session, api_url, model_id, accelerator, verify=not insecure)
        frame = normalize_rows(rows, model_id, accelerator)
        if "turn_index" not in frame:
            frame["turn_index"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        output_path = output_dir / f"{pair['id']}.parquet"
        write_parquet(frame, output_path)
        manifest_pairs.append(
            {
                "id": pair["id"],
                "model_id": model_id,
                "accelerator": accelerator,
                "path": str(output_path),
                "records": len(frame),
                "columns": list(frame.columns),
                "sha256": dataframe_sha256(frame),
            }
        )
        print(f"wrote {output_path}: {len(frame)} records")

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "ground_truth_api",
        "discovery": str(discovery_path),
        "pairs": manifest_pairs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote dataset manifest: {manifest_path}")
    return manifest


def main() -> int:
    args = parse_args()
    if not args.api_url:
        raise SystemExit("Provide --api-url or PERF_DATA_API_URL; endpoint values are never stored in manifests.")
    if args.insecure:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    extract(args.discovery, args.api_url, args.output_dir, args.manifest, insecure=args.insecure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

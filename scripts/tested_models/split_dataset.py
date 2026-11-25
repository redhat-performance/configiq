#!/usr/bin/env python3
"""Create deterministic, configuration-grouped train/validation/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from stage5_common import (
    aggregate_measurements,
    configuration_group,
    normalize_features,
    read_manifest,
)

_HELD_OUT_SPLITS = ("validation", "test")


def _ensure_aggregate_training_group(
    frame: pd.DataFrame,
    assignments: dict[str, str],
    groups: list[str],
) -> tuple[dict[str, str], str]:
    """Keep at least one aggregate configuration group in train.

    Envelope labels are fitted on train and only evaluated on the held-out
    splits, so a hash assignment that puts every aggregate group in validation
    or test leaves `label_pair` with nothing to fit. Move one aggregate group
    into train, preferring a split that still holds another group, then the
    smallest group, so evaluation keeps as many rows as possible.
    """
    aggregate = aggregate_measurements(frame)
    if not aggregate.any():
        return assignments, "no_aggregate_rows"
    aggregate_groups = set(frame.loc[aggregate, "configuration_group"])
    if any(assignments[group] == "train" for group in aggregate_groups):
        return assignments, "present"

    row_counts = frame["configuration_group"].value_counts().to_dict()
    for held_out in _HELD_OUT_SPLITS:
        candidates = [
            group
            for group in groups
            if group in aggregate_groups and assignments[group] == held_out
        ]
        if not candidates:
            continue
        split_groups = [group for group in groups if assignments[group] == held_out]
        # Moving the only group would leave the held-out split empty; use it
        # only when no other split can supply an aggregate group.
        pool = [group for group in candidates if len(split_groups) > 1] or candidates
        chosen = min(pool, key=lambda group: (row_counts[group], group))
        moved = dict(assignments)
        moved[chosen] = "train"
        return moved, f"moved_from_{held_out}"
    # Every group carries an assignment, so an aggregate group is either in
    # train (handled above) or in one of the held-out splits.
    raise RuntimeError("aggregate group is not assigned to any split")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/tested-models/dataset-manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tested-models/stage5/splits"))
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "pairs": {}}
    for frame in read_manifest(args.manifest):
        pair = str(frame["pair_id"].iloc[0])
        frame = frame.copy()
        features = normalize_features(frame)
        frame["configuration_group"] = configuration_group(frame, features)
        versions = features["version"]
        groups = sorted(frame["configuration_group"].unique())
        if len(groups) <= 4:
            assignments = {group: "train" for group in groups}
            split_status = "insufficient_groups"
        else:
            # Hash ordering makes assignment independent of parquet row order.
            groups = sorted(groups, key=lambda group: (hashlib.sha256(f"{args.seed}|{pair}|{group}".encode()).hexdigest(), group))
            assignments = {group: ("test" if index % 10 < 2 else "validation" if index % 10 < 4 else "train") for index, group in enumerate(groups)}
            split_status = "ok"
        assignments, aggregate_train = _ensure_aggregate_training_group(frame, assignments, groups)
        frame["split"] = frame["configuration_group"].map(assignments)
        for split in ("train", "validation", "test"):
            frame[frame["split"] == split].to_parquet(output / f"{pair}.{split}.parquet", index=False)
        # Per version family support shows whether a family can be learned and
        # evaluated at all. A family confined to one split produces either an
        # all-zero inference feature or an unevaluable test set.
        version_support = {}
        for family in sorted(versions.unique()):
            family_mask = versions == family
            version_support[str(family)] = {
                split: {
                    "rows": int((family_mask & (frame["split"] == split)).sum()),
                    "groups": int(frame.loc[family_mask & (frame["split"] == split), "configuration_group"].nunique()),
                }
                for split in ("train", "validation", "test")
            }
        summary["pairs"][pair] = {
            "rows": len(frame),
            "groups": len(groups),
            "status": split_status,
            "aggregate_train": aggregate_train,
            "counts": frame["split"].value_counts().to_dict(),
            "versions": version_support,
        }
    (output / "split-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

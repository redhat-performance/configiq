"""Fetch tested-model InferenceX results and publish a private HF snapshot."""

from __future__ import annotations

import json
import subprocess
from io import BytesIO
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "redhat-performance/configiq-inferencex-comparison"
FILENAME = "inferencex-8k1k.json"


def snapshot_content(snapshot: dict) -> dict:
    """Ignore the fetch timestamp when checking whether measurements changed."""
    return {key: value for key, value in snapshot.items() if key != "retrievedAt"}


def main() -> int:
    result = subprocess.run(
        ["node", "scripts/update-inferencex-snapshot.mjs"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    snapshot = json.loads(result.stdout)
    config = json.loads((ROOT / "public/config.json").read_text())
    # `approvedModels` is the durable name; `testedModels` is the transition key.
    approved_models = config.get("approvedModels", config.get("testedModels"))
    if (
        not isinstance(approved_models, list)
        or not approved_models
        or not all(isinstance(model, str) for model in approved_models)
    ):
        raise ValueError("public/config.json must contain a non-empty approved model list")
    approved_model_set = set(approved_models)
    rows = snapshot.get("rows")
    if not isinstance(rows, list) or not rows or any(
        not isinstance(row, dict)
        or row.get("model") not in approved_model_set
        or row.get("benchmark_source") != "inferencex"
        or row.get("isl") != 8192
        or row.get("osl") != 1024
        for row in rows
    ):
        raise ValueError("InferenceX snapshot is empty or contains unapproved benchmark rows")

    api = HfApi()
    try:
        repo = api.repo_info(REPO_ID, repo_type="dataset")
    except RepositoryNotFoundError as error:
        raise ValueError(
            f"Private InferenceX dataset {REPO_ID} does not exist; create it before publishing"
        ) from error
    if not repo.private:
        raise ValueError("Refusing to publish to a non-private InferenceX dataset")
    try:
        current_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=FILENAME)
    except EntryNotFoundError:
        # The first publish has no previous snapshot to compare against.
        current = None
    else:
        current = json.loads(Path(current_path).read_text())
    if current is not None and snapshot_content(current) == snapshot_content(snapshot):
        print(f"InferenceX measurements are unchanged ({len(rows)} rows); private HF dataset is current")
        return 0

    commit = api.upload_file(
        path_or_fileobj=BytesIO(result.stdout),
        path_in_repo=FILENAME,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Refresh tested-model InferenceX snapshot ({len(rows)} rows)",
    )
    published_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=FILENAME, revision=commit.oid)
    if Path(published_path).read_bytes() != result.stdout:
        raise ValueError("Published InferenceX snapshot does not match the validated results")
    print(f"Published {len(rows)} InferenceX rows to private HF dataset at {commit.oid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

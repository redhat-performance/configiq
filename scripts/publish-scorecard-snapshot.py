"""Publish the simulator scorecard artifact, or promote one channel to another.

The sweep writes `scorecard-<channel>.json` and `scorecard-points-<channel>.json`
locally. This uploads them as `<channel>/scorecard.json` and `<channel>/points.json`
so the page can read them without a token.

Promotion copies a channel that already exists (`--from-channel dev`) rather than
rebuilding: the prod snapshot should be the same data a reader already saw on
dev, not a second sweep that happens to run a day later.

Usage:
  python scripts/publish-scorecard-snapshot.py --channel dev --source-dir artifacts/scorecard
  python scripts/publish-scorecard-snapshot.py --channel prod --from-channel dev
"""

from __future__ import annotations

import argparse
import json
import os
from io import BytesIO
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

DEFAULT_REPO_ID = "redhat-performance/configiq-simulator-scorecard"
CHANNELS = ("dev", "prod")

SUMMARY_FILE = "scorecard.json"
POINTS_FILE = "points.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True, choices=CHANNELS)
    parser.add_argument(
        "--from-channel",
        choices=CHANNELS,
        help="copy this published channel instead of uploading local files",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="directory holding scorecard-<channel>.json and scorecard-points-<channel>.json",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("SCORECARD_DATASET_REPO", DEFAULT_REPO_ID),
        help="Hugging Face dataset repository",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="report whether the artifact would change, without uploading",
    )
    parser.add_argument("--github-output", help="append changed/revision to this file")
    return parser.parse_args()


def write_output(path: str | None, values: dict[str, str]) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def local_artifact(source_dir: Path, channel: str) -> tuple[bytes, bytes]:
    summary_path = source_dir / f"scorecard-{channel}.json"
    points_path = source_dir / f"scorecard-points-{channel}.json"
    for path in (summary_path, points_path):
        if not path.is_file():
            raise SystemExit(f"Missing artifact file: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    points = json.loads(points_path.read_text(encoding="utf-8"))
    if summary.get("channel") != channel:
        raise SystemExit(f"{summary_path} declares channel {summary.get('channel')!r}, expected {channel!r}")
    if not summary.get("simulators") or not summary.get("totals"):
        raise SystemExit("Refusing to publish a scorecard with no simulator totals")
    if not isinstance(points.get("points"), list):
        raise SystemExit("The point detail is missing its points array")
    return summary_path.read_bytes(), points_path.read_bytes()


def download_file(repo_id: str, path_in_repo: str) -> bytes | None:
    try:
        return Path(
            hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=path_in_repo)
        ).read_bytes()
    except EntryNotFoundError:
        return None


def published_artifact(repo_id: str, channel: str) -> tuple[bytes, bytes]:
    summary = download_file(repo_id, f"{channel}/{SUMMARY_FILE}")
    points = download_file(repo_id, f"{channel}/{POINTS_FILE}")
    if summary is None or points is None:
        raise SystemExit(f"Channel {channel!r} is not published in {repo_id}")
    return summary, points


def main() -> int:
    args = parse_args()

    api = HfApi()
    try:
        repo = api.repo_info(args.repo_id, repo_type="dataset")
    except RepositoryNotFoundError as error:
        raise SystemExit(
            f"Scorecard dataset {args.repo_id} does not exist; create it before publishing"
        ) from error
    if repo.private:
        # The page reads this without credentials, so a private repository would
        # break every deployment at once.
        raise SystemExit(f"Refusing to publish to private dataset {args.repo_id}; the page reads it anonymously")

    if args.from_channel:
        summary_bytes, points_bytes = published_artifact(args.repo_id, args.from_channel)
        message = f"Promote scorecard snapshot from {args.from_channel} to {args.channel}"
    else:
        source_dir = args.source_dir or Path("artifacts/scorecard")
        summary_bytes, points_bytes = local_artifact(source_dir, args.channel)
        message = f"Publish scorecard snapshot for {args.channel}"

    # Both files are the artifact. A retry after a partial upload must see the
    # missing or stale half as a change, not report the channel current just
    # because the summary landed.
    current_summary = download_file(args.repo_id, f"{args.channel}/{SUMMARY_FILE}")
    current_points = download_file(args.repo_id, f"{args.channel}/{POINTS_FILE}")
    if current_summary == summary_bytes and current_points == points_bytes:
        print(f"Channel {args.channel!r} is already current")
        write_output(args.github_output, {"changed": "false", "revision": ""})
        return 0

    if args.check_only:
        print(f"Channel {args.channel!r} would change")
        write_output(args.github_output, {"changed": "true", "revision": ""})
        return 0

    # One commit for both files, so a reader never sees a summary that
    # describes points from the previous snapshot.
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=message,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{args.channel}/{SUMMARY_FILE}",
                path_or_fileobj=BytesIO(summary_bytes),
            ),
            CommitOperationAdd(
                path_in_repo=f"{args.channel}/{POINTS_FILE}",
                path_or_fileobj=BytesIO(points_bytes),
            ),
        ],
    )
    revision = commit.oid

    print(f"Published {args.repo_id}/{args.channel} at {revision}")
    write_output(args.github_output, {"changed": "true", "revision": revision})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

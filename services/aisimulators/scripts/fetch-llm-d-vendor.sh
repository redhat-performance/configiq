#!/usr/bin/env bash
# Fetch the pinned llm-d subtrees the deployment render gate builds against.
#
# The llm-d deployment target renders Kustomize overlays that reference files in
# the llm-d repository by relative path. A bundle has to carry those files, so
# the service needs the same tree the overlay was validated against, at the
# exact revision the target pins. This script materialises that tree; the
# Containerfile runs it at image build time, and CI runs it before the tests so
# the render gate is exercised against the real files.
#
# Usage: scripts/fetch-llm-d-vendor.sh [revision]
#   AISIMULATORS_LLM_D_VENDOR_DIR  destination root (default: ../vendor/llm-d)
#
# The destination layout is <root>/<revision>/<repo-relative path>, which is
# what tools/api_service/deploy/catalog.py resolves.
set -euo pipefail

# Keep in step with LLM_D_REVISION in tools/api_service/deploy/catalog.py; a
# unit test asserts the two agree.
REVISION="${1:-${LLM_D_REVISION:-378b42d18dabe0b9c4ced4e6f3670d93b52db08e}}"
REPO="${LLM_D_REPOSITORY:-https://github.com/llm-d/llm-d.git}"

# Keep in step with LLM_D_VENDOR_PATHS in tools/api_service/deploy/catalog.py.
PATHS=(
  guides/recipes/modelserver
  guides/recipes/router
  guides/pd-disaggregation/modelserver
  guides/pd-disaggregation/router
  guides/optimized-baseline/modelserver
  guides/optimized-baseline/router
)

destination="${AISIMULATORS_LLM_D_VENDOR_DIR:-$(cd "$(dirname "$0")/.." && pwd)/vendor/llm-d}"
target="${destination}/${REVISION}"

if [ -f "${target}/.complete" ]; then
  printf 'llm-d %s is already vendored at %s\n' "${REVISION}" "${target}"
  exit 0
fi

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

git -C "${work}" init --quiet
git -C "${work}" remote add origin "${REPO}"
git -C "${work}" fetch --quiet --depth 1 origin "${REVISION}"

# A tree without the marker may be a half-extracted earlier attempt, so replace
# it rather than extracting on top of stale files.
rm -rf "${target}"
mkdir -p "${target}"
git -C "${work}" archive FETCH_HEAD -- "${PATHS[@]}" | tar -x -C "${target}"
: > "${target}/.complete"

printf 'vendored llm-d %s into %s\n' "${REVISION}" "${target}"

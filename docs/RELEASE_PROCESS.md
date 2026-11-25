# Release Process

## Overview

ConfigIQ uses semantic versioning with git tags to version the UI and its
container images. The image build workflows are stored in this repository under
`.github/workflows/`.

The deployment contract consists of five container images:

- `ghcr.io/redhat-performance/configiq` (Next.js webapp)
- `ghcr.io/redhat-performance/aisimulators` (GPU sizing / estimation API)
- `ghcr.io/redhat-performance/aicostings` (GPU / LLM pricing API)
- `ghcr.io/redhat-performance/configiq-mcp` (MCP gateway for the assistant)
- `ghcr.io/redhat-performance/tritonserver` (FIL classifiers; carries the
  assistant's vLLM runtime)

Keep the image versions compatible when releasing the frontend and services.

`build.yml` builds four of them; `tritonserver.yml` builds the fifth. The
`configiq-mcp` image is built by the matrix leg named `mcp-gateway`, so read a
leg's `image:` value rather than its `name:` when auditing this list.

## Creating a Release

1. **Update package.json version** (optional, for npm tracking):
   ```bash
   npm version patch --no-git-tag-version
   ```
   Use `patch`, `minor`, or `major` depending on the type of release.

2. **Create and push a signed git tag** (replace `vX.Y.Z` with the release version):
   ```bash
   git tag -s vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```
   The `-s` flag creates a signed tag for release verification. The `-m` flag provides the tag message.

3. Build, publish, and deploy the images using the workflows in
   `.github/workflows/build.yml` and `.github/workflows/tritonserver.yml`.
   A release tag builds matching tags for all five images:
   - `ghcr.io/redhat-performance/{configiq,aisimulators,aicostings,configiq-mcp,tritonserver}:X.Y.Z`
   - `ghcr.io/redhat-performance/{configiq,aisimulators,aicostings,configiq-mcp,tritonserver}:latest`

## Version Display in UI

The sidebar footer shows:
- **ConfigIQ build**: version (from git tag, via `NEXT_PUBLIC_GIT_VERSION`) and
  short commit (via `NEXT_PUBLIC_GIT_COMMIT`)
- **Every simulator**: label, version, and git commit, from `GET /simulators`,
  in that endpoint's order. Each simulator reports the commit of the artifact
  this image runs: AISim's comes from the SDK wheel's direct-URL release
  tag; BLIS's from the `BLIS_COMMIT` build arg. `revision` is null when the
  artifact does not report a commit, and the footer then shows the version
  only.

The footer lives in `components/layout/AppShell.tsx` and reads simulator identity
from the catalog API, so adding a simulator needs no footer change.

## Container Tags

When the five images are released together, they carry the same tag set:

- **`latest`** - Most recent tagged release (release-tracking; the `configiq.io` host follows this)
- **`dev`** - Latest commit from `main` (commit-tracking; the `.dev` host follows this)
- **`X.Y.Z`** - Specific version tags

The Triton image workflow is path-filtered for `services/tritonserver/**`. The
image starts from the standard `nvcr.io/nvidia/tritonserver:26.09-py3` image so
the FIL backend remains available, then installs vLLM `0.33.0` into an isolated
site directory. That vLLM is what the assistant's model server runs; Triton
itself serves FIL only. Build validation checks FIL, vLLM, and native K2
architecture support.

The two-host tag model (`.dev` → `:dev`, `.io` → `:latest`) is implemented in
the `configiq-deploy` repo; `podman auto-update` on each host polls whatever tag
its running containers use, so the per-host tag is the only lever.

## Container Registry

Published containers:

- https://github.com/redhat-performance/configiq/pkgs/container/configiq
- https://github.com/redhat-performance/configiq/pkgs/container/aisimulators
- https://github.com/redhat-performance/configiq/pkgs/container/aicostings
- https://github.com/redhat-performance/configiq/pkgs/container/tritonserver

## Bumping the simulation SDK

The `aisimulators` and `aicostings` services install the **aisim** SDK as a
single unified wheel from the Red Hat `redhat-performance/aisimulate` fork's
GitHub Release assets — not PyPI. The distribution is named `aisim`, while the
import packages keep their historical names: the services import the
`aisimulate` / `aisimulate_core` namespaces (`aisimulators` directly for
sizing; `aicostings` indirectly via `configiq.systems` for the GPU catalog). It
is referenced by **exact download URL** rather than `name==version`, because
the wheel is published only via the fork's Release (never PyPI); a direct URL
forces pip to install that exact artifact. The requirement name must match the
wheel metadata (`aisim @ …/aisim-<version>-…whl`), or the install is rejected.

To move to a new SDK build, update the single `aisim @ …` wheel URL — keep it
**identical in both files** — in `[project.dependencies]`, then regenerate each
`uv.lock` (`uv lock`):

1. `services/aisimulators/pyproject.toml`
2. `services/aicostings/pyproject.toml`

The URL points at a specific `deploy-api-v<version>+<hash>` release asset (the
`+` in the tag is URL-encoded as `%2B`); never the rolling `deploy-api-latest`,
so rebuilds are reproducible. The wheel is produced by the fork's
`deploy-api-wheels.yml` workflow on its `deploy/api` branch.

> **Note:** the aisim wheel pulls in a large JAX/optimization dependency
> stack (jax, jaxlib, optax, google-vizier, tfp-nightly) as base dependencies,
> so both service images are substantially larger than under the previous
> two-wheel aiconfigurator 0.11/0.12 SDK.

After bumping the SDK, rebuild the `aisimulators` image before assuming the new
model configs are live. BLIS does not read the wheel's model configs: it prices
only models in the pinned blis-catalog, so an SDK model absent from the catalog
is an explicit unsupported response rather than a build-time import.

## Bumping BLIS

BLIS normally comes from the upstream multi-arch release image
`ghcr.io/inference-sim/blis:<tag>`, pinned by index digest. Until a release
carries the reports the adapter reads (`blis version --json` with each modelled
engine release, each pool's memory on the run's Result, `blis catalog list`;
inference-sim#2693, #2694, #2695), the Containerfile builds
the binary from a pinned commit of the fork's `deploy/api` branch instead — the
upstream `main` commit that tag names, plus those changes, tagged in the fork so
a rewrite of the branch cannot orphan it. BLIS v0.19 embeds its catalog and
coefficient registry, so the Containerfile clones no data tree at all: the API
reads the same facts from the binary with `blis catalog list --json`, and the
image cannot pair a binary with a different build's catalog:

| Argument | Upstream |
|---|---|
| `BLIS_REPOSITORY` | the fork holding the `deploy/api` branch, while the build is a source build |
| `BLIS_COMMIT` | the commit the binary and `/simulators` revision report |
| `BLIS_BASE_TAG` | the highest release the pinned commit descends from; the fetch reaches back to it so the toolchain can name the commit's pseudo-version, and naming an older release makes that version claim a pre-release of code the build already contains |
| `BLIS_FETCH_DEPTH` | how far back that fetch goes; it must reach `BLIS_BASE_TAG` |
| `BLIS_GO_VERSION` | the `go` directive of that commit |
| `BLIS_VERSION` | what `blis version` prints for it, without the leading `v` (a Go pseudo-version while no release tag contains the commit) |

To move to a new BLIS build:

1. For a release, resolve the release image's multi-arch index digest and put
   the tag plus digest in the build stage, replacing the source build; set
   `BLIS_VERSION` to the human-facing release. For a commit build, set
   `BLIS_REPOSITORY`, `BLIS_COMMIT`, `BLIS_BASE_TAG` and `BLIS_GO_VERSION`, and
   `BLIS_VERSION` to the pseudo-version `blis version` prints. The reported
   version is a build fact, never a literal that can drift from the binary: the
   build fails if `blis version --json` reports a version other than the one
   `BLIS_VERSION` states, and the Python test pins the commit the footer shows.
2. Set `BLIS_COMMIT` to the commit the release tag points at (`git rev-parse
   vX.Y.Z`): the binary embeds no revision, so this is the only source for the
   git commit the footer shows. The default `BLIS_VERSION` may omit it, because
   the service resolves the default build's commit; any other build without a
   commit reports no revision.
3. Update `DEFAULT_BLIS_RELEASE` and `DEFAULT_BLIS_COMMIT` in
   `tools/api_service/blis/runtime.py`. Do not add an engine-version table: the
   build reports the release its kernel prices, and the adapter forwards it.
4. Rebuild the image and confirm the new binary reports the build:
   `/usr/local/bin/blis version --json`, `blis catalog list --json`, then
   `GET /simulators` and `GET /backends?simulator=blis`.
5. Exercise a catalog-backed request end to end (`GET /models?simulator=blis`,
   then a BLIS `/predict` for a listed model) so a catalog/binary mismatch fails
   before deployment.

BLIS never fetches at run time, so its catalog and registry must be baked into
the binary. The binary's own `catalog list` report is the only source of BLIS
support facts: model graphs, chips, and fabrics.


## Deployment

Deployment is handled by the `configiq-deploy` repo (rootless Podman + systemd
quadlets + nginx on IBM Cloud VMs). The `.dev` hosts follow `:dev`; the `.io`
hosts follow `:latest`. The Triton quadlet is pinned to an immutable GHCR
digest, and the same image digest is used by the K2 prefetch script.

Every image exposes `GET /api/version` with the version and git commit it was
built from. The live e2e workflows read it and check out that commit before
running, so the tests describe the deployed build rather than a branch head.
After deploying a release, trigger the **Release sanity (configiq.io)**
workflow; the daily and weekly full matrices run against `.dev` by design.

To pull an image manually:

**Production** (`:latest`):
```bash
podman pull ghcr.io/redhat-performance/configiq:latest
```

**Development** (`:dev`):
```bash
podman pull ghcr.io/redhat-performance/configiq:dev
```

## Tested-model bundle releases

Tested-model classifiers are data artifacts, not container-image contents. The
`tested-models.yml` workflow builds a sanitized bundle from the ground-truth
dataset's current default branch (resolved to an immutable commit SHA) and
uploads it as a workflow artifact. The publish workflow dispatches that build
when it uploads changed data; manual dispatches can pin a `revision`. Follow
[`docs/tested-models.md`](tested-models.md) to publish ground truth, reproduce
the build, and install the bundle with `configiq-deploy/deploy-tested-models.sh`.

Install the bundle independently on each host. The installer atomically
replaces `/var/lib/configiq/tested-models` and explicitly loads the generated
FIL models into Triton; do not rely on repository polling alone when Triton is
running in explicit model-control mode. The deployment-owned K2 and Qwen
models are separate from the generated bundle.

## Rollback

To roll back, deploy a previous tagged version (do so for all five images so
they stay in the version lockstep they shipped in):
```bash
podman pull ghcr.io/redhat-performance/configiq:X.Y.Z
```

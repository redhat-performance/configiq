# Release Process

## Overview

ConfigIQ uses semantic versioning with git tags to version the UI and its
container images. The image build workflows are stored in this repository under
`.github/workflows/`.

The deployment contract consists of four container images:

- `ghcr.io/redhat-performance/configiq` (Next.js webapp)
- `ghcr.io/redhat-performance/aisimulators` (GPU sizing / estimation API)
- `ghcr.io/redhat-performance/aicostings` (GPU / LLM pricing API)
- `ghcr.io/redhat-performance/tritonserver` (mixed FIL/vLLM inference server)

Keep the image versions compatible when releasing the frontend and services.

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
   A release tag builds matching tags for all four images:
   - `ghcr.io/redhat-performance/{configiq,aisimulators,aicostings,tritonserver}:X.Y.Z`
   - `ghcr.io/redhat-performance/{configiq,aisimulators,aicostings,tritonserver}:latest`

## Version Display in UI

The sidebar footer shows:
- **Version**: From git tag (via `NEXT_PUBLIC_GIT_VERSION`)
- **Commit**: Short git hash (via `NEXT_PUBLIC_GIT_COMMIT`)
- **Build time**: UTC timestamp (via `NEXT_PUBLIC_BUILD_TIME`)

These are injected at build time by `scripts/inject-build-metadata.js`.

## Container Tags

When the four images are released together, they carry the same tag set:

- **`latest`** - Most recent tagged release (release-tracking; the `.xyz` host follows this)
- **`dev`** - Latest commit from `main` (commit-tracking; the `.dev` host follows this)
- **`X.Y.Z`** - Specific version tags

The Triton image workflow is path-filtered for `services/tritonserver/**` and
`v0.30.0` is installed in an isolated image site directory. The image starts
from the standard `nvcr.io/nvidia/tritonserver:26.09-py3` image so the FIL
backend remains available, then adds the pinned Triton vLLM backend and vLLM
`0.30.0`. Build validation checks FIL, vLLM, and native K2 architecture support.

The two-host tag model (`.dev` → `:dev`, `.xyz` → `:latest`) is implemented in
the `configiq-deploy` repo; `podman auto-update` on each host polls whatever tag
its running containers use, so the per-host tag is the only lever.

## Container Registry

Published containers:

- https://github.com/redhat-performance/configiq/pkgs/container/configiq
- https://github.com/redhat-performance/configiq/pkgs/container/aisimulators
- https://github.com/redhat-performance/configiq/pkgs/container/aicostings
- https://github.com/redhat-performance/configiq/pkgs/container/tritonserver

## Bumping the simulation SDK

The `aisimulators` and `aicostings` services install the **aisimulate** SDK as a
single unified wheel from the Red Hat `redhat-performance/aisimulate` fork's
GitHub Release assets — not PyPI. The one wheel bundles the Rust-compiled core
and provides the `aisimulate` / `aisimulate_core` namespaces the services import
(`aisimulators` directly for sizing;
`aicostings` indirectly via `configiq.systems` for the GPU catalog). It is
referenced by **exact download URL** rather than `name==version`, because the
wheel is published only via the fork's Release (never PyPI); a direct URL forces
pip to install that exact artifact.

To move to a new SDK build, update the single `aisimulate @ …` wheel URL — keep
it **identical in both files** — in `[project.dependencies]`, then regenerate
each `uv.lock` (`uv lock`):

1. `services/aisimulators/pyproject.toml`
2. `services/aicostings/pyproject.toml`

The URL points at a specific `deploy-api-v<version>+<hash>` release asset (the
`+` in the tag is URL-encoded as `%2B`); never the rolling `deploy-api-latest`,
so rebuilds are reproducible. The wheel is produced by the fork's
`deploy-api-wheels.yml` workflow on its `deploy/api` branch.

> **Note:** the aisimulate wheel pulls in a large JAX/optimization dependency
> stack (jax, jaxlib, optax, google-vizier, tfp-nightly) as base dependencies,
> so both service images are substantially larger than under the previous
> two-wheel aiconfigurator 0.11/0.12 SDK.

## Deployment

Deployment is handled by the `configiq-deploy` repo (rootless Podman + systemd
quadlets + nginx on IBM Cloud VMs). The `.dev` hosts follow `:dev`; the `.xyz`
hosts follow `:latest`. The Triton quadlet is pinned to an immutable GHCR
digest, and the same image digest is used by the K2 prefetch script.

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
`tested-models.yml` workflow builds a sanitized bundle from the pinned
`TESTED_DATASET_REVISION` and uploads it as a workflow artifact. Follow
[`docs/tested-models.md`](tested-models.md) to publish ground truth, reproduce
the build, and install the bundle with `configiq-deploy/deploy-tested-models.sh`.

Install the bundle independently on each host. The installer atomically
replaces `/var/lib/configiq/tested-models` and explicitly loads the generated
FIL models into Triton; do not rely on repository polling alone when Triton is
running in explicit model-control mode. The deployment-owned K2 and Qwen
models are separate from the generated bundle.

## Rollback

To roll back, deploy a previous tagged version (do so for all four images so
they stay in the version lockstep they shipped in):
```bash
podman pull ghcr.io/redhat-performance/configiq:X.Y.Z
```

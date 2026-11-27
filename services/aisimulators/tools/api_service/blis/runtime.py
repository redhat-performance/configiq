"""Resolve the pinned BLIS runtime without accepting arbitrary executables."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .errors import BlisCatalogError

#: The BLIS build this image is pinned to. The image sets BLIS_VERSION from the
#: Containerfile's build arg; the literal is the value a source checkout gets, so
#: the reported version never silently drifts from the pin. Until a release
#: carries the commands the adapter reads, the image builds a pinned commit and
#: this is the pseudo-version `blis version` prints for it.
DEFAULT_BLIS_RELEASE = "0.19.2-0.20261127005804-694802f59d85"
#: The commit the pinned BLIS binary was built from. The binary embeds no
#: revision in the version string, so the image sets BLIS_COMMIT alongside
#: BLIS_VERSION; the literal mirrors the commit a source checkout gets.
DEFAULT_BLIS_COMMIT = "694802f59d852c3f1c4dfdac5dd2bd0249618642"


@dataclass(frozen=True)
class BlisRuntime:
    """The one deployment fact the adapter needs: which binary to run.

    The catalog, coefficient registry, and engine facts are all embedded in that
    binary and reported by it, so there is no data directory to configure and
    nothing can pair this binary with another build's data.
    """

    binary: str


def blis_release() -> str:
    """Return the BLIS version this image was built from.

    For a release, that is the tag without its `v`. For the pinned commit build
    it is the Go pseudo-version `blis version` prints, because no release tag
    contains that commit yet.
    """
    release = os.environ.get("BLIS_VERSION", "").strip()
    return release or DEFAULT_BLIS_RELEASE


def blis_commit() -> str | None:
    """Return the BLIS git commit this image was built from, if known.

    The binary embeds no revision. A build that overrides BLIS_VERSION without
    passing the matching BLIS_COMMIT must report no revision rather than the
    default build's commit.
    """
    commit = os.environ.get("BLIS_COMMIT", "").strip()
    if commit:
        return commit
    release = os.environ.get("BLIS_VERSION", "").strip()
    if release and release != DEFAULT_BLIS_RELEASE:
        return None
    return DEFAULT_BLIS_COMMIT


def resolve_runtime() -> BlisRuntime:
    raw_binary = os.environ.get("BLIS_BINARY") or shutil.which("blis")
    if not raw_binary:
        raise BlisCatalogError("BLIS binary is unavailable; set BLIS_BINARY or install blis on PATH")
    binary_path = Path(raw_binary).expanduser()
    if not binary_path.is_absolute():
        located = shutil.which(raw_binary)
        if not located:
            raise BlisCatalogError(f"BLIS_BINARY {raw_binary!r} is not on PATH")
        binary_path = Path(located)
    binary_path = binary_path.resolve()
    if binary_path.name not in {"blis", "blis.exe"}:
        raise BlisCatalogError("BLIS_BINARY must refer to the BLIS executable")
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        raise BlisCatalogError(f"BLIS_BINARY is not executable: {binary_path}")

    blis_home = os.environ.get("BLIS_HOME")
    if blis_home and not binary_path.is_relative_to(Path(blis_home).expanduser().resolve()):
        raise BlisCatalogError("BLIS_BINARY must be located inside BLIS_HOME")

    return BlisRuntime(binary=str(binary_path))

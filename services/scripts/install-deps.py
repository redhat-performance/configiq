#!/usr/bin/env python3
"""Install a service's runtime dependencies from its own pyproject.toml.

Shared by the service Containerfiles (aisimulators, aicostings) so each
service's [project.dependencies] list is the single source of truth. A
hand-copied duplicate in the Containerfile previously drifted from pyproject
and dropped a dependency (ijson), crash-looping the container at runtime.

configiq is the one dep filtered out here: it's the shared library, a path
dependency installed separately from the source copy at /src/configiq-py (there
is no `configiq` on PyPI to resolve). Everything else passes straight to pip,
including the aisimulate SDK + its Rust-compiled core — pyproject.toml pins
those by exact GitHub-Release download URL (`name @ https://…whl`) so pip installs
the fork's exact artifacts rather than the same-versioned, numpy-incompatible
packages NVIDIA ships to PyPI.

Runs under whichever interpreter invokes it (`sys.executable -m pip`), so it
targets the venv in aisimulators' image and the system Python in aicostings'.

Usage:
  python install-deps.py <path-to-pyproject.toml> [extra pip args ...]

Any extra argv (e.g. --find-links, --index-url) is forwarded to pip verbatim.
"""
import subprocess
import sys
import tomllib


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <pyproject.toml> [extra pip args ...]", file=sys.stderr)
        return 2

    pyproject_path = sys.argv[1]
    passthrough = sys.argv[2:]  # extra pip args (e.g. --index-url), forwarded as-is

    with open(pyproject_path, "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]

    # configiq is the path-dep installed from source elsewhere in the build.
    deps = [d for d in deps if not d.lower().startswith("configiq")]

    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", *passthrough, *deps]
    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0

"""GPU system ids and display names, sourced from the aisimulate SDK.

This is the single source of truth for "what GPU systems exist" and "what do we
call them" across services, so aisimulators and aicostings never drift.

The aisimulate SDK (`aisimulate_core`) is imported lazily inside the
functions rather than at module load: it ships a compiled, platform-specific
wheel, and not every consumer needs it at import time. Callers that use these
functions must declare the SDK as their own dependency. Everything here degrades
gracefully to an empty result if the SDK or its perf data is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def supported_systems() -> set[str]:
    """Return the set of GPU system ids known to the aisimulate SDK.

    Returns an empty set if the SDK is not importable.
    """
    try:
        from aisimulate_core.sdk.common import SupportedSystems
    except ImportError:
        logger.warning("aisimulate SDK not available; no supported systems")
        return set()
    return {str(s) for s in SupportedSystems}


def load_device_names_from_perf_data() -> dict[str, str]:
    """Load vendor GPU display names from the SDK's perf-data parquet files.

    Reads the `device` column (originally from torch.cuda.get_device_name()) from
    the SDK's bundled perf parquet files to get canonical vendor strings like
    "NVIDIA H200" or "NVIDIA A100-SXM4-80GB". Intended to be called once at API
    startup and cached.

    Degrades gracefully: if the SDK, its perf data, or pyarrow is unavailable,
    the returned map is empty and callers should fall back to the system id.
    """
    try:
        from aisimulate_core.sdk.perf_database import get_systems_paths
    except ImportError:
        logger.warning("aisimulate SDK not available; device display names unavailable")
        return {}

    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow not available; device display names unavailable")
        return {}

    device_names: dict[str, str] = {}
    for systems_root in get_systems_paths():
        data_dir = Path(systems_root) / "data"
        if not data_dir.exists():
            continue

        for sys_dir in data_dir.iterdir():
            if not sys_dir.is_dir():
                continue
            sys_id = sys_dir.name
            if sys_id in device_names:
                continue

            # Read the device name from the first available parquet file.
            for parquet_file in sys_dir.rglob("*_perf.parquet"):
                try:
                    table = pq.read_table(parquet_file, columns=["device"])
                    if len(table) > 0:
                        device_names[sys_id] = table.column("device")[0].as_py()
                        break
                except Exception:
                    logger.debug("could not read device name from %s", parquet_file)
                    continue

    logger.info("Loaded %d device display names from perf data", len(device_names))
    return device_names

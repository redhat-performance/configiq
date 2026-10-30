# SPDX-License-Identifier: Apache-2.0

"""Hardware purchase cost loader.

new_usd: curated YAML seed (manually updated quarterly).
used_usd: placeholder for eBay completed listings API (to be implemented).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

HARDWARE_COSTS_PATH = Path(__file__).parent.parent.parent.parent / "data" / "hardware-costs.yaml"


def load_hardware_costs() -> dict[str, dict[str, Any]]:
    if not HARDWARE_COSTS_PATH.exists():
        logger.warning("Hardware costs file not found: %s", HARDWARE_COSTS_PATH)
        return {}

    try:
        with open(HARDWARE_COSTS_PATH) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.error("Failed to load hardware costs: %s", e)
        return {}

    if not data or "systems" not in data:
        return {}

    result: dict[str, dict[str, Any]] = {}
    for system in data["systems"]:
        sid = system.get("id")
        if not sid:
            continue
        result[sid] = {
            "new_usd": system.get("new_usd"),
            "new_usd_low": system.get("new_usd_low"),
            "new_usd_high": system.get("new_usd_high"),
            "new_node_usd": system.get("new_node_usd"),
            "install_cost_node": system.get("install_cost_node"),
            "used_usd": system.get("used_usd"),
            "indicative": True,
            "source_label": system.get("source_label"),
            "source_url": system.get("source_url"),
            "source_date": system.get("source_date"),
            "tdp_watts": system.get("tdp_watts"),
            "gpus_per_node": system.get("gpus_per_node"),
        }

    logger.info("Hardware costs: loaded %d systems", len(result))
    return result

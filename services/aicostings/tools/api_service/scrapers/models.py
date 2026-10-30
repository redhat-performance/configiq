# SPDX-License-Identifier: Apache-2.0

"""Hosted LLM API pricing scrapers.

Two peer pricing sources, plus a curated override layer. Every model carries a
`source` field so consumers can trace where a price came from and filter by it:
  - OpenRouter API      (https://openrouter.ai/api/v1/models)  source="openrouter"
  - LiteLLM public cost map                                source="litellm"
  - Curated YAML overrides                                     source="override"

scrape_all_models() returns each source's catalog separately alongside a
`merged` view. The merged view is a union keyed by model id; on a duplicate id
the precedence is curated overrides > OpenRouter > LiteLLM. Overrides are
applied as *field-level patches*: an override overlays only the fields it
specifies onto the matching scraped record (so unspecified fields keep tracking
the live feeds), and an override for an id neither feed has is a full add. The
/models endpoint serves any single source or the merged view via ?source=.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import aiohttp
import yaml

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
CURATED_OVERRIDES_PATH = Path(__file__).parent.parent.parent.parent / "data" / "model-overrides.yaml"

TIER_KEYWORDS = {
    "fast": ["haiku", "flash", "mini", "luna", "fable", "small", "lite"],
    "frontier": ["opus", "sol", "ultra", "pro-max"],
}


def _classify_tier(model_id: str, model_name: str) -> str:
    lower = (model_id + " " + model_name).lower()
    for tier, keywords in TIER_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return tier
    return "balanced"


def _infer_provider(model_id: str) -> str:
    prefix = model_id.split("/")[0].lower() if "/" in model_id else ""
    mapping = {
        "anthropic": "Anthropic",
        "openai": "OpenAI",
        "google": "Google",
        "meta-llama": "Meta",
        "mistralai": "Mistral",
        "deepseek": "DeepSeek",
        "cohere": "Cohere",
        "groq": "Groq",
    }
    return mapping.get(prefix, prefix.title())


def _per_million(value: Any) -> float | None:
    """Normalize a token rate once; reject booleans, negatives and overflow."""
    if isinstance(value, bool):
        return None
    try:
        rate = float(value) * 1_000_000
    except (ValueError, TypeError, OverflowError):
        return None
    return round(rate, 4) if math.isfinite(rate) and rate >= 0 else None


def _context_window(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def _parse_openrouter_model(m: dict[str, Any]) -> dict[str, Any] | None:
    pricing = m.get("pricing", {})
    model_id = m.get("id")
    model_name = m.get("name")
    if not isinstance(pricing, dict) or not isinstance(model_id, str) or not model_id.strip():
        return None
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    price_per_m_input = _per_million(pricing.get("prompt"))
    price_per_m_output = _per_million(pricing.get("completion"))
    if price_per_m_input is None or price_per_m_output is None or not (price_per_m_input or price_per_m_output):
        return None

    checkpoint = m.get("hugging_face_id")
    if checkpoint is not None and not isinstance(checkpoint, str):
        return None

    return {
        "id": model_id,
        "name": model_name,
        "provider": _infer_provider(model_id),
        "tier": _classify_tier(model_id, model_name),
        "price_per_m_input": price_per_m_input,
        "price_per_m_output": price_per_m_output,
        "context_window": _context_window(m.get("context_length")),
        "source": "openrouter",
        **({"hugging_face_id": checkpoint.strip()} if checkpoint and checkpoint.strip() else {}),
    }


async def fetch_openrouter_models(session: aiohttp.ClientSession) -> list[dict[str, Any]]:
    # Raise on failure (rather than returning []) so scrape_all_models records
    # this source as errored, not as a successful empty scrape — mirroring
    # fetch_litellm_models. A silent [] would clear the scrape-error marker and
    # mark the source healthy while pricing quietly went stale.
    try:
        async with session.get(OPENROUTER_URL, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"OpenRouter returned {resp.status}")
            data = await resp.json()
    except Exception as e:
        logger.error("OpenRouter fetch failed: %s", e)
        raise

    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise TypeError("OpenRouter returned an invalid model catalogue")
    raw_models = data["data"]
    parsed = [result for m in raw_models if isinstance(m, dict) and (result := _parse_openrouter_model(m))]
    if not parsed:
        raise RuntimeError("OpenRouter returned no usable token-priced models")

    logger.info("OpenRouter: parsed %d models from %d total", len(parsed), len(raw_models))
    return parsed


def load_curated_overrides() -> list[dict[str, Any]]:
    if not CURATED_OVERRIDES_PATH.exists():
        return []
    try:
        with open(CURATED_OVERRIDES_PATH) as f:
            data = yaml.safe_load(f)
        return data.get("models", []) if data else []
    except Exception as e:
        logger.warning("Failed to load curated overrides: %s", e)
        return []


def _parse_litellm_model(model_id: str, m: dict[str, Any]) -> dict[str, Any] | None:
    provider = m.get("litellm_provider")
    if model_id == "sample_spec" or not model_id.strip() or m.get("mode") != "chat":
        return None
    if not isinstance(provider, str) or not provider.strip():
        return None
    price_per_m_input = _per_million(m.get("input_cost_per_token"))
    price_per_m_output = _per_million(m.get("output_cost_per_token"))
    if price_per_m_input is None or price_per_m_output is None or not (price_per_m_input or price_per_m_output):
        return None

    return {
        "id": model_id,
        "name": model_id.split("/")[-1] if "/" in model_id else model_id,
        "provider": provider.strip().title(),
        "tier": _classify_tier(model_id, model_id),
        "price_per_m_input": price_per_m_input,
        "price_per_m_output": price_per_m_output,
        "context_window": _context_window(m.get("max_input_tokens")),
        "source": "litellm",
    }


async def fetch_litellm_models(session: aiohttp.ClientSession) -> list[dict[str, Any]]:
    """Fetch standard chat token rates from LiteLLM's public, keyed cost map."""
    try:
        async with session.get(LITELLM_URL, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"LiteLLM returned {resp.status}")
            # raw.githubusercontent.com serves JSON files as text/plain.
            data = await resp.json(content_type=None)
    except Exception as e:
        logger.error("LiteLLM fetch failed: %s", e)
        raise
    if not isinstance(data, dict):
        raise TypeError("LiteLLM returned an invalid cost map")

    parsed: list[dict[str, Any]] = []
    for model_id, m in data.items():
        if isinstance(m, dict) and (result := _parse_litellm_model(model_id, m)):
            parsed.append(result)
    if not parsed:
        raise RuntimeError("LiteLLM returned no usable token-priced chat models")
    logger.info("LiteLLM: parsed %d models from %d entries", len(parsed), len(data))
    return parsed


# catalogs keyed by source ("openrouter", "litellm", "merged"); each value is a
# model list. errors maps the scrape source name → Exception (or None).
ModelScrapeResult = tuple[dict[str, list[dict[str, Any]]], dict[str, Exception | None]]


async def scrape_all_models(session: aiohttp.ClientSession) -> ModelScrapeResult:
    """Scrape model pricing from OpenRouter and LiteLLM (peer sources).

    Returns:
        catalogs: {"openrouter": [...], "litellm": [...], "merged": [...]} —
            each model tagged with its `source`. The merged view is a union
            keyed by id (precedence: overrides > OpenRouter > LiteLLM), with
            curated overrides applied as field-level patches.
        errors: mapping of source name → Exception (or None if successful)
    """
    import asyncio

    or_task = asyncio.create_task(fetch_openrouter_models(session))
    lt_task = asyncio.create_task(fetch_litellm_models(session))

    errors: dict[str, Exception | None] = {}
    or_models: list[dict[str, Any]] = []
    lt_models: list[dict[str, Any]] = []

    for source, task in [("models.openrouter", or_task), ("models.litellm", lt_task)]:
        try:
            result = await task
            errors[source] = None
            if source == "models.openrouter":
                or_models = result
            else:
                lt_models = result
        except Exception as e:
            logger.error("%s fetch failed: %s", source, e)
            errors[source] = e

    overrides = load_curated_overrides()

    # Merged view: union keyed by id. LiteLLM first, then OpenRouter overwrites
    # on collision (OpenRouter wins over LiteLLM), then curated overrides win
    # over both — applied as field-level patches so an override overlays only the
    # fields it specifies and unspecified fields keep tracking the live feeds. An
    # override whose id neither feed has is a full add.
    merged_by_id: dict[str, dict[str, Any]] = {}
    for m in lt_models:
        merged_by_id[m["id"]] = m
    for m in or_models:
        merged_by_id[m["id"]] = m
    for o in overrides:
        oid = o.get("id")
        if not oid:
            continue
        base = merged_by_id.get(oid)
        patched = {**base, **o} if base else dict(o)
        patched["source"] = "override"
        merged_by_id[oid] = patched

    merged = sorted(merged_by_id.values(), key=lambda m: m["id"])

    logger.info(
        "Models merged: %d total (OpenRouter: %d, LiteLLM: %d, overrides: %d)",
        len(merged), len(or_models), len(lt_models), len(overrides),
    )
    return {"openrouter": or_models, "litellm": lt_models, "merged": merged}, errors

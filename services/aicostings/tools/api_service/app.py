# SPDX-License-Identifier: Apache-2.0

"""aicostings REST API.

GPU and LLM pricing API for AI infrastructure cost modelling.
See docs/api/openapi.yaml for the full spec.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import aiohttp
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from configiq.systems import load_device_names_from_perf_data, supported_systems
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .scrapers.cloud_rates import scrape_all_cloud_rates
from .scrapers.hardware_costs import load_hardware_costs
from .scrapers.models import LITELLM_URL, OPENROUTER_URL, scrape_all_models
from .valkey import ValkeyStore

# Optional observability + MCP, provided by the shared configiq package
# (configiq[otel,mcp]). Kept out of the base install/container because otel
# tracing opens an OTLP exporter eagerly at import; every use is guarded so the
# app runs unchanged when the extras are absent.
try:
    from configiq import observability
    _OBS = True
except ImportError:
    _OBS = False

try:
    from configiq import mcp as mcp_support
    _MCP = True
except ImportError:
    _MCP = False

logger = logging.getLogger(__name__)

store = ValkeyStore()

# GPU vendor display names, keyed by system id, loaded once at startup from the
# aisimulate SDK (the same source the aisimulators API uses) so the two
# services never drift. Falls back to the system id when a name is unavailable.
_DEVICE_DISPLAY_NAMES: dict[str, str] = {}


# ── Domain metrics (scrape health) ────────────────────────────────────────────
#
# aicostings is a scheduled scraper/cache, so the signals that matter most are
# per-source scrape health and freshness, not just request throughput. These
# instruments are created after observability is enabled; record_scrape() no-ops
# until then, so the scrape jobs can call it unconditionally.
_scrape_total = None
_scrape_duration_seconds = None
_scrape_records = None


def _init_scrape_metrics() -> None:
    global _scrape_total, _scrape_duration_seconds, _scrape_records
    meter = observability.get_meter("aicostings.scrape")
    _scrape_total = meter.create_counter(
        name="costings.scrape.total",
        description="Total scrape attempts, labelled by source and result",
        unit="1",
    )
    _scrape_duration_seconds = meter.create_histogram(
        name="costings.scrape.duration",
        description="Scrape duration per source",
        unit="s",
    )
    _scrape_records = meter.create_counter(
        name="costings.scrape.records",
        description="Records stored per successful scrape, labelled by source",
        unit="1",
    )


def record_scrape(source: str, result: str, duration: float, records: int = 0) -> None:
    """Record the outcome of a scrape job (no-op until metrics are initialized)."""
    if not _scrape_total:
        return
    _scrape_total.add(1, {"source": source, "result": result})
    _scrape_duration_seconds.record(duration, {"source": source})
    if result == "success" and records:
        _scrape_records.add(records, {"source": source})


# ── Scrape jobs ───────────────────────────────────────────────────────────────


async def job_scrape_models() -> None:
    logger.info("Scraping hosted model pricing (OpenRouter + LiteLLM)...")
    start = time.monotonic()
    async with aiohttp.ClientSession() as session:
        catalogs, source_errors = await scrape_all_models(session)
    # Store each source's catalog separately (openrouter/litellm/merged) so the
    # /models endpoint can serve one feed or the merged view. Skip empties so a
    # failed source doesn't clobber its last-good data.
    for source, catalog in catalogs.items():
        if catalog:
            store.set_models(catalog, source=source)
    for source, error in source_errors.items():
        if error is None:
            store._mark_scrape(source)
        else:
            store.set_scrape_error(source, str(error))
    merged = catalogs.get("merged", [])
    error_count = sum(1 for e in source_errors.values() if e is not None)
    logger.info("Stored %d models (%d source errors)", len(merged), error_count)
    # Partial data is still useful, so "success" means we stored something.
    record_scrape("models", "success" if merged else "error",
                  time.monotonic() - start, records=len(merged))


async def job_scrape_cloud_rates() -> None:
    logger.info("Scraping cloud GPU rates...")
    start = time.monotonic()
    async with aiohttp.ClientSession() as session:
        rates_by_system, provider_errors = await scrape_all_cloud_rates(session)
    for system_id, rates in rates_by_system.items():
        store.set_cloud_rates(system_id, {
            "cloud_rates": rates,
            "rates_updated_at": datetime.now(UTC).isoformat(),
            "rates_stale": False,
        })
    for provider, error in provider_errors.items():
        source_key = f"cloud.{provider}"
        if error is None:
            store._mark_scrape(source_key)
        else:
            store.set_scrape_error(source_key, str(error))
    error_count = sum(1 for e in provider_errors.values() if e is not None)
    logger.info("Stored cloud rates for %d systems (%d provider errors)",
                len(rates_by_system), error_count)
    record_scrape("cloud_rates", "success" if rates_by_system else "error",
                  time.monotonic() - start, records=len(rates_by_system))


async def job_load_hardware_costs() -> None:
    logger.info("Loading hardware costs from seed data...")
    start = time.monotonic()
    costs = load_hardware_costs()
    for system_id, cost in costs.items():
        store.set_hardware_cost(system_id, {
            **cost,
            "hardware_updated_at": datetime.now(UTC).isoformat(),
        })
    # Hardware costs are loaded from seed data rather than scraped, but still
    # record a freshness timestamp so /health reports the source (and flags it
    # stale if the 7-day reload ever stops running).
    if costs:
        store._mark_scrape("hardware_costs")
    else:
        store.set_scrape_error("hardware_costs", "no hardware cost seed data found")
    logger.info("Loaded hardware costs for %d systems", len(costs))
    record_scrape("hardware_costs", "success" if costs else "error",
                  time.monotonic() - start, records=len(costs))


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.connect()

    # Load GPU display names from the aisimulate SDK once at startup.
    global _DEVICE_DISPLAY_NAMES
    _DEVICE_DISPLAY_NAMES = load_device_names_from_perf_data()

    # Unconditionally refresh all datasets on startup. This ensures fresh data
    # after every container restart, avoiding edge cases where data TTL expires
    # between restart and scheduler's first job run. Startup latency (30-40s)
    # is acceptable; restarts are infrequent. Failures are logged but don't
    # block startup — the scheduler will retry within 24h and /health reports staleness.
    try:
        await job_scrape_models()
    except Exception as e:
        logger.error("Startup scrape_models failed: %s", e)
    try:
        await job_scrape_cloud_rates()
    except Exception as e:
        logger.error("Startup scrape_cloud_rates failed: %s", e)
    try:
        await job_load_hardware_costs()
    except Exception as e:
        logger.error("Startup load_hardware_costs failed: %s", e)

    # Start scheduler for periodic refreshes
    scheduler = AsyncIOScheduler()
    scheduler.add_job(job_scrape_models, IntervalTrigger(hours=24), id="scrape_models")
    scheduler.add_job(job_scrape_cloud_rates, IntervalTrigger(hours=24), id="scrape_cloud_rates")
    scheduler.add_job(job_load_hardware_costs, IntervalTrigger(days=7), id="load_hardware_costs")
    scheduler.start()
    logger.info("Scheduler started")

    yield

    scheduler.shutdown()
    store.close()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="aicostings API",
    description="GPU and LLM pricing API for AI infrastructure cost modelling.",
    version="0.1.0",
    lifespan=lifespan,
    default_response_class=JSONResponse,
)

# Initialize OpenTelemetry (tracing + metrics + middleware) and domain metrics if
# the optional extra is present.
if _OBS:
    observability.enable(app, service_name="aicostings", service_version="0.1.0",
                         meter_name="aicostings.api")
    _init_scrape_metrics()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

# Expose the API as MCP tools if the optional extra is present.
if _MCP:
    mcp_support.mount(app, name="aicostings",
                      description="GPU and LLM pricing for AI infrastructure cost modelling")
else:
    logger.info("MCP server unavailable (install with: pip install '.[mcp]')")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_include(include: str | None) -> set[str]:
    if not include:
        return set()
    return {s.strip().lower() for s in include.split(",")}


def _stale_flag(source: str, max_age_seconds: int = 25 * 3600) -> bool:
    return store.is_stale(source, max_age_seconds)


# ── Endpoints ─────────────────────────────────────────────────────────────────


# Hosted-model pricing feeds, with display metadata so clients don't hardcode
# labels. 'merged' is the synthetic default (a union of all feeds, keyed by
# model id, with curated overrides applied). This tuple is the single source of
# truth for which ?source= values are valid — GET /sources exposes it so the
# frontend selector is API-driven and follows feeds added or retired here.
#
# 'url' is the exact upstream endpoint each feed is scraped from (reused from
# the scraper module so it can never drift), useful for diagnostics/attribution;
# 'merged' has no single URL.
MODEL_SOURCES: tuple[dict[str, str | None], ...] = (
    {
        "id": "merged",
        "label": "Merged",
        "description": "All feeds, deduplicated by model id with curated overrides applied",
        "url": None,
    },
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "description": "OpenRouter pricing feed only",
        "url": OPENROUTER_URL,
    },
    {
        "id": "litellm",
        "label": "LiteLLM",
        "description": "LiteLLM model catalog feed only",
        "url": LITELLM_URL,
    },
)
_MODEL_SOURCE_IDS = tuple(s["id"] for s in MODEL_SOURCES)


@app.get("/sources")
def get_sources():
    """List the available hosted-model pricing feeds with display metadata.

    Clients use this to build the pricing-source selector instead of hardcoding
    feed ids and labels; the list follows feeds added or retired in
    MODEL_SOURCES with no client change.
    """
    return {"sources": [dict(s) for s in MODEL_SOURCES]}


@app.get("/models")
def get_models(
    source: str = Query(
        default="merged",
        examples=["merged", "openrouter", "litellm"],
        description="Pricing source: merged (default), openrouter, or litellm.",
    ),
):
    """List hosted LLM API pricing ($/M tokens).

    Includes both proprietary frontier models (Claude, GPT, Gemini) and
    open-weight models available via inference APIs (Llama, Qwen, Mistral,
    DeepSeek). OpenRouter and LiteLLM are peer feeds; every model is tagged with
    its `source`. Use ?source= to select a single feed or the default merged
    view (a union keyed by model id, with curated overrides applied as
    field-level patches).
    """
    source = source.lower()
    if source not in _MODEL_SOURCE_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"invalid source '{source}'; expected one of {', '.join(_MODEL_SOURCE_IDS)}",
        )

    models = store.get_models(source)
    if models is None:
        return JSONResponse(
            {"models": [], "source": source, "stale": True, "updated_at": None},
            status_code=200,
        )

    scrape_times = store.get_scrape_times()
    if source == "merged":
        # merged is fresh if either feed is fresh (partial data is still useful)
        stale = _stale_flag("models.openrouter") and _stale_flag("models.litellm")
        updated_at = scrape_times.get("models.openrouter") or scrape_times.get("models.litellm")
    else:
        stale = _stale_flag(f"models.{source}")
        updated_at = scrape_times.get(f"models.{source}")
    return {
        "models": models,
        "source": source,
        "updated_at": updated_at,
        "stale": stale,
    }


@app.get("/systems")
def get_systems(
    include: str | None = Query(
        default=None,
        examples=["cloud", "hardware", "cloud,hardware"],
        description="Comma-separated extras: cloud, hardware.",
    ),
):
    """List GPU system pricing."""
    includes = _parse_include(include)
    want_cloud = "cloud" in includes
    want_hardware = "hardware" in includes

    cloud_data = store.get_all_cloud_rates() if want_cloud else {}
    hardware_data = store.get_all_hardware_costs() if want_hardware else {}

    all_ids = supported_systems() | set(cloud_data.keys()) | set(hardware_data.keys())

    systems: list[dict[str, Any]] = []
    for sid in sorted(all_ids):
        entry: dict[str, Any] = {
            "id": sid,
            "name": _DEVICE_DISPLAY_NAMES.get(sid, sid),
        }
        if want_cloud and sid in cloud_data:
            entry.update(cloud_data[sid])
        if want_hardware and sid in hardware_data:
            hw = hardware_data[sid]
            entry["hardware_cost"] = {
                "new_usd": hw.get("new_usd"),
                "new_usd_low": hw.get("new_usd_low"),
                "new_usd_high": hw.get("new_usd_high"),
                "used_usd": hw.get("used_usd"),
                "indicative": True,
                "source_label": hw.get("source_label"),
                "source_url": hw.get("source_url"),
                "source_date": hw.get("source_date"),
            }
            entry["tdp_watts"] = hw.get("tdp_watts")
            entry["gpus_per_node"] = hw.get("gpus_per_node")
            entry["hardware_updated_at"] = hw.get("hardware_updated_at")

        systems.append(entry)

    return {"systems": systems}


@app.get("/health")
def get_health():
    """Service health and per-source data freshness.

    stale: true when last successful scrape exceeds the expected interval.
    last_error: populated when the most recent scrape attempt failed.
    """
    scrape_times = store.get_scrape_times()
    scrape_errors = store.get_scrape_errors()
    sources = {}
    # Union of both key spaces so a source that has only ever errored (no
    # successful scrape yet) still shows up as unhealthy rather than vanishing.
    for source in sorted(set(scrape_times) | set(scrape_errors)):
        max_age = 7 * 24 * 3600 if source == "hardware_costs" else 25 * 3600
        sources[source] = {
            "last_success": scrape_times.get(source),
            "last_error": scrape_errors.get(source),
            "stale": store.is_stale(source, max_age),
        }

    return {
        "status": "ok",
        "version": "0.1.0",
        "sources": sources,
    }


@app.get("/metrics")
def get_metrics(request: Request):
    """Expose metrics in Prometheus or OTLP JSON format via content negotiation.

    - Accept: application/json → OTLP JSON (for pmdaopentelemetry and similar)
    - anything else (default)  → Prometheus/OpenMetrics text
    """
    if not _OBS:
        raise HTTPException(
            status_code=503,
            detail="Metrics unavailable (install with: pip install '.[otel]')",
        )
    return observability.metrics_response(request.headers.get("accept", "text/plain"))


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8080)

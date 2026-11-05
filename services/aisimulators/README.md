# AISimulators REST API

A thin FastAPI wrapper over the [aisimulate](https://github.com/redhat-performance/aisimulate)
SDK, exposing GPU recommendation, performance estimation, and memory estimation
as HTTP endpoints. ConfigIQ's `app/api/*` proxy routes call this service.

This directory contains **only** the REST wrapper. The SDK itself is not vendored
here — it is installed as a single wheel published by the Red Hat fork's GitHub
Releases. That one `aisimulate` wheel bundles the Rust-compiled core and exposes
the `aisimulate` and `aisimulate_core` APIs used by this service.

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/recommend` | GPU sizing recommendations |
| POST | `/predict` | Single-point performance prediction |
| POST | `/estimate` | Deprecated alias for `/predict` |
| POST | `/memory` | Memory / KV-cache breakdown |
| GET | `/models` | Supported models (per `simulator`) |
| GET | `/systems` | Supported GPU systems (per `simulator`) |
| GET | `/backends` | Backend/engine versions and memory defaults (per `simulator`) |
| GET | `/networks` | BLIS network fabrics |

Full spec: [`docs/api/openapi.yaml`](docs/api/openapi.yaml).

## Run with the container (recommended)

The `aisimulate` wheel ships only for `linux/amd64` (manylinux x86_64), so the
container is the portable way to run this service. The build context is
`services/` (not this directory) so the image can pull in the shared
`configiq-py` package; pass the Containerfile with `-f` and build for
`linux/amd64` (emulated on Apple Silicon):

```bash
docker build --platform linux/amd64 \
  -f services/aisimulators/Containerfile -t aisimulators services/
docker run --rm -p 7860:7860 aisimulators
curl http://localhost:7860/systems
```

The image also exposes the simulator CLIs at stable paths:

```text
/usr/local/bin/aisimulate
/usr/local/bin/blis
```

BLIS v0.14.0 is built from source in the multi-stage build (commit
`f2ba6bb1c189ed5a56933cfa3d0927eae3836ee9`) rather than copied from an image
digest. Upstream never published a GHCR image for v0.12.0 through v0.14.0: its
own `Dockerfile` names a Go base older than `go.mod` requires, so the build
failed at the tag. Building here keeps the Go base under our control
(`BLIS_GO_VERSION`). The image bundles the two catalog inputs a BLIS run reads,
both at pinned commits:

| Path | Contents | Pin |
|---|---|---|
| `/opt/blis/catalog` | declared facts: model graphs, chips, fabrics, workloads | blis-catalog `0.3.0`, `676326ec6c708e91ec4981bd6b03922a7fe60512` |
| `/opt/blis/registry` | fitted coefficient sets a scenario names | blis-registry `v0.2.0`, `7f6662d7188dc3dc873a37623e1f7bbfe1d6081b` |

No example-scenario directory is bundled: the adapter generates a Scenario and
Deployment per request, so those files would be dead weight.

The build then runs [`tools/export_blis_catalog.py`](tools/export_blis_catalog.py)
to augment the wheel's packaged AISimulate model configs into that catalog:

- **The upstream catalog always wins a shared name.** AISimulate can only add
  models the catalog does not already define, so a reviewed BLIS graph is never
  silently replaced.
- **An added model must earn its place.** The exporter runs blis-catalog's own
  `scripts/derive_graph.py` for every added model and keeps it only if a
  `graph.yaml` comes out. A model BLIS cannot price is removed and recorded.
- **Nothing is dropped silently.** Unsupported architectures, name collisions and
  malformed configs are recorded in `/opt/blis/support.json` with a stable reason
  code, so the API can explain an unavailable model instead of returning a generic
  error.

`/opt/blis/support.json` is the API contract for simulator selection. It records,
per catalog model, whether it came from upstream or AISimulate, whether it has a
graph, its quantization, how an AISimulate config relates to the upstream entry of
the same name (`identical`, `different_quantization`, `different`, with the
differing paths), and any `equivalent_catalog_name`. It also lists the skipped
AISimulate models with reasons, the bundled hardware and fabrics, and the exact
`aisimulate` wheel (`aisimulate_wheel`) whose configs were used. Derived
`model.yaml` files name that release build as their `revision`, so the manifest
and the catalog agree on provenance.

Because the upstream catalog wins, BLIS prices the catalog's reviewed config even
when the wheel's copy differs. For example, upstream `glm-5` is the FP8 variant
while AISimulate's `zai-org/GLM-5` is the unquantized base; the manifest marks that
shared entry `different_quantization` and records the FP8 AISimulate model
separately as `glm-5-fp8` with `equivalent_catalog_name: glm-5`.

## Simulator API behavior

`POST /predict` and `POST /recommend` accept `simulator: "aisimulate" | "blis"`;
the default remains `aisimulate`. A BLIS request is gated by `support.json` and
uses the manifest's catalog model and hardware identities. The service generates
a temporary Scenario + Deployment for the requested configuration and passes it
to `blis run` with the image-pinned catalog and registry. Unsupported models,
systems, or settings return an explicit client error. A BLIS failure is never
retried through AISimulate.

## AISimulate backend versions

`backend_version` is optional. When it is omitted, the service resolves it to the
system default — the SDK's `current` slot, the same value `/backends` reports in
`system_defaults`. The `current` alias means the same thing and behaves the same
way; `previous` and `next` name specific slots and are treated as pins. A model
can be absent from the default version's performance data while a newer version
covers it, so if the default cannot serve the request the service retries the
newest declared versions (at most three attempts) and reports the version it used
in `backend_version` and `provenance.engine_version`, plus a `warnings` entry
(`/predict`) or a warning on each returned config (`/recommend`). The resolved
fallback is cached per request shape — model, system, backend, and workload — so
a fallback for one shape never skips a default that covers another.

An explicit `backend_version` is never substituted. If the performance data
cannot serve it, `/predict`, `/estimate`, and `/recommend` return 422 and name
the versions the SDK ships for that system, so the caller can retry or omit the
field. The same 422 class covers a compute shape the selected data never
measured (`EmpiricalNotImplementedError` inside the replay engine), which
previously surfaced as a generic 500.

### MoE / FFN width resolution

AISimulate validates that `tp_size * attention_dp_size * cp_size` equals
`moe_tp_size * moe_ep_size` for model families that can host MoE layers,
including dense variants such as dense Gemma 4. The service therefore resolves
an omitted width pair per worker to
`(tp_size * attention_dp_size, 1)`, meaning the FFN or expert block is
tensor-sharded across the attention group with no expert parallelism. This
applies to dense and MoE models alike.

An explicit pair is passed through unchanged. If only one width is supplied,
the other is inferred when it divides the attention width. Disaggregated
requests resolve each pool independently from `prefill_moe_tp_size` /
`prefill_moe_ep_size` and `decode_moe_tp_size` / `decode_moe_ep_size`, falling
back to the top-level pair, so pools with different TP widths receive matching
pairs. The response reports the resolved layout: `moe_tp` / `moe_ep` for an
aggregated prediction, and `prefill_config.moe_tp` / `decode_config.moe_tp`
(plus the matching `_ep` fields) for a disaggregated one. Clients should send
only widths they explicitly chose and display the returned values rather than
reconstructing the default. BLIS has no width invariant: it receives only the
widths the caller supplied, and its aggregated response reports those explicit
values (or null).

BLIS models three vLLM releases: `0.29.0`, `0.31.0`, and `0.32.0`. The default is
`0.31.0`, the newest release that models prefill/decode disaggregation and
accepts `max_model_len: auto`. `0.32.0` is opt-in and colocated-only: a P/D
request returns 422, and the adapter states `max_model_len = isl + osl` instead of
`auto`. Any other `backend_version`, including AISimulate aliases such as
`current`, is refused with a 422 that names the modeled releases — BLIS treats
`engine_version` as a check, not a selector, so the service never substitutes a
different release silently.

`GET /backends`, `/models`, and `/systems` take `simulator` so a client can ask
what the selected simulator supports. `/backends?simulator=blis` reports the
modeled versions, the literal default, and a `version_support` map
(`disaggregated`, `max_model_len_auto`) per release. `/models` and `/systems`
with the same parameter return only the AISimulate IDs and systems BLIS can
price. The envelopes include `simulator`; `/backends` also reports
`blis_version`. Each backend entry reports
`disaggregation_requires_fabric`, so a client knows whether a P/D topology
needs a fabric selection without matching on the simulator ID.

Simulator provenance includes the requested and resolved model, hardware,
engine version, and any model-configuration relation reported by the selected
simulator. Non-identical model configs are surfaced in the response so clients
can label the comparison. BLIS predictions
do not include KV-cache memory breakdowns; the single-backend `/memory` endpoint
continues to use AISimulate and reports `simulator: "aisimulate"`.

BLIS `/predict` supports both colocated and prefill/decode-disaggregated vLLM
scenarios. P/D requests require a `fabric` from `GET /networks`, the same GPU
system for both pools, and `attention_dp_size=1`. The adapter generates separate
role pools, a NIXL transfer connector, and a router configuration that
disaggregates every request. BLIS `/recommend` always searches aggregate TP
candidates. It defaults to `roce-200g` when the `fabric` field is omitted and
also searches bounded P/D TP and worker-count combinations, with one node per
pool and DP=PP=1. Send `fabric: null` to request aggregate-only candidates; the
response then warns that P/D candidates were not searched.

BLIS recommendation sizes closed-loop concurrency by default. With
`objective=throughput_demand` `sla_mode=none` it instead replays open-loop
traffic: each candidate runs `blis run --rate <target_request_rate>` with
`num_requests = max(4, round(rate x 5 s))`, so the request count scales with the
offered rate rather than a fixed concurrency slot. `BLIS_DEMAND_WINDOW_SECONDS`
(default 5) and `BLIS_DEMAND_MAX_REQUESTS` (default 5000) bound that slice. The
achieved rate is measured over the arrival window:
`completed_requests / (vllm_estimated_duration_s - e2e_mean_ms / 1000)`, falling
back to `responses_per_sec` (then `tokens_per_sec / osl`) when duration is
missing. BLIS times a run to its last completion, so that subtraction removes
the final response drain and keeps the measured rate comparable to the offered
rate. A candidate is compliant only when `dropped_unservable == 0` and
`completed_requests >= num_requests`; when none comply, the response is 200 with
`demand.met=false` and the best observed rate. Demand searches are
aggregate-only, and the scheduler cap equals the run's request count, which is a
run bound rather than a modeled production queue. Any latency target on a
demand request is rejected with 422.

BLIS recommendation's candidate ceiling is the same
`AISIMULATORS_MAX_CANDIDATE_GPUS` service setting used by AISimulate, further
limited by one node for aggregate-only search or two nodes (one per P/D pool)
when a fabric is active. A total search budget bounds the individual simulations. The service
caps that budget at 60 seconds; the web
gateway may lower it further to leave room before its request deadline. The
response reports the configured, hardware, and effective ceilings plus the
actual candidate and evaluated GPU counts.

## Recommendation objectives

`POST /recommend` takes `objective` and `sla_mode`. The pair is validated
strictly; unsupported combinations return 422 rather than a silently relaxed
search.

| `objective` | `sla_mode` | Load input | Result |
|---|---|---|---|
| `min_gpus` (default) | `enforce` | exactly one of `target_request_rate` / `target_concurrency` | fewest GPUs that clears the latency SLA |
| `throughput_demand` | `none` | `target_request_rate` only | fewest GPUs among evaluated candidates that sustains the offered rate; no latency target |
| `pareto` | `enforce` or `none` | none; the sweep drives load itself | every sampled candidate plus a frontier view |

With `sla_mode=enforce`, send either `request_latency` or both `ttft` and
`tpot`. A demand request (`throughput_demand`) rejects `ttft`, `tpot`, and
`request_latency`; the service never accepts a bound the search does not apply.
`pareto` with `sla_mode=none` sweeps capacity without an SLA, while
`sla_mode=enforce` keeps only SLA-compliant candidates on the frontier.

### Throughput-demand sizing

A demand result carries a `demand` block whose values come from the selected
candidate's replay report, not from the requested target:

- `met` — the selected candidate sustained `target_rps` within a 0.1% tolerance.
- `achieved_rps` — request rate sustained over the arrival window. A replay is
  timed to its final completion, so `completed / duration` appends one response
  latency; `achieved_rps` uses `completed / (duration - mean_e2e)` instead.
- `best_observed_rps` — highest sustained rate seen across evaluated candidates;
  the selected value when the target is unmet.
- `observed_concurrency` — Little's-law estimate
  (`achieved_rps x mean_e2e_latency_ms / 1000`), omitted when the run completed
  no work.
- `window_seconds` — arrival window (`(duration - mean_e2e) / 1000`), excluding
  the final response drain.
- `evaluated_configurations` — candidates with a usable rate measurement.

Selection is: among compliant candidates, fewest GPUs, then highest sustained
rate, then lowest E2E latency. When none comply, the highest sustained rate wins
(then fewest GPUs) and the response is still 200 with `met=false`. A request
that produces no feasible candidate with a rate measurement returns 422.

AISimulate demand runs use `DEMAND_MAX_TRIALS` (default 16),
`DEMAND_PARALLELISM` (default 4), `DEMAND_CANDIDATE_TIMEOUT_S` (default 15) and
`DEMAND_REQUESTS_PER_LOAD_UNIT` (default 5). The SDK config uses open-loop
`constant_rate` traffic and stops after `requests_per_load_unit` units, so the
replay length grows with the requested rate. The trial budget is split across
the searched deployment modes; it is a total, not a per-mode count.

`min_gpus` is rejected without an SLA, so the SDK cannot order a demand search
smallest-first. The service therefore walks a growing GPU window, starting at
`DEMAND_WINDOW_START_GPUS` (default 8) and doubling until the offered rate is
met, the effective ceiling is reached, or the whole-search budget
`DEMAND_SEARCH_BUDGET_S` (default 50 seconds) is spent. A window that cannot
hold the model is skipped. An explicit `max_candidate_gpus` pins a single
window. Each window's trial/timeout budget is reduced to the time that remains,
so the ladder stays inside the gateway timeout.

### Pareto frontier provenance and availability

`objective=pareto` is AISimulate-only. BLIS returns 422 and the service never
falls back to AISimulate, because a frontier computed by another simulator would
be misattributed.

For a Pareto result:

- `candidates` contains every evaluated candidate, including infeasible,
  unsupported, timed-out, failed, and resource-limited records. SDK-provided raw
  fields such as `topology`, `metrics`, and `objectives` are copied verbatim;
  the service does not synthesize topology, transfer, or performance values.
  Service-derived fields are added alongside them: `mode`, `provenance_sources`,
  and `transfer_delay_modeled` are computed (or combined from request and
  candidate data) by the service, not passed through from the SDK.
- `views.pareto_front` and `views.top_n` are SDK-computed candidate-ID lists.
  `pareto_front` is authoritative for the simulator's own objectives
  (throughput per user vs throughput per GPU). Clients may recompute dominance
  only for a different axis or orientation, and must say so in the UI.
- `transfer_delay_modeled` is true only when a P/D transfer delay was actually
  modeled for that candidate. Omitting `kv_transfer_bandwidth_gb_per_second`
  means no modeled transfer delay; it does not imply a fabric default.
- `sweep` reports the algorithm, seed, trial budget, evaluated/feasible/timed-out
  counts, and the GPU window. `sweep.budget_exhausted=true` means the trial
  budget was spent; it never means the search was exhaustive. BLIS enumerates
  its candidate grid rather than sampling it, so its `sweep.algorithm` is
  `enumerate`.

Pareto sweeps use `PARETO_MAX_TRIALS` (default 20), `PARETO_PARALLELISM`
(default 8), `PARETO_CANDIDATE_TIMEOUT_S` (default 8), and
`PARETO_MAX_CONCURRENCY` (default 2 concurrent sweeps per worker, returning 429
when exceeded). `PARETO_ALGORITHM` selects `random` (default) or `bayesian`; the
Bayesian sampler enables JAX x64 process-wide and is not the default. `SWEEP_SEED`
defaults to 42. A one-GPU total window runs one aggregate trial, since it cannot
contain a P/D deployment.

The image sets `BLIS_VERSION`, `BLIS_CATALOG`, `BLIS_REGISTRY` and
`BLIS_SUPPORT`. `BLIS_VERSION` is the release reported by `/backends` and in
every prediction's provenance. BLIS itself only reads `BLIS_CATALOG` from the
environment; the registry has no environment fallback, so callers must pass
`--registry` explicitly as well:

```bash
blis run --catalog "$BLIS_CATALOG" --registry "$BLIS_REGISTRY" \
  --scenario generated-scenario.yaml \
  --rate 10 --num-requests 100
```

To move to a new BLIS release, bump `BLIS_COMMIT`, `BLIS_VERSION`,
`BLIS_CATALOG_COMMIT`, `BLIS_REGISTRY_COMMIT` and (if `go.mod` requires it)
`BLIS_GO_VERSION` together; update `MODELED_ENGINE_VERSIONS` in
`tools/api_service/blis/scenario.py` if that release changes which vLLM versions
it models. See
[docs/RELEASE_PROCESS.md](../../docs/RELEASE_PROCESS.md#bumping-blis).


The `aisimulate` SDK wheel is pinned in `pyproject.toml` by exact fork-release
download URL (the `aisimulate @ …` entry), so the image always installs that
exact artifact — no build args. To move to a newer SDK build, update that URL and
regenerate `uv.lock`; see
[docs/RELEASE_PROCESS.md](../../docs/RELEASE_PROCESS.md#bumping-the-simulation-sdk).

## Local development

The `aisimulate` wheel is amd64-only, so this works on a `linux/amd64` host:

```bash
cd services/aisimulators
uv venv && uv pip install -e . --group dev
uv run pytest          # tests mock the SDK; no perf DB required
uv run ruff check .
uvicorn tools.api_service.app:app --reload --port 7860
```

On other platforms (e.g. macOS/arm) the `aisimulate` wheel is not available; use
the container instead.

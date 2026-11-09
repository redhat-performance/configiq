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
| GET | `/networks` | Network fabrics (per `simulator`; empty when it models none) |
| GET | `/simulators` | Simulator identity, label, description, and version |

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

BLIS v0.15.0 is copied from the upstream multi-arch release image
`ghcr.io/inference-sim/blis:v0.15.0`, pinned by index digest
`sha256:ae02212a5c06d147f98e62f90d8101ec158bcd74c55cc3a424d9f2b213d89c33`.
Using the published release keeps the binary exactly the artifact upstream
tested and removes the Go toolchain from this build. The build fails if the
`BLIS_IMAGE` tag and the reported `BLIS_VERSION` disagree. The image bundles the
two catalog inputs a BLIS run reads, both at pinned commits:

| Path | Contents | Pin |
|---|---|---|
| `/opt/blis/catalog` | declared facts: model graphs, chips, fabrics, workloads | blis-catalog `0.3.0`, `676326ec6c708e91ec4981bd6b03922a7fe60512` |
| `/opt/blis/registry` | fitted coefficient sets a scenario names | blis-registry `v0.2.0`, `7f6662d7188dc3dc873a37623e1f7bbfe1d6081b` |

No example-scenario directory is bundled: the adapter generates a Scenario and
Deployment per request, so those files would be dead weight.

There is no build-time support manifest. The API reads `/opt/blis/catalog`
directly at request time, and the catalog is the only source of BLIS support
facts:

- A model is priceable only when blis-catalog ships its `models/<slug>/graph.yaml`.
  The public model identifier is the Hugging Face repo from that entry's
  `model.yaml` (`Qwen/Qwen3-32B`); the catalog slug (`qwen3-32b`) is accepted as
  an input alias.
- Hardware and network keys are the catalog's own files. ConfigIQ system IDs are
  translated to catalog keys (`h100_sxm` → `h100`, `gb200` → `gb200-nvl72`);
  PCIe parts are deliberately not translated.
- Nothing is fabricated for a model or system the catalog does not list: the
  request returns an explicit unsupported error naming what is available.

## Simulator API behavior

`POST /predict` and `POST /recommend` accept `simulator: "aisimulate" | "blis"`
(`GET /simulators` is the authoritative list). The default remains
`aisimulate`. A BLIS request resolves its model, hardware, and fabric from the
pinned catalog, generates a temporary Scenario + Deployment for the requested
configuration, and passes it to `blis run`. Unsupported models, systems, fabrics,
or settings return an explicit client error. A BLIS failure is never retried
through AISimulate, and an unknown `simulator` value is a 422 rather than a
fall-through to another branch.

`POST /memory` is an AISimulate capability. A request that names a simulator
without a memory capability (currently BLIS) returns 422 instead of borrowing
AISimulate's answer.

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
accepts `max_model_len: auto`. `0.32.0` is opt-in and models both colocated and
P/D serving, but requires `max_model_len` stated; the adapter then writes
`isl + osl` instead of `auto`. Any other `backend_version`, including AISimulate
aliases such as `current`, is refused with a 422 that names the modeled releases
— BLIS treats `engine_version` as a check, not a selector, so the service never
substitutes a different release silently.

`GET /backends`, `/models`, `/systems`, and `/networks` take `simulator` so a
client can ask what the selected simulator supports. `/backends?simulator=...`
reports `simulator_version` and `recommend_search` (`windowed` for AISimulate's
service-owned GPU window ladder, `single` for BLIS's one bounded search). The BLIS
branch also reports the modeled versions, the literal default, and a
`version_support` map (`disaggregated`, `max_model_len_auto`) per release.
`/models?simulator=blis` returns the catalog's Hugging Face repo IDs (catalog
slugs are accepted as input aliases), and `/systems?simulator=blis` returns only
systems that resolve to a catalog chip. `/networks?simulator=aisimulate` is an
empty list, because AISimulate does not model an inter-node fabric; each
response includes `simulator`, and the BLIS response names its `default_network`.
Each backend entry reports `disaggregation_requires_fabric`, so a client knows
whether a P/D topology needs a fabric selection without matching on the
simulator ID. Each entry also reports `capacity_options.disaggregated`, which
tells a client whether a concurrency-capacity run can return disaggregated
alternatives; BLIS reports false because its capacity sizing is aggregate-only.
`/backends` also returns the simulator's `objectives` list, and
clients gate a sizing-goal selector on that list rather than on the simulator ID.

Simulator provenance includes the requested and resolved model, hardware,
simulator version, and engine version. BLIS prices the catalog's own model
configuration, so there is no cross-simulator config relation to report.
BLIS predictions do not include KV-cache memory breakdowns; `/memory` refuses a
simulator without that capability rather than substituting AISimulate.

BLIS `/predict` supports both colocated and prefill/decode-disaggregated vLLM
scenarios. P/D requests require a `fabric` from `GET /networks`, the same GPU
system for both pools, and `attention_dp_size=1`. The adapter generates separate
role pools, a NIXL transfer connector, and a router configuration that
disaggregates every request. BLIS `/recommend` searches aggregate TP candidates
for `min_gpus` and `throughput_demand`. It defaults to `roce-200g` when the
`fabric` field is omitted and also searches bounded P/D TP and worker-count
combinations, with one node per pool and DP=PP=1. Send `fabric: null` to request
aggregate-only candidates; the response then warns that P/D candidates were not
searched. `concurrency_capacity` is aggregate-only, and `pareto` enumerates both
topology modes when a fabric is selected.

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
| `concurrency_capacity` | `none` | `target_concurrency` only | fewest GPUs whose simulator capacity answer holds N sequences at the per-sequence context length; no latency target |
| `pareto` | `enforce` or `none` | none; the sweep drives load itself | every evaluated candidate plus a frontier view |

With `sla_mode=enforce`, send either `request_latency` or both `ttft` and
`tpot`. A demand request (`throughput_demand`) rejects `ttft`, `tpot`, and
`request_latency`; the service never accepts a bound the search does not apply.
`pareto` with `sla_mode=none` sweeps capacity without an SLA, while
`sla_mode=enforce` keeps only SLA-compliant candidates on the frontier.

### Latency-constrained min-GPU sizing

`objective=min_gpus` searches the fewest GPUs that clear the latency SLA. The
service owns the whole search window, so a request carries workload intent only.
It first resolves the model's **fit floor**: the smallest aggregated worker
shape whose KV capacity holds one full-context sequence, using the same runtime
envelope as the sweep (engine `max_num_tokens` / `max_num_seqs` defaults and
the backend's default memory fraction). Aggregated is a valid lower bound for
both deployment modes because a disaggregated deployment replicates the full
weights in each pool.

The first window is wide — `floor .. min(ceiling, max(16, floor × 2))` — so
both aggregated and disaggregated shapes have room and the SDK's
smallest-first ordering can find the true minimum. The ladder then:

- narrows to `floor .. candidate − 1` when a feasible candidate is above the
  floor;
- grows to `upper + 1 .. upper × 2` when a window cannot hold the model or no
  candidate clears the SLA;
- stops when the candidate is at the window floor, the effective ceiling is
  reached, or `MIN_GPUS_SEARCH_BUDGET_S` (default 70 seconds, including fit
  discovery and every backend-version attempt) is spent.

Each window draws its trial budget from that window's own upper bound
(`MIN_GPUS_MAX_TRIALS`, default 32, `MIN_GPUS_PARALLELISM`, default 4,
`MIN_GPUS_CANDIDATE_TIMEOUT_S`, default 30), and the same parallelism is passed
to the SDK so the budget and the config cannot disagree.

`AISIMULATORS_MAX_CANDIDATE_GPUS` (default 1024) is a soft ceiling. When the
fit floor is at or above it, the effective ceiling is `max(configured,
floor × 2)`, so a large model still has room to search. Discovery of the fit
floor is bounded separately by `FIT_DISCOVERY_BUDGET_S` (default 15 seconds).
If no shape fits within that budget, or the preflight estimate fails, the
failure is logged and the search starts from the wide fallback window
`1 .. min(ceiling, 16)` instead of returning an early error.

`sweep.budget_exhausted` is true only when the wall-clock budget stopped the
ladder. A spent trial budget is visible in `sweep.max_trials` and
`sweep.evaluated`; it does not mean the search was exhaustive.

The response reports the window actually searched as
`sweep.window_min_gpus` / `sweep.window_max_gpus`, plus
`sweep.configured_max_gpus` and `sweep.effective_max_gpus`.

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
hold the model is skipped. The request carries no GPU bounds: the service owns
the demand ladder as well. Each window draws its own trial budget, and its
per-candidate timeout is fitted to the time that remains, so the ladder stays
inside the gateway timeout.
`sweep.budget_exhausted` is true only when that wall-clock budget stopped the
ladder; trial spend remains visible in `max_trials`/`evaluated`.

### Concurrency-capacity sizing

`objective=concurrency_capacity` answers a residency question: how few GPUs
hold `target_concurrency` sequences at the per-sequence context length with no
latency target. `target_concurrency` is a design input, not a derived demand; it
must be a positive whole number. `sla_mode` must be `none`, and `target_request_rate`,
`ttft`, `tpot`, and `request_latency` are rejected rather than reinterpreted.

The per-sequence context is the peak a resident sequence can occupy:
`max_seq_len` when supplied, otherwise `isl + osl`. Prefix caching is not
credited, because a shorter average prompt does not bound how large a resident
sequence can grow.

For **AISimulate**, selection enumerates the parallel search space (the same
backend filters, wideEP heuristic, and KV-feasibility rule as `min_gpus`) and
reads each shape's KV token capacity. A sequence cannot span replicas, so for
aggregated shapes the service counts whole per-replica sequences and adds the
fewest replicas that reach `target_concurrency`
(`ceil(target_concurrency / per_replica_sequences)`). The fewest total GPUs is
the minimum. Disaggregated pairs come from the SDK's P/D enumeration; the
decode pool must hold the resident sequences, while the prefill pool only needs
to hold one sequence transiently. Because each pool replicates the full
weights, a P/D deployment is only offered as a larger comparison option, never
as the minimum. This is deterministic and exhaustive, not a sampled sweep:
`sweep.algorithm` is `enumerate`, `budget_exhausted` is false, and every
enumerated shape counts as evaluated and feasible.

For **BLIS**, selection searches power-of-two tensor-parallel aggregate shapes
with data and pipeline parallelism fixed at 1, runs each once at concurrency 1,
and reads the engine's reported KV pool. It applies the same whole-sequence
per-replica rule described above. BLIS capacity also reports
`algorithm="enumerate"`, but its wall-clock search budget can stop the
enumeration: then `budget_exhausted=true` and `exhaustive=false`, and the
reported minimum is not confirmed. A shape run can also fail while budget
remains, which likewise reports `exhaustive=false`; clients must check
`capacity.exhaustive`, not only `budget_exhausted`, before treating the minimum
as confirmed.

The capacity evidence is always the selected simulator's own answer:

- **AISimulate** derives weights plus KV cache for each enumerated shape, then
  applies the replica math above. It also enumerates disaggregated pairs and
  returns `capacity_options`.
- **BLIS** runs each aggregate shape at concurrency 1 and reads the engine's
  reported KV pool
  (`engine.cluster.kv_capacity_tokens_per_replica`), then adds replicas of whole
  per-replica sequences until the target is held. It never re-derives KV bytes
  from weights.
  BLIS capacity is aggregate-only: it returns the primary `configs[0]` and
  `capacity`, but no `capacity_options`, and it rejects an explicit non-null
  `fabric`. It does not simulate the target concurrency, so its capacity result
  reports `performance_available=false` with a reason. BLIS also rejects
  `kv_transfer_bandwidth_gb_per_second`, because it models P/D transfer through
  the selected fabric rather than an explicit bandwidth.

`configs[0]` remains the single primary recommendation. For AISimulate,
`capacity_options` is a ranked comparison set: rank 1 is the same minimum, and
the initial selection adds up to three same-GPU ties, up to two larger
aggregated deployments, and up to two disaggregated representatives. Those are
selection quotas, not response caps: if slots remain, the service fills them
with further aggregate or disaggregated candidates ranked by the same order.
When `top_n` is omitted the service returns up to 8 options when disaggregated
candidates exist and 6 otherwise; an explicit `top_n` overrides the count. Each option carries
`option_id`, `rank`, `role` (`minimum`, `same_gpus`, `larger`, `disaggregated`,
or `shortfall`), `gpus_over_minimum`, a complete `config`, and its own
`capacity` block, so a client can promote any option without re-running the
search.

A capacity result carries a `capacity` block:

- `target_concurrency` — sequences requested.
- `context_per_sequence_tokens` — `max_seq_len` or `isl + osl`.
- `capacity_tokens` — `per_rank_tokens × attention_dp × replicas`; the
  aggregate pool. Unused tokens in one replica cannot serve another.
- `capacity_sequences` — whole sequences per replica multiplied by replicas.
  Sequences cannot span replica engines, so this is not the pooled token total
  divided by the context length.
- `fits` — `capacity_sequences >= target_concurrency`.
- `shortfall_sequences` / `shortfall_tokens` — missing sequences and the KV
  tokens for those sequences when `fits` is false.
- `minimum_gpus` — fewest GPUs in the requested window that hold one sequence.
- `per_rank_tokens`, `replicas`, `attention_dp` — resolved capacity inputs. For
  a disaggregated option these describe the decode pool, which holds resident
  sequences. BLIS is aggregate-only, and with attention DP 1 its per-replica
  pool is also the per-rank figure.
- `mode` — `agg` or `disagg`.
- `prefill_replicas`, `decode_replicas`, `prefill_gpus`, `decode_gpus` —
  per-pool detail for a disaggregated option; null for agg.
- `evaluated_shapes`, `feasible_shapes`, `exhaustive`.
- `performance_available`, `performance_unavailable_reason` — whether an
  informational simulation produced timing data.
- `simulator` — the simulator that produced this capacity result.

For AISimulate, each option is simulated at the requested concurrency.
Simulations share one wall-clock budget and run at most two at a time, with the
primary simulated first, then disaggregated representatives, then the remaining
alternatives. That simulation never gates `fits`: when timing data cannot cover
an option, the option stays in the comparison set with `performance_available`
false, `performance_unavailable_reason` explaining why, and a warning on its
config. No performance value is invented. BLIS capacity sizing does not run the
target concurrency at all, so it always reports `performance_available=false`.

When the target does not fit inside the GPU window, the response is still 200:
the service reports the best achievable deployment inside the window plus the
sequence and token shortfall, so a client can show how far the budget is from
the request. When no aggregated shape can be scheduled in the window at all
(for example, a service candidate ceiling that no worker shape plus replicas
can reach), the service returns 422 naming the window. It never claims
that one GPU is always feasible and never silently substitutes another
simulator.

### Pareto frontier provenance and availability

Both simulators report `objective=pareto` in `/backends`. AISimulate samples the
frontier; BLIS has no native Pareto command, so the service enumerates topology
and closed-loop concurrency combinations, runs each through BLIS, and computes
the non-dominated set over throughput per user and throughput per GPU itself.
The frontier is still BLIS's own measured output, and the response records
`simulator="blis"` and `sweep.algorithm="enumerate"`. Neither simulator ever
falls back to the other, because a frontier computed by another simulator would
be misattributed.

A sweep searches both aggregated and disaggregated deployments whenever the GPU
window can hold more than one GPU. Both modes contribute candidates and frontier
points when they are feasible; a workload may have no feasible disaggregated
point, and that absence is reported rather than filled in. BLIS includes P/D
topologies only when the request names a supported `fabric`; the web client
sends the simulator's announced default fabric.

When the caller sends a Pareto request, AISimulate resolves a model-aware
window before sampling. The floor is the smallest aggregated worker shape that
holds one full-context sequence (from the KV-feasibility enumeration), and the
ceiling is `max(configured ceiling, floor × 2)`. The first window ends at
`max(16, floor × 2)`, capped by that ceiling, so the sweep is not spent below
the model's size. Aggregated is a valid floor for both modes because a
disaggregated deployment replicates the full weights in both pools. The
resolved bounds are reported as `sweep.window_min_gpus` /
`sweep.window_max_gpus`, along with `sweep.configured_max_gpus` and
`sweep.effective_max_gpus`. The request has no window fields to override this.

Fit-floor discovery is bounded by `FIT_DISCOVERY_BUDGET_S` (default 15
seconds). If no aggregated shape fits inside that budget, or the preflight
estimate fails, the service logs the failure and uses the wide
`1 .. min(ceiling, 16)` fallback window rather than returning an early error.
BLIS uses the service candidate ceiling, further limited by its hardware
maximum; it does not resolve a model-aware fit floor.

For a Pareto result:

- `candidates` contains every evaluated candidate, including infeasible,
  unsupported, timed-out, failed, and resource-limited records. AISimulate's raw
  fields such as `topology`, `metrics`, and `objectives` are copied verbatim;
  the service does not synthesize topology, transfer, or performance values.
  BLIS candidates carry the service-built topology and the metrics BLIS
  measured. Service-derived fields are added alongside them: `mode`,
  `provenance_sources`, and `transfer_delay_modeled` are computed (or combined
  from request and candidate data) by the service, not passed through from the
  simulator.
- `views.pareto_front` and `views.top_n` are simulator-computed candidate-ID
  lists; BLIS's are computed by the service from its measured runs.
  `pareto_front` is authoritative for the simulator's own objectives
  (throughput per user vs throughput per GPU). Clients may recompute dominance
  only for a different axis or orientation, and must say so in the UI.
- `transfer_delay_modeled` is true only when a P/D transfer delay was actually
  modeled for that candidate. Under AISimulate, omitting
  `kv_transfer_bandwidth_gb_per_second` means no modeled transfer delay; it does
  not imply a fabric default. BLIS models transfer through the named `fabric`
  and rejects an explicit bandwidth.
- `sweep` reports the algorithm, seed, trial budget, evaluated/feasible/timed-out
  counts, and the GPU window. `sweep.budget_exhausted=true` means a wall-clock
  search time budget stopped the run: BLIS's Pareto enumeration or an
  AISimulate window ladder. Trial spend is reported separately by
  `max_trials`/`evaluated`; for a sampled sweep, `evaluated >= max_trials`
  means the trial budget was spent. Exhaustion never means the search was
  exhaustive. `BLIS_PARETO_CONCURRENCY_LEVELS` overrides the
  closed-loop load grid.

Pareto sweeps use `PARETO_MAX_TRIALS` (default 20), `PARETO_PARALLELISM`
(default 8), `PARETO_CANDIDATE_TIMEOUT_S` (default 8), and
`PARETO_MAX_CONCURRENCY` (default 2 concurrent sweeps per worker, returning 429
when exceeded). `PARETO_ALGORITHM` selects `random` (default) or `bayesian`; the
Bayesian sampler enables JAX x64 process-wide and is not the default.
`SWEEP_SEED` defaults to 42. A one-GPU total window runs one aggregate trial,
since it cannot contain a P/D deployment.

The image sets `BLIS_VERSION`, `BLIS_CATALOG` and `BLIS_REGISTRY`. `BLIS_VERSION`
is the release reported by `/simulators`, `/backends`, and in every prediction's
provenance. BLIS itself only reads `BLIS_CATALOG` from the environment; the
registry has no environment fallback, so callers must pass `--registry`
explicitly as well:

```bash
blis run --catalog "$BLIS_CATALOG" --registry "$BLIS_REGISTRY" \
  --scenario generated-scenario.yaml \
  --rate 10 --num-requests 100
```

To move to a new BLIS release, bump `BLIS_IMAGE` (tag and index digest),
`BLIS_VERSION`, `BLIS_CATALOG_COMMIT`, and `BLIS_REGISTRY_COMMIT` together;
update `MODELED_ENGINE_VERSIONS` in `tools/api_service/blis/scenario.py` if that
release changes which vLLM versions it models. See
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

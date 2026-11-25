# AISimulators REST API

A thin FastAPI wrapper over the [aisimulate](https://github.com/redhat-performance/aisimulate)
SDK, exposing GPU recommendation, performance estimation, and memory estimation
as HTTP endpoints. ConfigIQ's `app/api/*` proxy routes call this service.

This directory contains **only** the REST wrapper. The SDK itself is not vendored
here — it is installed as a single wheel published by the Red Hat fork's GitHub
Releases. That one `aisim` wheel bundles the Rust-compiled core and exposes
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
| GET | `/simulators` | Simulator identity, label, description, version, and revision |
| GET | `/deploy/targets` | Deployment targets, their pins, and whether this deployment can render them |
| POST | `/deploy` | Deployable artifact rendered from an evaluated `engine` section |

`/recommend`, `/predict`, and `/estimate` accept an `include` query parameter for
optional sections (`config`, `memory`, `engine`, `debug`); `/memory` accepts
`debug`. See [Engine limits](#engine-limits-includeengine) for `engine`,
[deployment artifacts](#deployment-artifacts-deploy) for `/deploy`, and
[Replay artifacts](#replay-artifacts-includedebug) for `debug`.

Full spec: [`docs/api/openapi.yaml`](docs/api/openapi.yaml).

## Run with the container (recommended)

The `aisim` wheel ships only for `linux/amd64` (manylinux x86_64), so the
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

The image also exposes the simulator CLIs at stable paths, plus the inputs the
deployment render gate needs:

```text
/usr/local/bin/aisimulate
/usr/local/bin/blis
/usr/local/bin/kustomize        # 5.7.1, pinned release
/opt/llm-d/<revision>/guides/   # pinned llm-d subtrees (AISIMULATORS_LLM_D_VENDOR_DIR)
```

BLIS is built from source at a pinned commit of the fork's `deploy/api` branch
that adds the reports this adapter reads. No release image carries them yet:
[inference-sim/inference-sim#2693](https://github.com/inference-sim/inference-sim/pull/2693)
(`blis version --json` reporting each modelled engine release),
[#2694](https://github.com/inference-sim/inference-sim/pull/2694)
(each pool's memory in the run's Result) and
[#2695](https://github.com/inference-sim/inference-sim/pull/2695)
(`blis catalog list`) are open, and the image builds the commit the
Containerfile pins (`natoscott/inference-sim@694802f5`, tagged
`deploy-api-20261127-694802f5` so a rewrite of the branch cannot orphan it).
Once they land in a tagged release, the
build stage returns to the published multi-arch image
(`ghcr.io/inference-sim/blis:vX.Y.Z`, pinned by index digest), which keeps the
binary exactly the artifact upstream tested and removes the Go toolchain from
this build. The version the image reports is the pseudo-version `blis version`
prints for the pinned commit, because no release tag contains it; the build
fails if the binary prints anything else.

BLIS v0.19 embeds its catalog and coefficient registry in the binary and
refuses the external clones earlier releases read, so a run takes neither
`--catalog` nor `--registry`. The API reads the same facts from the binary with
`blis catalog list --json`, so the image carries no data directory and cannot
pair a binary with a different build's catalog.

No example-scenario directory is bundled: the adapter generates a Scenario and
Deployment per request, so those files would be dead weight.

There is no build-time support manifest. The API asks the pinned binary for its
catalog at request time, and that report is the only source of BLIS support
facts:

- A model is priceable only when the build reports a graph for it. The public
  model identifier is the Hugging Face repo from the entry's `source` block
  (`Qwen/Qwen3-32B`); the catalog name (`qwen3-32b`) is accepted as an input
  alias.
- The context window is the graph's stated `context_window`, and nothing else.
  Only some graphs state one; a model whose graph leaves it open reports `null`
  rather than a value derived from a file the build never read.
- Hardware and network keys are the report's own chip and fabric names. ConfigIQ
  system IDs are translated to catalog keys (`h100_sxm` → `h100`, `gb200` →
  `gb200-nvl72`); PCIe parts are deliberately not translated.
- Nothing is fabricated for a model or system the catalog does not list: the
  request returns an explicit unsupported error naming what is available.

## Simulator API behavior

`POST /predict` and `POST /recommend` accept `simulator: "aisim" | "blis"`
(`GET /simulators` is the authoritative list). The default remains
`aisim`. A BLIS request resolves its model, hardware, and fabric from the
pinned catalog, generates a temporary Scenario + Deployment for the requested
configuration, and passes it to `blis run`. Unsupported models, systems, fabrics,
or settings return an explicit client error. A BLIS failure is never retried
through AISim, and an unknown `simulator` value is a 422 rather than a
fall-through to another branch.

`POST /memory` answers from the selected simulator. AISim uses its SDK memory
model. BLIS reads the block its own run attaches to a Result, using the build's
memory kernel, so the endpoint and a `/predict` run describe the same pool:

- `max_num_tokens` becomes the pool's context window, `max_batch_size` the batch
  it admits, and `memory_fraction_value` the fraction of device memory it may
  use. Omit the fraction to let the simulator resolve its own default: AISim
  applies 1.0, while BLIS keeps the engine's choice. The block is fixed when the
  engine is constructed, so the service replays the smallest run that reports it
  — one request at a constant rate — and no prompt length or larger request count
  is needed or charges anything.
- The pool record is returned verbatim under `pools`, and the per-GPU summary
  fields are that record's per-rank terms (`total_gpu_capacity_bytes` is the
  device, `total_kv_size_bytes` one rank's KV bytes, `kv_size_per_token_bytes` one
  rank's bytes per token, `total_kv_size_tokens` one rank's token capacity).
- The legacy four-bucket `memory_breakdown` is an exact partition of the record's
  fixed occupancy: `weights` plus a redundant expert copy is weights, the peak
  activation is activations, the CUDA graph plus non-torch and engine buffers are
  runtime overhead, and communication buffers are their own term.
- BLIS sizes from a fraction of device memory, so `memory_fraction_kind=of_free`
  and the options it does not model (`fmha_quant_mode`, `comm_quant_mode`,
  `tolerance_fraction`) return 422 rather than being mapped onto another setting.

A request that names a simulator without a memory capability returns 422 instead
of borrowing another simulator's answer. A simulator that reports no per-pool
record answers with an empty `pools` list, and `warnings` carries anything the
caller must know about the answer.

## Replay artifacts (`include=debug`)

`POST /recommend`, `POST /predict`, `POST /estimate`, and `POST /memory` accept
`include=debug`. The response then carries a `debug` record of what the request
executed: a command line per run, the input files that command reads, and the
simulator releases and data revisions the run was made against. The record is
meant to be reproducible outside ConfigIQ — write the files to a directory and
run the command, or run the record's `script` (`run.sh`), on a machine with the
pinned simulators installed.

The record travels inline in the response rather than being fetched by id. The
service runs more than one worker process, so a later request has no guarantee of
reaching the worker that made a record; an id-addressed store would answer 404
for a record that exists.

What a command means:

| Simulator | `execution` | Meaning |
|---|---|---|
| BLIS | `subprocess` | The exact `blis run` process, with the scenario files ConfigIQ generated. The binary path is read from `BLIS_BINARY`, so the command does not name this image's layout. BLIS reads its embedded catalog, registry, and engine facts, so the command carries no external data path. |
| BLIS (memory) | `subprocess` | The `blis run` process whose Result carries the pool's memory block, with the same generated scenario a prediction of that request would read. |
| AISim | `in_process` | An equivalent CLI invocation, `aisimulate predict -c <file>` or `aisimulate recommend -c <file>`. The service calls the Python API directly, so the command repeats the same work rather than the same process. |
| AISim (memory) | `in_process` | A Python snippet calling `estimate_kv_cache`. The memory model has no CLI verb. |

`debug: null` is a real answer, not an error: recording is best-effort so a
simulation never fails because its description could not be built. It means the
request has no single command of its own — a `concurrency_capacity`
recommendation is an enumeration over the search space, not one invocation — or
that the artifacts exceeded the size cap (`AISIMULATORS_REPLAY_MAX_BYTES`,
default 2 MiB). Over the cap, per-candidate commands are dropped first and the
record says how many, because the command that produced the answer is what
explains the answer.

Each Pareto candidate's recorded invocation carries that candidate's
`candidate_id`, so a candidate row copies its own command instead of the sweep's.
`script` leaves candidate invocations out: running every candidate is not the
sweep. A record holds at most 512 invocations, and a version-fallback attempt
that failed is dropped from it — the record describes the run that produced the
answer, not the ones that did not. A GPU window the search skipped because the
model did not fit is dropped the same way, so `run.sh` never stops before the
window that answered.


## Engine limits (`include=engine`)

`POST /predict` and `POST /recommend` accept `include=engine`. Each answer then
carries the exact engine limits it was evaluated with: parallelism, replica
counts, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, block size,
memory utilization, prefix caching, quantization, and the role layout
(`agg`, or `prefill` + `decode`).

Every field is an `EngineField`: a value plus a `source` that says why it holds
that value.

| Source | Meaning |
|---|---|
| `evaluated` | The simulator resolved and used it. A deployment must render it. |
| `requested` | The caller asked for it and it rides along to a role the simulator does not evaluate. |
| `backend-default` | No side stated a value, so the backend's own default applies. |
| `not-modeled` | The run states nothing. `value` is null; the field is not invented. |

The values are read from the compiled replay spec, not reconstructed from the
request: a limit the request leaves out (a block size, a scheduler ceiling) is
reported as the simulator resolved it. A result that carries no compiled spec —
an unavailable run, or a record the ledger did not keep a config for — reports
`engine: null`, because a reconstructed default would describe a run that never
happened. BLIS states what its scenario holds and marks the rest `not-modeled`
(for example block size, or a context window it resolved by policy).

This section is the sole input to `POST /deploy`, so the deployment is rendered
from the same numbers the prediction reported.


## Deployment artifacts (`/deploy`)

`GET /deploy/targets` lists the targets this service can render; `POST /deploy`
renders one. The request carries an `engine` section from `include=engine` and
nothing else that changes sizing: the artifact is a deployment of the evaluated
configuration, not a starting point for a new one.

| Target | Layout | Renders |
|---|---|---|
| `dynamo-dgd` | NVIDIA Dynamo | One `DynamoGraphDeployment` (aggregated: `VllmWorker`; disaggregated: `VllmPrefillWorker` + `VllmDecodeWorker`) |
| `llm-d` | llm-d well-lit path | A Kustomize overlay over the pinned llm-d v1.0.0 guide base: `Deployment` when aggregated, `DisaggregatedSet` when disaggregated, plus the pinned router Helm release |

An llm-d bundle is self-contained: it carries the subtrees of llm-d v1.0.0
(`378b42d18dabe0b9c4ced4e6f3670d93b52db08e`) that the overlay references,
under `llm-d/` at their repository-relative paths, so the overlay's
`../../../../../recipes/...` references resolve without a network fetch. The
bundle-root `kustomization.yaml` is post-processed to set the install namespace,
and, for a P/D layout, to replace the upstream base's moving `nightly` routing
sidecar with the pinned
`ghcr.io/llm-d/llm-d-router-disagg-sidecar:v1.0.0`.

### The render gate

Nothing is returned unverified. Every artifact is assembled in a scratch
directory and checked before the response is written:

| Check | What it proves |
|---|---|
| `kustomize-build` | A Kustomize target builds with the pinned tree, exactly as an installer would (skipped for a single-manifest target, whose assertions read the manifest directly). |
| `resources` | The expected kinds rendered (`DynamoGraphDeployment`, `Deployment`, or `DisaggregatedSet`). |
| `engine-limits` | Every evaluated limit and replica count appears in the rendered argv. |
| `images` | The model-server image is the pinned one, and a P/D layout also pins the routing sidecar. |
| `placeholders` | No upstream `REPLACE_*` placeholder survives. |
| `install-references` | Every file `install.sh` passes to Helm with `-f` is in the bundle (Kustomize targets only). |

A failed check is a 500 naming the check, not a bundle that breaks on apply.

### Pins and refusals

The `engine` section must be reproducible by the target. A target refuses with
422 when:

- a limit it must render is `not-modeled` (re-run the prediction, or choose a
  target that models it);
- the evaluated run states a limit the pinned generator does not render into the
  model server's argv (`enable_chunked_prefill`, `kv_cache_dtype`, or
  `quantization` when the source is `evaluated`), so deploying it would apply a
  backend default the prediction did not use. Re-run the prediction without that
  override;
- its pinned image runs a different engine release than the evaluated one. Set
  `allow_engine_version_mismatch: true` (and `image` when overriding the image)
  to accept the difference: it is recorded in `PROVENANCE.md` and printed by
  `preflight.sh`.
- the generator refuses the evaluated topology (for example a token budget that
  is not a multiple of 64).

A target whose pinned inputs are absent from this deployment answers 503, and
`GET /deploy/targets` reports `available: false` with the reason. Unknown target
ids answer 404 with the list of ids.

### Bundle contents

`README.md`, `PROVENANCE.md` (every pin, the evaluated limits, the gate's
checks, accepted warnings), `install.sh`, `preflight.sh`, and `verify.sh`, plus
the manifests or overlay. The overlay's own files are the ones the rendered
`kustomization.yaml` names — itself, the configurations it loads, and the
patches it applies — so a file the overlay references cannot be left behind
when a generator release adds one. `preflight.sh` is read-only and fails rather than
installing into a cluster whose CRDs, GPUs, or Helm are missing; `verify.sh`
waits for the rollout and queries `/v1/models`. `format=zip` returns the same
files with executable modes preserved.

The generator also produces benchmark artifacts (`k8s_bench.yaml`,
`bench_run.sh`, `sflow.yaml`, `cli_args_*`, `run_0.sh`). They reproduce the
measurement rather than deploy it, so they are left out and the artifact's
warnings say so.

### Running the render gate outside the container

The gate needs kustomize (or `kubectl`) and the pinned llm-d tree. Fetch the tree
once, then point the service at it:

```bash
export AISIMULATORS_LLM_D_VENDOR_DIR=/tmp/llm-d-vendor
scripts/fetch-llm-d-vendor.sh          # vendors <dir>/378b42d1.../guides/...
export AISIMULATORS_KUSTOMIZE_BIN=/usr/local/bin/kustomize   # optional
```

`AISIMULATORS_KUSTOMIZE_BIN` pins the binary when several are on `PATH`;
otherwise the service prefers `kustomize` and falls back to `kubectl kustomize`.
Without either, an llm-d render answers 503 rather than shipping unbuilt
manifests. The image vendors the tree at `/opt/llm-d` and installs kustomize
`5.7.1`; CI does the same before running the tests.


## AISim backend versions

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

AISim validates that `tp_size * attention_dp_size * cp_size` equals
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

BLIS v0.19 fixes the engine release its kernel prices: a Scenario states
`engine: {kind: vllm}` and no version, and a document that names one is refused.
`/backends?simulator=blis` reports that release from `blis version --json`
(provenance only fields such as the build's own engine snapshot are not shown).
A BLIS build that predates the report lists no version, and clients show the
version as unreported rather than substituting the snapshot. A request whose
`backend_version` names a release the build does not model is refused with a 422
naming the modeled release, when the build can report it; a blank version takes
the build's release. AISim aliases such as `current` are not meaningful here.
The adapter always
states a concrete `engine.max_model_len`: the request's `max_seq_len` when set,
otherwise the catalog model's native context, otherwise `isl + osl`. It does not
emit `auto`, because BLIS resolves that from the model graph and some catalog
graphs (for example `gpt-oss-120b`) omit the context window. A P/D scenario
gives each pool its own limit from `prefill_max_seq_len` /
`decode_max_seq_len` when set, falling back to `max_seq_len`; AISim applies
the same per-pool fallback. Any other
`backend_version`, including AISim aliases such as `current`, is refused
with a 422 that names the modeled releases — BLIS treats `engine_version` as a
check, not a selector, so the service never substitutes a different release
silently.

`GET /backends`, `/models`, `/systems`, and `/networks` take `simulator` so a
client can ask what the selected simulator supports. `/backends?simulator=...`
reports `simulator_version` and `recommend_search` (`windowed` for AISim's
service-owned GPU window ladder, `single` for BLIS's one bounded search). The BLIS
branch reports one entry per engine kind the pinned build models, with the
release it models as that entry's version; a build that cannot report its
engines lists no backends at all, and the client shows the catalogue as
unavailable rather than guessing. Each entry's `version_support` states the
serving roles the build reports for that release, so `disaggregated` is true only
when it models both prefill and decode. `runtime_field` names that engine's own
memory-fraction control (`gpu_memory_utilization` or `mem_fraction_static`).
`/models?simulator=blis` returns the catalog's Hugging Face repo IDs (catalog
names are accepted as input aliases), and `/systems?simulator=blis` returns only
systems that resolve to a catalog chip, with `include=specs` reporting the chip
facts the build states and leaving out (rather than filling in) anything it does
not. `/networks?simulator=aisim` is an
empty list, because AISim does not model an inter-node fabric; each
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
BLIS predictions carry the pool memory its own kernel sized for the run: with
`include=memory`, `memory_pools` holds one record per pool and each
disaggregated worker also gets its own record, its own four-bucket breakdown,
and its own per-GPU peak. The headline `memory` figure is the larger per-GPU
peak. A run whose build reports no pool memory says so in
`memory_unavailable_reason` rather than substituting another simulator's
answer. A `memory_breakdown` describes one GPU, not the cluster: weights are
already sharded across the deployment's parallelism and KV is one rank's pool,
so a client must not divide them again.

A `gpu_memory_utilization` / `memory_fraction_value` the request omits is left
out of the generated scenario, so the engine's own default applies. The response
reports `serving_config.gpu_memory_utilization` as null when the run did not
state one; a client shows the setting as unstated rather than substituting its
own default.

Both `/predict` and `/recommend` accept `prefix`, the shared prefix cached as a
subset of `isl`; a value larger than `isl` or a supplied context limit is
rejected with 422. `/predict` forwards it to the simulator so the cached tokens
reduce prefill, and `/recommend` applies the capacity credit described below.

BLIS `/predict` supports both colocated and prefill/decode-disaggregated vLLM
scenarios. P/D requests require a `fabric` from `GET /networks`, the same GPU
system for both pools, and `attention_dp_size=1`. The adapter generates separate
role pools, a NIXL transfer connector, and a router configuration that
disaggregates every request. Both modes accept `requests_per_load_unit` and
simulate `max(1, round(batch_size x requests_per_load_unit))` total requests
when it is specified, the same rule AISim uses. When omitted, both modes
simulate one total request.

The run also pins the prompt and output token distributions to the requested
ISL/OSL (`min = max = requested`, stdev 0), so a prompt longer than BLIS's
default 7000-token ceiling is simulated instead of rejected.
BLIS `/recommend` searches aggregate TP candidates
for `min_gpus` and `throughput_demand`. It defaults to `roce-200g` when the
`fabric` field is omitted and also searches bounded P/D TP and worker-count
combinations, with one node per pool and DP=PP=1. Send `fabric: null` or
`mode=agg` to request aggregate-only candidates; the response then warns that
P/D candidates were not searched. A named fabric is a topology assumption, so
an unknown one is rejected with 422 even when the search excludes P/D (for
example `mode=agg`). `concurrency_capacity` is aggregate-only. `pareto`
enumerates both topology modes when a fabric is selected and rejects `mode`,
because a sweep has no single serving mode to narrow.

`mode` names the deployments a sizing search covers, for both simulators.
`agg` searches aggregated candidates only; omitting it or sending `disagg`
keeps the simulator's existing search. Omission is not the same as
`fabric: null`: an omitted fabric takes the catalog default (BLIS:
`roce-200g`), while an explicit null asks for aggregate candidates only. The
field is rejected with 422 where it cannot narrow anything: with
`objective=pareto`, with `objective=concurrency_capacity` and `disagg`, and as
`mode=agg` combined with `prefill_max_seq_len`/`decode_max_seq_len`
(per-pool limits only exist in a P/D deployment). A simulator that models P/D
only through a fabric (`disaggregation_requires_fabric` on `/backends`, true
for BLIS) also rejects `mode=disagg` when the request resolves to no fabric.
Validation reads that capability flag, never the simulator ID.

`mode=disagg` is accepted with `objective=throughput_demand` because AISim
searches P/D for that goal. BLIS demand sizing replays open-loop traffic and
attaches no P/D transfer fabric, so its search stays aggregated; the response
says so in `sweep.modes_searched` (`["agg"]`) rather than echoing the request.
A client should read that field instead of assuming the requested mode was
searched.

Both simulators report what the search covered in `sweep.modes_searched`
(contract values `agg`/`disagg`; null when the simulator reported no search
facts). `chosen_mode` describes the recommendation, which can be aggregated
even when P/D candidates were evaluated. AISim reports the union over the
GPU windows that returned a result, so a one-GPU window contributes `agg` only;
BLIS reports the modes its candidate grid reached, where a candidate that ran
and failed still counts, and also lists them in
`search_summary.modes_searched`.

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
`AISIMULATORS_MAX_CANDIDATE_GPUS` service setting used by AISim, further
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
smallest-first ordering can find the true minimum. `mode=agg` limits that
search to aggregated candidates. Omit `mode`, or send `disagg`, to keep both
deployment modes. A Pareto sweep rejects `mode` and enumerates both. The ladder
reports the modes it covered in `sweep.modes_searched`, so a one-GPU window
shows `agg` alone even when the request kept the broader search. The ladder then:

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

AISim demand runs use `DEMAND_MAX_TRIALS` (default 16),
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
`max_seq_len` when supplied, otherwise `isl + osl`. A shared prefix does not
shorten that peak: holding one sequence still needs the full context. It does
change what an *additional* resident sequence needs. Each attention-DP rank
has its own KV budget and caches the shared prefix once for the sequences it
holds, so each rank holds one prefix plus whole suffixes and an additional
sequence costs only `context - prefix` tokens; a replica holds `attention_dp`
times that count. Credit is applied only when the prefix is shorter than the
peak context; with no prefix, or a prefix at least as long as the context, the
space an additional sequence needs is unchanged. The credit is reported as
`capacity.shared_prefix_tokens` so a client can show the assumption instead of
inferring it. `prefix` must not exceed `isl` or a supplied `max_seq_len`; the
service returns 422 rather than forwarding a prefix that cannot fit in the
input. The optional memory breakdown is a cold-request model: it charges every
sequence the full peak context and does not subtract the prefix credit, so the
capacity block, not the breakdown, carries the reuse assumption.

For **AISim**, selection enumerates the parallel search space (the same
backend filters, wideEP heuristic, and KV-feasibility rule as `min_gpus`) and
reads each shape's KV token capacity. A sequence cannot span ranks or replicas,
so aggregated shapes count whole per-rank sequences, multiply by attention DP
and the replica count, and add the fewest replicas that reach
`target_concurrency` (`ceil(target_concurrency / per_replica_sequences)`). The
fewest total GPUs is the minimum. Disaggregated pairs come from the SDK's P/D
enumeration; the decode pool must hold the resident sequences, while the
prefill pool only needs to hold one sequence transiently. Because each pool
replicates the full weights, a P/D deployment is only offered as a larger
comparison option, never as the minimum. This is deterministic and exhaustive,
not a sampled sweep: `sweep.algorithm` is `enumerate`, `budget_exhausted` is
false, and every enumerated shape counts as evaluated and feasible.

For **BLIS**, selection searches power-of-two tensor-parallel aggregate shapes
with data and pipeline parallelism fixed at 1, runs each once at concurrency 1,
and reads the engine's reported KV pool. It applies the same whole-sequence
rule described above; BLIS runs attention DP 1, so its per-replica pool is also
the per-rank pool. BLIS capacity also reports
`algorithm="enumerate"`, but its wall-clock search budget can stop the
enumeration: then `budget_exhausted=true` and `exhaustive=false`, and the
reported minimum is not confirmed. A shape run can also fail while budget
remains, which likewise reports `exhaustive=false`; clients must check
`capacity.exhaustive`, not only `budget_exhausted`, before treating the minimum
as confirmed.

The capacity evidence is always the selected simulator's own answer:

- **AISim** derives weights plus KV cache for each enumerated shape, then
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

`configs[0]` remains the single primary recommendation. For AISim,
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
- `context_per_sequence_tokens` — `max_seq_len` or `isl + osl`. This bound
  applies to aggregated and disaggregated options; per-pool sequence-length
  overrides do not change the residency estimate.
- `shared_prefix_tokens` — prefix tokens each attention-DP rank caches. Zero
  when no prefix was requested, or when the prefix is not shorter than the peak
  context.
- `capacity_tokens` — `per_rank_tokens × attention_dp × replicas`; the
  aggregate pool. Unused tokens in one rank or replica cannot serve another.
- `capacity_sequences` — whole sequences per rank multiplied by attention DP
  and the replica count. Sequences cannot span ranks or replica engines, so this
  is not the pooled token total divided by the context length. A shared prefix
  is stored once on each rank, so each extra sequence is charged
  `context_per_sequence_tokens - shared_prefix_tokens`.
- `fits` — `capacity_sequences >= target_concurrency`.
- `shortfall_sequences` / `shortfall_tokens` — missing sequences and the KV
  tokens for those sequences when `fits` is false. With a shared prefix the
  token shortfall is measured in suffix tokens.
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

For AISim, each option is simulated at the requested concurrency.
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

Both simulators report `objective=pareto` in `/backends`. AISim samples the
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

When the caller sends a Pareto request, AISim resolves a model-aware
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
  unsupported, timed-out, failed, and resource-limited records. AISim's raw
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
  modeled for that candidate. Under AISim, omitting
  `kv_transfer_bandwidth_gb_per_second` means no modeled transfer delay; it does
  not imply a fabric default. BLIS models transfer through the named `fabric`
  and rejects an explicit bandwidth.
- `sweep` reports the algorithm, seed, trial budget, evaluated/feasible/timed-out
  counts, and the GPU window. `sweep.budget_exhausted=true` means a wall-clock
  search time budget stopped the run: BLIS's Pareto enumeration or an
  AISim window ladder. Trial spend is reported separately by
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

The image sets `BLIS_VERSION` and `BLIS_COMMIT`. `BLIS_VERSION` is the build
reported by `/simulators`, `/backends`, and in every prediction's provenance;
`BLIS_COMMIT` is the git commit it was built from, reported as `revision` in
`/simulators`. BLIS reads its
embedded catalog and coefficient registry, so only its binary path is required:

```bash
blis run --scenarios "$BLIS_SCENARIOS" --scenario generated-scenario.yaml \
  --rate 10 --num-requests 100
```

To move to a new BLIS build, bump `BLIS_COMMIT` (and `BLIS_GO_VERSION` when the
commit's `go` directive moves), `BLIS_VERSION` to what `blis version` prints for
it, and `DEFAULT_BLIS_RELEASE` /
`DEFAULT_BLIS_COMMIT` in `tools/api_service/blis/runtime.py` together. The engine
release the build models is reported by the binary, so no code names it.
`BLIS_COMMIT` may be omitted only for the default build, whose commit the service
resolves; another build without it reports no revision. When a release carries
the commands, replace the build stage with its image and keep `BLIS_VERSION` the
human-facing release. See
[docs/RELEASE_PROCESS.md](../../docs/RELEASE_PROCESS.md#bumping-blis).


The `aisim` SDK wheel is pinned in `pyproject.toml` by exact fork-release
download URL (the `aisim @ …` entry), so the image always installs that exact
artifact — no build args. The reported `revision` comes from the release tag
suffix (`deploy-api-v<version>+<sha>`) recorded in the wheel's
`direct_url.json`. To move to a newer SDK build, update that URL and regenerate
`uv.lock`; see
[docs/RELEASE_PROCESS.md](../../docs/RELEASE_PROCESS.md#bumping-the-simulation-sdk).

## Local development

The `aisim` wheel is amd64-only, so this works on a `linux/amd64` host:

```bash
cd services/aisimulators
uv venv && uv pip install -e . --group dev
uv run pytest          # tests mock the SDK; no perf DB required
uv run ruff check .
uvicorn tools.api_service.app:app --reload --port 7860
```

On other platforms (e.g. macOS/arm) the `aisim` wheel is not available; use
the container instead.

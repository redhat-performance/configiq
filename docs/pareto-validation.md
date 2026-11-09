# Pareto frontier — AISimulate and BLIS validation

Validation of the `objective=pareto` topology-design feature against the real
aisimulate SDK (v0.13.0 wheel). The goal was not "does it run" but "are the
numbers physically correct" — every output was checked against a known physical
expectation. BLIS is validated on its own measured runs and reports
`sweep.algorithm="enumerate"`; see [BLIS sweeps](#blis-sweeps) below.

**How it was run:** the aisimulators service was run locally in an x86 container
with the pinned aisimulate wheel; requests went through `POST /recommend` with
`objective=pareto`. (Apple Silicon runs the wheel under x86 emulation, so wall
times below are emulation-slow and not representative of production.)

## Matrix sweep — scaling laws

Seven model × GPU × workload cells, loose SLA so throughput (not feasibility)
is the comparison axis. `maxTPG` = best throughput per GPU, `maxTPU` = best
throughput per user, both from the feasible set.

| Model | GPU | Workload | maxTPG (tok/s/GPU) | maxTPU (tok/s/user) |
|---|---|---|---|---|
| Qwen3-0.6B | H200 | chat | 22,207 | 214 |
| Qwen3-8B | H200 | chat | 6,513 | 182 |
| Qwen3-30B-A3B (MoE) | H200 | chat | 7,689 | 80 |
| Qwen3-8B | B200 | chat | 12,214 | 209 |
| Qwen3-8B | H100 | chat | 5,568 | 153 |
| Qwen3-8B | H200 | RAG (ISL 8k) | 1,286 | 172 |
| Qwen3-8B | H200 | coding (ISL 4k) | 3,233 | 144 |

Checks, all passing:

- **Model scaling** (H200/chat): 0.6B (22.2k) ≫ 8B (6.5k) tok/s/GPU. Larger dense
  model → lower per-GPU throughput.
- **MoE sparsity captured**: 30B-A3B (7.7k) beats dense 8B (6.5k) — correct,
  because the MoE activates ~3B params/token, so per-token compute is near a 3B
  dense model, not a 30B one. The simulator is not treating it as 30B dense.
- **Hardware ordering**: B200 (12.2k) > H200 (6.5k) > H100 (5.6k); B200 ≈ 1.9× H200.
- **Context scaling** (8B/H200): chat ISL1024 (6.5k) > coding ISL4096 (3.2k) >
  RAG ISL8192 (1.3k). Monotonic with prefill work.
- **The frontier is a real trade-off**: in every cell the max-per-GPU config
  sits at low per-user throughput and vice-versa (e.g. 8B/H200: max-TPG at 29
  tok/s/user, max-TPU at 2.3k tok/s/GPU). That anti-correlation is the actual
  batching-vs-interactivity tension a Pareto frontier should show.

## Heavy regime — disagg emergence

Large models + long context + 16-GPU budget push disaggregated (P/D) topologies
onto the frontier:

| Scenario | Frontier points | agg | disagg |
|---|---|---|---|
| 32B / H200 / RAG | 7 | 3 | 4 |
| 30B-A3B / H200 / RAG | 11 | 7 | 4 |
| 32B / H200 / coding | 5 | 2 | 3 |

Disagg points use 6–15 GPUs — the scale where splitting prefill and decode pools
pays off. For 32B/RAG, disagg e2e latency (24.1s) is well below agg (38.1s),
which is exactly why disagg wins a frontier region here.

## P/D transfer-delay modeling

The decisive test: sweep `kv_transfer_bandwidth` on the disagg-heavy 32B/RAG
scenario. If transfer is truly modeled, **disagg** latency responds to bandwidth
while **agg** (no P/D transfer) stays invariant.

| bw (GB/s) | AGG e2e (control) | DISAGG e2e | DISAGG TTFT |
|---|---|---|---|
| 5 | 38,132 | 24,293 | 12,450 |
| 25 | 38,132 | 24,165 | 12,296 |
| 200 | 38,132 | 24,139 | 12,265 |

- **Control holds**: AGG e2e and TTST are byte-identical across all bandwidths —
  bandwidth never leaks into aggregated serving.
- **Disagg responds monotonically** in the correct direction: more bandwidth →
  lower e2e and TTFT.
- **Magnitude matches `transfer_time ∝ 1/bandwidth`** (the real physics): the
  latency drop from 5→25 is ~5× the drop from 25→200. Predicted ratio
  (1/5−1/25)/(1/25−1/200) = 4.57; observed e2e 4.92, TTFT 4.97. Near-match to
  the 1/bw curve is strong evidence the SDK computes transfer time as
  `KV_bytes / bandwidth`, not a flat penalty.

Absolute magnitude is small (hundreds of ms over a 40× bandwidth range) because
Qwen3-32B's GQA keeps the KV cache small (~0.25 MB/token, ~2 GB for 8K context)
and transfer partially overlaps compute — physically consistent.

**Metric note for future tests:** transfer delay under `timing_mode:
destination_missing` shows up in **TTFT / e2e latency**, not in steady-state
`throughput_per_user` (a one-time transfer amortizes over the output length) and
not cleanly in `mean_ttst_ms`. Use `mean_e2e_latency_ms` / `mean_ttft_ms` as the
transfer-sensitive signal.

## ConfigIQ standalone-sweep contract

The Pareto frontier is a standalone action, not a companion scan. `Pareto
frontier` and `Calculate` are alternative buttons: either action replaces the
other's result, and retrying re-runs only the active action. The contract below
is what the UI and service rely on.

### Availability

Both simulators advertise `pareto` in `GET /backends`. AISimulate samples the
frontier with the SDK. BLIS has no native Pareto command, so the service
enumerates topology and closed-loop concurrency combinations, runs each through
BLIS, and computes the non-dominated set itself. Each result reports the
simulator that produced it; the UI surfaces an explicit unsupported/error
response rather than substituting the other simulator. There is no
cross-simulator frontier.

### Deployment modes

Every sweep searches both aggregated and disaggregated deployments (when the
window can hold more than one GPU). A workload may legitimately have no feasible
disaggregated point; that absence is reported, not hidden. Both modes contribute
candidates and frontier points when they are feasible.

### Automatic GPU window

The browser does not send `min_candidate_gpus` or `max_candidate_gpus`. For
AISimulate, when both are omitted, the service resolves a model-aware window
before sampling:

- **Floor** — the smallest aggregated worker shape that holds one full-context
  sequence (`min(shape.gpus_per_worker)` from the KV-feasibility enumeration).
  Aggregated is a valid floor for both modes because a disaggregated deployment
  replicates the full weights in both pools.
- **Ceiling** — `clamp(max(16, floor × 2), floor, 64,
  AISIMULATORS_MAX_CANDIDATE_GPUS)`. The Pareto cap is configurable with
  `PARETO_MAX_WINDOW_GPUS` (default 64).

The resolved window is reported as `sweep.window_min_gpus` /
`sweep.window_max_gpus`; there is no user-facing window control. If no shape fits
inside the cap, the response is an explicit 422 that asks the caller for wider
bounds. If the preflight estimate itself fails, the service logs a warning and
falls back to a bounded `1..min(16, ceiling)` window rather than failing the
sweep.

API callers may still pin either bound explicitly. Sending either
`min_candidate_gpus` or `max_candidate_gpus` disables automatic resolution and
keeps the historical semantics; the server clamps an explicit maximum to
`AISIMULATORS_MAX_CANDIDATE_GPUS` (default 1024). BLIS uses the requested bounds
and its hardware ceiling directly; it does not resolve a model-aware window.

### Budgets and sampler

| Setting | Default | Purpose |
|---|---|---|
| `PARETO_MAX_TRIALS` | 20 | Total trial budget, split across searched deployment modes |
| `PARETO_PARALLELISM` | 8 | Concurrent candidate evaluations |
| `PARETO_CANDIDATE_TIMEOUT_S` | 8 | Per-wave wall budget (applied per evaluation wave, not per candidate) |
| `PARETO_MAX_CONCURRENCY` | 2 | Concurrent sweeps per worker; excess returns 429 |
| `PARETO_ALGORITHM` | `random` | `random` or `bayesian` |
| `PARETO_MAX_WINDOW_GPUS` | 64 | Cap for an automatically resolved window |
| `SWEEP_SEED` | 42 | Reproducible sampling and optimizer seed |

`bayesian` maps to `SeededBayesianBranchSampler`, which flips `jax_enable_x64`
process-wide. At the default budget of ~20 trials it gets only one or two GPU
optimization rounds, so it is not the default. The trial budget is split across
aggregated and disaggregated modes via `divmod`, so `max_trials` is a total, not
a per-mode count.

`sweep.budget_exhausted` is true when `evaluated >= max_trials`. It means the
trial budget was spent, never that the search enumerated every configuration.
The UI caption reads "sampled — not exhaustive" for a sampled sweep and
"evaluated — exhaustive over the enumerated grid" for an enumerated sweep
(BLIS), downgraded to a budget-exhaustion warning when the enumeration stopped
early.

### BLIS sweeps

BLIS drives closed-loop concurrency, so the service enumerates a load grid
(`BLIS_PARETO_CONCURRENCY_LEVELS`, default `1,2,4,8,16,32,64`) for every
topology shape that fits the GPU window. It runs each combination through BLIS
and computes `views.pareto_front` over throughput per user and throughput per
GPU. The response reports `sweep.algorithm="enumerate"`,
`candidate_configurations`, and `evaluated_configurations`.

- BLIS includes P/D topologies only when the request names a supported `fabric`.
  The web client sends the simulator's announced default fabric. Without one,
  the sweep is aggregate-only.
- `target_request_rate` and `target_concurrency` are rejected: the sweep owns
  its load. `kv_transfer_bandwidth_gb_per_second` is also rejected, because BLIS
  models transfer through the fabric.
- Each candidate records `simulator="blis"` and its measured metrics. Failed and
  timed-out runs stay in the ledger with an explicit status; no values are
  synthesized.
- The service is the only Pareto implementation for BLIS; the browser never
  recomputes `views.pareto_front` for the served objective pair.

### Candidate presentation

The chart plots feasible, measured candidates only. Failed, timed-out,
infeasible, and unsupported attempts remain in the response for transparency but
are collapsed behind `Show failed attempts (N)` in the UI, with a status summary
(evaluated, feasible, timed out, …) visible above the chart. A sweep with no
feasible candidate shows the same summary plus a retry action instead of an
empty chart.

### Dominance authority

`views.pareto_front` is computed by the simulator and is authoritative for the
simulator's own objectives: maximize `throughput_per_user` and
`throughput_per_gpu`. The browser uses that view unchanged.

When the user flips the x-axis to TTFT, the orientation changes (minimize TTFT,
maximize throughput per GPU), so the browser recomputes the non-dominated set
locally and labels the view as client-recomputed. It does not overwrite or
replace the server's view.

The served objective pair is fixed to (`throughput_per_user`,
`throughput_per_gpu`) because `pareto_objectives` is not reachable through
`run_recommendation`. A native SLA-free demand frontier (`min_gpus` +
`min_goodput_rps`) is blocked upstream: `goodput_*` metrics are only emitted
when an SLA exists, and the validators require an SLA for `min_gpus`.

### Demand-mode budgets

Throughput-demand sizing uses its own run control so it does not borrow the
closed-loop sizing budget:

| Setting | Default | Purpose |
|---|---|---|
| `DEMAND_MAX_TRIALS` | 16 | Total demand trial budget |
| `DEMAND_PARALLELISM` | 4 | Concurrent candidate evaluations |
| `DEMAND_CANDIDATE_TIMEOUT_S` | 15 | Per-candidate wall budget |
| `DEMAND_REQUESTS_PER_LOAD_UNIT` | 5 | Requests per load unit for the constant-rate stop rule |
| `DEMAND_SEARCH_BUDGET_S` | 50 | Whole-ladder wall budget |
| `DEMAND_WINDOW_START_GPUS` | 8 | First GPU window; doubles until met or capped |

Demand search walks a growing GPU window (`8, 16, 32, …` up to the effective
ceiling) and stops at the first window that sustains the rate. A window that
cannot hold the model is skipped; each window's trial and timeout budgets are
reduced to the time remaining. An explicit `max_candidate_gpus` pins a single
window. The 50-second ladder budget stays inside the 90-second gateway timeout.

Demand selection prefers the fewest GPUs that sustain the rate, then the highest
sustained rate, then the lowest E2E latency. When nothing complies, the response
is 200 with `met=false` and the best observed rate; a request with no feasible
measured candidate returns 422.

Compliance uses the arrival-window rate `completed / (duration - mean_e2e)`,
not `completed / duration`. A replay is timed to its last completion, so the
latter appends one response latency and marks even a correctly sized cluster
unmet. `demand.window_seconds` reports that same arrival window.

### Timing

A sweep is the only request in flight for the Pareto action, so it must fit the
gateway timeout on its own. The auto window keeps the search near the model's
fit floor, which materially reduces sparse curves and timeout noise compared
with a one-GPU-to-ceiling sweep. Measured wall times against `aisimulators.dev`
for 0.6B/8B/32B sweeps over 1–16 GPUs were 20–29 seconds with 20/20 feasible
candidates; a Qwen3-235B-A22B sweep from its 4-GPU floor over 4–16 GPUs took
21 seconds with 10 aggregated and 10 disaggregated candidates. The frontend
shows elapsed time and the gateway limit while the sweep runs, and a timeout
degrades only the frontier panel.

## Conclusion

Across model size, MoE sparsity, GPU class, context length, the Pareto trade-off,
disagg emergence under load, and P/D transfer bandwidth, every output moves the
way the physics demands. The feature emits genuine, correctly-behaving AISimulate
output — not cached or constant values.

## Note on running heavy sweeps

The pareto per-candidate wall budget is env-configurable:
`PARETO_CANDIDATE_TIMEOUT_S` (default 8, which fits the production gateway) and
`PARETO_MAX_TRIALS` (default 20). Raise them for slow hosts (e.g. the emulated
local container used here) so heavy candidates complete instead of being marked
`resource_limited`. Production behavior is unchanged at the defaults.

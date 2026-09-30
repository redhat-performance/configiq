# Hybrid Savings end-to-end audit

## Verdict

Hybrid Savings now implements a coherent, industry-aligned planning framework.
It is not an official Red Hat costing standard, a customer quote, or a
workload benchmark. The formulas, unit conversions, ranking logic, capacity
steps, and crossover calculations have been independently checked and are
internally consistent. The remaining uncertainty is primarily the evidence
feeding the framework: simulated performance, public cloud prices, and
indicative purchase prices still require workload-specific benchmark and
commercial validation.

## Findings

- ConfigIQ has no pre-AISimulators Hybrid Savings implementation to restore.
  AISimulators landed in `ed1f641`; Hybrid Savings was introduced later in
  `689ce48`. The standalone cost calculator is a product reference, not this
  page's historical implementation.
- Hosted API, rented infrastructure, and purchased hardware use the same
  monthly input/output token workload and representative request shape.
- AISimulators is used for model fit, topology, latency, and performance. It is
  not used as a source of cloud billing semantics or complete purchase prices.
- Only complete, compatible cloud instances and complete purchase-server
  records enter the cost ranking. Sizing-only systems remain visible but are
  not treated as zero-cost offers.
- Hosted offers are ranked with the selected input/output mix. The cheapest
  exact compatible offer is the default; changing provider recalculates the
  card, chart, and crossover statements.
- Full TCO and marginal cost are applied to all three paths at the same time.
- Rented and purchased curves retain whole-instance/server capacity steps.
  They are not smoothed into fractional hardware.
- Textual crossovers and chart markers are generated from the same piecewise
  cost functions.

## Problems found and fixes made

| Problem | Previous effect | Correction and verification |
| --- | --- | --- |
| Sparse workload demand was sent to AISimulators as the performance probe | Very low demand could be mistaken for system capacity | Use a stable, SLA-constrained concurrency-32 capacity probe. Verified through wrapper tests and live small/medium/large model runs. |
| Cluster throughput/GPU totals could be treated as per-replica values | Replicas or GPUs could be multiplied twice | Retain aggregate GPU totals; divide cluster throughput once by returned replica count. Reject inconsistent aggregate/disaggregated topology. |
| Monthly token mix and representative request mix could disagree | Using `(I + O) / (i + o)` could silently omit input or output work | Size with `max(I / i, O / o)` and surface the mismatch in the UI. Hosted billing still uses the entered input/output totals. |
| Scale-to-zero runtime used peak replicas as if they ran continuously | Artificial sawtooth spikes and overcharging near capacity boundaries | Bill modeled average work plus the runtime buffer; retain whole hardware only where the billing mode requires it. |
| Purchased energy used only active model GPUs | Spare GPUs installed in a complete server were not charged for power | Charge every installed GPU plus base server power, PUE, and the selected electricity rate. |
| Loose hosted aliases could match a base checkpoint to an instruction-tuned or quantized offer | A different checkpoint's price could be presented as exact | Restrict matching to safe explicit aliases and reject incompatible suffix variants. |
| Incomplete AISimulators responses could enter sizing | Invalid throughput, latency, or topology could affect ranking | Require positive GPU count, replica count, throughput, TTFT, and TPOT, plus a consistent topology. |
| Chart/crossover boundary caps could omit late infrastructure steps | A transition after an arbitrary sampled boundary could be missed | Search piecewise cost intervals for the first whole-token crossover through 1T tokens; use bounded chart samples only for display. If the search work budget is exhausted, label transitions **Not verified**, never **Not reached**. |
| The UI allowed workloads beyond the crossover search horizon | The current workload could lie outside the validated chart range | Cap the comparison input at the documented 1T-token planning horizon. |
| Cost-source staleness was not clearly surfaced | A technically valid calculation could look commercially current | Surface shared costing-source freshness warnings and retain source dates on local complete offers. |

## Before versus after the AISimulators integration

There is no earlier ConfigIQ Hybrid Savings implementation before AISimulators.
The material comparison is therefore between the initial Hybrid Savings
integration and the audited implementation:

```text
initial: current workload request rate used as the performance probe
audited: stable concurrency-32 capacity probe with the selected SLA

initial: cluster values could be scaled again as per-replica values
audited: cluster topology is validated and normalized exactly once

initial: simple combined-token request count
audited: conservative input/output-aware request count

initial: scale-to-zero could inherit peak whole-replica steps
audited: modeled average runtime plus the explicit runtime buffer
```

These are correctness fixes, not changes designed to make one deployment path
win.

## End-to-end calculation walkthrough

Live validation used Qwen3-235B-A22B with 2B monthly input tokens and 500M
monthly output tokens under the General assistant profile:

```text
billed tokens = 2.0B + 0.5B = 2.5B
average request = 2,048 input + 512 output = 2,560 tokens
monthly requests = max(2.0B / 2,048, 0.5B / 512)
                 = 976,562.5
peak request rate = 976,562.5 / (730 * 3,600) * 1.0
                  = 0.3716 requests/second
```

### Hosted API

The selected DeepInfra offer was $0.18/M input and $0.54/M output:

```text
input usage  = 2,000M * $0.18 = $360
output usage =   500M * $0.54 = $270
token usage                         = $630
operations (0.05 FTE * $18,000)    = $900
implementation ($15,000 / 36)      = $416.67
Full TCO                            = $1,946.67 (~$1,947)
effective cost                     = $1,946.67 / 2,500M
                                   = $0.779/M (~$0.78/M)
```

### Rented infrastructure

AISimulators sized nine of ten systems and ConfigIQ independently selected the
lowest-cost complete rented offer: one CoreWeave 4x GB200 instance, representing
four GPUs and one replica. The estimated candidate used 15.8% of its planned
capacity and satisfied the displayed latency constraints.

```text
modeled scale-to-zero compute       = $5,312
infrastructure and observability   =   $500
platform/model operations          = $3,600
implementation ($40,000 / 36)      = $1,111
Full TCO                            = $10,523
effective cost                     = $10,523 / 2,500M = $4.21/M
```

### Purchased hardware

ConfigIQ independently selected one complete 8-GPU A100 server. Acquisition,
power, facilities, and lifecycle charges apply to the whole installed server,
not only the currently active model GPUs.

```text
depreciation net of residual       = $2,667
cost of capital                    =   $793
installation and commissioning    =   $233
maintenance and spares             =   $667
power including PUE                =   $448
rack/facilities/infrastructure     = $1,700
platform/model operations          = $4,500
implementation amortization       = $1,389
Full TCO                            = $12,397
effective cost                     = $12,397 / 2,500M = $4.96/M
```

The app displayed the same rounded values. This example is a planning result,
not proof that the quoted hardware or cloud rates are commercially available.

## Independent arithmetic scenarios

A deterministic audit fixture verifies the formulas without trusting the live
feeds. It uses an 80/20 token mix, 4,000/1,000-token request shape, 100 output
tokens/second, 90% planning capacity, 100 active hours, and 2x peak demand:

```text
capacity = 100 * 5 * 0.90 * 100 * 3,600 / 2 = 81M billed tokens
hosted blended price = 0.8 * $1 + 0.2 * $3 = $1.40/M
rented deployment = 100h * $2/h = $200
purchased depreciation = $24,000 / 60 months = $400
```

Verified cases:

- 1M tokens: hosted $1.40, rented $200, purchased $400;
- 40.5M tokens: hosted $56.70, rented $200 at 50% utilization, purchased $400;
- 80.19M tokens: one rented and one purchased deployment at 99% utilization;
- 81,000,001 tokens: two deployments, so rented jumps to $400 and purchased to $800;
- piecewise hosted/rented crossover: exactly 235,714,286 tokens, with rented
  still more expensive one token earlier.

## Graph verification

The Hybrid Savings page has one comparison chart:

- X-axis: billed tokens per month, beginning at zero;
- Y-axis: monthly USD under the selected Full TCO or Marginal lens;
- green: selected hosted provider;
- red: lowest-cost eligible rented configuration at each plotted volume;
- orange: lowest-cost eligible purchased configuration at each plotted volume.

The chart recalculates the best eligible self-managed candidate at every
sampled volume, so a line can switch hardware/provider shape as demand grows.
Whole-instance/server additions are represented as vertical steps. Boundary
tests verify the value immediately before, at, and after a capacity threshold.
The provider selector and cost lens both update the graph and textual
milestones. Live checks covered Qwen3-8B, Qwen3-32B, and Qwen3-235B-A22B.

## Validation evidence

- TypeScript/Vitest: 16 files and 200 tests passed after the Next.js 15 and
  PatternFly 6 merge.
- AISimulators wrapper and real integration: 95 tests passed, one optional
  telemetry test skipped. The candidate-GPU limit was reduced to eight for
  the local 4 GB Podman VM; the production default remains 1,024.
- TypeScript type-check passed after the production build generated Next types.
- Production Next.js build passed.
- ESLint passed with two pre-existing warnings outside Hybrid Savings
  (`cluster-cost` hook dependency and global font loading).
- Live UI: provider switching, Full TCO/Marginal switching, cost cards,
  breakdowns, progress, small/medium/large models, chart, and milestones were
  exercised successfully.

## Remaining assumptions and limitations

- AISimulators outputs are estimates unless backed by a ConfigIQ `Tested`
  configuration; production decisions still need a workload-matched benchmark.
- The complete cloud-instance and purchase-server records are currently local,
  dated planning records where the shared costing API lacks complete billing
  semantics. Indicative purchase prices require quotes.
- Default scale-to-zero and 1x peak are explicit scenario choices, not a
  neutral enterprise baseline. Startup time and provider minimum billing can
  make scale-to-zero less attractive in practice.
- No automatic high-availability/failure reserve is included.
- CPU serving, retry traffic, cache discounts, storage, network/data transfer,
  taxes, private contract prices, commitment discounts, and Red Hat
  subscriptions are outside the current default model unless entered as an
  explicit cost.
- Large-model sizing can take roughly two minutes when one GPU family consumes
  most of its allowed search time. Progress remains accurate, but latency is an
  operational limitation rather than a calculation error.
- `Not reached` means no transition within the 1T-token planning horizon.

# Hybrid Savings calculation methodology

## Status and purpose

Hybrid Savings is a **custom, industry-aligned planning model**. It is not an
official Red Hat costing standard, a customer quote, or a substitute for a
workload benchmark. It compares three ways of serving the same open-weight
model and workload:

1. a hosted token API;
2. rented GPU infrastructure; and
3. purchased GPU servers.

The ordering of results is an output, not an assumption. The implementation
does not force hosted, rented, or purchased infrastructure to win.

## Evidence boundary

The result combines several evidence types that must not be confused:

- AISimulators provides model fit, topology, latency, and estimated serving
  performance for a model/system combination.
- The costing service provides hosted-token prices and shared GPU cost data.
- ConfigIQ maps sized systems to complete rented instances and complete
  purchased-server configurations.
- User-editable assumptions add planning headroom, labour, implementation,
  facilities, energy, capital, and lifecycle treatment.

A configuration marked `Tested` elsewhere in ConfigIQ has real-hardware
benchmark evidence. An AISimulators estimate is still a planning estimate.
Current commercial rates and a workload-matched benchmark are required before
a production or purchasing decision.

## End-to-end calculation

```text
monthly input/output tokens
  -> request count and peak request rate
  -> AISimulators model/system fit and performance
  -> per-replica monthly token capacity
  -> whole replicas, cloud instances, or purchased servers
  -> direct cost plus the selected cost lens
  -> monthly cost and effective cost
  -> piecewise chart points and crossover statements
```

### Workload normalization

Let:

- `I` = monthly input tokens;
- `O` = monthly output tokens;
- `i` and `o` = average input and output tokens per request;
- `H` = processing hours per month; and
- `P` = peak-to-average demand multiplier.

Then:

```text
billed tokens = I + O
monthly requests = max(I / i, O / o)
peak requests/second = monthly requests / (H * 3,600) * P
```

The entered monthly token totals remain the hosted billing basis. The request
shape is used for AISimulators sizing and infrastructure capacity. When the
monthly input/output ratio differs from the representative request ratio, the
larger of the two implied request counts is used. This is conservative and
prevents either input or output work from silently disappearing; the UI asks
the user to align the ratios for a tighter estimate.

Hybrid Savings starts with 2.4B input and 150M output tokens/month (2.55B
total), matching the shared Default request shape of 2,048 input and 128 output
tokens. Both monthly volumes imply exactly 1,171,875 requests, so the default
workload mix is aligned. Selecting a shared preset changes only the input and
output lengths per request. It leaves monthly volumes, latency targets,
concurrency, and cached prefix unchanged.

## AISimulators field contract

Hybrid Savings requests each supported GPU system using the selected request
lengths, cached prefix, serving backend, simulator and reference concurrency
(one concurrent request initially), enforcing the selected TTFT and TPOT
limits. This is modeled performance at that reference load, not evidence of
maximum server throughput. A concurrency-one result describes that load;
increasing concurrency is not guaranteed to preserve latency or scale linearly.
Calibrate with a benchmark matching the intended workload and latency targets.

| AISimulators value | Meaning used by Hybrid Savings | Transformation |
| --- | --- | --- |
| `total_gpus_needed` / `used_gpus` | GPUs in the complete returned candidate cluster | Retained as the cluster total; never multiplied by replicas again |
| `replicas_needed` | Number of replicas in the returned candidate cluster | Adapted to a positive whole replica count; incomplete results cannot be priced |
| `tp`, `pp`, `dp`, `cp` | Parallelism dimensions for one aggregate-serving replica | Product gives GPUs per replica |
| `tokens_per_second` / output throughput | Output-token throughput for the complete candidate cluster | Divided by returned replicas to obtain scalable per-replica throughput |
| per-GPU output throughput | Diagnostic rate | Not used as a second capacity multiplier |
| `concurrency` | Concurrency represented by the complete candidate | Display/diagnostic value; not multiplied again |
| `ttft` | Time to first token, milliseconds | Must be positive and satisfy the requested limit |
| `tpot` | Time per output token, milliseconds/token | Must be positive and satisfy the requested limit |
| request latency | End-to-end request latency when that target is used | Must satisfy the requested limit |

Recommendations missing positive GPU count, replica count, throughput, TTFT,
or TPOT are rejected. A successfully sized system without a complete price is
retained as sizing-only and cannot enter a cost ranking. A response from a
different simulator is rejected, never used as a fallback.

### Capacity conversion

For a valid candidate:

```text
output tokens/second/replica = cluster output tokens/second / cluster replicas
requests/second/replica = output tokens/second/replica / o
requests/billed token = max(input share / i, output share / o)
capacity/replica = requests/second/replica
                   * planning capacity percentage
                   * H * 3,600
                   / P
                   / requests/billed token
required replicas = ceil(billed tokens / capacity/replica)
```

The default planning-capacity percentage is 90%, so 10% of estimated
throughput remains as headroom. This is applied once.

## Cost paths

### Hosted API

The selected checkpoint must have an exact compatible hosted offer. Prefix and
case normalization are allowed; a base model is not silently priced as its
instruction-tuned variant.

```text
hosted usage = input tokens / 1M * input price
             + output tokens / 1M * output price
             + monthly requests * price per API request
full TCO = usage + fixed fees + operations + implementation / analysis months
marginal = usage + fixed fees
```

Provider offers are ranked using the same input/output workload. The cheapest
valid offer is selected initially, and changing provider recalculates the
cards, chart, and crossovers.

The supported tariff is **standard synchronous, uncached text** in USD:

- AICostings normalizes USD/token to USD/million tokens once, without rounding
  very small non-zero rates to zero. Per-request prices remain USD/request.
- Applicable long-context rates use the representative input length, not
  monthly token volume. A threshold is strictly exceeded; ordered partial
  overrides inherit unchanged rates. Cards, provider ranking and crossovers
  use the same resolved tariff.
- Published input, output and combined-context limits are checked separately.
  LiteLLM's legacy `context_window` is an input limit. Its ambiguous
  `max_tokens` field is not guessed to be a combined-context limit.
- Cache read/write rates are retained as metadata but not applied: simulator
  prefix reuse is not a measured hosted cache-hit/write volume. Batch, tools,
  audio/image and other optional modalities are not included.
- Offers with unsupported token-pricing conditions (for example time-window
  rates), invalid additional charges or separately priced reasoning tokens are
  not eligible for this comparison. They are not silently charged at base rates.
- OpenRouter is a routing catalogue, not a fixed serving-endpoint quote. Its
  model-author namespace is distinct from a serving provider. Legacy API fields
  and source/override precedence are preserved for other consumers.
- Invalid hosted-catalogue responses retain the client's last-good data and
  expose refresh failure/staleness. Older responses without billing metadata
  remain compatible, but cannot convey the new conditions; deploy the updated
  AICostings service as well as the frontend to obtain these accuracy checks.

### Rented infrastructure

Only complete mapped cloud offers can be priced. Replica topology is packed
into whole instances, subject to any multi-instance/interconnect limit.

- `Scale to zero` bills modeled serving time plus the runtime buffer. Average
  work, rather than peak replicas running continuously, determines runtime.
- `Active window` keeps the required whole instances warm for `H` hours.
- `Always on` bills the required whole instances for the configured monthly
  hours (730 by default).
- Capacity-block offers are always-on because the committed block cannot be
  modeled as per-request scale-to-zero consumption.

Scale-to-zero is an optimistic assumption unless instances really stop between
workloads. The runtime buffer is an allowance, not a measured startup/idle
duration or an implementation of provider per-start minimum billing. Use Active
window or Always on when instances remain running. Persistent storage and
network charges must be supplied explicitly; stopping compute does not remove
them automatically.

Full TCO adds infrastructure/observability, operations, implementation, and
other explicit fixed costs. Marginal cost retains compute and recurring direct
infrastructure but excludes operations and initial implementation.

### Purchased hardware

Only complete compatible server configurations can be priced. Replicas are
packed into whole servers; spare installed GPUs remain part of acquisition,
facilities, maintenance, and energy planning.

```text
acquisition = complete server price * server count
hardware depreciation = hardware book value consumed across held lifecycle periods
                        / analysis months
capital charge = sum over held periods of
                 average(opening book value, ending book value)
                 * annual capital % * held months / 12
                 / analysis months
maintenance = acquisition * annual maintenance % / 12
energy = (base server watts * servers + GPU TDP watts * installed GPUs)
         / 1,000 * monthly hours * PUE * electricity price/kWh
```

Book value includes hardware and unamortized installation. The calculation
accounts for replacement cycles when the analysis period exceeds hardware
life, and subtracts the assumed hardware residual value.

Full TCO includes depreciation, capital, commissioning, maintenance, energy,
facilities, operations, implementation, and other explicit costs. Marginal
cost treats hardware acquisition and initial implementation as already
committed, while retaining maintenance, energy, facilities, and recurring
direct costs. GPU TDP is a conservative planning proxy, not a metered power
measurement.

The GPU TDP and base-server allowance are not whole-system AC measurements.
Missing/invalid GPU power is not treated as zero electricity cost: that
purchased option is excluded while electricity has a non-zero price. Facilities
charges must exclude electricity/cooling already modeled through power and
PUE, otherwise the same expense is counted twice. The complete server's
inclusions, source date and indicative/quote status are retained in pricing
data; no new configuration-specific quotes are inferred from vendor links.

Shared infrastructure and other monthly costs are flat entered deployment
totals, not automatic per-node or per-GPU subscriptions. Enter the correct
scaled contract total for the deployment. Without an explicit contract basis
and rate the calculator cannot safely scale licensing as the graph adds
hardware. Depreciation is an economic allocation, not an extra purchase: do not
add full CAPEX again to a total that already includes depreciation.

## Graph and crossover method

The growing-demand X-axis is total input + output tokens/month, at a fixed mix.
The Y-axis is effective USD per million requested output tokens for the selected
cost lens; the numerator includes input processing and all included costs.
Zero-output unit costs are unavailable, never shown as free. At every plotted
volume ConfigIQ recalculates:

- hosted cost using the selected provider;
- the lowest-cost eligible rented candidate; and
- the lowest-cost eligible purchased candidate.

Rented and purchased series are piecewise because whole replicas, instances,
and servers are discrete. The chart samples a bounded set of volumes, including
representative capacity boundaries and crossover points, so very dense steps
may be visually compressed; plotted costs still use whole deployments.

The separate crossover search checks cost-curve intervals from low to high
volume, prunes intervals that cannot contain a win, and finds the first
whole-token match within affine intervals. It does not stop after an arbitrary
number of capacity boundaries. A shared 10,000-interval work budget prevents
pathological inputs from blocking the page: **Not verified** means that budget
was exhausted, whereas **Not reached** means no crossover was found through
the 1T-token planning horizon. `Cheaper than hosted` and `lowest cost overall`
are separate milestones. Every plotted cost is recalculated from the same
hosted, rented, and purchased cost functions used for the current workload.

The hardware-usage view freezes the chosen physical deployment instead of
adding hardware with demand. Its percentage is modeled capacity use at the
selected request shape and reference concurrency, not measured GPU utilization.

## Full TCO and marginal cost

The selected lens is applied to all three deployment paths at the same time.
The application never compares Full TCO for one path with marginal cost for
another.

- Use **Full TCO** for a new sourcing or deployment decision.
- Use **Marginal** only when acquisition and initial implementation are
  already committed and the question is about incremental operating cost.

## Historical baseline and migration assessment

The ConfigIQ Hybrid Savings implementation was introduced in commit `689ce48`,
after the repository had already migrated sizing to AISimulators in commit
`ed1f641`. Therefore this repository has no pre-AISim Hybrid Savings behavior
to restore.

The earlier standalone LLM cost calculator remains a useful product baseline.
It modeled additional concerns such as retries, cache pricing, high-availability
reserve, CPU paths, Red Hat subscriptions, quality gates, and uncertainty. The
current ConfigIQ page intentionally began with a leaner GPU planning scope.
Those omissions are limitations to disclose, not evidence that AISimulators
replaced those calculations.

The material AISim integration correction is:

```text
old: the current workload's often-tiny target request rate was used as the
     performance probe, and cluster metrics could be treated as scalable
     per-replica metrics
new: selected reference-concurrency probe; cluster throughput is normalized once by
     the returned replica count; topology is not multiplied twice
result: sparse demand is no longer mistaken for server capacity and GPU cost is
        not duplicated
```

## Current limitations

- No automatic high-availability or failure-reserve replica is included.
- CPU serving paths are outside the present Hybrid Savings scope.
- Retry traffic, cache discounts, storage, data transfer, networking, taxes,
  commitment discounts, private contract rates, and Red Hat subscriptions are
  included only if represented by an explicit user-entered cost.
- Hosted reasoning-token volumes and cache-hit/write volumes are not inferred
  from visible output lengths or the self-managed prefix setting. Eligible
  prices represent only the supported standard-text tariff, not every possible
  vendor billing mode or production invoice.
- Scale-to-zero results depend on real startup time and provider minimum billing.
- Hosted services and self-managed deployments may provide different support,
  availability, and governance unless those requirements are added explicitly.
- Public and indicative prices must be commercially validated. Stale feed state
  is surfaced in the UI, but a fresh scraper run does not guarantee a negotiated
  customer rate.
- The chart searches up to 1T billed tokens/month. `Not reached` means not
  reached within that horizon, not that a transition is impossible forever.

## Review checklist

Before relying on a result, confirm:

1. the exact model checkpoint and hosted provider;
2. the latency and workload shape;
3. the rented billing policy and provider minimums;
4. complete instance/server pricing and source freshness;
5. the Full TCO or marginal decision context;
6. availability and failover requirements; and
7. a workload-matched benchmark and current commercial quote.

## Reference guidance

- [NVIDIA inference-cost methodology](https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/):
  workload/latency-specific throughput, deployment sizing, then cost.
- [MLPerf inference datacenter](https://mlcommons.org/benchmarks/inference-datacenter/):
  distinguish measured benchmark performance and whole-system power from proxies.
- [OpenRouter model/pricing contract](https://openrouter.ai/docs/guides/overview/models):
  token rates, request fees, context limits and conditional pricing overrides.
- [LiteLLM public cost map](https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json):
  provider-specific standard rates, token limits and long-context fields.
- [AWS instance stop/start billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-stop-start-works.html):
  running-instance billing and charges that persist after stopping compute.

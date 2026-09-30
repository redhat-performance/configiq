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

## AISimulators field contract

Hybrid Savings requests each supported GPU system at a stable capacity probe
of 32 concurrent requests while enforcing the selected TTFT and TPOT limits.
This measures multi-request serving capacity; concurrency 1 is a single-user
latency/decode result and materially understates server capacity.

| AISimulators value | Meaning used by Hybrid Savings | Transformation |
| --- | --- | --- |
| `total_gpus_needed` / `used_gpus` | GPUs in the complete returned candidate cluster | Retained as the cluster total; never multiplied by replicas again |
| `replicas_needed` | Number of replicas in the returned candidate cluster | Rounded up to a whole replica |
| `tp`, `pp`, `dp`, `cp` | Parallelism dimensions for one aggregate-serving replica | Product gives GPUs per replica |
| `tokens_per_second` / output throughput | Output-token throughput for the complete candidate cluster | Divided by returned replicas to obtain scalable per-replica throughput |
| per-GPU output throughput | Diagnostic rate | Not used as a second capacity multiplier |
| `concurrency` | Concurrency represented by the complete candidate | Display/diagnostic value; not multiplied again |
| `ttft` | Time to first token, milliseconds | Must be positive and satisfy the requested limit |
| `tpot` | Time per output token, milliseconds/token | Must be positive and satisfy the requested limit |
| request latency | End-to-end request latency when that target is used | Must satisfy the requested limit |

Recommendations missing positive GPU count, replica count, throughput, TTFT,
or TPOT are rejected. A successfully sized system without a complete price is
retained as sizing-only and cannot enter a cost ranking.

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
full TCO = usage + fixed fees + operations + implementation / analysis months
marginal = usage + fixed fees
```

Provider offers are ranked using the same input/output workload. The cheapest
valid offer is selected initially, and changing provider recalculates the
cards, chart, and crossovers.

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

## Graph and crossover method

The X-axis is billed tokens per month and the Y-axis is monthly cost in USD for
the selected cost lens. At every plotted volume ConfigIQ recalculates:

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
new: concurrency 32 capacity probe; cluster throughput is normalized once by
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

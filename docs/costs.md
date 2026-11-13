# Costs

ConfigIQ compares three ways to serve the same open-weight model and workload:

1. a **hosted** token API, billed per token;
2. **rented** GPU infrastructure, billed per instance-hour; and
3. **purchased** GPU servers, amortised over a hardware life.

The ordering is an output of the model, not an assumption. Nothing in the
calculation forces hosted, rented, or purchased infrastructure to win.

## What the number includes

| Lens | Includes | Excludes |
|---|---|---|
| Hosted | Input and output token prices for the selected model and feed | Egress, storage, rate-limit overage, managed-service fees |
| Rented | Complete instance price for the sized topology | Egress, storage, support uplift, staff |
| Purchased | Hardware amortisation, power and cooling (PUE), maintenance, and the planning overheads you enable | Networking, datacentre space, staff, unless you add them |

Cost is calculated only for whole replicas, instances, or servers. A partial GPU
is not a purchasable unit, so the result rounds up and states the headroom that
rounding created.

## Planning assumptions

Assumptions are edited on the Sources page under cost assumptions. They are
stored in this browser, not on the server, and they apply to this user's
calculations only. Defaults are defined in `lib/costing-assumptions.ts`.

The assumptions that move the result most are hardware life, electricity price
and PUE, annual maintenance, loaded staff cost, and the utilisation you assume.
Changing them changes the comparison; they are not hidden constants.

## Pricing data

Hosted model prices and shared GPU prices come from the aicostings service,
through selectable pricing feeds. The selected feed is named on the result, and
the Sources page records where each figure came from.

Pricing is cost data. It never changes a sized GPU count: the topology comes
from AISimulators, and the price is applied to it.

## What this is not

Hybrid Savings is an industry-aligned planning model, not a quote. A
configuration marked `Tested` elsewhere in ConfigIQ has real-hardware benchmark
evidence; a sizing estimate is still an estimate. Before a purchase or a
production commitment, confirm current commercial rates and run a
workload-matched benchmark.

## Related

- [User scenarios](user-scenarios.md): which sizing tool to use
- [Tested models](tested-models.md): what "validated" means

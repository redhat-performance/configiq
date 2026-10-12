# Shared cost planning assumptions

Sources (`/sources#cost-assumptions`) is the editor for planning overrides.
`CostAssumptionsProvider` owns the shared state; validated numeric overrides are
stored under `configiq_cost_assumptions_v1` in this browser. This is not a
server-wide or organisation-wide configuration. Navigation and reload retain
the values; storage events synchronize other tabs on the same origin.

`lib/costing-assumptions.ts` defines the defaults, editable fields and bounds.
Hybrid Savings uses all these inputs in both its result cards and chart. The
existing numeric defaults are unchanged. Cost view and rented billing policy
remain workload-specific controls on Hybrid Savings.

Cluster Cost consumes explicit overrides only for equivalent inputs:

- Hardware life in years → depreciation years.
- Electricity price and PUE → power calculation.
- Annual maintenance percent → annual hardware support/warranty fraction.
- Loaded monthly staff cost → annual staff cost (× 12).

Cluster Cost retains its existing defaults for unset overrides and its own
cluster-specific inputs. A shared override is read-only there, with a Sources
link, so two editors cannot silently disagree. Resetting Sources overrides
restores each tool's original defaults, not identical cross-tool defaults.

Routing Economics currently calculates API token usage and GPU rental charges,
not full TCO. No depreciation, staffing or implementation costs are injected
into that calculation. Existing on-prem customer profiles in Sources are a
separate feature; this change does not silently activate or reinterpret them.

Validation rejects blank, non-finite, negative and out-of-range values without
changing the last valid calculation. Malformed or inaccessible browser storage
falls back safely; the editor warns when persistence is unreliable. Pricing
feeds, catalogue prices and AISimulators sizing are unchanged.

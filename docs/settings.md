# Settings and simulators

Settings holds the choices that apply across every tool: the simulator, the
inference backend, the backend version, and the defaults a new view starts
from.

## Simulator

A **simulator** is the implementation that produces an estimate. It is not the
serving engine: `vllm`, `sglang`, and `tensorrt-llm` are backends that a
simulator models, and several simulators can model the same backend.

The simulator list comes from `GET /simulators`, not from this app. Each entry
has an id, a label, a version, and an optional revision. The entry marked
`default` is used when no explicit choice has been saved.

Switching simulators changes which models, GPU systems, backends, fabrics, and
objectives are offered. The app reads those capabilities from the catalog and
never assumes that two simulators support the same set. If a requested
combination is unsupported, the API returns an explicit error; the app does not
substitute another simulator or invent a value.

## Backend and version

A backend version pin means something different in each simulator, so versions
are stored per simulator. When you switch simulators, the app loads the version
saved for that simulator rather than carrying the previous one across.

Some simulators publish a `system_defaults` entry per backend version. Leaving
the version empty lets the service resolve that default. Pinning a version is
how you reproduce a specific engine release.

## The simulator badge

Every result carries a badge naming the simulator that produced it. The badge is
read from the response, not inferred from your settings, because the service is
the only place that decides which simulator ran. Treat it as provenance: two
results with the same inputs and different badges are not expected to match.

## Sharing a view

A shared link records the simulator, backend, and backend version used to build
it. Opening a link never changes your saved settings:

- If the link's simulator matches yours, its inputs are applied.
- If it differs, the page shows a banner instead of switching. The banner names
  both simulators and offers a one-click **Use … for this view** action, which
  applies the link's provenance for the current session only.
- If the link's simulator is unavailable in this deployment, the existing
  unavailable notice is shown. There is no fallback.

## Related

- [User scenarios](user-scenarios.md): which tool to use for which ask
- [REST API](api.md): catalog endpoints and response provenance

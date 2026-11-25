# ADR: Render deployment artifacts from evaluated engine limits

**Date**: 2026-11-25
**Status**: Accepted

## Context

ConfigIQ answers what a workload needs, not how to run it. Every result ends at
a configuration — GPUs, tensor and expert parallelism, sequence and token
limits — that a reader then hand-translates into a Kubernetes manifest. That
translation is where an evaluated answer stops being reproducible: a mistyped
`max-num-seqs`, a stale image, or a copied overlay for a different release
produces a deployment that is close enough to look right and different enough to
invalidate the measurement the user chose it for.

The tempting shortcut is a "starting configuration" template: string the request
into a manifest, tell the user to adjust it. That is worse than nothing, because
it looks authoritative while containing values nobody evaluated. The output we
want is the opposite — a bundle that either deploys exactly the measured
configuration or refuses with a reason.

Two upstream projects cover the well-lit paths we care about: NVIDIA Dynamo
(`DynamoGraphDeployment`) and llm-d v1.0.0 (a Kustomize model-server overlay
plus a Helm router). Both are pinned releases, not moving targets, and both
reject request shapes we would otherwise have to encode in the browser.

## Decision

**The service renders; the browser never assembles manifests.** Two endpoints
live in AISimulators (`services/aisimulators/tools/api_service/deploy/`):
`GET /deploy/targets` lists what this deployment can render with its pins and
availability, and `POST /deploy` renders one bundle from an
already-evaluated `engine` section. No new service, and no sizing formula moves
to the client. The Next.js pages call same-origin `app/api/deploy*` proxies, and
the MCP gateway exposes the same two operations from the first release.

**The input is a reported result, not a request.** `include=engine` publishes
the limits a run actually evaluated, each with a source: `evaluated`,
`requested`, `backend-default`, or `not-modeled`. `POST /deploy` renders that
section verbatim. A target that would have to render a `not-modeled` limit
refuses with 422 rather than filling in a default, so a rendered manifest is a
statement about the run that produced it. The service reads the limits from the
compiled replay spec (`spec.backend_deployment.<role>_engine_args`), never
re-deriving them from the original request.

**llm-d v1.0.0 is vendored; Dynamo is generated.** The llm-d overlay is only
valid against the exact commit its release tag points at, so the bundle carries
the requested mode's Kustomize closure (aggregated and disaggregated differ) at
repo-relative depth. Aggregated mode renders a `Deployment`; disaggregated mode
renders a `DisaggregatedSet` plus the pinned routing sidecar. The llm-d
`ModelService` chart is deliberately not used. The Dynamo bundle is a single
`DynamoGraphDeployment`, and its model-server image is discovered from the
rendered manifest instead of being hardcoded.

**A bundle is verified before it is returned.** Every artifact is assembled in a
scratch directory and gate-checked: `kustomize build` succeeds (or the
single-manifest assertions read it directly), the expected kinds render, every
evaluated limit and replica count appears in the rendered argv, the images match
the pins, and no upstream `REPLACE_*` placeholder survives. The bundle also
carries `install.sh`, a read-only `preflight.sh` that fails rather than
installing onto a cluster missing its CRDs, GPUs, or Helm, and a `verify.sh`
that waits for rollout and queries `/v1/models`. A failed check is a 500 naming
the check, never a bundle that breaks on apply. CI runs the same gate without a
cluster.

**Engine-release disagreement is explicit.** Each target's catalog entry pins
the engine release its image runs. A mismatch with the evaluated release is
refused unless the caller sets `allow_engine_version_mismatch` (and supplies the
`image` it is overriding with). The departure is recorded in `PROVENANCE.md`,
printed by `preflight.sh`, and returned as a warning, so it is visible in the
artifact rather than resolved silently.

**The UI gate is experimental and hides the action only.** Settings owns a
`deploymentEnabled` toggle. When it is off, Predict, Recommend, and Saved
results do not offer the deploy action. The REST and MCP endpoints are not
gated: they are the documented integration points and the artifact shape is
pinned by its own schema version.

**JSON for display, zip for download.** The dialog renders the JSON form, which
lets it show the checks, pins, and warnings and makes a refusal legible. The
download makes a second `?format=zip` request so the service's zip writer
preserves the scripts' executable bit instead of the client reconstructing it.

**Comparison snapshots carry the engine section.** A saved result stores the
evaluated limits and the model's `config.json` under an optional `deployment`
field (`schemaVersion` stays 1; old entries simply lack it). Saved before
artifacts existed means "reopen and save again"; saved with no reported limits
means the run cannot produce an artifact. The two are explained differently.

## Consequences

- A deployable bundle is only available for a result that reported engine
  limits. This is intentional: no limits, no artifact, rather than a plausible
  guess.
- Adding a deploy target means adding a pinned target entry, a renderer, a
  vendor/fetch step, and gate assertions — not a template in the browser.
- The vendored llm-d tree and a `kustomize` binary are deployment inputs. When
  they are absent, the target reports `available: false` with the reason and
  answers 503 instead of returning a partial bundle.
- The frontend stays simulator-generic: it forwards the `engine` section it was
  given and renders whatever provenance, pins, and checks the service returns.

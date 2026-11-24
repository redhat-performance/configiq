# Simulator scorecard

The simulator scorecard answers one question: when a simulator is asked to
predict a run that was measured, how close does it get? It covers every model,
every GPU, and every concurrency level in the measured datasets, for every
simulator the AISimulators service publishes.

A full sweep is thousands of simulator calls, so it cannot run in a page. A
scheduled job builds a derived artifact once a day and publishes it; the page
reads that artifact.

- Page: `/validation/scorecard`
- Artifact: `redhat-performance/configiq-simulator-scorecard` on Hugging Face,
  one directory per channel (`dev/`, `prod/`)
- Job: `.github/workflows/scorecard.yml`, daily against the dev gateway

## What is compared

Each point pairs a published measurement with a fresh prediction for the same
configuration:

| Field | Source |
| --- | --- |
| Model, GPU, input length, output length, tensor parallel size, concurrency | Measured row (sent as-is) |
| Serving backend and engine release | Measured row, when the catalogue lists that release |
| Pipeline parallel (`pp`), data parallel (`dp`) | Fixed at 1 — the dataset does not record them |
| Expert parallel and expert tensor widths | Resolved by the simulator |

Two things the measured row records are deliberately **not** sent, because the
request contract does not express them: precision or weight quantisation, and
speculative decoding or prefix caching. A run measured with those enabled is
therefore compared against a prediction made with the service's own defaults.
That mismatch shows up as bias rather than as a wrong measurement, and it is
largest on prefill-heavy workloads where precision moves compute time most.

The parallelism defaults make the number a statement about **agreement under
these inputs**, not an absolute accuracy claim. A reader's own parallelism and
quantisation will move the prediction.

### Release matching

The measured row names the engine release it ran. When the simulator's catalogue
lists that release for that GPU, the request names it and the point is tagged
`matched`. When it does not, the request omits the field so the service resolves
its own default, and the point is tagged as a mixed-release comparison.

`matched` is therefore a **dimension of the view, not a filter applied while
scoring**. Both slices are published:

- **Like for release** — only points where the simulator predicted with the
  measured release. This is the fair comparison.
- **All releases** — every point, including runs on releases the simulator does
  not model. A large error here can be the release gap rather than the model.

Both are shown because neither alone is honest: the first can cover very little
of the measured population, and the second mixes two different questions.

## Statistics

Errors are relative to the measurement:

```
APE        = |predicted − measured| / measured
signed     = (predicted − measured) / measured
```

Aggregation is **per configuration first, then across configurations**. A run
swept at twelve concurrency levels does not outvote one swept at three.

| Statistic | Meaning |
| --- | --- |
| MAPE | Mean of per-configuration mean APE |
| MdAPE | Median of per-configuration mean APE |
| P90 | 90th percentile of per-configuration mean APE |
| Bias | Mean of per-configuration mean signed error; negative means the simulator predicts faster than the measurement |

A point whose measured value is below a floor (1 ms TTFT, 0.05 ms TPOT) is not
scored: dividing by a sub-millisecond value turns measurement noise into a
900% error that dominates every mean.

The page opens on **TPOT** and **MdAPE**. TPOT is the metric a reader tunes at
steady state, and MdAPE is the headline because a handful of extreme misses
should not decide the ranking. MAPE, P90, and Bias sit beside it so the tail
stays visible: a wide gap between MdAPE and MAPE means those extremes, not the
typical run, drive the mean. The metric and statistic selectors change the
chart and tables too, so what is ranked is what is charted.

The error chart keeps every configuration with a prediction, worst first, and
scrolls inside its tile rather than capping the list. The table behind the flip
lists the same cells for exact reading.

## Coverage

Every attempt is counted, in four buckets:

| Bucket | Meaning |
| --- | --- |
| `supported` | The simulator returned a prediction |
| `unsupported` | The catalogue lists the parts, but the run is beyond what the simulator models |
| `inconclusive` | We asked and did not get a prediction — a timeout, a rate limit, or a service failure |
| `ineligible` | The run is outside the simulator's catalogue entirely (no such GPU, model, or backend) |

Keeping `inconclusive` separate from `unsupported` is deliberate: a timeout is
our outage, not the simulator's capability gap. Keeping `ineligible` separate
from `unsupported` is the same idea in the other direction: a GPU the catalogue
never lists is not a judgement about the hardware.

The job never falls back to a different simulator. If a requested simulator does
not answer, the point is `inconclusive` and the reason is published.

### Coverage map

The page draws the measured pairs as two two-set diagrams when the snapshot has
exactly two simulators: one for the simulator pair, and one for the measured
source pair. One dot is a model + GPU pair, not a model alone. A simulator can
score one GPU for a model and not another, and that difference is the
comparison.

The universe comes from `coverage`, so a pair that no simulator scored still
appears in the measured-data figure as a coverage gap. A pair sits in a
simulator's circle when that simulator returned a prediction for at least one
measured level of the pair. Selecting a scored dot sets the model and GPU
filters and clears the workload, because the dot names a pair, not a shape. The
table under the figures carries the exact four-way membership.

The measured-data lens is **pair overlap, not workload overlap**. One source can
measure a model and GPU at 8,000 input / 1,000 output while another measures the
same pair at 8,192 / 1,024. When both sources loaded, the artifact publishes
`sourceOverlap.sharedPairs` and `sourceOverlap.sharedWorkloads` as separate
counts, and the page shows the shared-workload count beside the figure rather
than letting the lens imply a like-for-like comparison. That count covers the
whole snapshot: the circles and lens follow the selected source and release
slice, but the workload note does not, and the note says so. The field is
omitted when a source could not be read, and it is also omitted from snapshots
published before it existed; in both cases the page says the count was not
published instead of treating it as zero.

Both figures follow the source and release filters. The metric and statistic
selectors change the error numbers elsewhere; they do not change pair
membership.

## Head to head

Comparing two independent MAPEs over different coverage is not a comparison: the
simulator that skipped the hard points would look better. The head-to-head view
uses only the measured levels **both** simulators predicted, and reports how many
those were.

A row in the artifact is one measured configuration (source, model, GPU, workload)
for one simulator pair. It carries the number of paired concurrency levels, how
often each simulator was best, the ties, and the median gap between their errors. A tie
is equality to six decimal places of relative error; it counts for neither
simulator and stays visible as its own count.

The page shows the paired view by model and by GPU. Each row in those panels is
one configuration, and a category's share is the mean of its configuration best
shares — one configuration, one vote — not the pooled level count. A run swept at
twelve concurrency levels must not outvote one swept at three. The bracket spans
the lowest and highest configuration share, so a category whose mean hides a
wide spread looks different from one where every configuration agreed. A
category with a single configuration has no range to draw and says
`1 configuration` instead.

The **Benchmark data** selector applies to this view. The release selector does
not: the paired rows are built from every level both simulators predicted, across
releases, because the published paired view carries no release slice. The pair
selector appears only when the snapshot contains more than one simulator pair.

## Narrowing the view

Model and GPU are the same searchable controls Benchmark comparison uses: the
model picker groups by publisher, and the GPU picker groups the measured parts by
the catalog's vendor and architecture and shows their memory. Both list only the
values present in the selected slice — a part the snapshot did not measure is
not offered, because picking it could only produce an empty table. Workload is a
plain dropdown over the same slice.

The three filters narrow the error chart and its table; they do **not** change
the headline tiles, and they do not narrow head to head, which has its own
by-model and by-GPU panels. The artifact publishes totals pooled over every
configuration and nothing finer, so a per-model or per-GPU total would have to be
recomputed in the browser, from a different population than the published one.
The page says this in place rather than letting a filtered table sit under totals
that look unrelated to it.

## Where the numbers come from

Measurements are read from the two sources the rest of the app uses:

- **Red Hat** — `redhat-performance/configiq-performance-upstream-data`
- **InferenceX** — the `configiq-inferencex-comparison` measured snapshot. The
  server reads it through its deployment configuration; the artifact contains
  only derived per-point values, never the raw measured rows.

The artifact records the revision of each source, the gateway it called, the
simulator releases it used, and any leg that did not finish. A partial snapshot
says so; it never silently scores a narrower population.

## Running a sweep locally

```bash
# One model, a handful of points, against the dev gateway
npm run scorecard:build -- --sources redhat --models google/gemma-4-31B-it --limit 20

# Everything, with the incremental prediction cache
npm run scorecard:build -- --concurrency 6 --out artifacts/scorecard
```

The CLI writes `scorecard-<channel>.json`, `scorecard-points-<channel>.json`, and
a prediction cache. The cache is keyed by simulator, simulator release, and
request body, so a re-run only pays for new points or a changed simulator
release. Nothing is uploaded.

To preview an unpublished snapshot, serve the artifact directory and point the
app at it:

```bash
mkdir -p artifacts/serve/dev
cp artifacts/scorecard/scorecard-dev.json artifacts/serve/dev/scorecard.json
cp artifacts/scorecard/scorecard-points-dev.json artifacts/serve/dev/points.json
python3 -m http.server 9099 --directory artifacts/serve &
SCORECARD_ARTIFACT_BASE_URL=http://localhost:9099 npm run dev
```

`SCORECARD_ARTIFACT_BASE_URL` is server-side only; the browser never sees it.

## Publishing

One daily job against the dev gateway writes the `dev` channel. A release
promotes the latest dev snapshot to `prod` unchanged — it is the same data, so
there is no reason to re-run a sweep and produce a second, slightly different
answer.

Promotion is a copy, not a rebuild:

```bash
hf auth login
python scripts/publish-scorecard-snapshot.py --channel prod --from-channel dev
```

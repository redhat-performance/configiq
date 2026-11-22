# Tested-model UI regression tests

Playwright covers the browser-to-AISimulators gap in [issue #186](https://github.com/redhat-performance/configiq/issues/186).

## Coverage

Breadth follows the branch, not the release host: the commit-tracking dev
deployment carries the large suites, and the release-tracking production host
gets a small sanity check.

- The dev nightly suite has 470 cases. It checks the same 270 baseline cases and 200 extra control cases, using one selected GPU per model. The workflow run number rotates the extra cases through workloads; all shards in a run use the same rotation. The baseline still covers every tested model and workload each run.
- The dev weekly suite runs the full 2,900 scheduled cases: each of the 10 approved models on all nine workloads and all 10 GPU systems in Predict, KV cache, and Recommend latency sizing at Small (10) concurrency (2,700 baseline cases). Each model also runs the other four concurrency choices on those three paths, all five capacity choices, throughput demand, and both Pareto modes on one rotated workload/GPU pairing per control (200 extra cases).
- The release sanity suite on `https://configiq.io` has 33 cases. It runs the built-in Default workload on one sampled GPU per approved model in Predict, KV cache, and Recommend latency sizing, plus one capacity, one throughput-demand, and one Pareto case. It is deliberately small: the heavy matrices belong on dev.
- The full matrices remain available for manual runs: 20,700 release-site cases (all GPU systems) and 2,070 dev cases (one sampled GPU per model). They cross each tested model, workload, and selected GPU with all five concurrency presets where applicable, plus the capacity, demand, and Pareto paths. Each case starts in a new browser context and compares its outgoing request and rendered result or error with the current API response. The tests do not judge the API's sizing accuracy.

Every live workflow resolves the deployed build from `/api/version` before
running any case, then checks out that commit. Specs and deployment therefore
describe the same code even when a host deploys asynchronously; a deployment
that reports no build metadata fails the preflight with one clear message
instead of thousands of assertion differences.

- Workload presets set ISL, OSL, and a shared prefix. Predict and KV cache send each concurrency preset as `batch_size` and `max_batch_size` respectively; KV cache sends `max_num_tokens = isl + osl`. These are test inputs for memory estimation, not a claim that the KV page implements the same workload semantics as Predict or Recommend.
- Recommend compares GPU count, memory, and the goal-specific result tile with the current API response. Throughput-demand requests carry an arrival rate instead of concurrency; capacity requests have no latency target. Pareto sweeps carry neither an arrival rate nor concurrency, and the tests compare the rendered candidate table or error with the response. Capacity and Pareto cases check that the control is unavailable and skip the API comparison when the deployed catalog does not advertise that capability. The release sanity suite fails instead of skipping, because a release that drops a capability is a regression.
- When browser settings explicitly contain a backend version, the tests check that the UI sends that version. Otherwise, they check that the request omits `backend_version` so the API can resolve its default. The live tests do not pin a version such as `0.27.1`.
- The larger tested models use fixed TP and EP test inputs entered through the parallelism controls. Some model/GPU combinations may be unsupported or need different topology settings; the tests check that the UI displays the API's error state for these cases. The test inputs are not sizing formulas or recommendations.

The `backend_version` in these requests is the inference backend version (for example, vLLM 0.27.1). The aisimulate SDK version is a separate server dependency pinned in `services/aisimulators/pyproject.toml`.

The live Predict performance tests compare displayed weight, KV cache, TTFT, TPOT, and throughput with the real response; KV cache tests compare displayed byte counts with `/api/memory` when supplied. Recommend sizing checks the streamed sizing response. When the API returns an error, the tests check for the corresponding UI error state. These tests compare against the current response rather than fixed performance numbers. Failed runs retain a screenshot, Playwright trace on retry, and request and response JSON where applicable.

The suite is UI detection infrastructure for issue #186. It does not change calculator behavior or validate AISimulators sizing accuracy. UI mismatches should be filed and fixed separately, with their model, workload, GPU system, request, response, and trace attached.

## Run manually

Install dependencies and the Chromium browser:

```bash
npm ci
npx playwright install chromium
```

Run the scheduled-size suites locally:

```bash
PLAYWRIGHT_BASE_URL=https://configiq.dev npm run test:e2e:scheduled:full
PLAYWRIGHT_BASE_URL=https://configiq.dev npm run test:e2e:scheduled:sample
PLAYWRIGHT_BASE_URL=https://configiq.io npm run test:e2e:sanity
```

Run the full matrices manually, one sampled GPU per model on dev or every GPU
on the release host:

```bash
PLAYWRIGHT_BASE_URL=https://configiq.io npm run test:e2e:live
PLAYWRIGHT_BASE_URL=https://configiq.dev npm run test:e2e:dev-full
```

For a smaller manual check, add a `--grep` filter to either scheduled command:

```bash
PLAYWRIGHT_BASE_URL=https://configiq.dev npm run test:e2e:scheduled:sample -- --grep 'RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic'
```

Local scheduled runs use rotation zero by default. Set `PLAYWRIGHT_MATRIX_ROTATION` to a non-negative integer to rotate the extra pairings; the GitHub workflows set it to their run number.

To split a manual run across machines, run one shard number per machine:

```bash
PLAYWRIGHT_BASE_URL=https://configiq.dev npm run test:e2e:scheduled:full -- --shard=1/12
```

All three live workflows first resolve the deployed commit from
`/api/version` and check out that commit, so the specs match the code being
tested. Each shard repeats the check before running and again after the tests,
so a deployment that moves during a shard fails instead of mixing two builds.
The final check runs even when a case fails, so a mid-run deployment change is
named as the cause rather than leaving its assertion failures to explain it. A
deployment that changes and reverts between the two checks is not detected. The
preflight reports when the triggering ref and the deployed commit differ.

- The **dev nightly** workflow runs at 2:00 a.m. every day in `America/New_York` and can also be started manually. It targets `https://configiq.dev` (the `:dev` images) and splits the 470 sampled cases into six Playwright shards, with at most three shard jobs at a time; each job runs two browser workers.
- The **dev weekly** workflow runs at 8:00 a.m. Saturday in `America/New_York` and can also be started manually. It targets `https://configiq.dev` and splits the full 2,900 scheduled cases into 12 Playwright shards, with at most three shard jobs at a time.
- The **release sanity** workflow runs at 6:00 a.m. Saturday in `America/New_York` and can also be started manually. It targets `https://configiq.io` (the `:latest` images) and runs the 33 sanity cases in one job with a single browser worker so the production service sees only light, sequential load. Unlike the matrices, it requires each case to complete: an API error fails the run instead of being accepted as an unsupported pairing. The one exception is a model whose Hugging Face repository is gated for the deployment; that case still checks that the API's error reaches the UI, and the report marks it `known-unavailable`.

Each nightly shard has a three-hour timeout, each weekly shard six hours, and the release sanity job one hour; every job retries a failed test once in CI. A local scheduled run uses the same selection without sharding. The local Playwright process drives the deployed UI; it does not start a local ConfigIQ server or API.

## When a case fails

Open the `tested-model-ui-dev-<run ID>-<shard>` (nightly), `tested-model-ui-dev-full-<run ID>-<shard>` (weekly), or `release-sanity-ui-<run ID>` (production) artifacts on the GitHub Actions run page. Each contains `test-results/` (failure screenshots, traces on retry, and request/response JSON) and `playwright-report/` (the HTML report). Local runs write these folders at the repository root; they are ignored by Git. The failed test name identifies the model, workload, and GPU system. Compare the request JSON with the selected UI values and the response JSON with the visible result or error; the trace shows browser interaction and network calls.

If the preflight fails before any test starts, the deployment is not reporting
the build the specs expect. Check `/api/version` on the host and deploy the
matching image; do not rerun the matrix hoping for a different result.

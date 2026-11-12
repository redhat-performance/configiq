# Tested-model UI regression tests

Playwright covers the browser-to-AISimulators gap in [issue #186](https://github.com/redhat-performance/configiq/issues/186).

## Coverage

- The scheduled weekly suite against `https://configiq.xyz` has 2,900 cases. Each of the 10 approved models runs on all nine workloads and all 10 GPU systems in Predict, KV cache, and Recommend latency sizing at Small (10) concurrency (2,700 baseline cases). Each model also runs the other four concurrency choices on those three paths, all five capacity choices, throughput demand, and both Pareto modes on one rotated workload/GPU pairing per control (200 extra cases).
- The scheduled nightly dev suite has 470 cases. It checks the same 270 baseline cases and 200 extra control cases, using one selected GPU per model. The workflow run number rotates the extra cases through workloads and, on XYZ, GPU systems; all shards in a run use the same rotation. The baseline still covers every tested model and workload each run. Some model/GPU combinations may return API errors, which pass only when the UI displays the corresponding error.
- The full matrices remain available for manual runs: 20,700 XYZ cases and 2,070 dev cases. They cross each tested model, workload, and selected GPU with all five concurrency presets where applicable, plus the capacity, demand, and Pareto paths. Each case starts in a new browser context and compares its outgoing request and rendered result or error with the current API response. The tests do not judge the API's sizing accuracy.
- Workload presets set ISL and OSL. Predict and KV cache send each concurrency preset as `batch_size` and `max_batch_size` respectively; KV cache sends `max_num_tokens = isl + osl`. These are test inputs for memory estimation, not a claim that the KV page implements the same workload semantics as Predict or Recommend.
- Recommend compares GPU count, memory, and the goal-specific result tile with the current API response. Throughput-demand requests carry an arrival rate instead of concurrency; capacity requests have no latency target. Pareto sweeps carry neither an arrival rate nor concurrency, and the tests compare the rendered candidate table or error with the response. Capacity and Pareto cases check that the control is unavailable and skip the API comparison when the selected simulator does not advertise that capability.
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
PLAYWRIGHT_BASE_URL=https://configiq.xyz npm run test:e2e:weekly
PLAYWRIGHT_BASE_URL=https://configiq.dev npm run test:e2e:nightly
```

Run the full 20,700-case XYZ or 2,070-case dev matrix manually:

```bash
PLAYWRIGHT_BASE_URL=https://configiq.xyz npm run test:e2e:live
PLAYWRIGHT_BASE_URL=https://configiq.dev npm run test:e2e:dev-full
```

For a smaller manual check, add a `--grep` filter to either scheduled command:

```bash
PLAYWRIGHT_BASE_URL=https://configiq.dev npm run test:e2e:nightly -- --grep 'RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic'
```

Local scheduled runs use rotation zero by default. Set `PLAYWRIGHT_MATRIX_ROTATION` to a non-negative integer to rotate the extra pairings; the GitHub workflows set it to their run number.

To split a manual run across machines, run one shard number per machine:

```bash
npm run test:e2e:weekly -- --shard=1/12
```

The weekly workflow runs at 12:00 a.m. Saturday in `America/New_York` and can also be started manually. It targets `https://configiq.xyz` and splits the 2,900 cases into 12 Playwright shards, with at most three shard jobs at a time; each job runs two browser workers. Each shard has a six-hour timeout and CI retries.

The nightly workflow runs at 2:00 a.m. every day in `America/New_York` and can also be started manually. It targets `https://configiq.dev` and splits the 470 cases into six shards. A local `test:e2e:nightly` run uses the same selection without sharding. The local Playwright process drives the deployed dev UI; it does not start a local ConfigIQ server or API.

## When a case fails

Open the `tested-model-ui-<run ID>-<shard>` weekly artifacts or `tested-model-ui-dev-<run ID>-<shard>` nightly artifacts on the GitHub Actions run page. Each contains `test-results/` (failure screenshots, traces on retry, and request/response JSON) and `playwright-report/` (the HTML report). Local runs write these folders at the repository root; they are ignored by Git. The failed test name identifies the model, workload, and GPU system. Compare the request JSON with the selected UI values and the response JSON with the visible result or error; the trace shows browser interaction and network calls.

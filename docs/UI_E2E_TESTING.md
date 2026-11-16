# Tested-model UI regression tests

Playwright covers the browser-to-AISimulators gap in [issue #186](https://github.com/redhat-performance/configiq/issues/186).

## Coverage

- The nightly and manual suite runs against `https://configiq.xyz` and AISimulators API. Predict performance, KV cache, and Recommend sizing each run a complete live matrix of 10 tested models × 6 workloads (Default plus five presets) × 10 GPU systems: 600 cases per page. Each case starts in a new browser context, selects its GPU, and checks the outgoing request and whether the rendered result or error reflects the current API response. The tests do not judge whether the API's sizing values are correct.
- KV cache has no workload preset buttons. For its six workload scenarios, the tests enter `max_num_tokens = isl + osl` and `max_batch_size = concurrency` in Advanced settings. These are test inputs for memory estimation, not a claim that the KV page implements the same workload semantics as Predict or Recommend.
- Recommend sizing runs all 600 live combinations and compares displayed GPU count, peak memory, TTFT, and throughput with the streamed response. Each case verifies that the deployed UI still lists the tested model and preset.
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

Run the full 1,800-case live UI suite against `configiq.xyz`:

```bash
npm run test:e2e:live
```

To split a manual run across machines, run one shard number per machine:

```bash
npm run test:e2e:live -- --shard=1/10
```

The nightly workflow can also be started manually. It targets `https://configiq.xyz` and splits the suite into 10 Playwright shards, with at most three shard jobs at a time; each job runs two browser workers. Each shard has a six-hour timeout and CI retries. A full local command runs all 1,800 cases without sharding.

## When a case fails

Open the `tested-model-ui-<run ID>-<shard>` artifacts on the GitHub Actions run page. Each contains `test-results/` (failure screenshots, traces on retry, and request/response JSON) and `playwright-report/` (the HTML report). Local runs write these folders at the repository root; they are ignored by Git. The failed test name identifies the model, workload, and GPU system. Compare the request JSON with the selected UI values and the response JSON with the visible result or error; the trace shows browser interaction and network calls.

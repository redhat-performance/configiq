# User scenarios and sizing paths

Which tool to use for which customer ask, the scenarios we test sizing against,
and how to size in bands. Cost, pricing, and economics content is out of scope
for this page and documented separately later.

## Which tool to use when

Three customer asks, each with a webapp path, REST endpoint, and MCP tool:

| Customer input | Ask | Webapp | REST | MCP tool | Output |
|---|---|---|---|---|---|
| Existing GPU count/type and models | "What performance will I get?" | Performance (`/predict`) | `POST /predict` | `predict` | Expected TTFT, TPOT, throughput, and memory |
| Models plus latency and throughput targets | "What should I buy?" | Recommend Sizing (`/recommend`) | `POST /recommend` | `recommend` | GPU count, TP/PP, and replicas |
| A use case or rough demand scale, without targets or hardware | "Where do I start?" | KV Cache Calculator (`/kv-cache`) | `POST /memory` | `memory` | Memory breakdown and KV headroom for candidate configurations |

The KV-cache quick start row currently expects explicit workload parameters.
Preset-based (use case + t-shirt size) quick-start sizing is planned in
[#183](https://github.com/redhat-performance/configiq/issues/183).

## Scenarios to test against

### Shipped today

- **Predict performance for an existing fleet.** Given a model, a GPU system,
  and a parallelism configuration, expect TTFT/TPOT/throughput. Use
  `/predict` (or the Performance page).
- **Recommend sizing from SLOs.** Given a model plus latency and throughput
  targets, expect GPU count, TP/PP, and replicas. Use `/recommend`.
- **KV-cache sizing for a known workload shape.** Given a model and workload
  parameters (ISL/OSL, concurrency), expect a memory breakdown and KV
  headroom. Use `/memory`.
- **Support lookups.** Which models, GPU systems, serving backends, and
  network fabrics are supported: `GET /models`, `GET /systems`,
  `GET /backends`, `GET /networks`.
- **Single-workload catalog shapes.** Size one workload at a time from the
  [workload catalog](https://github.com/kubernetes-sigs/inference-perf/tree/main/workload-catalog)
  shapes: Chat, Coding, RAG, Voice, Deep Research, Batch Summarization,
  Synthetic Data Gen, Deep Thinking. Reasoning is a per-row mode with an
  explicit output budget. Keep Coding distinct from Deep Research, and Batch
  Summarization distinct from Synthetic Data Gen.

### Planned

- **Shared mixed deployment sizing** ([#174](https://github.com/redhat-performance/configiq/issues/174)).
  Size a shared deployment for simultaneous workloads with independent
  concurrency and shapes — for example Chat at concurrency 100 plus Coding at
  concurrency 20, running together. Each row keeps its own concurrency,
  ISL/OSL, prefix assumptions, and TTFT/TPOT targets. The total is a read-only
  summary; an average shape or summed standalone GPU counts is not validated
  shared-pool sizing. A UI prototype exists at `/recommend/mix` (local
  prototype, not published).
- **T-shirt quick-start presets** ([#183](https://github.com/redhat-performance/configiq/issues/183),
  [#178](https://github.com/redhat-performance/configiq/issues/178)). S/M/L
  demand bands for the same workload/model (proposed 100 / 1,000 / 10,000
  active users), producing the top two or three configurations by KV headroom.
- **Model-aware KV smart defaults** ([#177](https://github.com/redhat-performance/configiq/issues/177)).
  Backend model/hardware/serving support suggests starting advanced settings
  (TP, PP, MoE TP, EP) with manual override and reset.
- **Catalog regression profiles** ([#175](https://github.com/redhat-performance/configiq/issues/175),
  [#176](https://github.com/redhat-performance/configiq/issues/176)).
  inference-perf workload-catalog shapes adopted as shared, versioned profiles,
  with UI presets aligned to the same profiles.

## Sizing in bands (batch sizing)

- Size one workload row or demand band per call. There is no single
  mixed-workload endpoint — until #174 ships, run `/recommend` (or `/memory`)
  once per row or band and record each result separately.
- For t-shirt bands, size every band separately. Active users differ from
  in-flight concurrency; the ~10 active-users-per-concurrent-request
  conversion in #183 is a rule of thumb, so record the request-interval
  assumptions alongside each band.
- Do not extrapolate GPU counts linearly between bands. GPU counts must come
  from recorded runs.

## What customers get

- **Predict:** expected TTFT, TPOT, and throughput for a given configuration.
- **Recommend:** a configuration — GPU count, TP/PP, and replicas — that meets
  the stated targets.
- **KV Cache Calculator:** a memory breakdown and KV headroom for candidate
  configurations.
- Mixed-workload runs (planned, #174) return per-workload rows plus a
  read-only aggregate summary.

## Related

- [MCP server](mcp-server.md) — connect Claude and other MCP clients
- [REST API](api.md) — endpoints and examples
- [Tested-model validation](tested-models.md)

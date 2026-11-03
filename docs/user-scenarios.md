# User scenarios and sizing paths

ConfigIQ helps you size LLM inference. This page covers which tool to use
for which ask, the scenarios we test sizing against, and how to size
multiple workloads. Cost, pricing, and economics content is out of scope for
this page and documented separately later.

## What is ConfigIQ

ConfigIQ is a web application for LLM inference sizing. It answers three
questions:

1. **What performance will I get?** Predict TTFT, TPOT, and throughput for a
   model on a given GPU configuration.
2. **What should I buy?** Recommend GPU counts and parallelism from latency
   and throughput targets.
3. **Will the model fit?** Estimate KV cache memory and headroom for a model
   configuration.

The webapp is powered by the AISimulators REST API and is also available as
MCP tools, so the same sizing is reachable from the UI, from scripts, and
from AI assistants such as Claude. You can also compare GPUs side by side in
the GPU Explorer.

## Which tool to use when

Three customer asks, each with a webapp path, REST endpoint, and MCP tool:

| Customer input | Ask | Webapp | REST | MCP tool | Output |
|---|---|---|---|---|---|
| Existing GPU count and type, plus models | What performance will I get? | Performance (`/predict`) | `POST /predict` | `predict` | Expected TTFT, TPOT, throughput, and memory |
| Models plus latency and throughput targets | What should I buy? | Recommend Sizing (`/recommend`) | `POST /recommend` | `recommend` | GPU count, TP/PP, and replicas |
| A use case or rough demand scale, without targets or hardware | Where do I start? | KV Cache Calculator (`/kv-cache`) | `POST /memory` | `memory` | Memory breakdown and KV headroom |

The KV cache quick start expects explicit workload parameters (ISL, OSL,
concurrency) today.

## Scenarios to test against

- **Predict performance for an existing fleet.** Given a model, a GPU
  system, and a parallelism configuration, expect TTFT, TPOT, and
  throughput. Use `/predict` (or the Performance page).
- **Recommend sizing from SLOs.** Given a model plus latency and throughput
  targets, expect GPU count, TP/PP, and replicas. Use `/recommend`.
- **KV cache sizing for a known workload shape.** Given a model and workload
  parameters (ISL, OSL, concurrency), expect a memory breakdown and KV
  headroom. Use `/memory`.
- **Support lookups.** Which models, GPU systems, serving backends, and
  network fabrics are supported: `GET /models`, `GET /systems`,
  `GET /backends`, `GET /networks`.
- **Single workload catalog shapes.** Size one workload at a time from the
  [workload catalog](https://github.com/kubernetes-sigs/inference-perf/tree/main/workload-catalog)
  shapes: Chat, Coding, RAG, Voice, Deep Research, Batch Summarization,
  Synthetic Data Gen, and Deep Thinking. Reasoning is selected per row with
  an explicit output budget. Keep Coding distinct from Deep Research, and
  Batch Summarization distinct from Synthetic Data Gen.

## Sizing multiple workloads

- Size one workload per call. There is no single mixed workload endpoint.
  Run `/recommend` (or `/memory`) once per workload and record each result
  separately.
- Do not sum standalone GPU counts for a shared deployment. That is not
  validated shared pool sizing.
- GPU counts must come from recorded runs, not extrapolation.

## What customers get

- **Predict:** expected TTFT, TPOT, and throughput for a given
  configuration.
- **Recommend:** a configuration with GPU count, TP/PP, and replicas that
  meets the stated targets.
- **KV Cache Calculator:** a memory breakdown and KV headroom for candidate
  configurations.

## Related

- [MCP server](mcp-server.md): connect Claude and other MCP clients
- [REST API](api.md): endpoints and examples
- [Tested-model validation](tested-models.md)

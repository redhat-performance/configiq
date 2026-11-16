# Project overview

ConfigIQ is a Next.js web application for LLM inference sizing, GPU comparison,
and cost modeling. The frontend uses same-origin Next.js API routes to reach the
AISimulators and aicostings services without exposing service URLs to the browser.

## Audience

- ML engineers sizing GPU clusters for inference
- Platform engineers evaluating cloud vs on-premise costs
- Finance and procurement teams modeling GPU spend

## Tools

| Tool | Description |
|---|---|
| Performance | Instant GPU memory and throughput estimate |
| Recommend Sizing | Full inference sizing with batching and quantization |
| KV Cache Calculator | Memory breakdown and KV cache capacity analysis |
| GPU Explorer | Side-by-side GPU comparison table and charts |
| Hybrid Savings | Cloud vs on-premise cost modeling |
| Routing Economics | Multi-tier model routing cost analysis |
| Cluster cost | Estimate costs for multi-node GPU clusters |

## Guides

- [In-app help](help.md): search, ask, and take a tour without leaving the page
- [User scenarios and sizing paths](user-scenarios.md): which tool to use
  for which ask, the scenarios we test sizing against, and how to size
  multiple workloads
- [Settings and simulators](settings.md): simulator, backend, and version
  choices, and how a shared link behaves
- [Costs](costs.md): hosted, rented, and purchased assumptions
- [Model and GPU caveats](model-gpu-caveats.md): known limits and estimates
- [Tested-model validation](tested-models.md): measured pairs and classifiers
- [Pareto validation](pareto-validation.md): predictions checked against
  measured data
- [MCP server](mcp-server.md): connect Claude and other MCP clients to ConfigIQ
- [REST API](api.md): endpoint reference and examples
- [Contributing](contributing-guide.md): run the app and prepare a change
- [DeepWiki](https://deepwiki.com/redhat-performance/configiq): AI-generated,
  third-party docs from the codebase; not curated documentation

# ConfigIQ

LLM inference sizing, GPU comparison, and cost modeling for engineers and infrastructure teams.

## Live

- [configiq.xyz](https://configiq.xyz) (latest release)
- [configiq.dev](https://configiq.dev) (latest commit)

Built with Next.js + PatternFly, powered by our [AISimulators](https://github.com/ai-dynamo/aisimulate) [REST API](https://aisimulators.dev/docs).

## What it does

| Tool | Description |
|------|-------------|
| **Performance** | Fast GPU memory and cost estimate from model + load profile |
| **Recommend Sizing** | Detailed sizing with batching, quantization, and cost modeling |
| **KV Cache Calculator** | Memory breakdown and KV cache capacity analysis |
| **GPU Explorer** | Compare GPUs across memory, throughput, cost, and availability |
| **Hybrid Savings** | Model cost savings across cloud, on-premise, and hybrid strategies |
| **Routing Economics** | Analyze request routing between model tiers |

## Getting started

### Prerequisites

- Node.js >= 20.19.0
- npm >= 10

### Setup

```bash
git clone https://github.com/openshift-psap/configiq.git
cd configiq
npm install
cp .env.example .env.local
npm run dev
```

App runs at **http://localhost:3000**.

### Available commands

```bash
npm run dev          # Start dev server (http://localhost:3000)
npm run build        # Production build
npm run type-check   # TypeScript check without building
npm run lint         # ESLint
```

## Tech stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 14 App Router + TypeScript |
| UI | PatternFly v5 |
| Backend API | [AISimulators](https://aisimulators.dev/docs) (GPU sizing + memory estimation) |

## Project structure

```
app/                  Next.js App Router pages
  layout.tsx          Root layout, fonts, PatternFly CSS imports
  page.tsx            Homepage
  recommend/          Recommend sizing tool
  kv-cache/           KV Cache Calculator
  predict/             Predict performance
  gpu-explorer/       GPU Explorer
  hybrid-savings/     Hybrid Savings
  routing/            Routing Economics
  settings/           App settings
  api/                Next.js API routes (proxy to REST APIs)
    recommend/        POST — GPU sizing via AISimulators /recommend
    predict/          POST — GPU performance via AISimulators /predict
    memory/           POST — memory breakdown via AISimulators /memory
    gpus/             GET — GPU catalog via AISimulators /systems
    hf-config/        GET — Hugging Face model config lookup
    health/           GET — health check
components/
  layout/
    AppShell.tsx      Top-nav masthead + sidebar navigation
lib/
  api/                AISimulators API clients
docs/                 Architecture docs and ADRs
```

## Contributing

### Before opening a PR

CI runs automatically and must pass:

```bash
npm run type-check   # Must be clean
npm run lint         # Must be clean
npm run build        # Must succeed
```

### Code conventions

1. **GPU math belongs in `aisimulators`** — never write sizing formulas inside React components.
1. **Pricing belongs in the `aicostings` service** — never add costing inside React components.
2. **PatternFly only** — do not add Tailwind, shadcn/ui, or any other component library.
3. **Sentence case everywhere** — no title case in headings or labels.
4. **Server components by default** — add `"use client"` only when needed.
5. **No `any` types** — TypeScript strict mode is enforced.

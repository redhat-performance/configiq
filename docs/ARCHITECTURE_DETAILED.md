# GPU Calc - Detailed Architecture

This document provides comprehensive component diagrams, data flow visualizations, and module relationships for the ConfigIQ system.

> **Note**: This supplements the high-level [architecture.md](./architecture.md) with detailed diagrams showing how components interact.

## Table of Contents
- [System Overview](#system-overview)
- [Inference Config Engine](#inference-config-engine)
- [Data Flow](#data-flow)
- [Module Dependencies](#module-dependencies)
- [Component Responsibilities](#component-responsibilities)
- [Current Integration Status](#current-integration-status)
- [API Architecture](#api-architecture)
- [Type System](#type-system)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (Next.js App Router)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐        ┌──────────────────────┐     │
│  │  Performance Page    │        │ Shared Components    │     │
│  │  app/predict/        │◄───────┤ components/          │     │
│  │                      │        │ - ProductTour        │     │
│  │  - Model input       │        │ - FlipTile           │     │
│  │  - GPU selector      │        │ - Term (glossary)    │     │
│  │  - Result tiles      │        └──────────────────────┘     │
│  │  - Test panel 🧪     │                                      │
│  └──────────┬───────────┘                                      │
│             │                                                  │
└─────────────┼──────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Business Logic Layer                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────┐        │
│  │  Inference Config Engine                           │        │
│  │  lib/gpu-math/inference-config/                    │        │
│  │                                                     │        │
│  │  ┌──────────────┐  ┌──────────────┐               │        │
│  │  │ core.ts      │  │ validation   │               │        │
│  │  │ (orchestrate)│─→│ (check input)│               │        │
│  │  └──────┬───────┘  └──────────────┘               │        │
│  │         │                                          │        │
│  │         ▼                                          │        │
│  │  ┌──────────────┐  ┌──────────────┐               │        │
│  │  │tensor-       │  │ vllm-        │               │        │
│  │  │parallel.ts   │─→│ defaults.ts  │               │        │
│  │  └──────────────┘  └──────┬───────┘               │        │
│  │                           │                        │        │
│  │                           ▼                        │        │
│  │  ┌──────────────┐  ┌──────────────┐               │        │
│  │  │ bottleneck.ts│  │ parallelism  │               │        │
│  │  │              │─→│              │               │        │
│  │  └──────────────┘  └──────────────┘               │        │
│  └────────────────────────────────────────────────────┘        │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ GPU Catalog  │  │ Model Catalog│  │ Other Helpers│        │
│  │ gpus.ts      │  │ models.ts    │  │ memory.ts    │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐ │
│  │ gpu-catalog    │  │ model-specs    │  │ aicostings       │ │
│  │ .json          │  │ .json          │  │ (Live Pricing)   │ │
│  │                │  │                │  │                  │ │
│  │ 15 GPUs        │  │ Model params   │  │ cloud GPU rates  │ │
│  │ VRAM, bandwidth│  │ layers, heads  │  │                  │ │
│  └────────────────┘  └────────────────┘  └──────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Inference Config Engine - Internal Architecture

The core calculation engine that determines GPU requirements.

```
                 ┌─────────────────────────┐
                 │   InferenceRequest      │
                 │                         │
                 │ - model_name            │
                 │ - gpu_type              │
                 │ - concurrent_users      │
                 │ - isl, osl              │
                 │ - workload_type         │
                 │ - sla_priority          │
                 │ - precision             │
                 └───────────┬─────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │  validation.ts                         │
        │  validateInferenceRequest()            │
        │  - Check all params valid              │
        │  - Return errors if invalid            │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │  core.ts                               │
        │  computeInferenceConfig()              │
        │  - Main orchestrator                   │
        │  - Calls all sub-modules in order     │
        └────────────────┬───────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│quantization  │ │ Get GPU/Model│ │Calculate     │
│.ts           │ │ Specs        │ │Weight Memory │
│              │ │              │ │              │
│Determine     │ │from catalogs │ │params × bytes│
│bytes/param   │ │              │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────────┐
        │  tensor-parallel.ts                    │
        │  computeTensorParallelSize()           │
        │  - Determine TP size (1, 2, 4, 8...)   │
        │  - Calculate replicas                  │
        │  - Compute usable HBM per GPU          │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │  vllm-defaults.ts                      │
        │  computeVLLMConfig()                   │
        │  - max_model_len (context window)      │
        │  - max_num_seqs (batch size)           │
        │  - enable_chunked_prefill (ISL > 1000?)│
        │  - enable_prefix_caching (workload?)   │
        │  - max_num_batched_tokens              │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │  bottleneck.ts                         │
        │  classifyBottleneck()                  │
        │  - Analyze TTFT vs TPOT vs throughput  │
        │  - Identify primary bottleneck         │
        │  - Generate fix suggestions            │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │  parallelism.ts                        │
        │  determineParallelismStrategy()        │
        │  - TP_ONLY vs PP_ACROSS_NODES          │
        │  - Disaggregated serving (llmd)        │
        │  - Network topology recommendations    │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │  llmd.ts (optional)                    │
        │  computeLLMDConfig()                   │
        │  - Prefill instances config            │
        │  - Decode instances config             │
        │  - KV transfer settings                │
        └────────────────┬───────────────────────┘
                         │
                         ▼
                 ┌─────────────────────────┐
                 │ InferenceConfigResult   │
                 │                         │
                 │ - memory_analysis       │
                 │ - vllm_config           │
                 │ - bottleneck_analysis   │
                 │ - parallelism_strategy  │
                 │ - llmd_config (opt)     │
                 │ - warnings              │
                 └─────────────────────────┘
```

---

## Data Flow - Performance Page

Shows how data flows from user input through calculations to display.

```
┌─────────┐
│  User   │
└────┬────┘
     │
     │ 1. Select Model: "Llama 3.1 70B"
     │ 2. Select GPU: "H100 80GB"
     │
     ▼
┌─────────────────────────────────────────┐
│  Performance UI                         │
│  (Performance.tsx)                      │
│                                         │
│  React State:                           │
│  - model = "meta-llama/..."             │
│  - gpu = "NVIDIA H100 80GB"             │
└────┬────────────────────────────────────┘
     │
     │ useEffect triggered on model/gpu change
     │
     ▼
┌─────────────────────────────────────────┐
│  GPU Name Mapping                       │
│  mapGpuToCatalogId()                    │
│                                         │
│  "NVIDIA H100 80GB" → "h100-80gb"       │
└────┬────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Call Inference Engine                  │
│  computeInferenceConfig({               │
│    model_name: "meta-llama/...",        │
│    gpu_type: "h100-80gb",               │
│    concurrent_users: 97,                │
│    isl: 1000,                           │
│    osl: 150,                            │
│    workload_type: "chat",               │
│    sla_priority: "ttft",                │
│    precision: "FP16"                    │
│  })                                     │
└────┬────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Inference Engine Processing            │
│                                         │
│  1. Validate inputs ✓                   │
│  2. Get model specs (70B params)        │
│  3. Get GPU specs (80 GB VRAM)          │
│  4. Calculate weight: 70B × 2 = 140 GB  │
│  5. Determine TP: 140 GB > 80 GB → TP=2 │
│  6. Calculate KV cache budget           │
│  7. Generate vLLM config                │
│  8. Analyze bottlenecks                 │
│  9. Determine parallelism strategy      │
└────┬────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Return Result                          │
│  {                                      │
│    memory_analysis: {                   │
│      tp_size: 2,                        │
│      replicas: 1,                       │
│      weight_gb: 140,                    │
│      kv_cache_budget_gb: 45             │
│    },                                   │
│    vllm_config: {...},                  │
│    bottleneck_analysis: {               │
│      primary: "TTFT"                    │
│    },                                   │
│    warnings: [...]                      │
│  }                                      │
└────┬────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Update React State                     │
│  setTestResult(result)                  │
└────┬────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Display in Test Panel                  │
│  🧪 TEST: Real Inference Engine         │
│                                         │
│  Mock GPU Count: 1                      │
│  ✨ Real GPU Count: 2                   │
│     (TP=2 × Replicas=1)                 │
│  Bottleneck: TTFT                       │
└─────────────────────────────────────────┘
```

---

## Module Dependencies

> **Stale**: this diagram describes the pre-REST-API client-side compute
> pipeline. None of the referenced modules below (`core.ts`, `validation.ts`,
> `quantization.ts`, `tensor-parallel.ts`, `vllm-defaults.ts`, `bottleneck.ts`,
> `parallelism.ts`, `llmd.ts`, `gpu-catalog.json`, `models.ts`) exist anymore —
> GPU math now runs through the AISimulators REST API via `lib/api/`
> ([see AGENTS.md](../AGENTS.md#critical-rules)). This section needs a full
> rewrite, not a rename.

```
Performance.tsx
    │
    ├─→ computeInferenceConfig (from core.ts)
    │       │
    │       ├─→ validateInferenceRequest (from validation.ts)
    │       ├─→ recommendQuantization (from quantization.ts)
    │       ├─→ computeTensorParallelSize (from tensor-parallel.ts)
    │       │       └─→ getGpuById (from gpus.ts)
    │       │               └─→ GPU_CATALOG (from gpu-catalog.json)
    │       ├─→ computeVLLMConfig (from vllm-defaults.ts)
    │       │       └─→ getModelById (from models.ts)
    │       ├─→ classifyBottleneck (from bottleneck.ts)
    │       ├─→ determineParallelismStrategy (from parallelism.ts)
    │       └─→ computeLLMDConfig (from llmd.ts)
    │
    ├─→ FlipTile (from performanceHelpers.tsx)
    ├─→ Term (from performanceHelpers.tsx)
    ├─→ useCountUp (from performanceHelpers.tsx)
    └─→ ProductTour (from components/ProductTour/)
```

---

## Component Responsibilities

### Frontend Components

| Component | File | Responsibility | Status |
|-----------|------|----------------|--------|
| **Performance** | `app/predict/Performance.tsx` | Main page, state management, orchestrates all other components | ✅ Active |
| **FlipTile** | `performanceHelpers.tsx` | Interactive card that flips to show formulas on click/Enter | ✅ Complete |
| **ProductTour** | `components/ProductTour/ProductTour.tsx` | Guided tour with spotlight and tooltips | ✅ Complete |
| **Term** | `performanceHelpers.tsx` | Glossary popover (? icon with explanation) | ✅ Complete |
| **useCountUp** | `performanceHelpers.tsx` | Animates numbers from 0 → target | ✅ Complete |

### Inference Engine Modules

| Module | File | Key Function | What It Does |
|--------|------|--------------|--------------|
| **Core** | `core.ts` | `computeInferenceConfig()` | Main orchestrator, calls all sub-modules |
| **Validation** | `validation.ts` | `validateInferenceRequest()` | Validates input parameters |
| **Tensor Parallel** | `tensor-parallel.ts` | `computeTensorParallelSize()` | Calculates TP size (1, 2, 4, 8...) and replicas |
| **vLLM Defaults** | `vllm-defaults.ts` | `computeVLLMConfig()` | Generates vLLM configuration |
| **Bottleneck** | `bottleneck.ts` | `classifyBottleneck()` | Identifies TTFT/TPOT/throughput bottlenecks |
| **Parallelism** | `parallelism.ts` | `determineParallelismStrategy()` | Selects parallelism strategy |
| **Quantization** | `quantization.ts` | `recommendQuantization()` | Recommends FP16/FP8/INT8/INT4 |
| **LLMD** | `llmd.ts` | `computeLLMDConfig()` | Disaggregated serving configuration |

### Data Modules

| Module | File | Purpose | Exports |
|--------|------|---------|---------|
| **GPU Catalog** | `gpus.ts` | Loads GPU specifications | `GPU_CATALOG`, `getGpuById()` |
| **Model Catalog** | `models.ts` | Loads model specifications | `MODEL_CATALOG`, `getModelById()` |
| **GPU JSON** | `gpu-catalog.json` | 15 GPU specs with VRAM, bandwidth, pricing | Raw JSON data |

---

## Current Integration Status

### ✅ Step 1 Complete (Today)

```
┌───────────────────────────────────────────────────────┐
│  Performance Page                                     │
│                                                       │
│  ┌─────────────────────────────────────────────┐    │
│  │  🧪 Test Panel (Blue Box)                    │    │
│  │                                              │    │
│  │  Mock GPU Count: 1                           │    │
│  │  ✨ Real GPU Count: 2 (TP=2 × Replicas=1)    │    │
│  │  Bottleneck: TTFT                            │    │
│  │  [View full result object ▼]                 │    │
│  └──────────────────┬───────────────────────────┘    │
│                     │                                │
│                     │ Calls                          │
│                     ▼                                │
│           ┌──────────────────┐                       │
│           │ Inference Engine │                       │
│           └──────────────────┘                       │
│                                                       │
│  ┌─────────────────────────────────────────────┐    │
│  │  Flip Tiles (Still using mock data)         │    │
│  │  - GPU count: 1 (mock)                       │    │
│  │  - Weight: 16 GB (mock)                      │    │
│  │  - KV cache: 245 MB (mock)                   │    │
│  │  - Cost: $2,500 (mock)                       │    │
│  └─────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────┘
```

### ⏳ Future State (Steps 2-8)

```
┌───────────────────────────────────────────────────────┐
│  Performance Page                                     │
│                                                       │
│  ┌─────────────────────────────────────────────┐    │
│  │  Workload Controls                           │    │
│  │  - Concurrent users slider                   │    │
│  │  - ISL/OSL sliders                           │    │
│  │  - Workload type dropdown                    │    │
│  │  - SLA priority toggles                      │    │
│  │  - Precision selector                        │    │
│  └──────────────────┬───────────────────────────┘    │
│                     │                                │
│                     │ All inputs                     │
│                     ▼                                │
│           ┌──────────────────┐                       │
│           │ Inference Engine │                       │
│           └─────────┬────────┘                       │
│                     │                                │
│                     │ Real calculations              │
│                     ▼                                │
│  ┌─────────────────────────────────────────────┐    │
│  │  Flip Tiles (Real data from engine)         │    │
│  │  - GPU count: 2 (TP=2 × 1)                   │    │
│  │  - Weight: 140 GB (70B × 2 bytes)            │    │
│  │  - KV cache: 45 GB                           │    │
│  │  - Cost: $4,800/mo (live pricing)            │    │
│  └─────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────┘
```

---

## API Architecture

```
┌──────────┐
│ Browser  │
└────┬─────┘
     │
     ├─→ GET /api/costings/systems?include=cloud
     │   │
     │   └─→ Proxy to aicostings /systems
     │       └─→ Returns live cloud GPU rates
     │
     ├─→ GET /api/v1/models
     │   │
     │   └─→ Load model-specs.json
     │
     └─→ POST /api/v1/config
         │
         └─→ Call computeInferenceConfig()
             └─→ Returns full inference result
```

---

## Type System

### Key Interfaces

```typescript
// INPUT to inference engine
interface InferenceRequest {
  model_name: string                // "meta-llama/Llama-3.1-70B-Instruct"
  precision: 'FP16' | 'FP8' | ...   // Weight precision
  gpu_type: string                  // "h100-80gb"
  concurrent_users: number          // 100
  isl: number                       // Input sequence length (2000)
  osl: number                       // Output sequence length (500)
  workload_type: 'chat' | 'rag' ... // Workload category
  sla_priority: 'ttft' | 'tpot' ... // Performance priority
}

// OUTPUT from inference engine
interface InferenceConfigResult {
  memory_analysis: {
    weight_gb: number               // 140
    tp_size: number                 // 2
    replicas: number                // 1
    kv_cache_budget_gb: number      // 45
    usable_hbm_per_gpu: number      // 72
  }
  vllm_config: {
    max_num_seqs: number            // 256
    max_model_len: number           // 8192
    enable_chunked_prefill: boolean // true
    enable_prefix_caching: boolean  // false
  }
  bottleneck_analysis: {
    primary: 'TTFT' | 'TPOT' | ...  // 'TTFT'
    risk: string                    // "high"
    fix_suggestions: string[]       // ["Reduce ISL", ...]
  }
  parallelism_strategy: {...}
  llmd_config?: {...}
  warnings: string[]                // ["Memory tight", ...]
}
```

---

## Calculation Example

**Input:**
- Model: Llama 3.1 70B
- GPU: H100 80GB
- Concurrent users: 100
- ISL: 2000, OSL: 500
- Workload: chat
- Priority: ttft

**Processing:**
1. **Quantization**: FP16 → 2 bytes/param
2. **Weight memory**: 70B × 2 = 140 GB
3. **TP size**: 140 GB needs 2× H100 (80 GB each) → TP=2
4. **Usable HBM**: 80 GB × 90% = 72 GB per GPU
5. **Weight per GPU**: 140 GB ÷ 2 = 70 GB
6. **KV cache budget**: 72 - 70 = 2 GB per GPU → 4 GB total
7. **KV per request**: ~450 MB (2000 tokens × layers × heads)
8. **Max sequences**: 4000 MB ÷ 450 MB = ~8 concurrent requests (low!)
9. **Bottleneck**: KV cache is tight → THROUGHPUT bottleneck
10. **Warnings**: "Consider reducing ISL or adding more GPUs"

**Output:**
- Total GPUs: 2 (TP=2, Replicas=1)
- vLLM config: max_num_seqs=8, enable_chunked_prefill=true
- Bottleneck: THROUGHPUT (not enough KV cache)
- Warnings: ["KV cache budget tight for 100 concurrent users"]

---

## Next Steps (Remaining Plan)

- **Step 2**: Add interactive sliders to test panel
- **Step 3**: Replace GPU count flip tile with real data
- **Step 4**: Add workload controls to main UI
- **Step 5**: Replace all flip tiles
- **Step 6**: Integrate live pricing
- **Step 7**: Remove mock data
- **Step 8**: Polish and finalize

---

**Last Updated**: After Step 1 completion  
**Status**: Test panel working, engine integrated, ready for Step 2

"use client";

import React from "react";
import { Popover } from "@patternfly/react-core";
import { HelpIcon } from "@patternfly/react-icons";

interface JargonPopoverProps {
  term: string;
  explanation: string;
  /** Portal target for use inside clipped or transformed containers such as flip cards. */
  appendTo?: HTMLElement | ((ref?: HTMLElement) => HTMLElement) | "inline";
}

export function JargonPopover({ term, explanation, appendTo }: JargonPopoverProps) {
  return (
    <Popover
      aria-label={`${term} explanation`}
      headerContent={<div className="gc-label">{term}</div>}
      bodyContent={<div className="gc-body">{explanation}</div>}
      position="auto"
      maxWidth="360px"
      appendTo={appendTo}
    >
      <button
        type="button"
        aria-label={`Help for ${term}`}
        style={{
          background: "none",
          border: "none",
          padding: "0 4px",
          cursor: "pointer",
          color: "var(--gc-text-3)",
          verticalAlign: "middle",
          display: "inline-flex",
          alignItems: "center",
        }}
      >
        <HelpIcon style={{ fontSize: "12px" }} />
      </button>
    </Popover>
  );
}

// Pre-defined jargon explanations
export const JARGON = {
  "KV cache": "Key-Value cache stores attention computations from previous tokens. Larger cache = more concurrent requests but uses more memory.",
  "max_num_seqs": "Maximum number of sequences (requests) the model can handle simultaneously. Higher = better throughput but needs more memory.",
  "prefills/step": "Number of new requests that can start processing in each iteration. Limited by available GPU compute.",
  "tensor parallel": "Splitting the model across multiple GPUs. Required when a single GPU doesn't have enough memory for the full model.",
  "active-request ratio": "Percentage of max_num_seqs slots actively being used. Higher is better for GPU utilization.",
  "range drivers": "Factors that cause GPU count to vary: batch size changes, context length, concurrency, precision.",
  "GQA": "Grouped Query Attention - uses fewer key/value heads than query heads. Saves memory compared to MHA.",
  "MHA": "Multi-Head Attention - standard attention mechanism with equal query/key/value heads. Uses most memory.",
  "MQA": "Multi-Query Attention - single key/value head shared across query heads. Saves most memory but may reduce quality.",
  "worst-case context": "Maximum memory needed when all requests use the full context window (ISL + OSL). Planning for this prevents OOM errors.",
  "ISL": "Input Sequence Length - the number of prompt tokens sent to the model.",
  "OSL": "Output Sequence Length - the number of tokens the model generates in response.",
  "TP": "Tensor Parallelism - distributing model layers across GPUs. TP=4 means the model is split across 4 GPUs.",
  "BF16": "BFloat16 precision - 16-bit format optimized for training. Balances memory usage and model quality.",
  "FP8": "8-bit floating point - half the memory of BF16. Newer GPUs support this with minimal quality loss.",
  "vLLM": "High-performance LLM inference engine that uses PagedAttention for efficient KV cache management.",
  "chunked prefill": "Breaking long prompts into chunks to interleave with decoding. Improves time-to-first-token for concurrent requests.",
  "prefix caching": "Reusing KV cache for repeated prompt prefixes (e.g., system prompts). Reduces compute and latency.",
};

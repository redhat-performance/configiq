import { KV_PRECISION_METADATA, PRECISION_METADATA } from './precision';

/**
 * Canonical glossary for in-app help.
 *
 * One definition per term, shared by the `<Term>` inline popovers, the
 * `JargonPopover` trigger, and the help search index. Adding a term here makes
 * it available everywhere; there is no second copy to keep in sync.
 */
export interface GlossaryEntry {
  id: GlossaryId;
  title: string;
  body: string;
  /** Extra search keywords that are not in the title. */
  tags?: string[];
}

const prevalence = Object.entries(PRECISION_METADATA)
  .map(([, value]) => `${value.label} = ${value.desc}`)
  .join('. ');
const kvPrecision = Object.entries(KV_PRECISION_METADATA)
  .map(([, value]) => `${value.label} = ${value.desc}`)
  .join('. ');

const ENTRIES = {
  kvCache: {
    title: 'KV cache',
    body: 'As the model generates text it remembers every previous token as "key/value" vectors. That memory is the KV cache. It grows with prompt length and the number of users active at once — usually the biggest driver of GPU memory after the model weights themselves.',
    tags: ['key value', 'memory'],
  },
  kvPerReq: {
    title: 'KV cache / request',
    body: 'Memory one in-flight request holds for its tokens. Multiply by the number of concurrent requests to get the total KV cache the GPUs must hold.',
    tags: ['memory', 'per request'],
  },
  weightMemory: {
    title: 'Weight memory',
    body: 'The model parameters loaded into GPU memory. Roughly parameters × bytes-per-number — e.g. an 8B model in BF16 (2 bytes) needs about 16 GB.',
    tags: ['parameters', 'vram'],
  },
  gqa: {
    title: 'GQA · grouped-query attention',
    body: 'A memory-saving attention design where several query heads share one key/value head. Fewer KV heads means a smaller KV cache — here 8 KV heads vs 32 attention heads, about 4× smaller than full multi-head attention (MHA).',
    tags: ['attention', 'grouped query'],
  },
  mha: {
    title: 'MHA · multi-head attention',
    body: 'The standard attention mechanism, with as many key/value heads as query heads. It uses the most KV cache memory of the common attention variants.',
    tags: ['attention'],
  },
  mqa: {
    title: 'MQA · multi-query attention',
    body: 'A single key/value head shared across all query heads. It saves the most KV cache memory but can reduce output quality compared with MHA or GQA.',
    tags: ['attention'],
  },
  isl: {
    title: 'ISL · input sequence length',
    body: 'How many tokens the prompt contains on average. Longer prompts use more KV cache per request.',
    tags: ['prompt', 'input tokens'],
  },
  osl: {
    title: 'OSL · output sequence length',
    body: 'How many tokens the model generates per response on average.',
    tags: ['output tokens', 'completion'],
  },
  ttft: {
    title: 'TTFT · time to first token',
    body: 'How long from request submission until the first output token arrives. Critical for interactive experiences — users notice delays above 1 second. Primarily driven by prompt length and concurrency (how many other requests are being processed simultaneously).',
    tags: ['latency', 'time to first token'],
  },
  tpot: {
    title: 'TPOT · time per output token',
    body: 'Milliseconds between each output token during generation. Affects streaming speed — lower values feel more responsive. Typical targets: 10-50ms for interactive chat, can be higher for batch jobs.',
    tags: ['latency', 'inter-token', 'streaming'],
  },
  throughput: {
    title: 'Throughput',
    body: 'The rate at which the serving configuration generates output tokens. Higher throughput means the GPUs can produce more tokens per second across active requests.',
    tags: ['tokens per second'],
  },
  requestLatency: {
    title: 'Request latency (E2E)',
    body: 'Total end-to-end time for a request, from submission to completion. Equals TTFT + (OSL × TPOT). Constrains both prefill and decode performance.',
    tags: ['end to end', 'e2e'],
  },
  concurrent: {
    title: 'Concurrent requests',
    body: 'How many requests are being processed at the same instant — not total daily traffic. Derived from requests/day, the peak multiplier, and how long each request runs. This sets the batch size the GPUs must hold in memory at once.',
    tags: ['concurrency', 'batch', 'traffic'],
  },
  maxNumSeqs: {
    title: 'max_num_seqs',
    body: 'The vLLM scheduler limit on how many sequences run in a batch together. If concurrent requests exceed it, requests queue and latency rises.',
    tags: ['batch size', 'scheduler'],
  },
  batchedTokens: {
    title: 'Batched tokens / step',
    body: 'Tokens the engine processes in a single forward pass (prompt "prefill" + ongoing "decode"). If prefills overflow the budget the engine splits them across steps (chunked prefill).',
    tags: ['prefill', 'decode'],
  },
  prefillsPerStep: {
    title: 'Prefills / step',
    body: 'How many new requests can start processing in each iteration. It is limited by the available GPU compute.',
    tags: ['prefill', 'scheduler'],
  },
  activeRequestRatio: {
    title: 'Active-request ratio',
    body: 'The share of max_num_seqs slots actively being used. A higher ratio means better GPU utilisation, but less headroom for bursts.',
    tags: ['utilisation', 'batch'],
  },
  tensorParallel: {
    title: 'Tensor parallel size (TP)',
    body: 'How many GPUs a single model copy is split across. TP = 1 means the whole model fits on one GPU; larger models need TP = 2, 4, 8…',
    tags: ['tp', 'parallelism', 'multi gpu'],
  },
  pipelineParallel: {
    title: 'Pipeline parallel size (PP)',
    body: 'How many GPUs to split the model layers across in a pipeline. PP = 1 means all layers on the same GPU(s). PP > 1 splits layers across GPUs for very large models. Total GPUs = TP × PP × replicas.',
    tags: ['pp', 'parallelism'],
  },
  maxModelLen: {
    title: 'max_model_len',
    body: 'Maximum total sequence length (input + output tokens) the engine will allow. Set to "auto" to use the model\'s native context window from its config. Lower values save KV cache memory.',
    tags: ['context window', 'max sequence length'],
  },
  prefillMaxSeqLen: {
    title: 'Prefill max sequence length',
    body: 'Maximum prompt-plus-output context reserved for prefill workers. Lower values reduce the KV cache allocation for the prefill pool when serving a specific context tier.',
    tags: ['prefill', 'context'],
  },
  decodeMaxSeqLen: {
    title: 'Decode max sequence length',
    body: 'Maximum prompt-plus-output context reserved for decode workers. Lower values reduce the KV cache allocation for the decode pool when serving a specific context tier.',
    tags: ['decode', 'context'],
  },
  chunkedPrefill: {
    title: 'enable_chunked_prefill',
    body: 'When enabled, vLLM splits long prompts into chunks to avoid stalling decode (generation). Improves latency for mixed workloads with very long prompts. Recommended when ISL > 4096 or batch size > 64.',
    tags: ['chunked prefill'],
  },
  prefixCaching: {
    title: 'enable_prefix_caching',
    body: 'When enabled, vLLM automatically detects and reuses shared prompt prefixes (e.g., system prompts). Saves compute and memory when many requests share the same prefix. Recommended when you have shared prefixes.',
    tags: ['prefix cache'],
  },
  rangeDrivers: {
    title: 'Range drivers',
    body: 'The estimate is a range, not a single number, because real traffic varies. Range drivers are the assumptions that move the GPU count the most — tune these first to make the estimate match your reality.',
    tags: ['range', 'sensitivity'],
  },
  worstCase: {
    title: 'Worst-case context',
    body: 'KV cache if every active request filled the model\'s entire context window (max_model_len) at the same time. A safety ceiling — most workloads never reach it.',
    tags: ['kv cache', 'safety'],
  },
  prefix: {
    title: 'Shared prefix tokens',
    body: 'Number of tokens shared across multiple requests (e.g., a system prompt). When prefix caching is enabled, these tokens are computed once and reused, saving memory and compute.',
    tags: ['system prompt', 'prefix'],
  },
  prefixCache: {
    title: 'Prefix-cache hit rate',
    body: 'Share of requests that reuse an already-computed prompt prefix (e.g. a shared system prompt). Higher hit rates reuse KV cache and cut memory.',
    tags: ['prefix', 'cache hit'],
  },
  weightPrecision: {
    title: 'Weight precision',
    body: `Number format for model parameters. ${prevalence}. Lower precision reduces memory but may impact quality.`,
    tags: ['quantization', 'fp8', 'fp16', 'int4'],
  },
  kvCachePrecision: {
    title: 'KV cache precision',
    body: `Number format for key-value cache vectors. ${kvPrecision}. Lower precision lets you fit more requests in the same memory.`,
    tags: ['quantization', 'fp8'],
  },
  bf16: {
    title: 'BF16 · bfloat16',
    body: 'A 16-bit floating-point format optimised for machine learning. It balances memory use and model quality, and is the usual full-precision serving baseline.',
    tags: ['precision', 'fp16'],
  },
  fp8: {
    title: 'FP8 · 8-bit floating point',
    body: 'Half the memory of BF16. Newer GPUs support FP8 weights and KV cache with minimal quality loss.',
    tags: ['precision', 'quantization'],
  },
  moeQuantization: {
    title: 'MoE quantization mode',
    body: 'Quantization strategy for Mixture-of-Experts models. W4A16 MXFP4 = 4-bit weights, 16-bit activations (best quality, H100+). W4A8 = 4-bit weights, 8-bit activations (~2× faster on B200, lower quality). CUTLASS = H100-optimized kernel. TRT-LLM = B200 TensorRT-LLM variant.',
    tags: ['mixture of experts', 'quantization'],
  },
  gpuUtil: {
    title: 'GPU memory utilization',
    body: 'Fraction of each GPU\'s memory available to the backend for weights and KV cache. AISim maps this generic value to the selected backend and uses a backend-specific default when left on auto.',
    tags: ['memory', 'vram'],
  },
  selfHosted: {
    title: 'Self-hosted (hardware only)',
    body: 'Hardware amortization only — excludes power, cooling, staff, and networking. Typical full TCO adds 40–80% to this number.',
    tags: ['cost', 'tco', 'on premises'],
  },
  cloudPricing: {
    title: 'Cloud pricing',
    body: 'GPU instance pricing only — excludes egress, storage, and managed service fees.',
    tags: ['cost', 'rented'],
  },
  kvCategory: {
    title: 'KV cache category',
    body: 'Different model architectures use different KV cache strategies. KV-1 = standard dense (GQA/MHA/MQA), KV-2 = MLA (Multi-head Latent Attention), KV-3a = sliding window, KV-4 = cross-layer sharing, KV-5b = SSM-based (state-space models). Fewer KV heads = smaller KV cache per request.',
    tags: ['mla', 'sliding window', 'architecture'],
  },
  vllm: {
    title: 'vLLM',
    body: 'A high-performance LLM inference engine that uses paged attention for efficient KV cache management.',
    tags: ['backend', 'serving engine'],
  },
  infoStrip: {
    title: 'Assumption summary',
    body: 'The strip under the inputs states the workload the next calculation will use: sequence lengths, KV cache precision, and concurrency. It is a summary of your current inputs, not the last result — use Adjust to change them.',
    tags: ['assumptions', 'summary', 'info strip'],
  },
  sizingGoal: {
    title: 'Sizing goal',
    body: 'What you are asking the recommender to optimise for. Latency finds the smallest configuration that meets a TTFT/TPOT target. Demand finds the fewest GPUs that sustain a request rate. Capacity finds the fewest GPUs that hold a number of concurrent requests. They answer different questions and can return different GPU counts.',
    tags: ['objective', 'recommend', 'latency', 'demand', 'capacity'],
  },
  pareto: {
    title: 'Pareto frontier',
    body: 'The set of configurations where you cannot improve one axis (for example lower cost) without making another worse (for example higher latency). Points off the frontier are dominated by a better option. Use it to compare trade-offs rather than to get one answer.',
    tags: ['trade-off', 'frontier', 'recommend'],
  },
  servingMode: {
    title: 'Serving mode · aggregated vs disaggregated',
    body: 'Aggregated serves prompt processing (prefill) and token generation (decode) on the same GPUs. Disaggregated runs them in separate pools so each can be sized for its own bottleneck, at the cost of moving KV cache between them. Disaggregated needs a network fabric and is only supported by some simulators.',
    tags: ['agg', 'disagg', 'prefill', 'decode', 'p/d'],
  },
  fabric: {
    title: 'Network fabric',
    body: 'The interconnect used between disaggregated prefill and decode pools. It sets the bandwidth and latency for moving KV cache, so it changes the predicted performance. It is a hardware and topology choice, not a simulator setting.',
    tags: ['network', 'roce', 'infiniband', 'interconnect'],
  },
  testedVerdict: {
    title: 'Tested-model verdict',
    body: 'How close a model-on-GPU combination is to the configurations that have published benchmark runs. Within validated range means the prediction interpolates measured data. Near validated boundary means it is an edge case. Outside validated range means the simulator is extrapolating. Validation unavailable means there is no benchmark data to compare against. None of these are guarantees; they describe confidence, not correctness.',
    tags: ['validation', 'benchmark', 'confidence'],
  },
  deepLink: {
    title: 'Shared link',
    body: 'A link that carries a page\'s inputs so someone else sees the same configuration. Generated links include the simulator, backend, and backend version used. Opening a link never runs a calculation and never overwrites your saved settings. If the link was made with a different simulator, you are asked before it is applied.',
    tags: ['share', 'url', 'permalink', 'copy link'],
  },
} as const satisfies Record<string, Omit<GlossaryEntry, 'id'>>;

export type GlossaryId = keyof typeof ENTRIES;

export const GLOSSARY: Record<GlossaryId, GlossaryEntry> = Object.fromEntries(
  Object.entries(ENTRIES).map(([id, entry]) => [id, { id: id as GlossaryId, ...entry }]),
) as unknown as Record<GlossaryId, GlossaryEntry>;

export const GLOSSARY_ENTRIES: GlossaryEntry[] = Object.values(GLOSSARY);

function normalise(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, ' ');
}

/**
 * Resolve a glossary entry from a display term, an id, or a title fragment.
 * Used by `JargonPopover` so call sites can keep passing human-readable terms.
 */
export function findGlossaryEntry(term: string): GlossaryEntry | undefined {
  const needle = normalise(term);
  if (!needle) return undefined;

  if (needle in GLOSSARY) return GLOSSARY[needle as GlossaryId];

  return GLOSSARY_ENTRIES.find((entry) => {
    const title = normalise(entry.title);
    const shortTitle = normalise(entry.title.split('·')[0]);
    return (
      title === needle ||
      shortTitle === needle ||
      entry.id.toLowerCase() === needle ||
      (entry.tags ?? []).some((tag) => normalise(tag) === needle)
    );
  });
}

/**
 * Curated map of user-visible controls.
 *
 * This is what lets search answer "where do I set concurrency?" rather than
 * only "what is concurrency?". One entry per control that a user might look
 * for by name. It is intentionally hand-written: a route scrape would describe
 * markup, not intent.
 */
export interface HelpAffordance {
  id: string;
  /** Route the control lives on. */
  route: string;
  /** Documentation section that explains the control. */
  doc?: string;
  /** Anchor within that documentation section. */
  anchor?: string;
  label: string;
  keywords: string[];
}

export const HELP_AFFORDANCES: HelpAffordance[] = [
  // ── Predict ──────────────────────────────────────────────────────────────
  {
    id: 'predict-model',
    route: '/predict',
    doc: 'user-scenarios',
    anchor: 'which-tool-to-use-when',
    label: 'Choose a model',
    keywords: ['model', 'hugging face', 'checkpoint', 'architecture'],
  },
  {
    id: 'predict-gpu',
    route: '/predict',
    doc: 'settings',
    anchor: 'simulator',
    label: 'Choose a GPU system',
    keywords: ['gpu', 'system', 'hardware', 'accelerator'],
  },
  {
    id: 'predict-workload-preset',
    route: '/predict',
    label: 'Workload presets',
    keywords: ['preset', 'isl', 'osl', 'prompt length', 'sequence length', 'workload'],
  },
  {
    id: 'predict-concurrency',
    route: '/predict',
    doc: 'user-scenarios',
    anchor: 'sizing-multiple-workloads',
    label: 'Concurrency presets',
    keywords: ['concurrency', 'concurrent users', 'batch', 'traffic'],
  },
  {
    id: 'predict-calculate',
    route: '/predict',
    label: 'Calculate',
    keywords: ['calculate', 'run', 'submit', 'results', 'refresh'],
  },
  {
    id: 'predict-assumptions',
    route: '/predict',
    doc: 'caveats',
    label: 'Change assumptions',
    keywords: ['assumptions', 'tp', 'pp', 'precision', 'max_model_len', 'chunked prefill'],
  },
  {
    id: 'predict-disagg',
    route: '/predict',
    doc: 'caveats',
    anchor: 'disaggregated-serving-needs-a-fabric',
    label: 'Disaggregated serving and fabric',
    keywords: ['disagg', 'disaggregated', 'prefill', 'decode', 'fabric', 'network'],
  },
  {
    id: 'predict-share',
    route: '/predict',
    doc: 'settings',
    anchor: 'sharing-a-view',
    label: 'Copy a link to this view',
    keywords: ['share', 'link', 'permalink', 'copy'],
  },

  // ── Recommend ───────────────────────────────���────────────────────────────
  {
    id: 'recommend-goal',
    route: '/recommend',
    label: 'Sizing goal',
    keywords: ['goal', 'objective', 'latency', 'demand', 'capacity', 'sla'],
  },
  {
    id: 'recommend-targets',
    route: '/recommend',
    label: 'Latency and throughput targets',
    keywords: ['ttft', 'tpot', 'latency', 'throughput', 'target', 'sla'],
  },
  {
    id: 'recommend-pareto',
    route: '/recommend',
    doc: 'pareto',
    label: 'Pareto frontier',
    keywords: ['pareto', 'trade-off', 'frontier', 'options'],
  },

  // ── KV cache ─────────────────────────────────────────────────────────────
  {
    id: 'kv-model',
    route: '/kv-cache',
    label: 'KV cache model and GPU',
    keywords: ['kv cache', 'memory', 'model', 'gpu'],
  },
  {
    id: 'kv-batch',
    route: '/kv-cache',
    doc: 'user-scenarios',
    label: 'KV cache batch size and tokens',
    keywords: ['max tokens', 'batch', 'concurrency', 'context window'],
  },

  // ── GPU explorer ─────────────────────────────────────────────────────────
  {
    id: 'gpu-explorer-axes',
    route: '/gpu-explorer',
    label: 'Bubble chart axes',
    keywords: ['chart', 'axes', 'vram', 'throughput', 'cost', 'compare'],
  },
  {
    id: 'gpu-explorer-vendor',
    route: '/gpu-explorer',
    label: 'Vendor filter',
    keywords: ['vendor', 'nvidia', 'amd', 'filter'],
  },

  // ── Hybrid savings ───────────────────────────────────────────────────────
  {
    id: 'hybrid-workload',
    route: '/hybrid-savings',
    doc: 'costs',
    label: 'Monthly token volumes',
    keywords: ['tokens', 'monthly', 'input', 'output', 'workload'],
  },
  {
    id: 'hybrid-lens',
    route: '/hybrid-savings',
    doc: 'costs',
    anchor: 'what-the-number-includes',
    label: 'Cost lens (hosted, rented, purchased)',
    keywords: ['cost', 'hosted', 'rented', 'purchased', 'tco', 'cloud'],
  },
  {
    id: 'hybrid-assumptions',
    route: '/hybrid-savings',
    doc: 'costs',
    anchor: 'planning-assumptions',
    label: 'Cost planning assumptions',
    keywords: ['assumptions', 'power', 'pue', 'amortisation', 'staff', 'maintenance'],
  },

  // ── Validation ───────────────────────────────────────────────────────────
  {
    id: 'validation-filters',
    route: '/validation/compare',
    doc: 'tested-models',
    label: 'Benchmark comparison filters',
    keywords: ['validation', 'benchmark', 'measured', 'predicted', 'dataset'],
  },

  // ── Settings ─────────────────────────────────────────────────────────────
  {
    id: 'settings-simulator',
    route: '/settings',
    doc: 'settings',
    anchor: 'simulator',
    label: 'Select a simulator',
    keywords: ['simulator', 'aisimulate', 'blis', 'default'],
  },
  {
    id: 'settings-backend',
    route: '/settings',
    doc: 'settings',
    anchor: 'backend-and-version',
    label: 'Select a backend and version',
    keywords: ['backend', 'vllm', 'sglang', 'tensorrt', 'version'],
  },

  // ── Docs ─────────────────────────────────────────────────────────────────
  {
    id: 'docs-home',
    route: '/docs',
    doc: 'overview',
    label: 'Documentation',
    keywords: ['docs', 'documentation', 'help', 'reference'],
  },
];

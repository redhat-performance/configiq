/**
 * Which page a question is about.
 *
 * One table feeds both the plain "Open the right tool" link and the proposal
 * that carries values into a page's link contract. Keeping them in one place
 * means a question cannot be routed to one page and proposed against another.
 */

export interface PageHint {
  /** First match wins, so order these from most to least specific. */
  pattern: RegExp;
  href: string;
  label: string;
}

export const PAGE_HINTS: readonly PageHint[] = [
  { pattern: /\brecommend|sizing goal|how many gpus|capacity|demand\b/i, href: '/recommend', label: 'Open Recommend sizing' },
  { pattern: /\bpredict|ttft|tpot|latency|performance\b/i, href: '/predict', label: 'Open Predict performance' },
  { pattern: /\bkv cache|kv-cache|vram|memory\b/i, href: '/kv-cache', label: 'Open KV cache calculator' },
  { pattern: /\bhybrid|hosted|purchased|rented\b/i, href: '/hybrid-savings', label: 'Open Hybrid savings' },
  { pattern: /\bbenchmark|validation|measured\b/i, href: '/validation/compare', label: 'Open Benchmark comparison' },
  { pattern: /\bgpu explorer|bubble|tflops|bandwidth\b/i, href: '/gpu-explorer', label: 'Open GPU explorer' },
  { pattern: /\bsettings|simulator|backend\b/i, href: '/settings', label: 'Open Settings' },
];

/**
 * The page a question is about, or null when the question names none.
 *
 * A proposal is only built against a page with a link contract, so this says
 * which page, not which fields: the contract stays the one authority on those.
 */
export function pageHintFor(question: string): PageHint | null {
  return PAGE_HINTS.find((hint) => hint.pattern.test(question)) ?? null;
}

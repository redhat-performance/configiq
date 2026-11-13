/**
 * Deterministic numeric-claim verifier.
 *
 * A 7–8B model on an L4 will occasionally invent a GPU count, a price, or a
 * latency target. Retrieval does not prevent that: the model can still produce
 * a number that appears nowhere in the material it was given. This module is
 * the last gate before prose reaches the user, and it is deliberately dumb —
 * every quantitative claim must trace to a retrieved section or a tool result.
 */

/**
 * Words that turn a number into a claim. `3` on its own is ordinary prose;
 * `3 GPUs`, `3 s`, and `3 tokens` are not.
 */
const UNIT_WORDS = new Set([
  'ms', 'us', 'µs', 's', 'sec', 'secs', 'second', 'seconds',
  'minute', 'minutes', 'hour', 'hours', 'day', 'days', 'month', 'months', 'year', 'years',
  'gb', 'gib', 'mb', 'mib', 'kb', 'tb', 'tb/s', 'gb/s', 'mb/s', 'gbps', 'tbps',
  'watts', 'w', 'kw', 'kwh', 'watt',
  'token', 'tokens', 'tok', 'tok/s', 't/s',
  'gpu', 'gpus', 'device', 'devices', 'node', 'nodes',
  'request', 'requests', 'req/s', 'rps', 'user', 'users', 'sequence', 'sequences',
  'copy', 'copies', 'batch', 'batches', 'worker', 'workers', 'replica', 'replicas',
  'tflop', 'tflops', 'flop', 'flops', 'tflop/s',
  'usd', 'dollar', 'dollars', 'cent', 'cents', 'hourly',
  'byte', 'bytes', 'b', 'percent',
]);

/** Numbers with at least two digits are always claims. */
const SIGNIFICANT_DIGITS = 2;

/**
 * A number plus the words immediately after it. The words are only used to
 * decide whether the number carries a unit (`8 H100 GPUs` is a claim, `3 ways`
 * is not) and to report the claim in readable form.
 */
const NUMBER_PATTERN = /-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s+[A-Za-z][\w./%-]*){0,3}/g;

const URL_PATTERN = /\bhttps?:\/\/[^\s)]+/gi;
const MARKDOWN_LINK_PATTERN = /\[([^\]]*)\]\([^)]*\)/g;
/** Ordered-list markers such as `1.` or `2)` are not claims. */
const LIST_MARKER_PATTERN = /^\s*\d+[.)]\s+/gm;

interface NumberClaim {
  /** The matched text, for the user-facing explanation. */
  raw: string;
  /** Digits only, so `1,000` and `1000` compare equal. */
  value: number;
  hasUnit: boolean;
}

function stripNonProse(text: string): string {
  return text
    .replace(URL_PATTERN, ' ')
    .replace(MARKDOWN_LINK_PATTERN, '$1')
    .replace(LIST_MARKER_PATTERN, ' ');
}

function claimsIn(text: string): NumberClaim[] {
  const claims: NumberClaim[] = [];

  for (const match of text.matchAll(NUMBER_PATTERN)) {
    const parts = match[0].trim().split(/\s+/);
    const digits = parts[0];
    const value = Number(digits.replace(/,/g, ''));
    if (!Number.isFinite(value)) continue;

    // Keep the words up to and including the first unit, then stop: a longer
    // tail would read as part of the claim when it is ordinary prose.
    const words: string[] = [];
    let hasUnit = false;
    for (const word of parts.slice(1)) {
      const normalised = word.toLowerCase().replace(/[.,;:]$/, '');
      words.push(word);
      if (UNIT_WORDS.has(normalised)) {
        hasUnit = true;
        break;
      }
    }

    claims.push({ raw: [digits, ...words].join(' '), value, hasUnit });
  }

  return claims;
}

/** A claim matters when it can be wrong in a way the user would notice. */
function isSignificant(claim: NumberClaim): boolean {
  if (claim.hasUnit) return true;
  return Math.abs(claim.value) >= 10 ** (SIGNIFICANT_DIGITS - 1);
}

/**
 * Collect every number the assistant was given. Values are compared as numbers
 * so formatting differences (`1,000` vs `1000`, `8.0` vs `8`) do not count as
 * unsupported claims.
 */
export function supportedNumbers(evidence: string[]): Set<number> {
  const values = new Set<number>();
  for (const entry of evidence) {
    for (const claim of claimsIn(stripNonProse(entry))) values.add(claim.value);
  }
  return values;
}

export interface VerificationResult {
  ok: boolean;
  /** Claim text that could not be traced to the evidence. */
  unsupported: string[];
}

/**
 * Verify every quantitative claim in `answer` against `evidence`.
 *
 * `evidence` is the retrieved section bodies plus every tool result the model
 * saw. Citations are added by the server, not the model, so they are not
 * checked here.
 */
export function verifyNumbers(answer: string, evidence: string[]): VerificationResult {
  const supported = supportedNumbers(evidence);
  const unsupported: string[] = [];

  for (const claim of claimsIn(stripNonProse(answer))) {
    if (!isSignificant(claim)) continue;
    if (supported.has(claim.value)) continue;
    if (!unsupported.includes(claim.raw)) unsupported.push(claim.raw);
  }

  return unsupported.length === 0 ? { ok: true, unsupported: [] } : { ok: false, unsupported };
}

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
 * `3 GPUs`, `3 s`, and `3 tokens` are not. Units named here are normalised
 * before comparison, so `sec`, `second`, and `seconds` are one unit while `ms`
 * and `s` stay distinct.
 */
const UNIT_SYNONYMS: Record<string, string> = {
  // time
  ms: 'ms', milli: 'ms', millis: 'ms', millisecond: 'ms', milliseconds: 'ms',
  us: 'us', 'µs': 'us', 'μs': 'us', micro: 'us', micros: 'us',
  microsecond: 'us', microseconds: 'us',
  s: 's', sec: 's', secs: 's', second: 's', seconds: 's',
  min: 'min', mins: 'min', minute: 'min', minutes: 'min',
  h: 'h', hr: 'h', hrs: 'h', hour: 'h', hours: 'h',
  d: 'd', day: 'd', days: 'd',
  mo: 'mo', month: 'mo', months: 'mo',
  y: 'yr', yr: 'yr', year: 'yr', years: 'yr',
  // memory and transfer
  b: 'b', byte: 'b', bytes: 'b',
  kb: 'kb', kib: 'kib', mb: 'mb', mib: 'mib', gb: 'gb', gib: 'gib', tb: 'tb', tib: 'tib',
  'gb/s': 'gb/s', gbps: 'gb/s', 'gbps/s': 'gb/s',
  'tb/s': 'tb/s', tbps: 'tb/s',
  'mb/s': 'mb/s',
  // tokens
  token: 'token', tokens: 'token', tok: 'token',
  'tok/s': 'token/s', 't/s': 'token/s', 'token/s': 'token/s', 'tokens/s': 'token/s',
  // counts
  gpu: 'gpu', gpus: 'gpu',
  device: 'device', devices: 'device',
  node: 'node', nodes: 'node',
  request: 'request', requests: 'request',
  'req/s': 'request/s', rps: 'request/s', 'request/s': 'request/s',
  user: 'user', users: 'user',
  sequence: 'sequence', sequences: 'sequence',
  copy: 'copy', copies: 'copy',
  batch: 'batch', batches: 'batch',
  worker: 'worker', workers: 'worker',
  replica: 'replica', replicas: 'replica',
  // money
  usd: 'usd', dollar: 'usd', dollars: 'usd',
  cent: 'cent', cents: 'cent',
  eur: 'eur', gbp: 'gbp', jpy: 'jpy',
  // power
  w: 'w', watt: 'w', watts: 'w', kw: 'kw', kwh: 'kwh',
  // compute
  flop: 'flop', flops: 'flop', tflop: 'tflop', tflops: 'tflop', 'tflop/s': 'tflop/s',
  // ratios share
  percent: 'percent', ratio: 'x', x: 'x',
};

/** Symbols that stand in for a unit in front of the number. */
const PREFIX_UNITS: Record<string, string> = {
  $: 'usd',
  '€': 'eur',
  '£': 'gbp',
  '¥': 'jpy',
};

/** Symbols that stand in for a unit after the number. */
const SUFFIX_SYMBOLS: Record<string, string> = {
  '%': 'percent',
  '×': 'x',
};

/** Numbers with at least two digits are always claims. */
const SIGNIFICANT_DIGITS = 2;

/**
 * Magnitude suffixes attached to a number (`128k`, `1.5M`, `70B`). Lowercase
 * `b` is deliberately absent: it is the byte unit matched through
 * `UNIT_SYNONYMS`, while uppercase `B` is the colloquial billion in a model
 * name such as `70B`.
 */
const MAGNITUDE_SUFFIXES: Record<string, number> = { k: 1e3, m: 1e6, t: 1e12 };

/** Magnitude words written after a number (`1.5 million tokens`). */
const MAGNITUDE_WORDS: Record<string, number> = {
  thousand: 1e3,
  million: 1e6,
  billion: 1e9,
  trillion: 1e12,
};

/**
 * A number with an optional currency symbol in front and an optional unit
 * attached to it (`8GB`, `5%`, `$3.50`, `2×`, `1e6`). A unit written as a
 * following word (`128 tokens`, `24 ms`) is looked for in `claimsIn`, because a
 * single regex capturing a repeated word group would only keep the last match.
 */
const NUMBER_PATTERN =
  /([$€£¥])?-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?(%|×|[A-Za-z][A-Za-z0-9/·._-]*)?/g;

/** The numeric part of a token, including a scientific exponent. */
const NUMBER_VALUE = /-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?/;

/** Words after the number that may name its unit. At most three are read. */
const FOLLOWING_WORD = /\s+([A-Za-z][\w./%-]*)/y;
/** A rate unit stated in words: `tokens per second`. */
const PER_RATE = /\s+per\s+([A-Za-z][\w./%-]*)/iy;
const MAX_FOLLOWING_WORDS = 3;

/**
 * Denominators that make a compound unit a rate (`tokens per second`). A
 * currency is not one: `$3.00 per hour` is a price, and the currency alone is
 * what the evidence and the answer share.
 */
const RATE_DENOMINATORS = new Set(['s', 'ms', 'us', 'min', 'h', 'd', 'mo', 'yr']);
const CURRENCIES = new Set(['usd', 'eur', 'gbp', 'jpy', 'cent']);

function isRateUnit(base: string, rate: string): boolean {
  return RATE_DENOMINATORS.has(rate) && !CURRENCIES.has(base);
}

/** A single-letter magnitude suffix, or undefined when the token is a unit. */
function magnitudeSuffix(attached: string): number | undefined {
  if (attached === 'B') return 1e9;
  if (attached.length !== 1) return undefined;
  return MAGNITUDE_SUFFIXES[attached.toLowerCase()];
}

const URL_PATTERN = /\bhttps?:\/\/[^\s)]+/gi;
const MARKDOWN_LINK_PATTERN = /\[([^\]]*)\]\([^)]*\)/g;
/** Ordered-list markers such as `1.` or `2)` are not claims. */
const LIST_MARKER_PATTERN = /^\s*\d+[.)]\s+/gm;

export interface NumberClaim {
  /** The matched text, for the user-facing explanation. */
  raw: string;
  /** Digits only, so `1,000` and `1000` compare equal. */
  value: number;
  /** Normalised unit, or null when the number carries none. */
  unit: string | null;
}

function stripNonProse(text: string): string {
  return text
    .replace(URL_PATTERN, ' ')
    .replace(MARKDOWN_LINK_PATTERN, '$1')
    .replace(LIST_MARKER_PATTERN, ' ');
}

function normalizeUnitWord(word: string): string | null {
  const lowered = word.toLowerCase();
  return UNIT_SYNONYMS[lowered] ?? UNIT_SYNONYMS[lowered.replace(/s$/, '')] ?? null;
}

function normalizeUnit(raw: string): string | null {
  const trimmed = raw.trim().replace(/[.,;:!?]+$/, '');
  if (!trimmed) return null;
  const symbol = PREFIX_UNITS[trimmed] ?? SUFFIX_SYMBOLS[trimmed];
  if (symbol) return symbol;
  const direct = normalizeUnitWord(trimmed);
  if (direct) return direct;
  // A rate such as `tokens/sec` or `GB/s`: normalise both halves so the same
  // rate written with a symbol and with a word compares equal.
  const slash = trimmed.indexOf('/');
  if (slash > 0 && slash < trimmed.length - 1) {
    const base = normalizeUnitWord(trimmed.slice(0, slash));
    const rate = normalizeUnitWord(trimmed.slice(slash + 1));
    if (base && rate && isRateUnit(base, rate)) return `${base}/${rate}`;
  }
  return null;
}

function claimsIn(text: string): NumberClaim[] {
  const claims: NumberClaim[] = [];
  const pattern = new RegExp(NUMBER_PATTERN.source, 'g');
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    const [token, prefix, attached] = match;
    const digits = token.match(NUMBER_VALUE)?.[0] ?? '';
    let value = Number(digits.replace(/,/g, ''));
    if (!Number.isFinite(value)) continue;

    // `128k`, `1.5M`, and `70B` mean a different number, not 128, 1.5, or 70.
    // A magnitude suffix is therefore part of the value, and the unit, if any,
    // is the following word (`128k tokens`).
    const magnitude = attached ? magnitudeSuffix(attached.replace(/[.,;:!?]+$/, '')) : undefined;
    if (magnitude !== undefined) value *= magnitude;

    // A currency symbol or an attached unit is unambiguous and is the whole
    // claim. An attached magnitude suffix is not a unit.
    let attachedUnit = normalizeUnit(prefix ?? '') ?? (magnitude === undefined ? normalizeUnit(attached ?? '') : null);
    if (attachedUnit) {
      // A prefixed amount can carry a magnitude word (`$1 million`) or a rate
      // phrase (`8GB per second`); both belong to this claim, not the next.
      let raw = token;
      let cursor = match.index + token.length;
      let sentenceEnded = false;
      for (let count = 0; count < MAX_FOLLOWING_WORDS; count += 1) {
        FOLLOWING_WORD.lastIndex = cursor;
        const word = FOLLOWING_WORD.exec(text);
        if (!word) break;
        const bare = word[1].replace(/[.,;:!?]+$/, '');
        const magnitudeWord = MAGNITUDE_WORDS[bare.toLowerCase()];
        if (magnitudeWord === undefined) break;
        value *= magnitudeWord;
        raw += ` ${bare}`;
        cursor = word.index + word[0].length;
        pattern.lastIndex = cursor;
        if (/[.!?]$/.test(word[1])) {
          sentenceEnded = true;
          break;
        }
      }
      if (!sentenceEnded) {
        PER_RATE.lastIndex = cursor;
        const per = PER_RATE.exec(text);
        if (per) {
          const rate = normalizeUnit(per[1]);
          if (rate && isRateUnit(attachedUnit, rate)) {
            attachedUnit = `${attachedUnit}/${rate}`;
            raw += ` per ${per[1].replace(/[.,;:!?]+$/, '')}`;
            pattern.lastIndex = per.index + per[0].length;
          }
        }
      }
      claims.push({ raw, value, unit: attachedUnit });
      continue;
    }

    // Otherwise the unit, if any, is the first recognised word that follows.
    // Everything up to it stays in the claim text so `8 H100 GPUs` is reported
    // as one unsupported claim rather than three, and the words that were read
    // are skipped so a model name inside them is not counted twice. A word
    // that ends a sentence ends the scan: `3 steps. GPUs` is two thoughts.
    let cursor = match.index + token.length;
    const kept: string[] = [];
    let unit: string | null = null;
    for (let count = 0; count < MAX_FOLLOWING_WORDS; count += 1) {
      FOLLOWING_WORD.lastIndex = cursor;
      const word = FOLLOWING_WORD.exec(text);
      if (!word) break;
      kept.push(word[1].replace(/[.,;:!?]+$/, ''));
      cursor = word.index + word[0].length;

      const magnitudeWord = MAGNITUDE_WORDS[word[1].replace(/[.,;:!?]+$/, '').toLowerCase()];
      if (magnitudeWord !== undefined) {
        value *= magnitudeWord;
        if (/[.!?]$/.test(word[1])) break;
        continue;
      }

      const candidate = normalizeUnit(word[1]);
      if (!candidate) {
        if (/[.!?]$/.test(word[1])) break;
        continue;
      }

      // `128 tokens per second` states the rate; `per second` is part of the
      // unit, not the next claim's prose. A rate cannot cross a sentence.
      unit = candidate;
      if (/[.!?]$/.test(word[1])) break;
      PER_RATE.lastIndex = cursor;
      const per = PER_RATE.exec(text);
      if (per) {
        const rate = normalizeUnit(per[1]);
        if (rate && isRateUnit(candidate, rate)) {
          unit = `${candidate}/${rate}`;
          kept.push('per', per[1].replace(/[.,;:!?]+$/, ''));
          cursor = per.index + per[0].length;
        }
      }
      break;
    }
    if (unit) pattern.lastIndex = cursor;

    claims.push({
      raw: unit ? [token, ...kept].join(' ') : token,
      value,
      unit,
    });
  }

  return claims;
}

/** A claim matters when it can be wrong in a way the user would notice. */
function isSignificant(claim: NumberClaim): boolean {
  if (claim.unit) return true;
  return Math.abs(claim.value) >= 10 ** (SIGNIFICANT_DIGITS - 1);
}

/**
 * Collect every number the assistant was given, with the units the evidence
 * attached to it. A value can appear both with and without a unit, so the set
 * carries `null` alongside any units.
 */
export function supportedClaims(evidence: string[]): Map<number, Set<string | null>> {
  const support = new Map<number, Set<string | null>>();
  for (const entry of evidence) {
    for (const claim of claimsIn(stripNonProse(entry))) {
      const units = support.get(claim.value) ?? new Set<string | null>();
      units.add(claim.unit);
      support.set(claim.value, units);
    }
  }
  return support;
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
 *
 * Units must agree when both sides state one: `128 GPUs` is not supported by
 * `128 tokens`. Evidence that gives the number with no unit attached is
 * compatible with any unit, because a table cell or a bullet frequently states
 * the unit in a heading instead of next to the number. Absence of a unit is
 * therefore allowed, while a contradictory unit is not — the gate should never
 * suppress an answer that the evidence does support.
 */
export function verifyNumbers(answer: string, evidence: string[]): VerificationResult {
  const support = supportedClaims(evidence);
  const unsupported: string[] = [];

  for (const claim of claimsIn(stripNonProse(answer))) {
    if (!isSignificant(claim)) continue;
    const units = support.get(claim.value);
    if (units) {
      // A claim with no unit is a bare value the evidence also contains, and
      // evidence with no unit cannot contradict the claim's unit.
      if (!claim.unit || units.has(claim.unit) || units.has(null)) continue;
    }
    if (!unsupported.includes(claim.raw)) unsupported.push(claim.raw);
  }

  return unsupported.length === 0 ? { ok: true, unsupported: [] } : { ok: false, unsupported };
}

/**
 * View proposals.
 *
 * A sizing question should end in a link that carries the values the user
 * already named, so the page opens with their question in it and they only
 * press Calculate. The model never writes that URL: the question is read here
 * with a small set of explicit patterns, and every value goes through the
 * destination page's own link contract before it is serialized. A proposal can
 * therefore never introduce a parameter the page does not accept, or a value
 * outside the bounds it parses with.
 *
 * Nothing here runs sizing. The link is inputs only, and the page's Calculate
 * button is still the only thing that produces a result.
 *
 * The page's current model and GPU system come from the browser (settings are
 * client state), so a question that names neither still opens a complete view.
 * A value written in the question wins over the setting.
 */

import { kvCacheLink, recommendLink } from '@/lib/deep-links/pages';
import { predictLink } from '@/lib/deep-links/predict';
import { buildValidated, type LinkFieldMap, type PageLink } from '@/lib/deep-links/common';
import { pageHintFor } from './intent';
import type { AssistantLink } from './types';

export interface ProposalContext {
  question: string;
  /** The model the page is currently set to, as the browser reports it. */
  model?: string;
  /** The GPU system the page is currently set to, as the browser reports it. */
  system?: string;
  simulator?: string;
  backend?: string;
  backendVersion?: string;
}

export interface IgnoredValue {
  key: string;
  /** The value as it was read, or the alternatives when they conflict. */
  value: string;
  /**
   * `rejected` — the page's contract refused the value.
   * `ambiguous` — the question named more than one value for the field.
   * `unsupported` — the destination page has no field for it.
   * `unused` — the page has the field, but the goal the view selected ignores it.
   */
  reason: 'rejected' | 'ambiguous' | 'unsupported' | 'unused';
}

export interface ProposedView {
  link: AssistantLink;
  /** Contract fields the link carries, without the values themselves. */
  fields: string[];
  /** Values read from the question that are not in the link, with the reason. */
  ignored: IgnoredValue[];
  /**
   * Sentences naming the carried values with their units, so an answer that
   * repeats a number from the question is not mistaken for an invented one.
   */
  evidence: string[];
}

/** Contract keys that describe where a view came from, not what it asks for. */
const PROVENANCE_KEYS = ['simulator', 'backend', 'backendVersion'];

/** Plain-language names, so the link can say what it will set. */
const FIELD_LABELS: Record<string, string> = {
  model: 'model',
  system: 'GPU system',
  goal: 'goal',
  isl: 'input length',
  osl: 'output length',
  concurrency: 'concurrency',
  demandRate: 'request rate',
  ttft: 'time to first token',
  tpot: 'time per output token',
  e2eLatency: 'end-to-end latency',
  prefix: 'prefix length',
  // The same limit, under the name each page gives its control.
  maxSeqLen: 'max sequence length',
  maxModelLen: 'max sequence length',
  maxNumTokens: 'maximum token count',
  maxBatchSize: 'batch size',
};

interface ProposablePage {
  name: string;
  /** The fields the page's contract declares, so a stray one can be reported. */
  fields: Set<string>;
  /**
   * The page's own name for the model's maximum sequence length. The three
   * tools ask the same question with different field names, so the extractor
   * reads one concept and the page says where it goes.
   */
  contextField: string;
  build: (values: Record<string, unknown>) => {
    href: string;
    values: Record<string, unknown>;
    ignored: Array<{ key: string; value: unknown }>;
  };
}

function proposable<F extends LinkFieldMap>(
  name: string,
  link: PageLink<F>,
  contextField: keyof F & string,
): ProposablePage {
  return {
    name,
    fields: new Set(Object.keys(link.fields)),
    contextField,
    build: (values) => {
      const result = buildValidated(link, values);
      return {
        href: result.href,
        values: result.values as unknown as Record<string, unknown>,
        ignored: result.ignored,
      };
    },
  };
}

/**
 * Pages whose contract can carry what a question names. The rest of the hints
 * stay plain links: a rail chart or a measured run is not a request a
 * paragraph can describe well enough to fill in.
 */
const PROPOSABLE_PAGES: Record<string, ProposablePage> = {
  '/recommend': proposable('Recommend sizing', recommendLink, 'maxSeqLen'),
  '/predict': proposable('Predict performance', predictLink, 'maxModelLen'),
  '/kv-cache': proposable('KV cache calculator', kvCacheLink, 'maxNumTokens'),
};

const NUMBER = String.raw`(\d[\d,]*(?:\.\d+)?)`;
/**
 * A count, but not the front half of `4k` or of a longer identifier: `input
 * length 4k` is not 4 tokens, and quietly proposing 4 would be a different
 * question from the one asked.
 */
const COUNT = String.raw`(\d[\d,]*(?:\.\d+)?)(?!\w)`;
// Longest form first, so `seconds` is consumed whole rather than leaving
// `econds` behind when the span is blanked out later.
const TIME_UNIT = String.raw`(milliseconds?|ms|seconds?|secs?|s)`;

/**
 * What may stand between a latency label and its number.
 *
 * Whitespace always may, so `TTFT 2 s` reads like `TTFT under 2 s`. A plain
 * bounded wildcard would also swallow the next clause: in "2 s to first token
 * and 50 ms per output token", "first token and 50 ms" is the *second* token's
 * latency, not another time-to-first-token. So whitespace and a connective that
 * introduces a target are the only things allowed before the number.
 */
const TARGET_LEAD =
  String.raw`(?:\s*(?:of|at|is|in|under|below|within|about|around|less\s+than|:|=|<|<=|≤))?\s*`;

/**
 * `4096/512 input/output` states both lengths in one place. Without this the
 * second number would be read as the input length, because it sits immediately
 * before the word "input".
 */
const LENGTH_RATIO = new RegExp(
  String.raw`${COUNT}\s*\/\s*(\d[\d,]*(?:\.\d+)?)\s*(?:tokens?\s+)?(?:input|prompt)\s*(?:\/|and|to)\s*(?:tokens?\s+)?(?:output|completion)`,
  'i',
);

/** `input length 4096`, `ISL=4096`, `input tokens: 4096`. */
const BEFORE_ISL = new RegExp(
  String.raw`\b(?:isl|input\s+(?:length|size|tokens?)|prompt\s+(?:length|size|tokens?))\b\s*(?:=|:|of)?\s*${COUNT}`,
  'i',
);
/** `input=4096`. */
const EXPLICIT_ISL = new RegExp(String.raw`\b(?:isl|input|prompt)\s*[=:]\s*${COUNT}`, 'i');
/**
 * `4096 input tokens`, `4,000 token inputs`. The word order varies, and the
 * second form is easy to read backwards: the number is the length either way.
 */
const AFTER_ISL = new RegExp(
  String.raw`${COUNT}\s*(?:tokens?\s+(?:of\s+)?(?:input|prompt)s?|(?:input|prompt)\s*(?:tokens?|tok)?)\b`,
  'i',
);

const BEFORE_OSL = new RegExp(
  String.raw`\b(?:osl|output\s+(?:length|size|tokens?)|completion\s+(?:length|tokens?)|response\s+(?:length|tokens?))\b\s*(?:=|:|of)?\s*${COUNT}`,
  'i',
);
const EXPLICIT_OSL = new RegExp(String.raw`\b(?:osl|output|completion)\s*[=:]\s*${COUNT}`, 'i');
const AFTER_OSL = new RegExp(
  String.raw`${COUNT}\s*(?:tokens?\s+(?:of\s+)?(?:output|completion)s?|(?:output|completion)\s*(?:tokens?|tok)?)\b`,
  'i',
);

const BEFORE_CONCURRENCY = new RegExp(
  String.raw`\bconcurrency\b\s*(?:=|:|of)?\s*${COUNT}`,
  'i',
);
const AFTER_CONCURRENCY = new RegExp(String.raw`${COUNT}\s*(?:concurrent|simultaneous)\b`, 'i');

/**
 * A rate is only carried when it is stated per second, the unit the page
 * stores. `60 requests per minute` is left out rather than quietly written as
 * 60 per second.
 */
const REQUEST_RATE = new RegExp(
  String.raw`${COUNT}\s*(?:requests?|reqs?)\s*(?:per\s+|\/\s*)(?:second|sec|s)\b`,
  'i',
);
const RATE_SHORTHAND = new RegExp(String.raw`${COUNT}\s*(?:rps|qps)\b`, 'i');

const BEFORE_TTFT = new RegExp(
  String.raw`\b(?:ttft|time\s+to\s+first\s+token|first\s+token)\b${TARGET_LEAD}${NUMBER}\s*${TIME_UNIT}?`,
  'i',
);
const AFTER_TTFT = new RegExp(
  String.raw`${NUMBER}\s*${TIME_UNIT}\s*(?:to|for)\s+(?:the\s+)?first\s+token`,
  'i',
);

/**
 * Inter-token latency is the same quantity the tools call time per output
 * token: the average gap between two output tokens. A question that states ITL
 * is stating TPOT in another name, and the page only has the latter.
 */
const TPOT_LABEL = String.raw`(?:tpot|itl|inter[- ]?token\s+latency|time\s+per\s+(?:output\s+)?token|per\s+(?:output\s+)?token)`;

const BEFORE_TPOT = new RegExp(
  String.raw`\b${TPOT_LABEL}\b${TARGET_LEAD}${NUMBER}\s*${TIME_UNIT}?`,
  'i',
);
const AFTER_TPOT = new RegExp(
  String.raw`${NUMBER}\s*${TIME_UNIT}\s*(?:per|for\s+each)\s+(?:output\s+)?token`,
  'i',
);

/**
 * A latency written before its label: `1second TTFT`, `100ms ITL`. The
 * label-first patterns cannot read these, because no connective ties the
 * number to the word that follows it, and the number would otherwise be read
 * as a length or ignored.
 *
 * A minus sign in front of the number is not a target, and a digit or decimal
 * point means the match would be the tail of a longer number, so neither may
 * start the number.
 */
const TRAILING_TTFT = new RegExp(
  String.raw`(?<![\d.,-])${NUMBER}\s*${TIME_UNIT}\s*(?:ttft|time\s+to\s+first\s+token|first\s+token)\b`,
  'i',
);
const TRAILING_TPOT = new RegExp(
  String.raw`(?<![\d.,-])${NUMBER}\s*${TIME_UNIT}\s*${TPOT_LABEL}\b`,
  'i',
);

/**
 * The ways a question names the whole-request latency. The bare "end-to-end"
 * and "e2e" forms take an optional "latency"; "request" and "total" do not
 * stand alone, because "total 4 s" is as likely to be something else.
 */
const E2E_LABEL = String.raw`(?:(?:end[- ]to[- ]end|e2e)(?:\s+latency)?|(?:request|total)\s+latency)`;

const BEFORE_E2E = new RegExp(
  String.raw`\b${E2E_LABEL}\b${TARGET_LEAD}${NUMBER}\s*${TIME_UNIT}?`,
  'i',
);
const AFTER_E2E = new RegExp(
  String.raw`${NUMBER}\s*${TIME_UNIT}\s*(?:end[- ]to[- ]end|in\s+total)`,
  'i',
);

const BEFORE_PREFIX = new RegExp(
  String.raw`\bprefix(?:\s+cache)?(?:\s+length)?\b\s*(?:=|:|of)?\s*${COUNT}`,
  'i',
);

const BEFORE_CONTEXT = new RegExp(
  String.raw`\b(?:context\s+(?:length|window|size)|max(?:imum)?\s+(?:model\s+)?(?:length|tokens))\b\s*(?:=|:|of)?\s*${COUNT}`,
  'i',
);
const AFTER_CONTEXT = new RegExp(String.raw`${COUNT}\s*(?:tokens?\s+)?(?:of\s+)?context\b`, 'i');

/**
 * A Hugging Face path, which is the only model reference we can carry without
 * a catalog lookup. Both segments must be at least three characters and the
 * match must contain a letter, so a ratio (`4096/512`) and a unit (`req/s`) are
 * not read as a model.
 */
const MODEL_PATH = /\b([A-Za-z0-9][A-Za-z0-9._-]{2,}\/[A-Za-z0-9][A-Za-z0-9._-]{2,})\b/;

/** Segments that make a slash-separated token prose rather than a model id. */
const NOT_A_MODEL = new Set([
  'input',
  'output',
  'prompt',
  'token',
  'tokens',
  'request',
  'requests',
  'req',
  'rps',
  'qps',
  'per',
  'sec',
  'second',
  'seconds',
  'ms',
  'gpu',
  'gpus',
  'model',
  'models',
  'ratio',
  'percent',
  // Runtimes and tools are written with a slash far more often than a model
  // repository is, and carrying one as a model would open the page on a name
  // no catalog knows.
  'vllm',
  'sglang',
  'tensorrt',
  'triton',
  'tgi',
  'ollama',
  'lmdeploy',
  'docker',
  'podman',
  'kubernetes',
  'nvidia',
  'amd',
  'intel',
]);

function toNumber(raw: string): number | null {
  const value = Number(raw.replace(/,/g, ''));
  return Number.isFinite(value) ? value : null;
}

/** One value read from the question, with the sentence that makes it quotable. */
interface Reading {
  value: number;
  text: string;
}

/** Every match of every pattern, in pattern then text order. */
function eachMatch(text: string, patterns: RegExp[]): RegExpExecArray[] {
  const matches: RegExpExecArray[] = [];
  for (const pattern of patterns) {
    const flags = pattern.flags.includes('g') ? pattern.flags : `${pattern.flags}g`;
    for (const match of text.matchAll(new RegExp(pattern.source, flags))) matches.push(match);
  }
  return matches;
}

interface Span {
  start: number;
  end: number;
}

/**
 * The number each pattern captured, not the whole match.
 *
 * Two patterns can share a label without sharing a target: in `1second TTFT at
 * 4096 input tokens`, the explicit pattern starts at `TTFT` and reads `4096`,
 * while the trailing pattern reads `1second`. Only the numbers decide whether
 * one pattern is reusing another's target.
 */
function numberSpans(text: string, patterns: RegExp[]): Span[] {
  const spans: Span[] = [];
  for (const match of eachMatch(text, patterns)) {
    const number = match[1];
    if (!number) continue;
    const offset = match[0].indexOf(number);
    if (offset === -1) continue;
    spans.push({ start: match.index + offset, end: match.index + offset + number.length });
  }
  return spans;
}

/**
 * Every count the question states for one field.
 *
 * More than one value means the question offered alternatives ("input length
 * 1024 or 8192"), and picking the first would present one of them as the
 * request. The caller reports the conflict instead of carrying a choice the
 * user did not make.
 */
function counts(text: string, patterns: RegExp[]): Reading[] {
  const readings: Reading[] = [];
  for (const match of eachMatch(text, patterns)) {
    const value = toNumber(match[1]);
    if (value === null || readings.some((reading) => reading.value === value)) continue;
    readings.push({ value, text: String(value) });
  }
  return readings;
}

/**
 * Every latency target the question states for one field, normalised to
 * milliseconds because that is the unit the page stores. A bare number with no
 * unit is not carried: `TTFT 2` could be two seconds or two milliseconds, and
 * picking one would invent a target.
 *
 * `claimed` holds the spans another field already read. A trailing label can
 * sit immediately after another field's target ("TTFT 2 s TPOT 50 ms"), and
 * that earlier number is not this field's target.
 */
function durations(text: string, patterns: RegExp[], claimed: Span[] = []): Reading[] {
  const readings: Reading[] = [];
  for (const match of eachMatch(text, patterns)) {
    const start = match.index;
    const end = start + (match[1]?.length ?? 0);
    if (claimed.some((span) => start < span.end && span.start < end)) continue;
    const unit = match[2]?.toLowerCase();
    if (!unit) continue;
    const value = toNumber(match[1]);
    if (value === null) continue;
    const milliseconds = /^(?:ms|milliseconds?)$/.test(unit) ? value : value * 1000;
    if (readings.some((reading) => reading.value === milliseconds)) continue;
    const spoken = `${match[1]} ${unit}`;
    readings.push({
      value: milliseconds,
      text: milliseconds === value ? spoken : `${spoken} (${milliseconds} ms)`,
    });
  }
  return readings;
}

/**
 * The readings from several pattern sets, with the same value kept once.
 *
 * A question can state one target in both forms ("50 ms per output token" is
 * read by the explicit and the trailing pattern), and the same value is one
 * target, not an alternative.
 */
function uniqueReadings(...groups: Reading[][]): Reading[] {
  const readings: Reading[] = [];
  for (const reading of groups.flat()) {
    if (readings.some((existing) => existing.value === reading.value)) continue;
    readings.push(reading);
  }
  return readings;
}

/**
 * The latency patterns that name their field explicitly, by label before the
 * number or by a connective after it. The number they read is authoritative: a
 * trailing label may not reuse it.
 */
const EXPLICIT_LATENCY = [
  BEFORE_TTFT,
  AFTER_TTFT,
  BEFORE_TPOT,
  AFTER_TPOT,
  BEFORE_E2E,
  AFTER_E2E,
];

/**
 * Every pattern that names a latency. Used to blank those expressions out
 * before the token-count patterns run.
 */
const LATENCY_PATTERNS = [
  BEFORE_TTFT,
  AFTER_TTFT,
  TRAILING_TTFT,
  BEFORE_TPOT,
  AFTER_TPOT,
  TRAILING_TPOT,
  BEFORE_E2E,
  AFTER_E2E,
];

/**
 * The question with its latency expressions blanked out.
 *
 * "time per output token 25 ms with output length 512" names one latency and
 * one length. Read as counts, `output token 25` looks like 25 output tokens, so
 * the real length would be dropped as ambiguous. Blanking the latency spans
 * keeps each number with the field that owns it, without narrowing the count
 * patterns and losing ordinary phrasing.
 */
function withoutLatencies(text: string, patterns: RegExp[]): string {
  let masked = text;
  // Every replacement has the same length as its match, so the indices found in
  // the original text stay valid.
  for (const match of eachMatch(text, patterns)) {
    // A label with no time unit ("ITL at 4096 input tokens") names no latency.
    // Blanking its number would also delete the input length that follows.
    if (!match[2]) continue;
    const start = match.index;
    masked = `${masked.slice(0, start)}${' '.repeat(match[0].length)}${masked.slice(start + match[0].length)}`;
  }
  return masked;
}

/**
 * Accelerator ids a question can name, spelled as the catalog spells them.
 *
 * Only ids the catalog actually has, and only ones that resolve to a single
 * system: "H100" alone is absent because it ships as SXM and PCIe, and pinning
 * either would name a machine the question did not. A multi-part id is matched
 * inside a longer run, after normalising hyphens, so
 * `nvidia-h100-sxm-80gb` resolves to `h100_sxm`.
 */
const SYSTEM_IDS = [
  'l40s',
  'h100_sxm',
  'h200_sxm',
  'a100_sxm',
  'a100_pcie',
  'a100_40',
  'a100_80',
  'b200_sxm',
  'mi300x',
];

/**
 * Bare accelerator names the catalog carries only one system for.
 *
 * "H200" resolves to the 141 GB SXM part, but only when the question did not
 * qualify it: "H200 NVL" and "H200 PCIe" are different machines, and pinning
 * SXM for them would open the page on hardware the question did not name.
 * "H100" is deliberately absent for the same reason: it ships as SXM and PCIe,
 * and pinning either would name a machine the question did not.
 */
const SYSTEM_ALIASES: ReadonlyArray<{ id: string; pattern: RegExp }> = [
  { id: 'h200_sxm', pattern: /\bh200\b(?!\s*(?:nvl|pcie))/i },
];

/**
 * The GPU system the question names, if it names exactly one.
 *
 * The system the page is set to is the default, and a named one has to win:
 * otherwise "on l40s" would open the page on the configured system and quietly
 * describe a different machine. Two names is a question, not a choice, so the
 * caller reports the conflict.
 */
function namedSystems(question: string): string[] {
  const normalised = question.replace(/-/g, '_');
  const found: string[] = [];
  for (const id of SYSTEM_IDS) {
    // `l40s` needs word boundaries (`l4` would otherwise match inside it); a
    // multi-part id is specific enough to match inside a longer run.
    const pattern = id.includes('_') ? new RegExp(id, 'i') : new RegExp(`\\b${id}\\b`, 'i');
    if (pattern.test(normalised) && !found.includes(id)) found.push(id);
  }
  // A bare name counts only when it did not resolve through an id above, so
  // "h200_sxm" is not also read as the alias and reported twice.
  for (const alias of SYSTEM_ALIASES) {
    if (alias.pattern.test(normalised) && !found.includes(alias.id)) found.push(alias.id);
  }
  return found;
}

/**
 * Both lengths when the question states them as a pair, so the second number is
 * not mistaken for the first. Numbers that are not counts (`4k`) match nothing
 * here either, because `COUNT` refuses a trailing word character.
 */
function lengthPair(question: string): { isl: number; osl: number } | null {
  const match = LENGTH_RATIO.exec(question);
  if (!match) return null;
  const isl = toNumber(match[1]);
  const osl = toNumber(match[2]);
  if (isl === null || osl === null) return null;
  return { isl, osl };
}

/**
 * Every model path written in the question, in order and without repeats.
 *
 * Every match is considered, because the first slash-separated token in a
 * question is often a runtime (`vllm/sglang`) rather than the model. A question
 * can also offer a choice between two models; returning both lets the caller
 * report that instead of answering with the first one.
 */
function namedModels(question: string): string[] {
  const found: string[] = [];
  const pattern = new RegExp(MODEL_PATH.source, 'gi');
  for (const match of question.matchAll(pattern)) {
    const whole = match[1];
    const [left, right] = whole.split('/');
    if (!/[A-Za-z]/.test(whole)) continue;
    if (NOT_A_MODEL.has(left.toLowerCase()) || NOT_A_MODEL.has(right.toLowerCase())) continue;
    if (!found.includes(whole)) found.push(whole);
  }
  return found;
}

/**
 * Which goal the values imply. The page picks a default and ignores numbers
 * that belong to another goal, so leaving this out would open a link whose
 * inputs are present but not used.
 */
function impliedGoal(values: Record<string, unknown>): 'latency' | 'demand' | 'capacity' | null {
  if (values.ttft !== undefined || values.tpot !== undefined || values.e2eLatency !== undefined) {
    return 'latency';
  }
  if (values.demandRate !== undefined) return 'demand';
  if (values.concurrency !== undefined) return 'capacity';
  return null;
}

/**
 * The values each sizing goal actually reads.
 *
 * The page writes goal-specific inputs into its link accessors, so a request
 * rate carried beside a latency target would sit in the URL and change nothing.
 * Naming the split here keeps the description honest about what the view uses.
 */
const GOAL_INPUTS: Record<'latency' | 'demand' | 'capacity', readonly string[]> = {
  latency: ['ttft', 'tpot', 'e2eLatency'],
  demand: ['demandRate'],
  capacity: ['concurrency'],
};

/** Every goal-specific input, so the unused ones can be found by difference. */
const GOAL_SPECIFIC_INPUTS = ['ttft', 'tpot', 'e2eLatency', 'demandRate', 'concurrency'];

/**
 * Build the view a question describes, or null when it describes none.
 *
 * Returning null is the common case: a question about what a control means has
 * no values to carry, and a plain page link is the honest answer.
 */
export function proposeView(context: ProposalContext): ProposedView | null {
  // A URL is not a value: its path can look like a model id without being one.
  const question = context.question.replace(/\bhttps?:\/\/\S+/gi, ' ');
  const hint = pageHintFor(question);
  if (!hint) return null;
  const page = PROPOSABLE_PAGES[hint.href];
  if (!page) return null;

  const candidates: Record<string, unknown> = {};
  const evidence = new Map<string, string>();
  /** Why a value read from the question is not in the link. */
  const ignored: IgnoredValue[] = [];
  /** The value as read, kept so a rejected or unsupported field can be quoted. */
  const rawValue = new Map<string, string>();
  /**
   * Values read from the question, as opposed to the page's current settings.
   * At least one is required: a proposal that only restates the defaults
   * changes nothing the page did not already show.
   */
  const fromQuestion = new Set<string>();

  /**
   * Record why a value is not in the link, once per field.
   *
   * The first reason wins: an ambiguous `context length` is reported as
   * ambiguous, not also as a field the page does not have.
   */
  const report = (entry: IgnoredValue) => {
    if (ignored.some((existing) => existing.key === entry.key)) return;
    ignored.push(entry);
  };

  /**
   * Carry one field, or report why it cannot be carried.
   *
   * A single value becomes a candidate; several mean the question offered
   * alternatives, and choosing between them would be our answer rather than
   * the user's. The field is left for the page's own value and named in the
   * link description.
   */
  const place = (key: string, readings: Reading[], sentence: (reading: Reading) => string) => {
    const values = readings.map((reading) => reading.value);
    if (values.length === 1) {
      const [reading] = readings;
      candidates[key] = reading.value;
      fromQuestion.add(key);
      rawValue.set(key, reading.text);
      evidence.set(key, sentence(reading));
      return;
    }
    if (values.length > 1) {
      fromQuestion.add(key);
      report({ key, value: values.join(', '), reason: 'ambiguous' });
    }
  };

  // A model path in the question wins over the setting, but a question that
  // offers two models has not chosen one. The setting is not substituted: it
  // would answer about a model the question never mentioned.
  const models = namedModels(question);
  if (models.length === 1) {
    candidates.model = models[0];
    fromQuestion.add('model');
  } else if (models.length > 1) {
    fromQuestion.add('model');
    report({ key: 'model', value: models.join(', '), reason: 'ambiguous' });
  } else if (context.model?.trim()) {
    candidates.model = context.model.trim();
  }

  // The configured system is the default, but a system named in the question
  // wins, for the same reason the model does.
  const systems = namedSystems(question);
  if (systems.length === 1) {
    candidates.system = systems[0];
    fromQuestion.add('system');
  } else if (systems.length > 1) {
    fromQuestion.add('system');
    report({ key: 'system', value: systems.join(', '), reason: 'ambiguous' });
  } else if (context.system?.trim()) {
    candidates.system = context.system.trim();
  }

  if (context.simulator?.trim()) candidates.simulator = context.simulator.trim();
  if (context.backend?.trim()) candidates.backend = context.backend.trim();
  if (context.backendVersion?.trim()) candidates.backendVersion = context.backendVersion.trim();

  // A paired statement wins over the single-value patterns: they would read the
  // second number as the input length, since it precedes the word "input".
  const pair = lengthPair(question);
  // Latencies are read from the whole question, then blanked out, so a number
  // that belongs to one cannot also be counted as a length or a rate.
  const countsFrom = withoutLatencies(question, LATENCY_PATTERNS);

  place(
    'isl',
    pair
      ? [{ value: pair.isl, text: String(pair.isl) }]
      : counts(countsFrom, [BEFORE_ISL, EXPLICIT_ISL, AFTER_ISL]),
    (reading) => `Input length: ${reading.text} tokens.`,
  );
  place(
    'osl',
    pair
      ? [{ value: pair.osl, text: String(pair.osl) }]
      : counts(countsFrom, [BEFORE_OSL, EXPLICIT_OSL, AFTER_OSL]),
    (reading) => `Output length: ${reading.text} tokens.`,
  );
  place('concurrency', counts(countsFrom, [BEFORE_CONCURRENCY, AFTER_CONCURRENCY]), (reading) =>
    `Concurrency: ${reading.text} concurrent requests.`,
  );
  place('demandRate', counts(countsFrom, [REQUEST_RATE, RATE_SHORTHAND]), (reading) =>
    `Request rate: ${reading.text} requests per second.`,
  );
  // A trailing label may follow another field's target ("TTFT 2 s TPOT
  // 50 ms"). The explicit span owns that number, so the trailing form does not
  // read it as a second target and turn one value into an ambiguity.
  const explicitSpans = numberSpans(question, EXPLICIT_LATENCY);
  place(
    'ttft',
    uniqueReadings(
      durations(question, [AFTER_TTFT, BEFORE_TTFT]),
      durations(question, [TRAILING_TTFT], explicitSpans),
    ),
    (reading) => `Time to first token: ${reading.text}.`,
  );
  place(
    'tpot',
    uniqueReadings(
      durations(question, [AFTER_TPOT, BEFORE_TPOT]),
      durations(question, [TRAILING_TPOT], explicitSpans),
    ),
    (reading) => `Time per output token: ${reading.text}.`,
  );
  place('e2eLatency', durations(question, [AFTER_E2E, BEFORE_E2E]), (reading) =>
    `End-to-end latency: ${reading.text}.`,
  );
  place('prefix', counts(countsFrom, [BEFORE_PREFIX]), (reading) =>
    `Prefix length: ${reading.text} tokens.`,
  );
  // The three tools store the same limit under different names, so the value
  // goes to the field this page actually has.
  place(page.contextField, counts(countsFrom, [BEFORE_CONTEXT, AFTER_CONTEXT]), (reading) =>
    `Context length: ${reading.text} tokens.`,
  );

  if (fromQuestion.size === 0) return null;

  // A field the destination page has no room for would be dropped without a
  // trace by the contract, so it is reported here instead.
  for (const key of fromQuestion) {
    if (page.fields.has(key)) continue;
    report({ key, value: rawValue.get(key) ?? '', reason: 'unsupported' });
  }

  // The goal follows the values the contract actually kept, not the ones the
  // question offered: a rejected concurrency must not produce a capacity view.
  const goal = impliedGoal(page.build(candidates).values);
  if (goal) candidates.goal = goal;

  // A goal-specific input the selected goal does not read is not carried: the
  // page would ignore it, and the link would claim to set something it does not.
  if (goal && page.fields.has('goal')) {
    const used = new Set(GOAL_INPUTS[goal]);
    for (const key of GOAL_SPECIFIC_INPUTS) {
      if (used.has(key) || candidates[key] === undefined) continue;
      delete candidates[key];
      report({ key, value: rawValue.get(key) ?? '', reason: 'unused' });
    }
  }

  const built = page.build(candidates);
  for (const entry of built.ignored) {
    report({ key: entry.key, value: String(entry.value), reason: 'rejected' });
  }

  const fields = Object.keys(built.values).filter((key) => !PROVENANCE_KEYS.includes(key));

  // A value that did not reach the link is not silently dropped: the link says
  // what it left out and why, so the opened view is not mistaken for the one
  // the answer described.
  const reasonText: Record<IgnoredValue['reason'], string> = {
    rejected: 'the page does not accept it',
    ambiguous: 'the question names more than one',
    unsupported: `${page.name} has no such field`,
    unused: `the ${goal} goal does not use it`,
  };
  const dropped = ignored.map(
    (entry) =>
      `${FIELD_LABELS[entry.key] ?? entry.key} "${entry.value}" (${reasonText[entry.reason]})`,
  );

  // Every value the question named was refused. The page still opens, and the
  // report is still shown: discarding it with the proposal would leave the
  // question looking answered when nothing of it was carried.
  if (fields.length === 0) {
    if (dropped.length === 0) return null;
    return {
      link: {
        kind: 'proposal',
        label: `Open ${page.name}`,
        href: built.href,
        description: `Left out ${dropped.join(', ')}.`,
      },
      fields,
      ignored,
      evidence: [],
    };
  }

  const labels = fields.map((field) => FIELD_LABELS[field] ?? field);
  const quoted = fields
    .map((field) => evidence.get(field))
    .filter((line): line is string => Boolean(line));
  const description =
    dropped.length === 0
      ? `Sets ${labels.join(', ')}.`
      : `Sets ${labels.join(', ')}. Left out ${dropped.join(', ')}.`;

  return {
    link: {
      kind: 'proposal',
      label: `Open ${page.name} with these values`,
      href: built.href,
      description,
    },
    fields,
    ignored,
    evidence: quoted,
  };
}

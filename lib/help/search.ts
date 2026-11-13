import { GLOSSARY_ENTRIES } from './glossary';
import { HELP_AFFORDANCES } from './affordances';

/**
 * Help search.
 *
 * The corpus is documentation sections, glossary entries, and page controls.
 * It deliberately excludes the model and GPU catalog: live facts belong to the
 * API, and copying them into a search index would let them go stale.
 *
 * The index is built at runtime from three local sources plus the
 * build-time-generated docs index (`public/help-index.json`).
 */

export interface DocsSearchEntry {
  id: string;
  title: string;
  section: string;
  body: string;
  url: string;
}

export type HelpResultKind = 'page' | 'glossary' | 'docs';

export interface HelpResult {
  id: string;
  kind: HelpResultKind;
  title: string;
  /** Where the result lives: a page name, `Glossary`, or a doc section. */
  context: string;
  body: string;
  /** Present for anything that can be opened. Glossary results have no page. */
  url?: string;
  /** Documentation that explains this control, when one is mapped to it. */
  docUrl?: string;
  score: number;
}

interface SourceEntry {
  id: string;
  source: 'docs' | 'glossary' | 'affordance';
  title: string;
  context: string;
  body: string;
  url?: string;
  docUrl?: string;
  /** Route an affordance lives on. */
  route?: string;
  keywords?: string[];
}

interface IndexedEntry extends SourceEntry {
  titleTokens: string[];
  contextTokens: string[];
  keywordTokens: string[];
  bodyTokens: string[];
  titleNormalised: string;
  contextNormalised: string;
  bodyNormalised: string;
}

export interface HelpIndex {
  entries: IndexedEntry[];
}

/** Human-readable names for routes, used to label cross-page results. */
export const ROUTE_LABELS: Record<string, string> = {
  '/': 'Home',
  '/predict': 'Predict performance',
  '/recommend': 'Recommend sizing',
  '/kv-cache': 'KV cache calculator',
  '/gpu-explorer': 'GPU explorer',
  '/compare': 'Compare',
  '/validation/compare': 'Benchmark comparison',
  '/hybrid-savings': 'Hybrid savings',
  '/cluster-cost': 'Cluster cost',
  '/routing': 'Routing economics',
  '/settings': 'Settings',
  '/sources': 'Sources',
  '/docs': 'Documentation',
};

export function routeLabel(route: string): string {
  return ROUTE_LABELS[route] ?? route;
}

/**
 * Words that carry no signal on their own. Dropping them keeps "how do I
 * change the concurrency" from ranking a page that merely says "the".
 */
const STOPWORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can', 'do', 'does', 'for',
  'from', 'how', 'i', 'if', 'in', 'is', 'it', 'its', 'me', 'my', 'of', 'on',
  'or', 'that', 'the', 'then', 'there', 'this', 'to', 'was', 'what', 'when',
  'where', 'which', 'why', 'with', 'you', 'your',
]);

const TITLE_EXACT = 12;
const TITLE_PREFIX = 7;
const KEYWORD_EXACT = 8;
const KEYWORD_PREFIX = 5;
const CONTEXT_EXACT = 4;
const CONTEXT_PREFIX = 2;
const BODY_EXACT = 2;
const BODY_PREFIX = 1;

const MIN_PREFIX_LENGTH = 3;

export function normaliseText(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9_+#.-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function tokenise(value: string): string[] {
  const normalised = normaliseText(value);
  if (!normalised) return [];
  return normalised.split(' ').filter((token) => token && !STOPWORDS.has(token));
}

/**
 * Validate the build-time docs index. A malformed or partial file degrades to
 * "no docs results" instead of breaking the help popover.
 */
export function parseHelpIndex(raw: unknown): DocsSearchEntry[] {
  if (!raw || typeof raw !== 'object') return [];
  const entries = (raw as { entries?: unknown }).entries;
  if (!Array.isArray(entries)) return [];

  const parsed: DocsSearchEntry[] = [];
  for (const entry of entries) {
    if (!entry || typeof entry !== 'object') continue;
    const record = entry as Record<string, unknown>;
    if (
      typeof record.id !== 'string' ||
      typeof record.title !== 'string' ||
      typeof record.url !== 'string'
    ) {
      continue;
    }
    parsed.push({
      id: record.id,
      title: record.title,
      section: typeof record.section === 'string' ? record.section : '',
      body: typeof record.body === 'string' ? record.body : '',
      url: record.url,
    });
  }
  return parsed;
}

function sourceEntries(docs: DocsSearchEntry[]): SourceEntry[] {
  const entries: SourceEntry[] = [];

  for (const doc of docs) {
    entries.push({
      id: doc.id,
      source: 'docs',
      title: doc.title,
      context: doc.section,
      body: doc.body,
      url: doc.url,
    });
  }

  for (const glossary of GLOSSARY_ENTRIES) {
    entries.push({
      id: `glossary/${glossary.id}`,
      source: 'glossary',
      title: glossary.title,
      context: 'Glossary',
      body: glossary.body,
      keywords: glossary.tags,
    });
  }

  for (const affordance of HELP_AFFORDANCES) {
    entries.push({
      id: `control/${affordance.id}`,
      source: 'affordance',
      title: affordance.label,
      context: routeLabel(affordance.route),
      // A control entry points at the control, not the docs. Documentation
      // about the same topic is a separate entry and ranks on its own.
      body: '',
      url: affordance.route,
      docUrl: affordance.doc
        ? `/docs/${affordance.doc}${affordance.anchor ? `#${affordance.anchor}` : ''}`
        : undefined,
      route: affordance.route,
      keywords: affordance.keywords,
    });
  }

  return entries;
}

function indexEntry(entry: SourceEntry): IndexedEntry {
  return {
    ...entry,
    titleTokens: tokenise(entry.title),
    contextTokens: tokenise(entry.context),
    keywordTokens: tokenise((entry.keywords ?? []).join(' ')),
    bodyTokens: tokenise(entry.body),
    titleNormalised: normaliseText(entry.title),
    contextNormalised: normaliseText(entry.context),
    bodyNormalised: normaliseText(entry.body),
  };
}

export function buildHelpIndex(docs: DocsSearchEntry[] = []): HelpIndex {
  return { entries: sourceEntries(docs).map(indexEntry) };
}

/** Best weight for one query token against one field. */
function fieldScore(
  token: string,
  tokens: string[],
  exact: number,
  prefix: number,
): number {
  if (tokens.length === 0) return 0;
  if (tokens.includes(token)) return exact;
  if (token.length < MIN_PREFIX_LENGTH) return 0;

  const matchesPrefix = tokens.some(
    (candidate) =>
      candidate.startsWith(token) ||
      (token.length >= MIN_PREFIX_LENGTH + 1 && token.startsWith(candidate)),
  );
  return matchesPrefix ? prefix : 0;
}

function scoreEntry(entry: IndexedEntry, queryTokens: string[], phrase: string): number {
  let score = 0;
  let matched = 0;

  for (const token of queryTokens) {
    const tokenScore = Math.max(
      fieldScore(token, entry.titleTokens, TITLE_EXACT, TITLE_PREFIX),
      fieldScore(token, entry.keywordTokens, KEYWORD_EXACT, KEYWORD_PREFIX),
      fieldScore(token, entry.contextTokens, CONTEXT_EXACT, CONTEXT_PREFIX),
      fieldScore(token, entry.bodyTokens, BODY_EXACT, BODY_PREFIX),
    );
    if (tokenScore > 0) {
      matched += 1;
      score += tokenScore;
    }
  }

  if (matched === 0) return 0;

  // Requiring every token would fail on natural language; rewarding coverage
  // keeps a full match ahead of a page that happens to contain one word.
  score *= (matched / queryTokens.length) ** 1.5;

  if (phrase.length >= 4) {
    if (entry.titleNormalised.includes(phrase)) score += 15;
    else if (entry.contextNormalised.includes(phrase)) score += 6;
    else if (entry.bodyNormalised.includes(phrase)) score += 3;
  }

  return score;
}

const KIND_ORDER: Record<HelpResultKind, number> = {
  page: 0,
  glossary: 1,
  docs: 2,
};

function resultKind(entry: IndexedEntry, currentRoute?: string): HelpResultKind {
  if (entry.source === 'glossary') return 'glossary';
  if (entry.source === 'affordance' && currentRoute && entry.route === currentRoute) {
    return 'page';
  }
  return 'docs';
}

/**
 * Rank help entries for a query. Returns an empty list for an empty query so
 * the caller can show suggestions instead of an arbitrary dump.
 */
export function searchHelp(
  query: string,
  index: HelpIndex,
  currentRoute?: string,
  limit = 12,
): HelpResult[] {
  const queryTokens = tokenise(query);
  if (queryTokens.length === 0) return [];
  const phrase = normaliseText(query);

  const results: HelpResult[] = [];
  for (const entry of index.entries) {
    const score = scoreEntry(entry, queryTokens, phrase);
    if (score <= 0) continue;
    results.push({
      id: entry.id,
      kind: resultKind(entry, currentRoute),
      title: entry.title,
      context: entry.context,
      body: entry.body,
      url: entry.url,
      docUrl: entry.docUrl,
      score,
    });
  }

  results.sort((left, right) => {
    if (right.score !== left.score) return right.score - left.score;
    if (KIND_ORDER[left.kind] !== KIND_ORDER[right.kind]) {
      return KIND_ORDER[left.kind] - KIND_ORDER[right.kind];
    }
    return left.title.localeCompare(right.title);
  });

  return results.slice(0, limit);
}

/** Controls on the current page, for the popover's empty state. */
export function suggestionsForRoute(
  index: HelpIndex,
  currentRoute: string,
  limit = 5,
): HelpResult[] {
  return index.entries
    .filter((entry) => entry.source === 'affordance' && entry.route === currentRoute)
    .slice(0, limit)
    .map((entry) => ({
      id: entry.id,
      kind: 'page',
      title: entry.title,
      context: entry.context,
      body: entry.body,
      url: entry.url,
      docUrl: entry.docUrl,
      score: 0,
    }));
}

import { tokenise } from '@/lib/help/search';
import type { RetrievedSection } from './types';

/**
 * BM25 retrieval over the build-time help index.
 *
 * There is deliberately no embedding model: the deployment has one L4 and the
 * corpus is a few hundred short sections, where BM25 is both sufficient and
 * explainable. The index never contains catalog facts, so retrieved text cannot
 * go stale the way a copied GPU table would.
 */

export interface RetrievalDoc {
  id: string;
  title: string;
  section: string;
  body: string;
  url: string;
}

interface IndexedDoc extends RetrievalDoc {
  tokens: string[];
  titleTokens: string[];
  length: number;
}

const K1 = 1.2;
const B = 0.75;
/** A title match is a stronger signal than a mention in the body. */
const TITLE_BOOST = 2.5;

function indexDoc(doc: RetrievalDoc): IndexedDoc {
  const tokens = tokenise(`${doc.title} ${doc.section} ${doc.body}`);
  return {
    ...doc,
    tokens,
    titleTokens: tokenise(doc.title),
    length: tokens.length,
  };
}

function termFrequency(tokens: string[], term: string): number {
  let count = 0;
  for (const token of tokens) if (token === term) count += 1;
  return count;
}

/**
 * Rank sections for a question. Returns an empty list for a question with no
 * usable terms so the caller can answer with links rather than an arbitrary
 * dump of the corpus.
 */
export function retrieve(
  question: string,
  docs: RetrievalDoc[],
  limit = 6,
): RetrievedSection[] {
  const queryTerms = tokenise(question);
  if (queryTerms.length === 0 || docs.length === 0) return [];

  const indexed = docs.map(indexDoc);
  const averageLength =
    indexed.reduce((total, doc) => total + doc.length, 0) / indexed.length || 1;

  const uniqueTerms = Array.from(new Set(queryTerms));
  const documentFrequency = new Map<string, number>();
  for (const term of uniqueTerms) {
    documentFrequency.set(
      term,
      indexed.reduce((count, doc) => count + (termFrequency(doc.tokens, term) > 0 ? 1 : 0), 0),
    );
  }

  const scored = indexed.map((doc) => {
    let score = 0;
    for (const term of uniqueTerms) {
      const tf = termFrequency(doc.tokens, term);
      if (tf === 0) continue;
      const df = documentFrequency.get(term) ?? 0;
      const idf = Math.log(1 + (indexed.length - df + 0.5) / (df + 0.5));
      const denominator = tf + K1 * (1 - B + (B * doc.length) / averageLength);
      score += idf * ((tf * (K1 + 1)) / denominator);
      if (doc.titleTokens.includes(term)) score += TITLE_BOOST;
    }
    return { doc, score };
  });

  return scored
    .filter((entry) => entry.score > 0)
    .sort((left, right) => right.score - left.score)
    .slice(0, limit)
    .map(({ doc, score }) => ({
      id: doc.id,
      title: doc.title,
      section: doc.section,
      url: doc.url,
      body: doc.body,
      score,
    }));
}

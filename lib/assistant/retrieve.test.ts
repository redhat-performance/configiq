import { describe, expect, it } from 'vitest';
import { retrieve, type RetrievalDoc } from './retrieve';

const DOCS: RetrievalDoc[] = [
  {
    id: 'docs/settings#simulator',
    title: 'Simulator',
    section: 'Settings and simulators',
    body: 'A simulator is the implementation that produces an estimate. A backend is the serving runtime being modelled.',
    url: '/docs/settings#simulator',
  },
  {
    id: 'docs/costs#planning-assumptions',
    title: 'Planning assumptions',
    section: 'Costs',
    body: 'Hardware life, electricity price, and PUE change the purchased-hardware comparison. Edit them on the Sources page.',
    url: '/docs/costs#planning-assumptions',
  },
  {
    id: 'docs/pareto#frontier',
    title: 'Pareto frontier',
    section: 'Pareto',
    body: 'The frontier lists the trade-offs between GPU count and latency instead of one recommendation.',
    url: '/docs/pareto#frontier',
  },
];

describe('assistant retrieval', () => {
  it('ranks the section that defines the term first', () => {
    const results = retrieve('what is a simulator', DOCS);
    expect(results[0]?.id).toBe('docs/settings#simulator');
  });

  it('returns sections with their anchor so citations can link', () => {
    const results = retrieve('PUE electricity', DOCS);
    expect(results[0]?.url).toBe('/docs/costs#planning-assumptions');
  });

  it('returns nothing for a question made only of stopwords', () => {
    expect(retrieve('how do I do that', DOCS)).toEqual([]);
  });

  it('respects the limit', () => {
    expect(retrieve('simulator backend costs frontier', DOCS, 2)).toHaveLength(2);
  });

  it('returns nothing rather than guessing when no section matches', () => {
    expect(retrieve('kubernetes ingress controller', DOCS)).toEqual([]);
  });

  it('handles an empty corpus', () => {
    expect(retrieve('simulator', [])).toEqual([]);
  });
});

import { describe, expect, it } from 'vitest';
import {
  buildHelpIndex,
  parseHelpIndex,
  routeLabel,
  searchHelp,
  suggestionsForRoute,
  tokenise,
  type DocsSearchEntry,
} from './search';

const docs: DocsSearchEntry[] = [
  {
    id: 'docs/settings#simulator',
    title: 'Simulator',
    section: 'Settings and simulators',
    body: 'A simulator is the implementation that produces an estimate.',
    url: '/docs/settings#simulator',
  },
  {
    id: 'docs/costs#planning-assumptions',
    title: 'Planning assumptions',
    section: 'Costs',
    body: 'Hardware life, electricity price, and PUE change the comparison.',
    url: '/docs/costs#planning-assumptions',
  },
  {
    id: 'docs/overview#cost-assumptions',
    title: 'Overview',
    section: 'Overview',
    body: 'Cost assumptions are edited on the Sources page.',
    url: '/docs/overview#cost-assumptions',
  },
];

describe('help search', () => {
  it('drops malformed build output instead of failing', () => {
    expect(parseHelpIndex(null)).toEqual([]);
    expect(parseHelpIndex({ entries: 'nope' })).toEqual([]);
    expect(
      parseHelpIndex({
        entries: [
          { id: 'a', title: 'Ok', url: '/docs/a', section: 'A', body: 'body' },
          { id: 'b', title: 'No url' },
          'garbage',
        ],
      }),
    ).toEqual([{ id: 'a', title: 'Ok', section: 'A', body: 'body', url: '/docs/a' }]);
  });

  it('drops an index entry whose URL is not local documentation', () => {
    const parsed = parseHelpIndex({
      entries: [
        { id: 'ok', title: 'Ok', url: '/docs/settings#sharing-a-view', section: 'Settings', body: '' },
        { id: 'js', title: 'Script', url: 'javascript:alert(1)', section: '', body: '' },
        { id: 'offsite', title: 'Other', url: 'https://evil.example/docs/x', section: '', body: '' },
        { id: 'proto', title: 'Proto-relative', url: '//evil.example/docs/x', section: '', body: '' },
        { id: 'root', title: 'Root', url: '/settings', section: '', body: '' },
      ],
    });

    expect(parsed.map((entry) => entry.id)).toEqual(['ok']);
  });

  it('drops stopwords so natural-language queries still match', () => {
    expect(tokenise('How do I change the concurrency?')).toEqual(['change', 'concurrency']);
  });

  it('finds a documentation section by title', () => {
    const index = buildHelpIndex(docs);
    const results = searchHelp('simulator', index);
    const doc = results.find((result) => result.id === 'docs/settings#simulator');

    expect(doc).toBeDefined();
    expect(doc?.url).toBe('/docs/settings#simulator');
    expect(doc?.kind).toBe('docs');
  });

  it('ranks a title match above a passing mention in a body', () => {
    const index = buildHelpIndex(docs);
    const results = searchHelp('assumptions', index);
    const titleMatch = results.find((result) => result.id === 'docs/costs#planning-assumptions');
    const bodyOnly = results.find((result) => result.id === 'docs/overview#cost-assumptions');

    expect(titleMatch).toBeDefined();
    expect(bodyOnly).toBeDefined();
    expect(titleMatch!.score).toBeGreaterThan(bodyOnly!.score);
  });

  it('returns glossary definitions with no page to open', () => {
    const index = buildHelpIndex(docs);
    const glossary = searchHelp('KV cache', index).find((result) => result.kind === 'glossary');

    expect(glossary?.title).toBe('KV cache');
    expect(glossary?.url).toBeUndefined();
    expect(glossary?.body).toContain('key/value');
  });

  it('marks a control on the current route as a page result', () => {
    const index = buildHelpIndex(docs);
    const onPage = searchHelp('concurrency', index, '/predict');
    expect(onPage.some((result) => result.kind === 'page')).toBe(true);

    const elsewhere = searchHelp('concurrency', index, '/settings');
    expect(elsewhere.some((result) => result.kind === 'page')).toBe(false);
  });

  it('attaches the mapped documentation to a control result', () => {
    const index = buildHelpIndex(docs);
    const result = searchHelp('concurrency presets', index)
      .find((entry) => entry.id === 'control/predict-concurrency');

    expect(result).toBeDefined();
    expect(result?.url).toBe('/predict');
    expect(result?.docUrl).toBe('/docs/user-scenarios#sizing-multiple-workloads');
  });

  it('leaves docUrl off controls with no mapped documentation', () => {
    const index = buildHelpIndex(docs);
    const result = suggestionsForRoute(index, '/predict')
      .find((entry) => entry.id === 'control/predict-calculate');

    expect(result).toBeDefined();
    expect(result?.docUrl).toBeUndefined();
  });

  it('returns nothing for an empty query', () => {
    const index = buildHelpIndex(docs);
    expect(searchHelp('   ', index)).toEqual([]);
  });

  it('suggests only controls for the given route', () => {
    const index = buildHelpIndex(docs);
    const suggestions = suggestionsForRoute(index, '/settings');

    expect(suggestions.length).toBeGreaterThan(0);
    expect(suggestions.every((result) => result.kind === 'page')).toBe(true);
    expect(suggestions.map((result) => result.title)).toContain('Select a simulator');
  });

  it('labels a route for display without inventing a name', () => {
    expect(routeLabel('/predict')).toBe('Predict performance');
    expect(routeLabel('/unknown-route')).toBe('/unknown-route');
  });
});

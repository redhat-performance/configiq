import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  readPublishedScorecard,
  resetPublishedScorecardCache,
  SCORECARD_DATASET_REPO,
  scorecardArtifactUrls,
} from './published';
import { point, summary } from './__fixtures__/points';

const pointsDocument = {
  schemaVersion: 1,
  channel: 'dev',
  generatedAt: '2026-11-20T10:20:00.000Z',
  points: [point()],
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function stubFetch(handler: (url: string) => Response | Promise<Response>) {
  const mock = vi.fn(async (input: RequestInfo | URL) => handler(String(input)));
  vi.stubGlobal('fetch', mock);
  return mock;
}

beforeEach(() => {
  resetPublishedScorecardCache();
});

afterEach(() => {
  vi.unstubAllGlobals();
  resetPublishedScorecardCache();
});

describe('scorecardArtifactUrls', () => {
  it('puts each channel in its own directory', () => {
    const urls = scorecardArtifactUrls('prod', 'org/scorecard');
    expect(urls.summary).toBe(
      'https://huggingface.co/datasets/org/scorecard/resolve/main/prod/scorecard.json',
    );
    expect(urls.points).toBe(
      'https://huggingface.co/datasets/org/scorecard/resolve/main/prod/points.json',
    );
  });

  it('defaults to the public artifact repository', () => {
    expect(SCORECARD_DATASET_REPO).toBe('redhat-performance/configiq-simulator-scorecard');
  });
});

describe('readPublishedScorecard', () => {
  it('reads the summary without fetching point detail', async () => {
    const fetchMock = stubFetch(url => (
      url.endsWith('/scorecard.json') ? jsonResponse(summary()) : jsonResponse({}, 404)
    ));

    const result = await readPublishedScorecard({ channel: 'dev' });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.summary.schemaVersion).toBe(1);
    expect(result.value.points).toEqual([]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('reads and validates point detail when asked', async () => {
    stubFetch(url => (
      url.endsWith('/points.json') ? jsonResponse(pointsDocument) : jsonResponse(summary())
    ));

    const result = await readPublishedScorecard({ channel: 'dev', includePoints: true });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.points).toHaveLength(1);
    expect(result.value.points[0].identity).toBe('point-1');
  });

  it('reports a missing snapshot rather than an empty one', async () => {
    stubFetch(() => jsonResponse({ error: 'not found' }, 404));
    const result = await readPublishedScorecard({ channel: 'prod' });
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error.kind).toBe('missing');
  });

  it('reports an unreachable host separately from a missing snapshot', async () => {
    stubFetch(() => jsonResponse({ error: 'busy' }, 503));
    const result = await readPublishedScorecard({ channel: 'dev' });
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error.kind).toBe('unavailable');
  });

  it('reports a transport failure as unavailable', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new Error('getaddrinfo ENOTFOUND');
    }));
    const result = await readPublishedScorecard({ channel: 'dev' });
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error.kind).toBe('unavailable');
    expect(result.error.message).toContain('ENOTFOUND');
  });

  it('rejects a body that is not JSON', async () => {
    stubFetch(() => new Response('<!doctype html>', { status: 200 }));
    const result = await readPublishedScorecard({ channel: 'dev' });
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error.kind).toBe('invalid');
  });

  it('fails closed on an artifact that does not match the contract', async () => {
    stubFetch(() => jsonResponse({ ...summary(), schemaVersion: 99 }));
    const result = await readPublishedScorecard({ channel: 'dev' });
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error.kind).toBe('invalid');
    expect(result.error.message).toContain('contract');
  });

  it('fails closed on point detail that does not match the contract', async () => {
    stubFetch(url => (
      url.endsWith('/points.json')
        ? jsonResponse({ ...pointsDocument, points: [{ identity: '' }] })
        : jsonResponse(summary())
    ));
    const result = await readPublishedScorecard({ channel: 'dev', includePoints: true });
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error.kind).toBe('invalid');
  });

  it('reports a missing point document when the summary exists', async () => {
    stubFetch(url => (
      url.endsWith('/points.json') ? jsonResponse({}, 404) : jsonResponse(summary())
    ));
    const result = await readPublishedScorecard({ channel: 'dev', includePoints: true });
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error.kind).toBe('missing');
  });

  it('caches a successful read until the cache expires or a refresh is requested', async () => {
    const fetchMock = stubFetch(url => (
      url.endsWith('/scorecard.json') ? jsonResponse(summary()) : jsonResponse({}, 404)
    ));

    await readPublishedScorecard({ channel: 'dev' });
    await readPublishedScorecard({ channel: 'dev' });
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await readPublishedScorecard({ channel: 'dev', refresh: true });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('does not cache a failure', async () => {
    const fetchMock = stubFetch(() => jsonResponse({}, 503));
    await readPublishedScorecard({ channel: 'dev' });
    await readPublishedScorecard({ channel: 'dev' });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('shares one read between concurrent callers', async () => {
    let release: (response: Response) => void = () => {};
    const gate = new Promise<Response>(resolve => {
      release = resolve;
    });
    const fetchMock = vi.fn(() => gate);
    vi.stubGlobal('fetch', fetchMock);

    const first = readPublishedScorecard({ channel: 'dev' });
    const second = readPublishedScorecard({ channel: 'dev' });
    release(jsonResponse(summary()));

    const [left, right] = await Promise.all([first, second]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(left.ok && right.ok).toBe(true);
    if (!left.ok || !right.ok) return;
    expect(left.value.summary.generatedAt).toBe(right.value.summary.generatedAt);
  });
});

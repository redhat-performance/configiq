import {
  scorecardPointsSchema,
  scorecardSummarySchema,
  type ScorecardChannel,
  type ScorecardPoint,
  type ScorecardSummary,
} from './types';

/**
 * Reader for the published scorecard artifact.
 *
 * The sweep is far too expensive to run per request, and a fully prerendered
 * scorecard would need private dataset credentials at image-build time. So the
 * build job publishes a derived, sanitized artifact to a public Hugging Face
 * dataset instead, and the page reads that. Public means no token, which means
 * static rendering and ISR keep working.
 */

/** Repository the build job publishes to. Overridable for a fork or a mirror. */
export const SCORECARD_DATASET_REPO = process.env.SCORECARD_DATASET_REPO
  ?? 'redhat-performance/configiq-simulator-scorecard';

/**
 * Base URL for the artifact directory, when a maintainer wants to preview a
 * snapshot that is not published yet (a local static server over `artifacts/`).
 * Server-side only: the browser never learns this value.
 */
const ARTIFACT_BASE_URL = process.env.SCORECARD_ARTIFACT_BASE_URL ?? null;

const CACHE_TTL_MS = 5 * 60 * 1000;

export interface PublishedScorecard {
  summary: ScorecardSummary;
  points: ScorecardPoint[];
  /** When this process read the artifact, not when the sweep ran. */
  fetchedAt: string;
}

export type ScorecardReadError =
  /** Nobody has published a snapshot for this channel yet. */
  | { kind: 'missing'; message: string }
  /** Something is published but it does not match the contract. */
  | { kind: 'invalid'; message: string }
  /** The artifact host could not be reached. */
  | { kind: 'unavailable'; message: string };

export type ScorecardReadResult =
  | { ok: true; value: PublishedScorecard }
  | { ok: false; error: ScorecardReadError };

export interface ReadOptions {
  channel: ScorecardChannel;
  /** Skip the per-point document when only the aggregates are needed. */
  includePoints?: boolean;
  /** Bypass the short-lived process cache. */
  refresh?: boolean;
}

interface CacheEntry {
  value: PublishedScorecard;
  expiresAt: number;
}

const cache = new Map<string, CacheEntry>();
const inFlight = new Map<string, Promise<ScorecardReadResult>>();

export function scorecardArtifactUrls(
  channel: ScorecardChannel,
  repo: string = SCORECARD_DATASET_REPO,
): { summary: string; points: string } {
  const base = ARTIFACT_BASE_URL
    ? `${ARTIFACT_BASE_URL.replace(/\/+$/, '')}/${channel}`
    : `https://huggingface.co/datasets/${repo}/resolve/main/${channel}`;
  return { summary: `${base}/scorecard.json`, points: `${base}/points.json` };
}

async function fetchJson(url: string): Promise<{ ok: true; data: unknown } | { ok: false; error: ScorecardReadError }> {
  let response: Response;
  try {
    response = await fetch(url, {
      headers: { Accept: 'application/json' },
      cache: 'no-store',
      signal: AbortSignal.timeout(20_000),
    });
  } catch (error: unknown) {
    return {
      ok: false,
      error: {
        kind: 'unavailable',
        message: error instanceof Error ? error.message : 'The scorecard host could not be reached.',
      },
    };
  }

  if (response.status === 404) {
    return { ok: false, error: { kind: 'missing', message: `No scorecard snapshot is published at ${url}.` } };
  }
  if (!response.ok) {
    return {
      ok: false,
      error: { kind: 'unavailable', message: `The scorecard host returned HTTP ${response.status}.` },
    };
  }
  try {
    return { ok: true, data: await response.json() };
  } catch {
    return { ok: false, error: { kind: 'invalid', message: 'The published scorecard is not valid JSON.' } };
  }
}

function parseSummary(data: unknown): ScorecardSummary {
  const parsed = scorecardSummarySchema.safeParse(data);
  if (!parsed.success) {
    throw new Error(`The published scorecard does not match the contract: ${parsed.error.issues[0]?.message ?? 'invalid'}`);
  }
  return parsed.data;
}

function parsePoints(data: unknown): ScorecardPoint[] {
  const parsed = scorecardPointsSchema.safeParse(data);
  if (!parsed.success) {
    throw new Error(`The published point detail does not match the contract: ${parsed.error.issues[0]?.message ?? 'invalid'}`);
  }
  return parsed.data.points;
}

async function read(options: ReadOptions): Promise<ScorecardReadResult> {
  const urls = scorecardArtifactUrls(options.channel);
  const summaryResponse = await fetchJson(urls.summary);
  if (!summaryResponse.ok) return { ok: false, error: summaryResponse.error };

  let summary: ScorecardSummary;
  try {
    summary = parseSummary(summaryResponse.data);
  } catch (error: unknown) {
    return {
      ok: false,
      error: { kind: 'invalid', message: error instanceof Error ? error.message : 'The scorecard is invalid.' },
    };
  }

  let points: ScorecardPoint[] = [];
  if (options.includePoints) {
    const pointsResponse = await fetchJson(urls.points);
    if (!pointsResponse.ok) return { ok: false, error: pointsResponse.error };
    try {
      points = parsePoints(pointsResponse.data);
    } catch (error: unknown) {
      return {
        ok: false,
        error: { kind: 'invalid', message: error instanceof Error ? error.message : 'The point detail is invalid.' },
      };
    }
  }

  return {
    ok: true,
    value: { summary, points, fetchedAt: new Date().toISOString() },
  };
}

/**
 * Read the published scorecard for one channel.
 *
 * Concurrent callers share one read, and a successful read is reused for a few
 * minutes. Failures are not cached: a transient host error should not keep the
 * page broken for the next five minutes.
 */
export async function readPublishedScorecard(options: ReadOptions): Promise<ScorecardReadResult> {
  const key = `${options.channel}:${options.includePoints ? 'full' : 'summary'}`;
  const now = Date.now();
  if (!options.refresh) {
    const hit = cache.get(key);
    if (hit && hit.expiresAt > now) return { ok: true, value: hit.value };
  }

  const pending = inFlight.get(key);
  if (pending) return pending;

  const request = read(options)
    .then(result => {
      if (result.ok) cache.set(key, { value: result.value, expiresAt: Date.now() + CACHE_TTL_MS });
      return result;
    })
    .finally(() => {
      inFlight.delete(key);
    });

  inFlight.set(key, request);
  return request;
}

/** Clear the process cache. Tests use this; nothing in the app should. */
export function resetPublishedScorecardCache(): void {
  cache.clear();
  inFlight.clear();
}

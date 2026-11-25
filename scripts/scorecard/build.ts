/**
 * Build the simulator scorecard artifact.
 *
 * Usage:
 *   npm run scorecard:build -- --models Qwen/Qwen3-32B --limit 20
 *   npm run scorecard:build -- --sources redhat --concurrency 2
 *
 * Writes `<out>/scorecard-<channel>.json`, `<out>/scorecard-points-<channel>.json`,
 * and the incremental prediction cache. Nothing is uploaded: publishing is a
 * separate, explicitly approved step.
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { buildScorecard, type PredictionCache } from '../../lib/scorecard/build';
import {
  SCORECARD_CHANNELS,
  SCORECARD_SOURCES,
  scorecardPointsSchema,
  scorecardSummarySchema,
  type ScorecardChannel,
  type ScorecardPoints,
  type ScorecardSource,
} from '../../lib/scorecard/types';

function option(name: string): string | undefined {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? undefined : process.argv[index + 1];
}

function flag(name: string): boolean {
  return process.argv.includes(`--${name}`);
}

function integerOption(name: string): number | undefined {
  const raw = option(name);
  if (raw === undefined) return undefined;
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`--${name} must be a positive integer, got ${raw}`);
  }
  return parsed;
}

function listOption<T extends string>(name: string, allowed: readonly T[]): T[] | undefined {
  const raw = option(name);
  if (raw === undefined) return undefined;
  const values = raw.split(',').map(value => value.trim()).filter(Boolean);
  const invalid = values.filter(value => !allowed.includes(value as T));
  if (invalid.length > 0) {
    throw new Error(`--${name} accepts ${allowed.join(', ')}, got ${invalid.join(', ')}`);
  }
  return values as T[];
}

function seconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}

async function readCache(path: string): Promise<PredictionCache> {
  try {
    const parsed: unknown = JSON.parse(await readFile(path, 'utf8'));
    if (parsed && typeof parsed === 'object' && 'entries' in parsed) {
      const entries = (parsed as PredictionCache).entries;
      if (entries && typeof entries === 'object') return { entries };
    }
  } catch {
    // A missing or unreadable cache is a cold start, not a failure.
  }
  return { entries: {} };
}

async function main(): Promise<void> {
  const gatewayUrl = option('gateway')
    ?? process.env.AISIMULATORS_GATEWAY_URL
    ?? 'https://aisimulators.configiq.dev';
  const channel = (option('channel') ?? 'dev') as ScorecardChannel;
  if (!SCORECARD_CHANNELS.includes(channel)) {
    throw new Error(`--channel must be one of ${SCORECARD_CHANNELS.join(', ')}`);
  }

  const outDir = resolve(option('out') ?? join('artifacts', 'scorecard'));
  await mkdir(outDir, { recursive: true });
  const cachePath = resolve(option('cache') ?? join(outDir, 'scorecard-cache.json'));
  const cache = flag('no-cache') ? { entries: {} } : await readCache(cachePath);

  const startedAt = Date.now();
  const result = await buildScorecard({
    gatewayUrl,
    channel,
    timeoutSeconds: integerOption('timeout'),
    concurrency: integerOption('concurrency'),
    sources: listOption<ScorecardSource>('sources', SCORECARD_SOURCES),
    models: option('models')?.split(',').map(value => value.trim()).filter(Boolean),
    limit: integerOption('limit'),
    cache,
    log: message => console.error(`[scorecard] ${message}`),
  });
  const elapsed = Date.now() - startedAt;

  const summary = scorecardSummarySchema.parse(result.summary);
  const points: ScorecardPoints = scorecardPointsSchema.parse({
    schemaVersion: summary.schemaVersion,
    channel: summary.channel,
    generatedAt: summary.generatedAt,
    points: result.points,
  });

  const summaryPath = join(outDir, `scorecard-${channel}.json`);
  const pointsPath = join(outDir, `scorecard-points-${channel}.json`);
  await writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
  await writeFile(pointsPath, `${JSON.stringify(points)}\n`);
  if (!flag('no-cache')) {
    await writeFile(cachePath, `${JSON.stringify(result.cache)}\n`);
  }

  const cacheEntries = Object.keys(result.cache.entries).length;
  console.error('');
  console.error('[scorecard] done');
  console.error(`  wall clock          ${seconds(elapsed)}`);
  console.error(`  simulators          ${summary.simulators.map(s => `${s.label} ${s.version ?? '?'}`).join(', ')}`);
  console.error(`  measured rows       ${summary.datasets.map(d => `${d.source} ${d.rowCount}`).join(', ')}`);
  console.error(`  points attempted    ${summary.attemptedPoints}`);
  console.error(`  points answered     ${summary.answeredPoints}`);
  console.error(`  cache               ${result.cacheHits} hit / ${result.cacheMisses} miss (${cacheEntries} entries)`);
  console.error(`  outcomes            ${Object.entries(result.outcomes).map(([key, value]) => `${key} ${value}`).join(', ')}`);
  for (const [simulator, timing] of Object.entries(result.timings)) {
    console.error(
      `  latency ${simulator.padEnd(10)} n=${timing.count}`
      + ` mean ${seconds(timing.meanMs)} p50 ${seconds(timing.p50Ms)}`
      + ` p90 ${seconds(timing.p90Ms)} max ${seconds(timing.maxMs)}`,
    );
  }
  console.error(`  aggregate cells     ${summary.cells.length}`);
  console.error(`  coverage rows       ${summary.coverage.length}`);
  console.error(`  wrote               ${summaryPath}`);
  console.error(`  wrote               ${pointsPath}`);
  if (summary.incomplete.length > 0) {
    console.error(`  incomplete          ${summary.incomplete.map(item => `${item.leg}: ${item.reason}`).join('; ')}`);
  }
}

main().catch(error => {
  console.error(`[scorecard] failed: ${error instanceof Error ? error.stack ?? error.message : String(error)}`);
  process.exitCode = 1;
});

import { readdirSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { describe, expect, it } from 'vitest';
import { LINK_VERSION } from './common';
import {
  gpuExplorerLink,
  hybridSavingsLink,
  kvCacheLink,
  recommendLink,
  scorecardLink,
  validationCompareLink,
} from './pages';
import { predictLink, buildPredictLink } from './predict';
import { PAGE_REGISTRY, sharedViewRoutes } from './registry';

/**
 * Every route under `app/` that renders a page.
 *
 * Read from the filesystem rather than listed, so a page added tomorrow is
 * covered without anyone remembering this file.
 */
function appRoutes(): string[] {
  const appDir = join(process.cwd(), 'app');

  function walk(dir: string): string[] {
    return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) return walk(path);
      if (entry.name !== 'page.tsx') return [];
      const segments = relative(appDir, dir)
        .split(sep)
        .filter(Boolean)
        // A route group is a layout concern, not part of the URL.
        .filter(segment => !(segment.startsWith('(') && segment.endsWith(')')));
      return [segments.length === 0 ? '/' : `/${segments.join('/')}`];
    });
  }

  return walk(appDir).sort();
}

describe('the page registry', () => {
  it('lists every route the app serves', () => {
    const registered = PAGE_REGISTRY.map(route => route.path).sort();

    expect(registered).toEqual(appRoutes());
  });

  it('gives every route without a shared view a reason', () => {
    for (const route of PAGE_REGISTRY) {
      if (route.sharedView.kind === 'none') {
        expect(route.sharedView.reason.trim(), route.path).not.toBe('');
      }
    }
  });

  it('names at least one field for every route that shares a view', () => {
    for (const route of sharedViewRoutes()) {
      if (route.sharedView.kind !== 'link') continue;
      expect(route.sharedView.fields.length, route.path).toBeGreaterThan(0);
    }
  });

  it('keeps shared-view paths in step with the contracts they name', () => {
    const paths = sharedViewRoutes().map(route => route.path).sort();

    expect(paths).toEqual(
      [
        predictLink.path,
        recommendLink.path,
        kvCacheLink.path,
        gpuExplorerLink.path,
        hybridSavingsLink.path,
        validationCompareLink.path,
        scorecardLink.path,
      ].sort(),
    );
  });
});

describe('golden links', () => {
  /**
   * A link is a published interface: someone pasted one into a ticket, a
   * browser bookmark, or a colleague's chat. These fixtures pin the encoding of
   * a representative view per page. A failure here is a format change, so the
   * fix is to decide whether it is additive (update the fixture) or a break
   * (bump `LINK_VERSION` and keep reading the old version), not to re-record it
   * by reflex.
   */
  it('keeps the format version at 1 while fields are only added', () => {
    // The version is written into every URL, so it is a string here.
    expect(LINK_VERSION).toBe('1');
  });

  it('/predict', () => {
    expect(buildPredictLink({
      model: 'Qwen/Qwen3-32B',
      system: 'h200_sxm',
      isl: 4096,
      osl: 512,
      concurrency: 32,
      servingMode: 'disagg',
      tp: 8,
      pp: 1,
      weightPrecision: 'FP8',
      kvPrecision: 'FP8',
      prefix: 512,
      requestsPerLoadUnit: 4,
      maxNumSeqs: 256,
      chunkedPrefill: 'on',
      gpuMemUtil: 0.9,
      moeTp: 1,
      moeEp: 8,
      moePrecision: 'w4a16_mxfp4',
      prefillTp: 4,
      prefillPp: 1,
      prefillWorkers: 2,
      prefillBatch: 16,
      decodeTp: 8,
      decodePp: 1,
      decodeWorkers: 4,
      decodeBatch: 256,
      fabric: 'ib-400g',
      simulator: 'blis',
      backend: 'vllm',
      backendVersion: '0.31.0',
    })).toMatchInlineSnapshot(`"/predict?v=1&model=Qwen%2FQwen3-32B&system=h200_sxm&isl=4096&osl=512&concurrency=32&backend=vllm&backendVersion=0.31.0&simulator=blis&servingMode=disagg&fabric=ib-400g&tp=8&pp=1&weightPrecision=FP8&kvPrecision=FP8&prefix=512&requestsPerLoadUnit=4&maxNumSeqs=256&chunkedPrefill=on&gpuMemUtil=0.9&moeTp=1&moeEp=8&moePrecision=w4a16_mxfp4&prefillTp=4&prefillPp=1&prefillWorkers=2&prefillBatch=16&decodeTp=8&decodePp=1&decodeWorkers=4&decodeBatch=256"`);
  });

  it('/recommend', () => {
    expect(recommendLink.build({
      model: 'Qwen/Qwen3-32B',
      system: 'h200_sxm',
      goal: 'latency',
      isl: 4096,
      osl: 512,
      concurrency: 32,
      ttft: 2000,
      tpot: 50,
      prefix: 512,
      maxSeqLen: 32768,
      // A disaggregated view carries per-pool limits and the fabric that makes
      // those pools searchable. Only a simulator that reports the fabric
      // capability exports one, so this view uses the fixture that reports it.
      servingMode: 'disagg',
      prefillMaxSeqLen: 8192,
      decodeMaxSeqLen: 24576,
      fabric: 'ib-400g',
      simulator: 'blis',
      backend: 'vllm',
      backendVersion: '0.31.0',
    })).toMatchInlineSnapshot(`"/recommend?v=1&model=Qwen%2FQwen3-32B&system=h200_sxm&goal=latency&isl=4096&osl=512&concurrency=32&ttft=2000&tpot=50&prefix=512&maxSeqLen=32768&servingMode=disagg&prefillMaxSeqLen=8192&decodeMaxSeqLen=24576&fabric=ib-400g&backend=vllm&backendVersion=0.31.0&simulator=blis"`);
  });

  it('/recommend aggregate view', () => {
    // The other shape the page produces: an aggregated view with no per-pool
    // limits and an explicit "no fabric". Pairing limits with `agg` would
    // describe a request the page rejects.
    expect(recommendLink.build({
      model: 'Qwen/Qwen3-32B',
      system: 'h200_sxm',
      goal: 'latency',
      servingMode: 'agg',
      fabric: 'none',
      simulator: 'blis',
      backend: 'vllm',
      backendVersion: '0.31.0',
    })).toMatchInlineSnapshot(`"/recommend?v=1&model=Qwen%2FQwen3-32B&system=h200_sxm&goal=latency&servingMode=agg&fabric=none&backend=vllm&backendVersion=0.31.0&simulator=blis"`);
  });

  it('/kv-cache', () => {
    expect(kvCacheLink.build({
      model: 'Qwen/Qwen3-32B',
      system: 'h200_sxm',
      backend: 'vllm',
      backendVersion: '0.31.0',
      maxNumTokens: 32768,
      maxBatchSize: 256,
      tp: 1,
      pp: 1,
      moeTp: 1,
      moeEp: 8,
      servingMode: 'disagg',
      prefillTp: 4,
      prefillPp: 1,
      prefillMoeTp: 1,
      prefillMoeEp: 8,
      decodeTp: 8,
      decodePp: 1,
      decodeMoeTp: 1,
      decodeMoeEp: 8,
      memFractionKind: 'of_free',
      memFractionValue: 0.9,
    })).toMatchInlineSnapshot(`"/kv-cache?v=1&model=Qwen%2FQwen3-32B&system=h200_sxm&backend=vllm&backendVersion=0.31.0&maxNumTokens=32768&maxBatchSize=256&tp=1&pp=1&moeTp=1&moeEp=8&servingMode=disagg&prefillTp=4&prefillPp=1&prefillMoeTp=1&prefillMoeEp=8&decodeTp=8&decodePp=1&decodeMoeTp=1&decodeMoeEp=8&memFractionKind=of_free&memFractionValue=0.9"`);
  });

  it('/gpu-explorer', () => {
    expect(gpuExplorerLink.build({
      preset: 'performance',
      vendor: 'nvidia',
      x: 'mem-bw',
      y: 'throughput-index',
    })).toMatchInlineSnapshot(`"/gpu-explorer?v=1&preset=performance&vendor=nvidia&x=mem-bw&y=throughput-index"`);
  });

  it('/hybrid-savings', () => {
    expect(hybridSavingsLink.build({
      model: 'Qwen/Qwen3-32B',
      apiModel: 'Qwen/Qwen3-32B',
      monthlyInputTokens: 900_000_000,
      monthlyOutputTokens: 90_000_000,
      averageInputTokens: 4096,
      averageOutputTokens: 512,
      ttft: 2000,
      tpot: 50,
      concurrency: 32,
      prefix: 512,
      hours: 120,
      peak: 2.5,
      billingMode: 'always-on',
      costLens: 'marginal',
      simulator: 'aisimulate',
      backend: 'vllm',
      backendVersion: '0.31.0',
    })).toMatchInlineSnapshot(`"/hybrid-savings?v=1&model=Qwen%2FQwen3-32B&apiModel=Qwen%2FQwen3-32B&monthlyInputTokens=900000000&monthlyOutputTokens=90000000&averageInputTokens=4096&averageOutputTokens=512&ttft=2000&tpot=50&concurrency=32&prefix=512&hours=120&peak=2.5&billingMode=always-on&costLens=marginal&backend=vllm&backendVersion=0.31.0&simulator=aisimulate"`);
  });

  it('/validation/compare', () => {
    expect(validationCompareLink.build({
      model: 'Qwen/Qwen3-32B',
      accelerator: 'h200_sxm',
      isl: 4096,
      osl: 512,
      tp: 8,
      configurationKey: 'fp8-tp8',
      simulator: 'aisimulate',
      backend: 'vllm',
      backendVersion: '0.31.0',
    })).toMatchInlineSnapshot(`"/validation/compare?v=1&model=Qwen%2FQwen3-32B&accelerator=h200_sxm&isl=4096&osl=512&tp=8&configurationKey=fp8-tp8&backend=vllm&backendVersion=0.31.0&simulator=aisimulate"`);
  });

  it('/validation/scorecard', () => {
    expect(scorecardLink.build({
      metric: 'tpot',
      statistic: 'mdape',
      releaseMatch: 'any',
      source: 'redhat',
      model: 'openai/gpt-oss-120b',
      gpu: 'H200',
      workload: '8192/1024',
    })).toMatchInlineSnapshot(`"/validation/scorecard?v=1&metric=tpot&statistic=mdape&releaseMatch=any&source=redhat&model=openai%2Fgpt-oss-120b&gpu=H200&workload=8192%2F1024"`);
  });
});

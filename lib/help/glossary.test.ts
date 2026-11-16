import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { GLOSSARY, GLOSSARY_ENTRIES, findGlossaryEntry } from './glossary';

const ROOT = process.cwd();

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) return sourceFiles(full);
    return /\.(tsx?|md)$/.test(entry) ? [full] : [];
  });
}

describe('glossary', () => {
  it('has a title and body for every entry', () => {
    for (const entry of GLOSSARY_ENTRIES) {
      expect(entry.title.trim().length).toBeGreaterThan(0);
      expect(entry.body.trim().length).toBeGreaterThan(0);
    }
  });

  it('keeps ids unique and equal to the registry keys', () => {
    const ids = GLOSSARY_ENTRIES.map((entry) => entry.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(new Set(ids)).toEqual(new Set(Object.keys(GLOSSARY)));
  });

  it('defines every term a page renders', () => {
    const used = new Set<string>();
    for (const file of sourceFiles(path.join(ROOT, 'app'))) {
      for (const match of readFileSync(file, 'utf8').matchAll(/<Term\s+k="([^"]+)"/g)) {
        used.add(match[1]);
      }
    }

    expect(used.size).toBeGreaterThan(0);
    for (const id of used) {
      expect(GLOSSARY[id as keyof typeof GLOSSARY], `missing glossary entry for "${id}"`).toBeDefined();
    }
  });

  it('resolves an entry by its id', () => {
    expect(findGlossaryEntry('kvCache')?.title).toBe('KV cache');
  });

  it('resolves an entry by the display term a page passes', () => {
    expect(findGlossaryEntry('KV cache')?.id).toBe('kvCache');
    expect(findGlossaryEntry('  kv CACHE  ')?.id).toBe('kvCache');
  });

  it('resolves a short title before the separator', () => {
    expect(findGlossaryEntry('MHA')?.id).toBe('mha');
    expect(findGlossaryEntry('GQA')?.id).toBe('gqa');
  });

  it('returns undefined for a term it does not know', () => {
    expect(findGlossaryEntry('definitely-not-a-term')).toBeUndefined();
    expect(findGlossaryEntry('')).toBeUndefined();
  });

  it('builds precision text from the shared precision metadata', () => {
    // Guards against a second copy of the precision table drifting from
    // lib/help/precision.ts.
    expect(GLOSSARY.weightPrecision.body).toContain('FP8 = 1 byte (8-bit float)');
    expect(GLOSSARY.kvCachePrecision.body).toContain('FP16 = 2 bytes per element (standard)');
  });
});

import { describe, expect, it } from 'vitest';
import { GLOSSARY, GLOSSARY_ENTRIES, findGlossaryEntry } from './glossary';

describe('glossary', () => {
  it('has a title and body for every entry', () => {
    for (const entry of GLOSSARY_ENTRIES) {
      expect(entry.title.trim().length).toBeGreaterThan(0);
      expect(entry.body.trim().length).toBeGreaterThan(0);
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

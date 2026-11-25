import { describe, expect, it } from 'vitest';
import type { DatasetRow } from '@/lib/validation/benchmark-types';
import { groupConfigurations } from './build';

/**
 * A measured row with one runtime configuration. `configurationKey` covers
 * only these fields, so the model, accelerator, shape, and parallel size are
 * what separate one series from another.
 */
function row(overrides: Partial<Record<string, unknown>> = {}): DatasetRow {
  return {
    benchmark_source: 'redhat',
    backend: 'vllm',
    backend_version: 'v0.31.0',
    model: 'org/model',
    accelerator: 'H200',
    isl: 1024,
    osl: 128,
    tp: 8,
    concurrency: 1,
    ...overrides,
  };
}

describe('groupConfigurations', () => {
  it('keeps rows that differ only by model in separate series', () => {
    // The runtime configuration is identical; pooling the rows would take the
    // model from the first one and score another model's measurements as it.
    const groups = groupConfigurations([
      row({ model: 'org/first' }),
      row({ model: 'org/second' }),
    ]);

    expect(groups.size).toBe(2);
    expect([...groups.values()].map(group => group.rows[0].model).sort())
      .toEqual(['org/first', 'org/second']);
  });

  it('keeps rows that differ only by accelerator in separate series', () => {
    const groups = groupConfigurations([
      row({ accelerator: 'H200' }),
      row({ accelerator: 'B200' }),
    ]);

    expect(groups.size).toBe(2);
  });

  it('keeps every shape and parallel size of one configuration apart', () => {
    const groups = groupConfigurations([
      row({ isl: 1024, osl: 128, tp: 8 }),
      row({ isl: 8192, osl: 128, tp: 8 }),
      row({ isl: 1024, osl: 1024, tp: 8 }),
      row({ isl: 1024, osl: 128, tp: 4 }),
    ]);

    expect(groups.size).toBe(4);
    expect([...groups.values()].map(group => group.rows.length)).toEqual([1, 1, 1, 1]);
  });

  it('keeps concurrency levels of one series together', () => {
    // Concurrency is the level a series is measured at, not part of its
    // identity: one prediction is made per level.
    const groups = groupConfigurations([
      row({ concurrency: 1 }),
      row({ concurrency: 8 }),
      row({ concurrency: 64 }),
    ]);

    expect(groups.size).toBe(1);
    expect([...groups.values()][0].rows).toHaveLength(3);
  });

  it('treats numeric and string shapes as one series', () => {
    const groups = groupConfigurations([
      row({ isl: 8192, osl: '1024', tp: 8 }),
      row({ isl: '8192', osl: 1024, tp: '8' }),
    ]);

    expect(groups.size).toBe(1);
  });
});

import { describe, expect, it } from 'vitest';
import {
  normalizeSimulatorId,
  normalizeSimulatorLabel,
  simulatorIdMigrations,
} from './simulator-ids';

describe('normalizeSimulatorId', () => {
  it('reads the retired id as the current one', () => {
    expect(normalizeSimulatorId('aisimulate')).toBe('aisim');
    expect(normalizeSimulatorId('aisim')).toBe('aisim');
    expect(normalizeSimulatorId('blis')).toBe('blis');
  });

  it('passes an unknown id through unchanged', () => {
    expect(normalizeSimulatorId('future-simulator')).toBe('future-simulator');
  });

  it('does not resolve inherited object properties', () => {
    // A shared link is external input: `__proto__` and `constructor` are valid
    // text but not simulator ids, and must not become objects.
    expect(normalizeSimulatorId('__proto__')).toBe('__proto__');
    expect(normalizeSimulatorId('constructor')).toBe('constructor');
  });
});

describe('normalizeSimulatorLabel', () => {
  it('replaces the label a retired id was published with', () => {
    expect(normalizeSimulatorLabel('aisimulate', 'AISimulate')).toBe('AISim');
  });

  it('keeps the label a current id carries', () => {
    expect(normalizeSimulatorLabel('aisim', 'AISim')).toBe('AISim');
    expect(normalizeSimulatorLabel('blis', 'BLIS')).toBe('BLIS');
  });

  it('does not resolve inherited object properties', () => {
    expect(normalizeSimulatorLabel('__proto__', 'AISim')).toBe('AISim');
    expect(normalizeSimulatorLabel('constructor', 'AISim')).toBe('AISim');
  });
});

describe('simulatorIdMigrations', () => {
  it('lists every retired id with the id that replaced it', () => {
    expect(simulatorIdMigrations()).toEqual([['aisimulate', 'aisim']]);
  });
});

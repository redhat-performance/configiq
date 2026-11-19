import { describe, expect, it } from 'vitest';
import { parseSweep } from './parse';

describe('parseSweep', () => {
  it('keeps the deployment modes the service reported', () => {
    expect(parseSweep({ modes_searched: ['agg', 'disagg'] })?.modesSearched).toEqual(['agg', 'disagg']);
    expect(parseSweep({ modes_searched: ['disagg'] })?.modesSearched).toEqual(['disagg']);
  });

  it('drops values outside the contract instead of forwarding them', () => {
    expect(parseSweep({ modes_searched: ['agg', 'pd', 3] })?.modesSearched).toEqual(['agg']);
  });

  it('reports no mode facts as null rather than inferring them', () => {
    expect(parseSweep({})?.modesSearched).toBeNull();
    expect(parseSweep({ modes_searched: null })?.modesSearched).toBeNull();
  });

  it('returns no sweep facts when the service sent none', () => {
    expect(parseSweep(null)).toBeNull();
  });
});

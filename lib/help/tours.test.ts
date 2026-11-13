import { describe, expect, it } from 'vitest';
import { HELP_TOURS, getTourById, getTourForRoute, normaliseRoute } from './tours';

describe('tour registry', () => {
  it('normalises a pathname before lookup', () => {
    expect(normaliseRoute('/predict')).toBe('/predict');
    expect(normaliseRoute('/predict/')).toBe('/predict');
    expect(normaliseRoute('/predict?model=x#top')).toBe('/predict');
    expect(normaliseRoute('')).toBe('/');
    expect(normaliseRoute(null)).toBe('/');
  });

  it('finds the tour registered for a route', () => {
    expect(getTourForRoute('/predict')?.id).toBe('predict-performance');
    expect(getTourForRoute('/predict/')?.id).toBe('predict-performance');
  });

  it('returns undefined for a route with no tour', () => {
    expect(getTourForRoute('/settings')).toBeUndefined();
  });

  it('finds a tour by id', () => {
    expect(getTourById('predict-performance')?.route).toBe('/predict');
    expect(getTourById('missing')).toBeUndefined();
  });

  it('only points steps at data-tour anchors', () => {
    // The registry is the single source of tour content; pages opt in by
    // rendering a matching `data-tour` attribute. Keeping the selector shape
    // uniform is what lets a missing target be skipped safely.
    for (const tour of HELP_TOURS) {
      expect(tour.route.startsWith('/')).toBe(true);
      expect(tour.steps.length).toBeGreaterThan(0);
      for (const step of tour.steps) {
        expect(step.target).toMatch(/^\[data-tour="[a-z0-9-]+"\]$/);
        expect(step.title.trim().length).toBeGreaterThan(0);
        expect(step.description.trim().length).toBeGreaterThan(0);
      }
    }
  });
});

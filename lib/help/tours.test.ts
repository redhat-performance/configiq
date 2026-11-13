import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { HELP_TOURS, getTourById, getTourForRoute, normaliseRoute } from './tours';

const ROOT = process.cwd();

/**
 * The component that renders each tour's route.
 *
 * A step whose target is missing still renders (centred), so this mapping is
 * what keeps a renamed `data-tour` attribute from quietly degrading a tour.
 */
const ROUTE_SOURCES: Record<string, string> = {
  '/predict': 'app/predict/Performance.tsx',
  '/recommend': 'app/recommend/Sizing.tsx',
  '/kv-cache': 'app/kv-cache/KvCacheCalc.tsx',
  '/gpu-explorer': 'app/gpu-explorer/page.tsx',
  '/hybrid-savings': 'app/hybrid-savings/HybridSavings.tsx',
  '/validation/compare': 'app/validation/compare/page.tsx',
};

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

  it('points every step at a target the page actually renders', () => {
    for (const tour of HELP_TOURS) {
      const source = ROUTE_SOURCES[tour.route];
      expect(source, `${tour.id} has no source mapping; add one`).toBeDefined();

      const contents = readFileSync(path.join(ROOT, source), 'utf8');
      for (const step of tour.steps) {
        const target = /data-tour="([a-z0-9-]+)"/.exec(step.target)?.[1];
        expect(contents.includes(`data-tour="${target}"`), `${tour.route} ${step.target}`).toBe(true);
      }
    }
  });

  it('registers each route only once', () => {
    const routes = HELP_TOURS.map((tour) => tour.route);
    expect(new Set(routes).size).toBe(routes.length);
  });
});

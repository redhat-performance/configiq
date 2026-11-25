import { describe, expect, it } from 'vitest';
import {
  layoutVenn,
  lensArea,
  placeVennDots,
  type VennLayout,
  type VennRegion,
} from './coverage-geometry';

const SIZE = { width: 320, height: 210 };

function distance(layout: VennLayout): number {
  return Math.hypot(layout.second.cx - layout.first.cx, layout.second.cy - layout.first.cy);
}

function inFirst(layout: VennLayout, x: number, y: number): boolean {
  return Math.hypot(x - layout.first.cx, y - layout.first.cy) <= layout.first.r + 0.001;
}

function inSecond(layout: VennLayout, x: number, y: number): boolean {
  return Math.hypot(x - layout.second.cx, y - layout.second.cy) <= layout.second.r + 0.001;
}

function inRegion(layout: VennLayout, region: VennRegion, x: number, y: number): boolean {
  const first = inFirst(layout, x, y);
  const second = inSecond(layout, x, y);
  if (region === 'firstOnly') return first && !second;
  if (region === 'secondOnly') return second && !first;
  return first && second;
}

describe('lensArea', () => {
  it('is zero when the circles are apart and the smaller area when one contains the other', () => {
    expect(lensArea(10, 5, 20)).toBe(0);
    expect(lensArea(10, 3, 5)).toBeCloseTo(Math.PI * 3 ** 2);
  });

  it('does not depend on which circle is named first', () => {
    expect(lensArea(10, 5, 8)).toBeCloseTo(lensArea(5, 10, 8));
  });

  it('grows as the centres move together', () => {
    expect(lensArea(10, 10, 19)).toBeLessThan(lensArea(10, 10, 10));
  });
});

describe('layoutVenn', () => {
  it('sizes each circle by area rather than by count alone', () => {
    const layout = layoutVenn({ first: 12, second: 3, shared: 0, ...SIZE });

    // Area is proportional to the count, so the radius ratio is the square
    // root of the count ratio.
    expect(layout.first.r / layout.second.r).toBeCloseTo(2);
  });

  it('solves the centre distance so the lens is proportional to the shared count', () => {
    const layout = layoutVenn({ first: 10, second: 10, shared: 4, ...SIZE });
    const ratio = lensArea(layout.first.r, layout.second.r, distance(layout))
      / (Math.PI * layout.first.r ** 2);

    expect(layout.nested).toBe(false);
    expect(ratio).toBeCloseTo(0.4);
    expect(distance(layout)).toBeLessThan(layout.first.r + layout.second.r);
  });

  it('draws the smaller circle inside the larger when one set contains the other', () => {
    const layout = layoutVenn({ first: 12, second: 3, shared: 3, ...SIZE });

    expect(layout.nested).toBe(true);
    expect(distance(layout)).toBeCloseTo(layout.first.r - layout.second.r);
    expect(layout.anchors.firstOnly).not.toBeNull();
    expect(layout.anchors.secondOnly).toBeNull();
    // The exclusive anchor has to fall in the crescent, not in the inner set.
    expect(inRegion(layout, 'firstOnly', layout.anchors.firstOnly!.x, layout.anchors.firstOnly!.y))
      .toBe(true);
  });

  it('keeps two disjoint sets apart when they share nothing', () => {
    const layout = layoutVenn({ first: 4, second: 4, shared: 0, ...SIZE });

    expect(layout.nested).toBe(false);
    expect(distance(layout)).toBeCloseTo(layout.first.r + layout.second.r);
    expect(lensArea(layout.first.r, layout.second.r, distance(layout))).toBeCloseTo(0);
  });

  it('stays inside the viewBox at both ends of the shared range', () => {
    for (const shared of [0, 1, 5, 10]) {
      const layout = layoutVenn({ first: 10, second: 10, shared, ...SIZE });
      for (const circle of [layout.first, layout.second]) {
        expect(circle.cx - circle.r).toBeGreaterThanOrEqual(-0.001);
        expect(circle.cx + circle.r).toBeLessThanOrEqual(SIZE.width + 0.001);
        expect(circle.cy - circle.r).toBeGreaterThanOrEqual(-0.001);
        expect(circle.cy + circle.r).toBeLessThanOrEqual(SIZE.height + 0.001);
      }
    }
  });

  it('returns finite geometry rather than NaN when one set is empty', () => {
    const layout = layoutVenn({ first: 0, second: 5, shared: 0, ...SIZE });

    expect(Number.isFinite(layout.first.r)).toBe(true);
    expect(Number.isFinite(layout.second.r)).toBe(true);
    expect(layout.anchors.firstOnly).toBeNull();
    expect(layout.anchors.secondOnly).toBeNull();
  });
});

describe('placeVennDots', () => {
  it('places the requested number of dots in each region', () => {
    const layout = layoutVenn({ first: 10, second: 10, shared: 4, ...SIZE });

    for (const [region, count] of [['firstOnly', 6], ['shared', 4], ['secondOnly', 6]] as const) {
      const dots = placeVennDots(layout, region, count);
      expect(dots).toHaveLength(count);
      for (const dot of dots) {
        expect(inRegion(layout, region, dot.x, dot.y)).toBe(true);
        expect(dot.region).toBe(region);
      }
    }
  });

  it('keeps dots apart instead of stacking them on the anchor', () => {
    const layout = layoutVenn({ first: 10, second: 10, shared: 4, ...SIZE });
    const dots = placeVennDots(layout, 'shared', 4);

    for (let left = 0; left < dots.length; left += 1) {
      for (let right = left + 1; right < dots.length; right += 1) {
        const gap = Math.hypot(dots[left].x - dots[right].x, dots[left].y - dots[right].y);
        expect(gap).toBeGreaterThan(5);
      }
    }
  });

  it('fills a thin crescent, not only the easy centre of it', () => {
    const layout = layoutVenn({ first: 12, second: 3, shared: 3, ...SIZE });
    const dots = placeVennDots(layout, 'firstOnly', 9);

    expect(dots).toHaveLength(9);
    for (const dot of dots) expect(inRegion(layout, 'firstOnly', dot.x, dot.y)).toBe(true);
  });

  it('returns nothing for an absent region or a zero count', () => {
    const nested = layoutVenn({ first: 12, second: 3, shared: 3, ...SIZE });

    expect(placeVennDots(nested, 'secondOnly', 3)).toEqual([]);
    expect(placeVennDots(nested, 'shared', 0)).toEqual([]);
  });

  it('keeps every dot inside the viewBox', () => {
    const layout = layoutVenn({ first: 30, second: 30, shared: 6, ...SIZE });
    const dots = [
      ...placeVennDots(layout, 'firstOnly', 24),
      ...placeVennDots(layout, 'shared', 6),
      ...placeVennDots(layout, 'secondOnly', 24),
    ];

    expect(dots).toHaveLength(54);
    for (const dot of dots) {
      expect(dot.x).toBeGreaterThanOrEqual(6);
      expect(dot.x).toBeLessThanOrEqual(SIZE.width - 6);
      expect(dot.y).toBeGreaterThanOrEqual(6);
      expect(dot.y).toBeLessThanOrEqual(SIZE.height - 6);
    }
  });
});

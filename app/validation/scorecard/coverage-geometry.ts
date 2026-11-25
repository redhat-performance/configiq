/**
 * Geometry for the coverage map's two-set circles.
 *
 * Circle area is proportional to the number of pairs, so a reader can compare
 * coverage by eye instead of reading the counts. The overlap is not chosen by
 * hand: the centre distance is solved so the lens area is proportional to the
 * shared count. When one set contains the other, the smaller circle sits inside
 * the larger and the lens is the whole smaller circle.
 *
 * Kept pure and separate from the component so the placement rules can be
 * tested without a DOM.
 */

export interface VennPoint {
  x: number;
  y: number;
}

export interface VennCircle {
  cx: number;
  cy: number;
  r: number;
}

export interface VennAnchors {
  /** Where the exclusive part of the first circle is labelled or packed. */
  firstOnly: VennPoint | null;
  /** Where the shared lens is labelled or packed. */
  shared: VennPoint;
  /** Where the exclusive part of the second circle is labelled or packed. */
  secondOnly: VennPoint | null;
}

export interface VennLayout {
  width: number;
  height: number;
  first: VennCircle;
  second: VennCircle;
  /** True when one set contains the other, so only one exclusive region exists. */
  nested: boolean;
  anchors: VennAnchors;
}

export interface VennInput {
  first: number;
  second: number;
  shared: number;
  width: number;
  height: number;
}

const PADDING = 14;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/** Area of the lens where two circles overlap. */
export function lensArea(firstRadius: number, secondRadius: number, distance: number): number {
  if (distance >= firstRadius + secondRadius) return 0;
  if (distance <= Math.abs(firstRadius - secondRadius)) {
    return Math.PI * Math.min(firstRadius, secondRadius) ** 2;
  }

  const firstAngle = Math.acos(
    (distance ** 2 + firstRadius ** 2 - secondRadius ** 2) / (2 * distance * firstRadius),
  );
  const secondAngle = Math.acos(
    (distance ** 2 + secondRadius ** 2 - firstRadius ** 2) / (2 * distance * secondRadius),
  );
  const product =
    (-distance + firstRadius + secondRadius)
    * (distance + firstRadius - secondRadius)
    * (distance - firstRadius + secondRadius)
    * (distance + firstRadius + secondRadius);
  return firstRadius ** 2 * firstAngle
    + secondRadius ** 2 * secondAngle
    - 0.5 * Math.sqrt(Math.max(0, product));
}

/**
 * Lay out two circles and their region anchors.
 *
 * A zero count has no circle to draw; callers should render a single-set
 * summary instead. The function still returns finite geometry so a caller that
 * forgets does not produce NaN coordinates.
 */
export function layoutVenn(input: VennInput): VennLayout {
  const firstCount = Math.max(0, input.first);
  const secondCount = Math.max(0, input.second);
  const shared = Math.max(0, Math.min(input.shared, firstCount, secondCount));
  const width = Math.max(1, input.width);
  const height = Math.max(1, input.height);
  // The circles have to fit both when nested (two diameters) and when apart
  // (both diameters plus the gap between them). Taking the tighter of the two
  // constraints keeps the drawing inside the viewBox at every overlap.
  const availableWidth = width - PADDING * 2;
  const availableHeight = height - PADDING * 2;
  const firstRoot = Math.sqrt(firstCount);
  const secondRoot = Math.sqrt(secondCount);
  const scale = firstCount === 0 && secondCount === 0
    ? 0
    : Math.min(
      availableWidth / (2 * (firstRoot + secondRoot)),
      availableHeight / (2 * Math.max(firstRoot, secondRoot)),
    );
  const firstRadius = firstRoot * scale;
  const secondRadius = secondRoot * scale;
  const cy = height / 2;

  if (firstRadius === 0 || secondRadius === 0) {
    const circle = {
      cx: width / 2,
      cy,
      r: firstRadius || secondRadius,
    };
    return {
      width,
      height,
      first: { ...circle, r: firstRadius },
      second: { ...circle, r: secondRadius },
      nested: false,
      anchors: { firstOnly: null, shared: { x: width / 2, y: cy }, secondOnly: null },
    };
  }

  const nested = shared >= Math.min(firstCount, secondCount);
  const distance = nested
    ? Math.abs(firstRadius - secondRadius)
    : solveDistance(firstRadius, secondRadius, shared * Math.PI * scale ** 2);

  const span = firstRadius + distance + secondRadius;
  const left = (width - span) / 2;
  const first: VennCircle = { cx: left + firstRadius, cy, r: firstRadius };
  const second: VennCircle = { cx: left + firstRadius + distance, cy, r: secondRadius };

  // Anchor each region at the midpoint of its horizontal extent. A dot spiral
  // that starts inside its own region finds a stable packing sooner than one
  // that starts on a boundary and has to reject its first candidates.
  const exclusive = (firstRadius + secondRadius - distance) / 2;
  return {
    width,
    height,
    first,
    second,
    nested,
    anchors: {
      firstOnly: nested && firstRadius <= secondRadius
        ? null
        : { x: first.cx - exclusive, y: cy },
      shared: { x: first.cx + (distance + firstRadius - secondRadius) / 2, y: cy },
      secondOnly: nested && secondRadius <= firstRadius
        ? null
        : { x: second.cx + exclusive, y: cy },
    },
  };
}

function solveDistance(firstRadius: number, secondRadius: number, target: number): number {
  if (target <= 0) return firstRadius + secondRadius;
  let low = Math.abs(firstRadius - secondRadius);
  let high = firstRadius + secondRadius;
  for (let index = 0; index < 60; index += 1) {
    const middle = (low + high) / 2;
    if (lensArea(firstRadius, secondRadius, middle) > target) low = middle;
    else high = middle;
  }
  return (low + high) / 2;
}

export type VennRegion = 'firstOnly' | 'shared' | 'secondOnly';

export interface VennDot extends VennPoint {
  region: VennRegion;
}

/**
 * Pack `count` dots into one region.
 *
 * The spiral starts at the region anchor and walks outward at the golden angle,
 * keeping a minimum spacing. A grid sweep fills any shortfall, so a narrow
 * crescent still reaches its count instead of silently drawing fewer dots than
 * the legend claims. Every returned point is checked against the region.
 */
export function placeVennDots(
  layout: VennLayout,
  region: VennRegion,
  count: number,
): VennDot[] {
  if (count <= 0) return [];
  const anchor = layout.anchors[region];
  if (!anchor) return [];

  const spacing = 17;
  const dots: VennDot[] = [];
  const limit = count * 60 + 240;

  for (let index = 0; index < limit && dots.length < count; index += 1) {
    const radius = spacing * Math.sqrt(index) * 0.82;
    const angle = index * GOLDEN_ANGLE;
    consider({
      x: anchor.x + Math.cos(angle) * radius,
      y: anchor.y + Math.sin(angle) * radius,
    });
  }

  if (dots.length < count) {
    for (let y = PADDING; y <= layout.height - PADDING && dots.length < count; y += spacing) {
      for (let x = PADDING; x <= layout.width - PADDING && dots.length < count; x += spacing) {
        consider({ x, y });
      }
    }
  }

  return dots;

  function consider(point: VennPoint): void {
    if (!inside(layout, point.x, point.y, 6)) return;
    if (!inRegion(point.x, point.y, layout, region)) return;
    if (dots.some(dot => Math.hypot(dot.x - point.x, dot.y - point.y) < spacing * 0.85)) return;
    dots.push({ x: point.x, y: point.y, region });
  }
}

function inside(layout: VennLayout, x: number, y: number, margin: number): boolean {
  return x >= margin && y >= margin
    && x <= layout.width - margin && y <= layout.height - margin;
}

function inRegion(x: number, y: number, layout: VennLayout, region: VennRegion): boolean {
  const first = Math.hypot(x - layout.first.cx, y - layout.first.cy) <= layout.first.r + 0.001;
  const second = Math.hypot(x - layout.second.cx, y - layout.second.cy) <= layout.second.r + 0.001;
  if (region === 'firstOnly') return first && !second;
  if (region === 'secondOnly') return second && !first;
  return first && second;
}

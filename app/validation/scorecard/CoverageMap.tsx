import * as React from 'react';
import { ExpandableSection } from '@patternfly/react-core';
import {
  SCORECARD_SOURCES,
  type ScorecardCoverage,
  type ScorecardReleaseMatch,
  type ScorecardSource,
  type ScorecardSourceFilter,
} from '@/lib/scorecard/types';
import { buildCoverageSets, type CoveragePair } from './scorecard-view';
import {
  layoutSingleSet,
  layoutVenn,
  placeVennDots,
  type VennLayout,
  type VennRegion,
} from './coverage-geometry';
import styles from './scorecard.module.css';

/**
 * Coverage map: one dot per measured model + GPU pair.
 *
 * Two diagrams answer two different questions. The simulator circles show which
 * pairs each simulator could score, so the area is a coverage comparison. The
 * dataset circles show which pairs each measured source holds. Each draws two
 * overlapping circles when both sets have pairs in the slice, and one circle
 * when only one does, so a slice narrowed to a single source or a single scored
 * simulator still shows its pairs instead of dropping the figure. A shared pair
 * is not a shared workload, so the dataset figure states the shared-workload
 * count separately rather than letting the lens imply a like-for-like
 * comparison.
 *
 * A four-circle Venn was rejected: the simulator set is a subset of another in
 * the current data, and two of the sets are simulators while two are datasets,
 * so the drawing would be unreadable and the all-four lens would overstate
 * comparability. The table below carries the exact four-way membership.
 */

const VENN_WIDTH = 320;
const VENN_HEIGHT = 210;

/** Data marks, distinct from the logo and from the simulator brand hues. */
const DATASET_COLORS: Record<ScorecardSource, string> = {
  redhat: '#c9190b',
  inferencex: '#a86e00',
};

const DATASET_LABELS: Record<ScorecardSource, string> = {
  redhat: 'Red Hat',
  inferencex: 'InferenceX',
};

export interface CoverageSimulator {
  id: string;
  label: string;
  color: string;
}

export interface CoverageMapProps {
  /** The coverage rows for the slice the page is showing. */
  coverage: readonly ScorecardCoverage[];
  /** The snapshot-level shared-workload count, when it was published. */
  sharedWorkloads?: number;
  source: ScorecardSourceFilter;
  releaseMatch: ScorecardReleaseMatch;
  simulators: readonly CoverageSimulator[];
  onSelectPair: (pair: CoveragePair) => void;
}

export default function CoverageMap({
  coverage,
  sharedWorkloads,
  source,
  releaseMatch,
  simulators,
  onSelectPair,
}: CoverageMapProps) {
  const sets = React.useMemo(
    () => buildCoverageSets(coverage, { source, releaseMatch }),
    [coverage, source, releaseMatch],
  );
  const [hovered, setHovered] = React.useState<CoveragePair | null>(null);

  const sources = React.useMemo<ScorecardSource[]>(
    () => (source === 'all' ? [...SCORECARD_SOURCES] : [source]),
    [source],
  );
  const selectable = React.useCallback(
    (pair: CoveragePair) => pair.simulators.size > 0,
    [],
  );

  if (sets.pairs.length === 0) {
    return <p className={styles.emptyRows}>No measured model + GPU pair in this view.</p>;
  }

  const simulatorSets = simulators.map(entry => ({
    id: entry.id,
    label: entry.label,
    color: entry.color,
    count: sets.pairs.filter(pair => pair.simulators.has(entry.id)).length,
  }));
  // The layout draws two circles. Keep the shared count on the same two sets the
  // figure draws, so a snapshot that ever listed more cannot report an overlap
  // the drawing does not show.
  const figureSimulators = simulatorSets.slice(0, 2);
  const simulatorShared = sharedCount(
    sets.pairs,
    figureSimulators.map(entry => entry.id),
    pair => pair.simulators,
  );
  const sourceSets = sources.map(entry => ({
    id: entry,
    label: DATASET_LABELS[entry],
    color: DATASET_COLORS[entry],
    count: sets.pairs.filter(pair => pair.sources.has(entry)).length,
  }));
  const sourceShared = sharedCount(sets.pairs, sources, pair => pair.sources);

  // A figure needs at least one set with something in it. A second set with no
  // pairs still keeps its legend entry, so an absent source or simulator reads
  // as empty rather than as a missing figure the reader has to explain.
  const showSimulatorFigure = figureSimulators.some(entry => entry.count > 0);
  const showSourceFigure = sourceSets.some(entry => entry.count > 0);

  return (
    <div className={styles.coverageMap}>
      <div className={styles.coverageFigures}>
        {showSimulatorFigure && (
          <VennFigure
            title="Simulator coverage"
            sets={figureSimulators}
            shared={simulatorShared}
            pairs={sets.pairs}
            dimension="simulators"
            selectable={selectable}
            onSelect={onSelectPair}
            onHover={setHovered}
            note={sets.notScored > 0
              ? `${formatCount(sets.notScored)} measured, not scored`
              : undefined}
          />
        )}

        {showSourceFigure && (
          <VennFigure
            title="Measured-data coverage"
            sets={sourceSets}
            shared={sourceShared}
            pairs={sets.pairs}
            dimension="sources"
            selectable={selectable}
            onSelect={onSelectPair}
            onHover={setHovered}
            note={sharedWorkloads === undefined
              ? 'Workload overlap not published in this snapshot'
              : sharedWorkloads === 0
                ? 'No shared workload shapes in this snapshot'
                : `${formatCount(sharedWorkloads)} shared workload ${sharedWorkloads === 1 ? 'shape' : 'shapes'} in this snapshot`}
          />
        )}
      </div>

      <p className={styles.coverageHover} aria-live="polite">
        {hovered
          ? <><span className={styles.coverageHoverName}>{hovered.model}</span> · {hovered.accelerator}</>
          : 'One dot per measured model + GPU pair'}
      </p>

      <ExpandableSection
        toggleText={`All measured pairs (${formatCount(sets.pairs.length)})`}
        className={styles.coverageTableSection}
        isIndented
      >
        <CoverageTable
          pairs={sets.pairs}
          simulators={simulators}
          sources={sources}
          selectable={selectable}
          onSelect={onSelectPair}
        />
      </ExpandableSection>
    </div>
  );
}

function VennFigure({
  title,
  sets,
  shared,
  pairs,
  dimension,
  selectable,
  onSelect,
  onHover,
  note,
}: {
  title: string;
  /** One set, or two. A second set with no pairs keeps its legend entry. */
  sets: readonly VennSet[];
  shared: number;
  pairs: readonly CoveragePair[];
  dimension: 'simulators' | 'sources';
  selectable: (pair: CoveragePair) => boolean;
  onSelect: (pair: CoveragePair) => void;
  onHover: (pair: CoveragePair | null) => void;
  note?: string;
}) {
  const first = sets[0];
  const second = sets[1];
  const hasSecond = second !== undefined;
  const secondId = second?.id;
  const secondCount = second?.count ?? 0;
  const secondColor = second?.color;
  const secondLabel = second?.label;
  // One circle when there is nothing to overlap: only one set exists, or one of
  // the two has no pair in this slice. Drawing the remaining set beats dropping
  // the figure, which hid both the pairs and their selectable dots.
  const single = !hasSecond || first.count === 0 || secondCount === 0;
  const shown = single && first.count === 0 && second ? second : first;
  const empty = single && second && (first.count === 0 || secondCount === 0)
    ? (first.count === 0 ? first : second)
    : null;

  const layout = React.useMemo(
    () => (single || !hasSecond
      ? layoutSingleSet(shown.count, VENN_WIDTH, VENN_HEIGHT)
      : layoutVenn({
        first: first.count,
        second: secondCount,
        shared,
        width: VENN_WIDTH,
        height: VENN_HEIGHT,
      })),
    [single, hasSecond, shown.count, first.count, secondCount, shared],
  );

  const regions = React.useMemo(() => {
    if (single || secondId === undefined) return null;
    const members = (pair: CoveragePair): ReadonlySet<string> =>
      dimension === 'simulators' ? pair.simulators : pair.sources;
    const firstOnly = pairs.filter(pair =>
      members(pair).has(first.id) && !members(pair).has(secondId));
    const both = pairs.filter(pair =>
      members(pair).has(first.id) && members(pair).has(secondId));
    const secondOnly = pairs.filter(pair =>
      members(pair).has(secondId) && !members(pair).has(first.id));
    return { firstOnly, both, secondOnly };
  }, [pairs, dimension, first.id, secondId, single]);

  const dots = React.useMemo(() => {
    if (single || !regions || secondId === undefined || secondColor === undefined) {
      // Pack only the pairs the drawn set actually contains, so a measured pair
      // no simulator scored stays out of the simulator circle.
      const members = (pair: CoveragePair): ReadonlySet<string> =>
        dimension === 'simulators' ? pair.simulators : pair.sources;
      return packRegion(
        layout,
        'shared',
        pairs.filter(pair => members(pair).has(shown.id)),
        shown.color,
      );
    }
    return [
      ...packRegion(layout, 'firstOnly', regions.firstOnly, first.color),
      ...packRegion(layout, 'shared', regions.both, sharedColor(first.color, secondColor)),
      ...packRegion(layout, 'secondOnly', regions.secondOnly, secondColor),
    ];
  }, [single, regions, layout, pairs, dimension, shown.id, shown.color, first.color, secondId, secondColor]);

  const description = single
    ? `${shown.label} covers ${formatCount(shown.count)} measured model and GPU pairs`
      + (empty ? `; ${empty.label} covers none in this view.` : '.')
    : `${first.label} covers ${formatCount(first.count)} measured model and GPU pairs; `
      + `${secondLabel} covers ${formatCount(secondCount)}; `
      + `${formatCount(shared)} are covered by both.`;

  return (
    <figure className={styles.vennFigure}>
      <figcaption className={styles.vennTitle}>{title}</figcaption>
      <div
        className={styles.vennCanvas}
        style={{ aspectRatio: `${VENN_WIDTH} / ${VENN_HEIGHT}` }}
        role="group"
        aria-label={description}
      >
        <VennDrawing layout={layout} first={shown} second={single ? null : second ?? null} />
        {dots.map(({ dot, pair, color }) => (
          <PointDot
            key={`${dot.region}:${pair.key}`}
            layout={layout}
            dot={dot}
            color={color}
            pair={pair}
            selectable={selectable(pair)}
            onSelect={onSelect}
            onHover={onHover}
          />
        ))}
      </div>
      <ul className={styles.vennLegend}>
        <LegendItem color={first.color} label={first.label} count={first.count} />
        {second && <LegendItem color={second.color} label={second.label} count={second.count} />}
        {!single && second && (
          <li>
            <span
              className={styles.vennSharedSwatch}
              style={{ '--first': first.color, '--second': second.color } as React.CSSProperties}
              aria-hidden="true"
            />
            Shared
            {' '}
            <span className={styles.vennCount}>{formatCount(shared)}</span>
          </li>
        )}
      </ul>
      {empty && (
        <p className={styles.vennNote}>
          No {empty.label} model + GPU pair in this view.
        </p>
      )}
      {note && <p className={styles.vennNote}>{note}</p>}
    </figure>
  );
}

interface VennSet {
  id: string;
  label: string;
  color: string;
  count: number;
}

function VennDrawing({
  layout,
  first,
  second,
}: {
  layout: VennLayout;
  first: VennSet;
  /** Null when the layout shows one set, so only one circle is drawn. */
  second: VennSet | null;
}) {
  return (
    <svg
      className={styles.vennSvg}
      viewBox={`0 0 ${layout.width} ${layout.height}`}
      aria-hidden="true"
      focusable="false"
    >
      <circle
        cx={layout.first.cx}
        cy={layout.first.cy}
        r={layout.first.r}
        fill={first.color}
        fillOpacity={0.12}
        stroke={first.color}
        strokeWidth={2}
      />
      {second && (
        <circle
          cx={layout.second.cx}
          cy={layout.second.cy}
          r={layout.second.r}
          fill={second.color}
          fillOpacity={0.14}
          stroke={second.color}
          strokeWidth={2}
        />
      )}
    </svg>
  );
}

function PointDot({
  layout,
  dot,
  color,
  pair,
  selectable,
  onSelect,
  onHover,
}: {
  layout: VennLayout;
  dot: { x: number; y: number; region: VennRegion };
  color: string;
  pair: CoveragePair;
  selectable: boolean;
  onSelect: (pair: CoveragePair) => void;
  onHover: (pair: CoveragePair | null) => void;
}) {
  const position = {
    left: `${(dot.x / layout.width) * 100}%`,
    top: `${(dot.y / layout.height) * 100}%`,
    '--dot-color': color,
  } as React.CSSProperties;
  const label = `${pair.model}, ${pair.accelerator}`;

  if (!selectable) {
    return (
      <span
        className={`${styles.vennDot} ${styles.vennDotStatic}`}
        style={position}
        title={`${label} · not scored`}
        aria-hidden="true"
      />
    );
  }

  return (
    <button
      type="button"
      className={styles.vennDot}
      style={position}
      aria-label={label}
      title={label}
      onClick={() => onSelect(pair)}
      onMouseEnter={() => onHover(pair)}
      onMouseLeave={() => onHover(null)}
      onFocus={() => onHover(pair)}
      onBlur={() => onHover(null)}
    />
  );
}

function LegendItem({ color, label, count }: { color: string; label: string; count: number }) {
  return (
    <li>
      <span className={styles.vennSwatch} style={{ background: color }} aria-hidden="true" />
      {label}
      {' '}
      <span className={styles.vennCount}>{formatCount(count)}</span>
    </li>
  );
}

function CoverageTable({
  pairs,
  simulators,
  sources,
  selectable,
  onSelect,
}: {
  pairs: readonly CoveragePair[];
  simulators: readonly CoverageSimulator[];
  sources: readonly ScorecardSource[];
  selectable: (pair: CoveragePair) => boolean;
  onSelect: (pair: CoveragePair) => void;
}) {
  const ordered = React.useMemo(() => [...pairs].sort((left, right) =>
    right.simulators.size - left.simulators.size
    || right.sources.size - left.sources.size
    || left.model.localeCompare(right.model)
    || left.accelerator.localeCompare(right.accelerator)), [pairs]);

  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <caption className={styles.tableCaption}>
          {formatCount(ordered.length)} measured model + GPU pairs
        </caption>
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">GPU</th>
            {simulators.map(entry => (
              <th key={entry.id} scope="col" className={styles.markCell}>{entry.label}</th>
            ))}
            {sources.map(entry => (
              <th key={entry} scope="col" className={styles.markCell}>{DATASET_LABELS[entry]}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ordered.map(pair => (
            <tr key={pair.key}>
              <td className={styles.modelCell}>
                {selectable(pair) ? (
                  <button
                    type="button"
                    className={styles.pairButton}
                    aria-label={`Show ${pair.model} on ${pair.accelerator}`}
                    onClick={() => onSelect(pair)}
                  >
                    {pair.model}
                  </button>
                ) : (
                  <span className={styles.mutedCell}>{pair.model}</span>
                )}
              </td>
              <td className={styles.workloadCell}>{pair.accelerator}</td>
              {simulators.map(entry => (
                <td key={entry.id} className={styles.markCell}>
                  <Mark
                    covered={pair.simulators.has(entry.id)}
                    label={`${entry.label} scored`}
                    color={entry.color}
                  />
                </td>
              ))}
              {sources.map(entry => (
                <td key={entry} className={styles.markCell}>
                  <Mark
                    covered={pair.sources.has(entry)}
                    label={`${DATASET_LABELS[entry]} measured`}
                    color={DATASET_COLORS[entry]}
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Mark({ covered, label, color }: { covered: boolean; label: string; color: string }) {
  return (
    <span
      className={covered ? styles.markOn : styles.markOff}
      style={covered ? { '--mark-color': color } as React.CSSProperties : undefined}
      role="img"
      aria-label={`${label}: ${covered ? 'yes' : 'no'}`}
    />
  );
}

function packRegion(
  layout: VennLayout,
  region: VennRegion,
  pairs: readonly CoveragePair[],
  color: string,
): { dot: { x: number; y: number; region: VennRegion }; pair: CoveragePair; color: string }[] {
  const dots = placeVennDots(layout, region, pairs.length);
  return dots.flatMap((dot, index) => {
    const pair = pairs[index];
    return pair ? [{ dot, pair, color }] : [];
  });
}

function sharedColor(first: string, second: string): string {
  return `linear-gradient(135deg, ${first} 0 50%, ${second} 50% 100%)`;
}

function sharedCount<T extends string>(
  pairs: readonly CoveragePair[],
  ids: readonly T[],
  members: (pair: CoveragePair) => ReadonlySet<T>,
): number {
  if (ids.length === 0) return 0;
  return pairs.filter(pair => ids.every(id => members(pair).has(id))).length;
}

function formatCount(value: number): string {
  return value.toLocaleString('en-US');
}

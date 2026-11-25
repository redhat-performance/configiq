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
  layoutVenn,
  placeVennDots,
  type VennLayout,
  type VennRegion,
} from './coverage-geometry';
import styles from './scorecard.module.css';

/**
 * Coverage map: one dot per measured model + GPU pair.
 *
 * Two two-set diagrams answer two different questions. The simulator circles
 * show which pairs each simulator could score, so the area is a coverage
 * comparison. The dataset circles show which pairs each measured source holds.
 * A shared pair is not a shared workload, so the dataset figure states the
 * shared-workload count separately rather than letting the lens imply a
 * like-for-like comparison.
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
  const simulatorShared = sharedCount(
    sets.pairs,
    simulatorSets.map(entry => entry.id),
    pair => pair.simulators,
  );
  const sourceSets = sources.map(entry => ({
    id: entry,
    label: DATASET_LABELS[entry],
    color: DATASET_COLORS[entry],
    count: sets.pairs.filter(pair => pair.sources.has(entry)).length,
  }));
  const sourceShared = sharedCount(sets.pairs, sources, pair => pair.sources);

  const showSimulatorVenn = simulatorSets.length === 2
    && simulatorSets.every(entry => entry.count > 0);
  const showSourceVenn = sourceSets.length === 2
    && sourceSets.every(entry => entry.count > 0);

  return (
    <div className={styles.coverageMap}>
      <div className={styles.coverageFigures}>
        {showSimulatorVenn && (
          <VennFigure
            title="Simulator coverage"
            first={simulatorSets[0]}
            second={simulatorSets[1]}
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

        {showSourceVenn && (
          <VennFigure
            title="Measured-data coverage"
            first={sourceSets[0]}
            second={sourceSets[1]}
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
  first,
  second,
  shared,
  pairs,
  dimension,
  selectable,
  onSelect,
  onHover,
  note,
}: {
  title: string;
  first: VennSet;
  second: VennSet;
  shared: number;
  pairs: readonly CoveragePair[];
  dimension: 'simulators' | 'sources';
  selectable: (pair: CoveragePair) => boolean;
  onSelect: (pair: CoveragePair) => void;
  onHover: (pair: CoveragePair | null) => void;
  note?: string;
}) {
  const layout = React.useMemo(
    () => layoutVenn({
      first: first.count,
      second: second.count,
      shared,
      width: VENN_WIDTH,
      height: VENN_HEIGHT,
    }),
    [first.count, second.count, shared],
  );

  const regions = React.useMemo(() => {
    const members = (pair: CoveragePair): ReadonlySet<string> =>
      dimension === 'simulators' ? pair.simulators : pair.sources;
    const firstOnly = pairs.filter(pair =>
      members(pair).has(first.id) && !members(pair).has(second.id));
    const both = pairs.filter(pair =>
      members(pair).has(first.id) && members(pair).has(second.id));
    const secondOnly = pairs.filter(pair =>
      members(pair).has(second.id) && !members(pair).has(first.id));
    return { firstOnly, both, secondOnly };
  }, [pairs, dimension, first.id, second.id]);

  const dots = React.useMemo(() => [
    ...packRegion(layout, 'firstOnly', regions.firstOnly, first.color),
    ...packRegion(layout, 'shared', regions.both, sharedColor(first.color, second.color)),
    ...packRegion(layout, 'secondOnly', regions.secondOnly, second.color),
  ], [layout, regions, first.color, second.color]);

  const description = `${first.label} covers ${formatCount(first.count)} measured model and GPU pairs; `
    + `${second.label} covers ${formatCount(second.count)}; `
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
        <VennDrawing layout={layout} first={first} second={second} />
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
        <LegendItem color={second.color} label={second.label} count={second.count} />
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
      </ul>
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
  second: VennSet;
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
      <circle
        cx={layout.second.cx}
        cy={layout.second.cy}
        r={layout.second.r}
        fill={second.color}
        fillOpacity={0.14}
        stroke={second.color}
        strokeWidth={2}
      />
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

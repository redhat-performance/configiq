'use client';

import * as React from 'react';
import {
  Alert,
  Button,
  ExpandableSection,
  FormSelect,
  FormSelectOption,
  PageSection,
  Spinner,
  ToggleGroup,
  ToggleGroupItem,
  Tooltip,
} from '@patternfly/react-core';
import Link from 'next/link';
import { ComboBox, type ComboBoxItem } from '@/components/ModelComboBox/ModelComboBox';
import { GpuSystemInput } from '@/components/ui/GpuSystemInput';
import { CopyLinkButton } from '@/components/help/CopyLinkButton';
import { TourButton } from '@/components/help/TourButton';
import { JargonPopover } from '@/components/ui/JargonPopover';
import { useCatalog, type GpuOption } from '@/lib/hooks/useCatalog';
import { useScorecard } from '@/lib/hooks/useScorecard';
import { defineAccessors } from '@/lib/deep-links/shared-view';
import { useSharedView } from '@/lib/deep-links/useSharedView';
import { scorecardLink } from '@/lib/deep-links/pages';
import { findCatalogGpu } from '@/lib/validation/gpu-catalog';
import { predictionColor } from '@/lib/ui/simulator-colors';
import {
  SCORECARD_SOURCE_FILTERS,
  type ScorecardCell,
  type ScorecardMetric,
  type ScorecardReleaseMatch,
  type ScorecardSourceFilter,
  type ScorecardStatistic,
  type ScorecardSummary,
} from '@/lib/scorecard/types';
import ErrorBarChart from './ErrorBarChart';
import HeadToHeadChart from './HeadToHeadChart';
import {
  buildChartRows,
  buildFilterOptions,
  buildHeadToHeadPairs,
  buildHeadToHeadSummary,
  buildMatrix,
  emptyReason,
  formatPercent,
  formatSignedPercent,
  selectHeadToHeadRows,
  selectTotals,
  sumCoverage,
  type ChartRow,
  type CoverageCounts,
  type HeadToHeadPairOption,
  type HeadToHeadSummary,
  type MatrixRow,
} from './scorecard-view';
import styles from './scorecard.module.css';

/**
 * Simulator scorecard.
 *
 * Leads with agreement: for every measured run, how close does each simulator's
 * prediction come? The headline number is a median APE over measured
 * configurations, because a mean is dominated by the handful of extreme misses
 * the tail column already reports.
 *
 * The page is Settings-independent on purpose. A scorecard answers "which
 * simulator should I trust", which is not a question the reader's current
 * settings should narrow.
 */

const METRICS: { value: ScorecardMetric; label: string }[] = [
  { value: 'ttft', label: 'TTFT' },
  { value: 'tpot', label: 'TPOT' },
];

const STATISTICS: { value: ScorecardStatistic; label: string; hint: string }[] = [
  { value: 'mape', label: 'MAPE', hint: 'Mean absolute percentage error. Every configuration counts once.' },
  { value: 'mdape', label: 'MdAPE', hint: 'Median absolute percentage error. Less moved by a single bad configuration.' },
  { value: 'p90', label: 'p90', hint: 'The error a bad configuration reaches: 90% of configurations are better than this.' },
  { value: 'bias', label: 'Bias', hint: 'Signed error. Negative means the simulator predicts faster than the measurement.' },
];

const SOURCE_LABELS: Record<ScorecardSourceFilter, string> = {
  all: 'All sources',
  redhat: 'Red Hat',
  inferencex: 'InferenceX',
};

const DEFAULT_SOURCE: ScorecardSourceFilter = 'all';
const DEFAULT_METRIC: ScorecardMetric = 'tpot';
const DEFAULT_STATISTIC: ScorecardStatistic = 'mdape';
const DEFAULT_RELEASE_MATCH: ScorecardReleaseMatch = 'any';

/** The card reserves one fixed box; both faces scroll inside it. */
const CARD_HEIGHT = 560;

function formatCount(value: number): string {
  return value.toLocaleString('en-US');
}

function formatSnapshotTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

function metricLabel(metric: ScorecardMetric): string {
  return metric === 'ttft' ? 'time to first token' : 'time per output token';
}

export default function SimulatorScorecard() {
  const { summary, notPublished, error, isLoading, reload } = useScorecard();
  // Only the GPU catalog is read from the gateway: it supplies the vendor and
  // architecture groups and the memory line for a measured accelerator. It is
  // not the reader's Settings, so the scorecard stays Settings-independent.
  const { gpuOptions: catalogGpus } = useCatalog();
  const [mounted, setMounted] = React.useState(false);
  const [metric, setMetric] = React.useState<ScorecardMetric>(DEFAULT_METRIC);
  const [statistic, setStatistic] = React.useState<ScorecardStatistic>(DEFAULT_STATISTIC);
  const [releaseMatch, setReleaseMatch] = React.useState<ScorecardReleaseMatch>(DEFAULT_RELEASE_MATCH);
  const [source, setSource] = React.useState<ScorecardSourceFilter>(DEFAULT_SOURCE);
  const [model, setModel] = React.useState('');
  const [gpu, setGpu] = React.useState('');
  const [workload, setWorkload] = React.useState('');
  // A pair key, not the pair itself: the snapshot decides which pairs exist,
  // and a key that disappears falls back to the first pair in the new slice.
  const [pairKey, setPairKey] = React.useState<string | null>(null);

  React.useEffect(() => setMounted(true), []);

  const accessors = defineAccessors(scorecardLink, {
    metric: { read: () => metric, write: value => setMetric(value) },
    statistic: { read: () => statistic, write: value => setStatistic(value) },
    releaseMatch: { read: () => releaseMatch, write: value => setReleaseMatch(value) },
    source: { read: () => source, write: value => setSource(value) },
    model: { read: () => (model || undefined), write: value => setModel(value) },
    gpu: { read: () => (gpu || undefined), write: value => setGpu(value) },
    workload: { read: () => (workload || undefined), write: value => setWorkload(value) },
  });

  const shared = useSharedView(scorecardLink, {
    fields: accessors,
    ready: mounted,
    defaults: {
      metric: DEFAULT_METRIC,
      statistic: DEFAULT_STATISTIC,
      releaseMatch: DEFAULT_RELEASE_MATCH,
      source: DEFAULT_SOURCE,
    },
  });

  const hasMatched = React.useMemo(
    () => summary?.totals.some(total => total.releaseMatch === 'matched') ?? false,
    [summary],
  );

  // The release dimension only becomes a choice once something matched. Before
  // that, the matched slice is empty and selecting it would show a blank page
  // for a reason the reader did not ask about.
  const effectiveReleaseMatch: ScorecardReleaseMatch = releaseMatch === 'matched' && !hasMatched
    ? 'any'
    : releaseMatch;

  const filterOptions = React.useMemo(
    () => (summary
      ? buildFilterOptions(summary, {
        metric,
        statistic,
        releaseMatch: effectiveReleaseMatch,
        source,
      })
      : { models: [], accelerators: [], workloads: [] }),
    [summary, metric, statistic, effectiveReleaseMatch, source],
  );

  // Switching source or statistic can remove the model, GPU, or workload the
  // reader picked. Drop it rather than leaving a filter that matches nothing.
  // Wait for the snapshot first: while it loads there is nothing to compare a
  // selection against, and a link's filters would be cleared before it arrived.
  React.useEffect(() => {
    if (!summary) return;
    if (model && !filterOptions.models.includes(model)) setModel('');
    if (gpu && !filterOptions.accelerators.includes(gpu)) setGpu('');
    if (workload && !filterOptions.workloads.includes(workload)) setWorkload('');
  }, [summary, filterOptions, model, gpu, workload]);

  const modelItems = React.useMemo<ComboBoxItem[]>(
    () => filterOptions.models.map(entry => ({
      value: entry,
      label: entry,
      group: entry.includes('/') ? entry.slice(0, entry.indexOf('/')) : 'Models',
    })),
    [filterOptions.models],
  );

  /**
   * The measured accelerators, dressed in the catalog's GPU shapes.
   *
   * The option keeps the measured label as both its value (`systemId`) and its
   * display text: that label is what the matrix filters on and what a shared
   * link carries, so showing the catalog's longer name would make the chosen
   * control disagree with the table column. Catalog metadata still supplies
   * the vendor/architecture group and the memory line. A part the catalog does
   * not know remains selectable, because the snapshot measured it.
   */
  const gpuInputOptions = React.useMemo<GpuOption[]>(
    () => filterOptions.accelerators.map(accelerator => {
      const catalog = findCatalogGpu(accelerator, catalogGpus);
      if (catalog) return { ...catalog, systemId: accelerator, label: accelerator };
      return {
        systemId: accelerator,
        label: accelerator,
        vendor: null,
        architecture: null,
        vramGb: null,
        bandwidthTbps: null,
        tflopsBf16: null,
        tdpWatts: null,
        gpusPerNode: null,
      };
    }),
    [filterOptions.accelerators, catalogGpus],
  );

  // A row filter narrows the matrix and the chart. The published snapshot has
  // no per-model, per-GPU, or per-workload totals, so the tiles stay pooled.
  const rowFilterActive = Boolean(model || gpu || workload);

  const simulators = React.useMemo(() => {
    if (!summary) return [];
    return summary.simulators
      .map((simulator, index) => {
        const totals = selectTotals(summary, {
          simulator: simulator.id,
          source,
          metric,
          releaseMatch: effectiveReleaseMatch,
        });
        const coverage = sumCoverage(summary.coverage, {
          simulator: simulator.id,
          source,
          releaseMatch: effectiveReleaseMatch,
        });
        return { simulator, index, totals, coverage };
      })
      .sort((left, right) => {
        if (!left.totals && !right.totals) return left.index - right.index;
        if (!left.totals) return 1;
        if (!right.totals) return -1;
        return left.totals.values.mdape - right.totals.values.mdape || left.index - right.index;
      });
  }, [summary, source, metric, effectiveReleaseMatch]);

  const matrix = React.useMemo(
    () => (summary
      ? buildMatrix(summary, { metric, statistic, releaseMatch: effectiveReleaseMatch, source, model, gpu, workload })
      : []),
    [summary, metric, statistic, effectiveReleaseMatch, source, model, gpu, workload],
  );

  const headToHeadPairs = React.useMemo(
    () => (summary ? buildHeadToHeadPairs(summary.headToHead, { metric, source }) : []),
    [summary, metric, source],
  );

  const activePair = React.useMemo<HeadToHeadPairOption | null>(() => {
    if (headToHeadPairs.length === 0) return null;
    return headToHeadPairs.find(entry => entry.key === pairKey) ?? headToHeadPairs[0];
  }, [headToHeadPairs, pairKey]);

  const headToHeadRows = React.useMemo(
    () => (summary && activePair
      ? selectHeadToHeadRows(summary.headToHead, { metric, source, pair: activePair.pair })
      : []),
    [summary, metric, source, activePair],
  );

  const headToHeadByModel = React.useMemo(
    () => buildHeadToHeadSummary(headToHeadRows, {
      metric,
      source,
      pair: activePair?.pair ?? null,
      dimension: 'model',
    }),
    [headToHeadRows, metric, source, activePair],
  );

  const headToHeadByGpu = React.useMemo(
    () => buildHeadToHeadSummary(headToHeadRows, {
      metric,
      source,
      pair: activePair?.pair ?? null,
      dimension: 'accelerator',
    }),
    [headToHeadRows, metric, source, activePair],
  );

  const linkedSimulators = React.useMemo(
    () => (summary ? summary.simulators.filter(simulator => simulator.available) : []),
    [summary],
  );

  const chartRows = React.useMemo(
    () => buildChartRows(matrix, linkedSimulators.map(simulator => simulator.id)),
    [matrix, linkedSimulators],
  );

  const simulatorColor = React.useCallback(
    (id: string) => {
      const index = summary?.simulators.findIndex(simulator => simulator.id === id) ?? 0;
      return predictionColor(id, Math.max(0, index));
    },
    [summary],
  );

  const simulatorLabel = React.useCallback(
    (id: string) => summary?.simulators.find(simulator => simulator.id === id)?.label ?? id,
    [summary],
  );

  const formatStatistic = statistic === 'bias' ? formatSignedPercent : formatPercent;
  const statisticLabel = STATISTICS.find(entry => entry.value === statistic)?.label ?? statistic;

  return (
    <PageSection className={styles.page}>
      <div className={styles.heading}>
        <div>
          <p className={styles.eyebrow}>Validation</p>
          <h1>Simulator scorecard</h1>
          <p className={styles.intro}>
            How closely each simulator&apos;s prediction agrees with a measurement of the
            same run, across every benchmarked model and GPU. A lower number is closer.
          </p>
        </div>
        <div className={styles.headingAside}>
          <div className={styles.headingActions}>
            <CopyLinkButton getUrl={() => shared.build()} />
            <TourButton />
          </div>
          {summary && <SourceBadges summary={summary} />}
        </div>
      </div>

      {shared.warnings.length > 0 && (
        <Alert variant="warning" isInline title="Part of this link was ignored" className={styles.notice}>
          <ul className={styles.warningList}>
            {shared.warnings.map(warning => <li key={warning}>{warning}</li>)}
          </ul>
        </Alert>
      )}

      {isLoading && (
        <Alert variant="info" isInline title="Loading the published scorecard" className={styles.notice}>
          <span className={styles.loadingCopy}><Spinner size="sm" /> Reading the latest snapshot.</span>
        </Alert>
      )}

      {error && !isLoading && (
        <Alert variant="danger" isInline title="The scorecard could not be loaded" className={styles.notice}>
          <p>{error}</p>
          <Button variant="link" onClick={reload}>Try again</Button>
        </Alert>
      )}

      {notPublished && !isLoading && (
        <Alert variant="info" isInline title="No scorecard snapshot is published yet" className={styles.notice}>
          <p>
            The sweep publishes one snapshot per day. Until the first run completes there is
            nothing to show here.
          </p>
        </Alert>
      )}

      {summary && (
        <>
          <Provenance summary={summary} />

          <section className={styles.controls} aria-label="Scorecard filters" data-tour="filters">
            <div className={styles.controlGroup}>
              <span className={styles.controlLabel}>Metric</span>
              <ToggleGroup aria-label="Metric">
                {METRICS.map(entry => (
                  <ToggleGroupItem
                    key={entry.value}
                    text={entry.label}
                    isSelected={metric === entry.value}
                    onChange={() => setMetric(entry.value)}
                  />
                ))}
              </ToggleGroup>
            </div>

            <div className={styles.controlGroup}>
              <span className={styles.controlLabel}>
                Statistic
                <JargonPopover
                  term="Statistic"
                  explanation={STATISTICS.map(entry => `${entry.label}: ${entry.hint}`).join(' ')}
                />
              </span>
              <FormSelect
                value={statistic}
                onChange={(_event, value) => setStatistic(value as ScorecardStatistic)}
                aria-label="Statistic"
              >
                {STATISTICS.map(entry => (
                  <FormSelectOption key={entry.value} value={entry.value} label={entry.label} />
                ))}
              </FormSelect>
            </div>

            <div className={styles.controlGroup}>
              <span className={styles.controlLabel}>
                Releases
                <JargonPopover
                  term="Releases"
                  explanation={hasMatched
                    ? 'Like for release keeps only runs measured on the engine release the simulator models, which is the fair comparison. All releases includes runs on other releases, where part of the difference is the release gap rather than the model.'
                    : 'No run matches the release this simulator models, so only the mixed comparison is available. Every configuration is scored, but a large part of the error can be the release gap rather than the model.'}
                />
              </span>
              <ToggleGroup aria-label="Release comparison">
                <ToggleGroupItem
                  text="Like for release"
                  isSelected={effectiveReleaseMatch === 'matched'}
                  isDisabled={!hasMatched}
                  onChange={() => setReleaseMatch('matched')}
                />
                <ToggleGroupItem
                  text="All releases"
                  isSelected={effectiveReleaseMatch === 'any'}
                  onChange={() => setReleaseMatch('any')}
                />
              </ToggleGroup>
            </div>

            <div className={styles.controlGroup}>
              <span className={styles.controlLabel}>Benchmark data</span>
              <FormSelect
                value={source}
                onChange={(_event, value) => setSource(value as ScorecardSourceFilter)}
                aria-label="Benchmark data"
              >
                {SCORECARD_SOURCE_FILTERS.map(entry => (
                  <FormSelectOption key={entry} value={entry} label={SOURCE_LABELS[entry]} />
                ))}
              </FormSelect>
            </div>

            <span className={styles.controlBreak} aria-hidden="true" />
            <span className={styles.filterRowLabel}>Filter rows</span>

            <div className={`${styles.controlGroup} ${styles.filterGroup}`}>
              <ComboBox
                id="scorecard-model"
                label="Model"
                value={model}
                onChange={value => setModel(value)}
                items={modelItems}
                allLabel="All models"
                placeholder="Search measured models..."
              />
            </div>

            <div className={`${styles.controlGroup} ${styles.filterGroup}`}>
              <GpuSystemInput
                id="scorecard-gpu"
                value={gpu}
                onChange={value => setGpu(value)}
                gpuOptions={gpuInputOptions}
                allLabel="All GPUs"
                emptyLabel="No measured GPU in this slice"
              />
            </div>

            <div className={`${styles.controlGroup} ${styles.filterGroup}`}>
              <span className={styles.controlLabel}>Workload</span>
              <FormSelect
                value={workload}
                onChange={(_event, value) => setWorkload(String(value))}
                aria-label="Workload"
                className={styles.filterSelect}
              >
                <FormSelectOption value="" label="All workloads" />
                {filterOptions.workloads.map(entry => (
                  <FormSelectOption key={entry} value={entry} label={entry} />
                ))}
              </FormSelect>
            </div>

            {rowFilterActive && (
              <Button
                variant="link"
                isInline
                className={styles.clearFilters}
                onClick={() => {
                  setModel('');
                  setGpu('');
                  setWorkload('');
                }}
              >
                Clear filters
              </Button>
            )}
          </section>

          <section aria-labelledby="scorecard-headline-title" data-tour="agreement">
            <div className={styles.sectionHead}>
              <h2 id="scorecard-headline-title">Agreement by simulator</h2>
              <JargonPopover
                term="Agreement and coverage"
                explanation={[
                  `${metricLabel(metric)}, pooled over every measured configuration.`,
                  effectiveReleaseMatch === 'matched'
                    ? 'Restricted to runs measured on the release the simulator models.'
                    : 'Measured on any release, so a large part of the error can be the release gap rather than the model.',
                  'The headline is MdAPE, the median configuration, so a few extreme misses do not decide the ranking. MAPE is shown beside it: a wide gap between the two means those extremes, not the typical run, drive the mean. p90 shows what a bad configuration reaches.',
                  'Engine settings the request does not carry, such as precision and speculative decoding, are not part of the comparison either.',
                  'Coverage counts every attempt once, in four buckets.',
                  'Predicted: the simulator returned a prediction, so the run is scored.',
                  'Unsupported: the catalog lists the parts, but the run is beyond what the simulator models.',
                  'Inconclusive: we asked and got no prediction — a timeout, a rate limit, or a service failure. That is our outage, not a capability gap.',
                  'Ineligible: the run is outside the catalogue entirely — no such GPU, model, or backend — so it says nothing about the hardware.',
                ].join(' ')}
              />
            </div>

            {rowFilterActive && (
              <p className={styles.filterNote}>
                These totals cover every configuration in the snapshot. The model, GPU, and
                workload choices narrow the chart and the table below.
              </p>
            )}

            <div className={styles.tiles}>
              {simulators.map(({ simulator, index, totals, coverage }) => (
                <article key={simulator.id} className={styles.tile}>
                  <header className={styles.tileHead}>
                    <span
                      className={styles.swatch}
                      style={{ background: predictionColor(simulator.id, index) }}
                      aria-hidden="true"
                    />
                    <h3>{simulator.label}</h3>
                    <span className={styles.tileVersion}>
                      {simulator.version ? `${simulator.id} ${simulator.version}` : simulator.id}
                    </span>
                  </header>

                  <div className={styles.tileValueRow}>
                    <span className={styles.tileValue}>{formatPercent(totals?.values.mdape)}</span>
                    <span className={styles.tileValueLabel}>MdAPE</span>
                  </div>

                  {totals ? (
                    <>
                      <dl className={styles.tileStats}>
                        <div>
                          <dt>MAPE</dt>
                          <dd>{formatPercent(totals.values.mape)}</dd>
                        </div>
                        <div>
                          <dt>p90</dt>
                          <dd>{formatPercent(totals.values.p90)}</dd>
                        </div>
                        <div>
                          <dt>Bias</dt>
                          <dd>{formatSignedPercent(totals.values.bias)}</dd>
                        </div>
                      </dl>
                      <p className={styles.tileFoot}>
                        {formatCount(totals.configurations)} configurations
                        {' · '}
                        {formatCount(totals.points)} measured levels
                      </p>
                    </>
                  ) : (
                    <p className={styles.tileEmpty}>{emptyReason(coverage)}</p>
                  )}

                  <CoverageBar coverage={coverage} />
                </article>
              ))}
            </div>
          </section>

          <section aria-label="What this measures, and what it does not">
            <ExpandableSection
              toggleText="What this measures, and what it does not"
              className={styles.methodSection}
              isIndented
            >
              <div className={styles.method}>
                <ul>
                  <li>
                    Each point is a published measurement and the prediction for the same
                    configuration: model, GPU, input and output length, tensor parallel size,
                    concurrency, and engine release.
                  </li>
                  <li>
                    The measured row does not record pipeline or data parallel size, so these
                    predictions use {summary.assumptions.ppSize} and {summary.assumptions.dpSize}
                    {' '}respectively. Expert-parallel widths come from the simulator.
                  </li>
                  <li>
                    Statistics are computed per configuration, then across configurations, so a
                    run swept at many concurrency levels does not dominate one swept at few.
                  </li>
                  <li>
                    This is agreement with a measurement under these inputs, not an absolute
                    accuracy claim. A user&apos;s own parallelism and quantisation will move the
                    prediction.
                  </li>
                  <li>
                    Coverage counts every attempt. A point is <strong>unsupported</strong> when the
                    catalogue says the run is beyond the simulator, <strong>inconclusive</strong> when
                    the call never returned a prediction, and <strong>ineligible</strong> when the run
                    is outside the simulator&apos;s catalogue entirely.
                  </li>
                </ul>
                <p className={styles.methodLinks}>
                  <Link href="/validation/compare">Compare one configuration in detail</Link>
                  {' · '}
                  <Link href="/docs/validation-scorecard">How the scorecard is built</Link>
                </p>
              </div>
            </ExpandableSection>
          </section>

          <section aria-labelledby="scorecard-matrix-title" data-tour="error">
            <div className={styles.sectionHead}>
              <h2 id="scorecard-matrix-title">Where the error sits</h2>
              <JargonPopover
                term="Where the error sits"
                explanation={`One bar per measured configuration, grouped by simulator, showing each simulator's ${statisticLabel.toLowerCase()} for that run. The table behind the flip lists every configuration; a dash means that simulator returned no prediction for the run.`}
              />
            </div>

            <MatrixFlipCard
              matrix={matrix}
              simulators={linkedSimulators.map(simulator => ({ id: simulator.id, label: simulator.label }))}
              chartRows={chartRows}
              color={simulatorColor}
              statistic={statistic}
              statisticLabel={statisticLabel}
              formatStatistic={formatStatistic}
              filtered={rowFilterActive}
            />
          </section>

          {activePair && (
            <section aria-labelledby="scorecard-paired-title" data-tour="paired">
              <div className={styles.sectionHead}>
                <h2 id="scorecard-paired-title">Head to head</h2>
                <JargonPopover
                  term="Head to head"
                  explanation="Only the measured levels both simulators predicted, so neither can look better by skipping the hard ones. Best is the closer prediction at that level. Each bar is the mean best share of a model's or a GPU's measured configurations, so one long sweep cannot outvote one short run. The bracket spans the lowest and highest configuration share; a category with one configuration has no range and says so. Ties count for neither simulator. This paired view spans every release; the Releases choice narrows the error statistics, not these pairs."
                />
                {headToHeadPairs.length > 1 && (
                  <div className={styles.sectionControl}>
                    <span className={styles.controlLabel}>Pair</span>
                    <FormSelect
                      value={activePair.key}
                      onChange={(_event, value) => setPairKey(String(value))}
                      aria-label="Simulator pair"
                      className={styles.pairSelect}
                    >
                      {headToHeadPairs.map(entry => (
                        <FormSelectOption
                          key={entry.key}
                          value={entry.key}
                          label={`${simulatorLabel(entry.pair[0])} vs ${simulatorLabel(entry.pair[1])}`}
                        />
                      ))}
                    </FormSelect>
                  </div>
                )}
              </div>
              <HeadToHeadFlipCard
                rows={headToHeadRows}
                byModel={headToHeadByModel}
                byGpu={headToHeadByGpu}
                pair={activePair.pair}
                color={simulatorColor}
                label={simulatorLabel}
                formatGap={gap => formatSignedPercent(gap / 100)}
                source={source}
              />
            </section>
          )}

        </>
      )}
    </PageSection>
  );
}

/**
 * Chart on the front, table on the back.
 *
 * The chart ranks by magnitude so the configurations that dominate the mean
 * appear first; the table still lists every row, including the ones the chart
 * leaves out. This is the Benchmark comparison flip-tile pattern, so the dense
 * table is available without being the default reading.
 */
function MatrixFlipCard({
  matrix,
  simulators,
  chartRows,
  color,
  statistic,
  statisticLabel,
  formatStatistic,
  filtered,
}: {
  matrix: MatrixRow[];
  simulators: { id: string; label: string }[];
  chartRows: ChartRow[];
  color: (id: string) => string;
  statistic: ScorecardStatistic;
  statisticLabel: string;
  formatStatistic: (value: number | null) => string;
  filtered: boolean;
}) {
  const [flipped, setFlipped] = React.useState(false);
  const frontRef = React.useRef<HTMLDivElement>(null);
  const backRef = React.useRef<HTMLDivElement>(null);
  const tableId = 'scorecard-error-table';
  const series = simulators.map(entry => ({
    simulator: entry.id,
    label: entry.label,
    color: color(entry.id),
  }));
  // Faces are absolutely positioned, so the card reserves a fixed height and
  // each face scrolls its own content inside the same box.
  const cardHeight = CARD_HEIGHT;

  React.useEffect(() => {
    // Only the visible face may receive focus or be read by assistive tech.
    if (frontRef.current) frontRef.current.inert = flipped;
    if (backRef.current) backRef.current.inert = !flipped;
  }, [flipped]);

  return (
    <div
      className={`${styles.flipCard} ${flipped ? styles.flipped : ''}`}
      data-flip-card="scorecard-error"
      style={{ height: cardHeight }}
      onClick={event => {
        const target = event.target as HTMLElement;
        // Keep tables and their controls interactive without flipping back.
        if (target.closest('table, button, a, input, select, textarea')) return;
        setFlipped(value => !value);
      }}
    >
      <div ref={frontRef} className={`${styles.flipFace} ${styles.flipFront}`} aria-hidden={flipped}>
        <div className={styles.chartHead}>
          <div>
            <h3>{statisticLabel} by measured configuration</h3>
            <p className={styles.chartHint}>
              {formatCount(chartRows.length)} {chartRows.length === 1 ? 'configuration' : 'configurations'} with a prediction, worst first
            </p>
          </div>
          <span className={styles.chartUnit}>Lower is closer</span>
        </div>
        <ul className={styles.chartLegend} aria-label="Chart legend">
          {series.map(entry => (
            <li key={entry.simulator}>
              <span className={styles.chartSwatch} style={{ background: entry.color }} aria-hidden="true" />
              {entry.label}
            </li>
          ))}
        </ul>
        {chartRows.length > 0 ? (
          <ErrorBarChart
            rows={chartRows}
            series={series}
            statisticLabel={statisticLabel}
            formatValue={formatStatistic}
          />
        ) : (
          <p className={styles.emptyRows}>No configuration in this view has a prediction to compare.</p>
        )}
        <button
          type="button"
          className={styles.flipButton}
          aria-expanded={flipped}
          aria-controls={tableId}
          onClick={() => setFlipped(true)}
        >
          <span aria-hidden="true">↻ </span>table
        </button>
      </div>

      <div ref={backRef} id={tableId} className={`${styles.flipFace} ${styles.flipBack}`} aria-hidden={!flipped}>
        <div className={styles.backHead}>
          <div>
            <h3>{statisticLabel} by measured configuration</h3>
            <p className={styles.chartHint}>Every configuration in this view</p>
          </div>
          <button
            type="button"
            className={styles.flipButton}
            aria-expanded={flipped}
            aria-controls={tableId}
            onClick={() => setFlipped(false)}
          >
            <span aria-hidden="true">↻ </span>chart
          </button>
        </div>
        <div className={`${styles.tableWrap} ${styles.tableWrapFill}`}>
          <table className={styles.table}>
            <caption className={styles.tableCaption}>
              {formatCount(matrix.length)} measured configurations
              {filtered ? ' · filtered' : ''}
              <span className={styles.scrollHint}> · scroll sideways for all columns</span>
            </caption>
            <thead>
              <tr>
                <th scope="col">Source</th>
                <th scope="col">Model</th>
                <th scope="col">GPU</th>
                <th scope="col">Workload</th>
                {simulators.map(simulator => (
                  <th key={simulator.id} scope="col" className={styles.numericCell}>
                    {simulator.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map(row => (
                <tr key={row.key}>
                  <td className={styles.mutedCell}>{row.source}</td>
                  <td className={styles.modelCell}>{row.model}</td>
                  <td>{row.accelerator}</td>
                  <td className={styles.workloadCell}>{row.workload}</td>
                  {simulators.map(simulator => {
                    const cell = row.values.get(simulator.id);
                    return (
                      <td key={simulator.id} className={styles.numericCell}>
                        {cell ? (
                          <Tooltip content={`${formatCount(cell.configurations)} configurations · ${formatCount(cell.points)} measured levels`}>
                            <span>
                              {statistic === 'bias' ? formatSignedPercent(cell.value) : formatPercent(cell.value)}
                            </span>
                          </Tooltip>
                        ) : (
                          <span className={styles.mutedCell}>—</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          {matrix.length === 0 && (
            <p className={styles.emptyRows}>
              No configuration in this view has a prediction to compare.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function Provenance({ summary }: { summary: ScorecardSummary }) {
  const attempted = summary.attemptedPoints;
  const answered = summary.answeredPoints;
  return (
    <section className={styles.provenance} aria-label="Snapshot provenance">
      <div className={styles.provenanceItem}>
        <span className={styles.provenanceLabel}>Snapshot</span>
        <span>{formatSnapshotTime(summary.generatedAt)}</span>
      </div>
      <div className={styles.provenanceItem}>
        <span className={styles.provenanceLabel}>Attempts</span>
        <span>
          {formatCount(answered)} answered of {formatCount(attempted)}
          {summary.cachedPoints > 0 ? ` · ${formatCount(summary.cachedPoints)} reused from cache` : ''}
        </span>
      </div>
    </section>
  );
}

/**
 * Measured-data provenance, presented the way Benchmark comparison does it:
 * one badge per source, naming the dataset, its size, and the revision scored.
 */
function SourceBadges({ summary }: { summary: ScorecardSummary }) {
  return (
    <div className={styles.sourceBadges} data-tour="sources">
      {summary.datasets.map(dataset => {
        const inferencex = dataset.source === 'inferencex';
        const revision = dataset.revision && dataset.revision.length > 0
          ? inferencex
            ? `snapshot ${new Date(dataset.revision).toLocaleDateString(undefined, { dateStyle: 'medium' })}`
            : `revision ${dataset.revision.slice(0, 7)}`
          : null;
        return (
          <div key={dataset.source} className={styles.sourceBadge}>
            {inferencex ? (
              <span>Benchmark data from SemiAnalysis InferenceX</span>
            ) : (
              <a href={`https://huggingface.co/datasets/${dataset.id}`} target="_blank" rel="noreferrer">
                Benchmark data from Red Hat · Hugging Face
              </a>
            )}
            <span className={styles.sourceDetail}>
              {formatCount(dataset.rowCount)} rows
              {inferencex ? ' · 8k/1k' : ''}
              {revision ? ` · ${revision}` : ''}
            </span>
          </div>
        );
      })}
    </div>
  );
}

const COVERAGE_SEGMENTS: {
  key: keyof CoverageCounts;
  label: string;
  color: string;
  hint: string;
}[] = [
  {
    key: 'supported',
    label: 'Predicted',
    color: 'var(--success)',
    hint: 'The simulator returned a prediction, so the run is scored.',
  },
  {
    key: 'unsupported',
    label: 'Unsupported',
    color: 'var(--warn)',
    hint: 'The catalog lists the parts, but the run is beyond what this simulator models.',
  },
  {
    key: 'inconclusive',
    label: 'Inconclusive',
    color: 'var(--danger)',
    hint: 'We asked and got no prediction: a timeout, a rate limit, or a service failure. Our outage, not a capability gap.',
  },
  {
    key: 'ineligible',
    label: 'Ineligible',
    color: 'var(--border-strong, #b8bbbe)',
    hint: 'The run is outside the catalogue entirely — no such GPU, model, or backend — so it says nothing about the hardware.',
  },
];

function CoverageBar({ coverage }: { coverage: CoverageCounts }) {
  const total = coverage.total;
  return (
    <div className={styles.coverage}>
      <div
        className={styles.coverageTrack}
        role="img"
        aria-label={COVERAGE_SEGMENTS
          .map(segment => `${coverage[segment.key]} ${segment.label.toLowerCase()}`)
          .join(', ')}
      >
        {COVERAGE_SEGMENTS.map(segment => {
          const count = coverage[segment.key];
          if (count === 0 || total === 0) return null;
          return (
            <span
              key={segment.key}
              className={styles.coverageSegment}
              style={{ width: `${(count / total) * 100}%`, background: segment.color }}
            />
          );
        })}
      </div>
      <ul className={styles.coverageLegend}>
        {COVERAGE_SEGMENTS.map(segment => (
          <li key={segment.key} title={segment.hint}>
            <span className={styles.legendSwatch} style={{ background: segment.color }} aria-hidden="true" />
            {segment.label}
            {' '}
            <span className={styles.legendCount}>
              {formatCount(coverage[segment.key])}
              {total > 0 ? ` (${Math.round((coverage[segment.key] / total) * 100)}%)` : ''}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Head-to-head chart on the front, every paired configuration on the back.
 *
 * The front answers "where does one simulator have the edge, and how firm is
 * it"; the table answers "which measured configurations made that number". The
 * card sizes itself from the chart, so two panels with different row counts do
 * not need a hand-computed height.
 */
function HeadToHeadFlipCard({
  rows,
  byModel,
  byGpu,
  pair,
  color,
  label,
  formatGap,
  source,
}: {
  rows: ScorecardSummary['headToHead'];
  byModel: HeadToHeadSummary;
  byGpu: HeadToHeadSummary;
  pair: [string, string];
  color: (id: string) => string;
  label: (id: string) => string;
  formatGap: (pct: number) => string;
  source: ScorecardSourceFilter;
}) {
  const [flipped, setFlipped] = React.useState(false);
  const frontRef = React.useRef<HTMLDivElement>(null);
  const backRef = React.useRef<HTMLDivElement>(null);
  const tableId = 'scorecard-paired-table';
  const first = { id: pair[0], label: label(pair[0]), color: color(pair[0]) };
  const second = { id: pair[1], label: label(pair[1]), color: color(pair[1]) };

  React.useEffect(() => {
    // Only the visible face may receive focus or be read by assistive tech.
    if (frontRef.current) frontRef.current.inert = flipped;
    if (backRef.current) backRef.current.inert = !flipped;
  }, [flipped]);

  return (
    <div
      className={`${styles.flipCard} ${styles.flipCardAuto} ${flipped ? styles.flipped : ''}`}
      data-flip-card="scorecard-paired"
      onClick={event => {
        const target = event.target as HTMLElement;
        // Keep tables and their controls interactive without flipping back.
        if (target.closest('table, button, a, input, select, textarea')) return;
        setFlipped(value => !value);
      }}
    >
      <div ref={frontRef} className={`${styles.flipFace} ${styles.flipFront}`} aria-hidden={flipped}>
        <div className={styles.chartHead}>
          <div>
            <h3>Best share by model and GPU</h3>
            <p className={styles.chartHint}>
              Only levels both simulators predicted; each category is the mean of its measured
              configurations.
            </p>
          </div>
          <span className={styles.chartUnit}>Higher is closer</span>
        </div>
        <ul className={styles.chartLegend} aria-label="Chart legend">
          <li>
            <span className={styles.chartSwatch} style={{ background: first.color }} aria-hidden="true" />
            {first.label}
          </li>
          <li>
            <span className={styles.chartSwatch} style={{ background: second.color }} aria-hidden="true" />
            {second.label}
          </li>
          <li>
            <span className={styles.candleSwatch} aria-hidden="true" />
            Configuration range
          </li>
        </ul>
        <div className={styles.h2hPanels}>
          <HeadToHeadChart
            title="Best share by model"
            summary={byModel}
            first={first}
            second={second}
            formatGap={formatGap}
          />
          <HeadToHeadChart
            title="Best share by GPU"
            summary={byGpu}
            first={first}
            second={second}
            formatGap={formatGap}
          />
        </div>
        <button
          type="button"
          className={styles.flipButton}
          aria-expanded={flipped}
          aria-controls={tableId}
          onClick={() => setFlipped(true)}
        >
          <span aria-hidden="true">↻ </span>table
        </button>
      </div>

      <div ref={backRef} id={tableId} className={`${styles.flipFace} ${styles.flipBack}`} aria-hidden={!flipped}>
        <div className={styles.backHead}>
          <div>
            <h3>Every paired configuration</h3>
            <p className={styles.chartHint}>
              {first.label} vs {second.label}
              {source === 'all' ? '' : ` · ${SOURCE_LABELS[source]}`}
            </p>
          </div>
          <button
            type="button"
            className={styles.flipButton}
            aria-expanded={flipped}
            aria-controls={tableId}
            onClick={() => setFlipped(false)}
          >
            <span aria-hidden="true">↻ </span>chart
          </button>
        </div>
        <div className={`${styles.tableWrap} ${styles.tableWrapFill}`}>
          <table className={`${styles.table} ${styles.h2hTable}`}>
            <caption className={styles.tableCaption}>
              {formatCount(rows.length)} measured {rows.length === 1 ? 'configuration' : 'configurations'}
              <span className={styles.scrollHint}> · scroll sideways for all columns</span>
            </caption>
            <thead>
              <tr>
                <th scope="col">Source</th>
                <th scope="col">Model</th>
                <th scope="col">GPU</th>
                <th scope="col">Workload</th>
                <th scope="col">Outcome</th>
                <th scope="col" className={styles.numericCell}>{first.label} best</th>
                <th scope="col" className={styles.numericCell}>{second.label} best</th>
                <th scope="col" className={styles.numericCell}>Ties</th>
                <th scope="col" className={styles.numericCell}>Median gap</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(row => (
                <tr key={[row.source, row.model, row.accelerator, row.workload, row.metric, ...row.pair].join('|')}>
                  <td className={styles.mutedCell}>{row.source === 'inferencex' ? 'InferenceX' : 'Red Hat'}</td>
                  <td className={styles.modelCell}>{row.model}</td>
                  <td>{row.accelerator}</td>
                  <td className={styles.workloadCell}>{row.workload}</td>
                  <td>
                    <OutcomeBar
                      first={first}
                      second={second}
                      firstWins={row.firstWins}
                      secondWins={row.secondWins}
                      ties={row.ties}
                    />
                  </td>
                  <td className={styles.numericCell}>{formatCount(row.firstWins)}</td>
                  <td className={styles.numericCell}>{formatCount(row.secondWins)}</td>
                  <td className={styles.numericCell}>{formatCount(row.ties)}</td>
                  <td className={styles.numericCell}>{formatSignedPercent(row.medianDeltaPct / 100)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length === 0 && (
            <p className={styles.emptyRows}>No paired configuration in this view.</p>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Share of paired levels where each simulator was best.
 *
 * The counts are already in adjacent columns for exact reading; the bar is the
 * at-a-glance form, colored with the same positional series colors as the
 * scorecard tiles and chart. Ties stay unfilled rather than being awarded to
 * either simulator.
 */
function OutcomeBar({
  first,
  second,
  firstWins,
  secondWins,
  ties,
}: {
  first: { id: string; label: string; color: string };
  second: { id: string; label: string; color: string };
  firstWins: number;
  secondWins: number;
  ties: number;
}) {
  const total = firstWins + secondWins + ties;
  if (total === 0) return <span className={styles.mutedCell}>—</span>;
  const firstShare = (firstWins / total) * 100;
  const secondShare = (secondWins / total) * 100;

  return (
    <div
      className={styles.outcomeBar}
      role="img"
      aria-label={[
        `${first.label} best on ${formatCount(firstWins)} of ${formatCount(total)} paired levels`,
        `${second.label} best on ${formatCount(secondWins)}`,
        ties > 0 ? `${formatCount(ties)} tied` : null,
      ].filter(Boolean).join('; ')}
    >
      {firstWins > 0 && (
        <span
          className={styles.outcomeSegment}
          style={{ width: `${firstShare}%`, background: first.color }}
        />
      )}
      {secondWins > 0 && (
        <span
          className={styles.outcomeSegment}
          style={{ width: `${secondShare}%`, background: second.color }}
        />
      )}
    </div>
  );
}

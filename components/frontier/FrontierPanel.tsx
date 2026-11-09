'use client';

import * as React from 'react';
import {
  Button,
  Card,
  CardBody,
  CardTitle,
  EmptyState,
  EmptyStateBody,
  Label,
  Spinner,
  ToggleGroup,
  ToggleGroupItem,
} from '@patternfly/react-core';
import {
  Chart,
  ChartAxis,
  ChartLine,
  ChartScatter,
  ChartVoronoiContainer,
} from '@patternfly/react-charts/victory';
import CheckCircleIcon from '@patternfly/react-icons/dist/esm/icons/check-circle-icon';
import ExclamationTriangleIcon from '@patternfly/react-icons/dist/esm/icons/exclamation-triangle-icon';
import { computeFrontier } from '@/lib/pareto/frontier';
import { formatCandidateId } from '@/lib/pareto/format';
import { describeSweepCoverage } from '@/lib/pareto/sweep-caption';
import { topologySummary, xpydLabel } from '@/lib/pareto/recipe';
import type { Candidate, FrontierState } from '@/lib/types/pareto';

import styles from './FrontierPanel.module.css';

interface FrontierPanelProps {
  state: FrontierState;
  /** Display label of the GPU system the sweep was submitted for. */
  gpuLabel: string;
  /** Re-run only the Pareto sweep. */
  onRetry: () => void;
  /** Seconds elapsed since the sweep started, while it is loading. */
  elapsed?: number;
  /** Gateway timeout the sweep must finish inside, when known. */
  timeoutSeconds?: number | null;
}

type AxisMode = 'throughput_per_user' | 'ttft';

// Categorical palette shared with the former /design reference. Frontier and
// feasible pass the dataviz contrast checks on a light surface.
const C_FRONTIER = '#0066cc';
const C_FEASIBLE = '#009596';
const C_SELECTED = '#f0ab00';

const FEASIBLE_STATUSES = new Set(['feasible']);

function throughputPerUser(candidate: Candidate): number | null {
  const value = candidate.objectives?.throughput_per_user;
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function throughputPerGpu(candidate: Candidate): number | null {
  const value = candidate.objectives?.throughput_per_gpu;
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function ttftOf(candidate: Candidate): number | null {
  const value = candidate.metrics.mean_ttft_ms ?? candidate.metrics.ttft_ms;
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function tpotOf(candidate: Candidate): number | null {
  const value = candidate.metrics.mean_tpot_ms ?? candidate.metrics.tpot_ms;
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function requestLatencyOf(candidate: Candidate): number | null {
  const value = candidate.metrics.mean_e2e_latency_ms ?? candidate.metrics.request_latency_ms;
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function isFeasible(candidate: Candidate): boolean {
  return FEASIBLE_STATUSES.has(candidate.status);
}

/**
 * TTFT is minimized while throughput per GPU is maximized, which is the
 * opposite orientation from the server's (throughput per user, throughput per
 * GPU) Pareto view. The server view stays authoritative for its own axes; this
 * recomputation is only used when the user flips the x-axis to TTFT.
 */
function computeTtftFrontier(candidates: Candidate[]): Set<string> {
  const ids = new Set<string>();
  for (const a of candidates) {
    const ax = ttftOf(a);
    const ay = throughputPerGpu(a);
    if (ax == null || ay == null) continue;
    const dominated = candidates.some(b => {
      if (b === a) return false;
      const bx = ttftOf(b);
      const by = throughputPerGpu(b);
      if (bx == null || by == null) return false;
      return bx <= ax && by >= ay && (bx < ax || by > ay);
    });
    if (!dominated) ids.add(a.candidate_id);
  }
  return ids;
}

function fmt(n: number | null, digits = 1): string {
  if (n == null) return '—';
  return n.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtThousands(n: number | null): string {
  if (n == null) return '—';
  if (n >= 10000) return `${Math.round(n / 1000)}k`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return `${Math.round(n)}`;
}

function candidateLabel(candidate: Candidate): string {
  const role = xpydLabel(candidate) ?? (candidate.mode === 'disagg' ? 'disagg' : 'agg');
  return `${role} · ${candidate.used_gpus ?? '?'} GPU`;
}

function useContainerWidth(): [React.RefObject<HTMLDivElement>, number] {
  const ref = React.useRef<HTMLDivElement>(null);
  const [width, setWidth] = React.useState(640);
  React.useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const observer = new ResizeObserver(entries => {
      const next = entries[0]?.contentRect.width;
      if (next && next > 0) setWidth(Math.floor(next));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className={styles.legendItem}>
      <span className={styles.legendDot} style={{ background: color }} />
      <span>{label}</span>
    </span>
  );
}

interface FrontierChartProps {
  candidates: Candidate[];
  frontierIds: Set<string>;
  axisMode: AxisMode;
  selectedId: string | null;
  onSelect: (id: string) => void;
}

interface ChartPoint {
  x: number;
  y: number;
  candidate: Candidate;
  name: string;
}

function FrontierChart({ candidates, frontierIds, axisMode, selectedId, onSelect }: FrontierChartProps) {
  const [wrapRef, width] = useContainerWidth();

  // The chart shows only candidates that actually produced a measurement.
  // Failed and timed-out attempts stay in the table behind a disclosure.
  const withObjectives = candidates.filter(c => c.objectives !== null && isFeasible(c));

  const xOf = (candidate: Candidate): number | null =>
    axisMode === 'ttft' ? ttftOf(candidate) : throughputPerUser(candidate);

  const toPoint = (candidate: Candidate): ChartPoint | null => {
    const x = xOf(candidate);
    const y = throughputPerGpu(candidate);
    if (x == null || y == null) return null;
    const xLabel = axisMode === 'ttft' ? `${fmt(x, 0)} ms TTFT` : `${fmt(x, 0)} tok/s/user`;
    return {
      x,
      y,
      candidate,
      name: `${candidateLabel(candidate)} · ${xLabel} · ${fmtThousands(y)} tok/s/GPU`,
    };
  };

  const feasiblePoints = withObjectives.map(toPoint).filter((p): p is ChartPoint => p !== null);
  const frontierPoints = feasiblePoints
    .filter(point => frontierIds.has(point.candidate.candidate_id))
    .sort((a, b) => a.x - b.x);
  const feasibleOff = feasiblePoints.filter(point => !frontierIds.has(point.candidate.candidate_id));

  const xs = feasiblePoints.map(point => point.x);
  const ys = feasiblePoints.map(point => point.y);
  const xMin = xs.length ? Math.min(...xs) : 0;
  const xMax = xs.length ? Math.max(...xs) : 1;
  const yMin = ys.length ? Math.min(...ys) : 0;
  const yMax = ys.length ? Math.max(...ys) : 1;
  const xPad = Math.max((xMax - xMin) * 0.08, axisMode === 'ttft' ? 10 : 1);
  const yPad = Math.max((yMax - yMin) * 0.1, 1);
  const domain = {
    x: [Math.max(0, xMin - xPad), xMax + xPad] as [number, number],
    y: [Math.max(0, yMin - yPad), yMax + yPad] as [number, number],
  };

  const handlePointClick = (_: unknown, props: { datum?: ChartPoint }): string[] => {
    if (props.datum) onSelect(props.datum.candidate.candidate_id);
    return [];
  };

  return (
    <div ref={wrapRef} className={styles.chartWrap}>
      <Chart
        containerComponent={
          <ChartVoronoiContainer
            labels={({ datum }) => (datum as ChartPoint).name ?? ''}
            constrainToVisibleArea
          />
        }
        domain={domain}
        height={340}
        padding={{ top: 20, bottom: 56, left: 76, right: 24 }}
        width={width}
      >
        <ChartAxis
          label={axisMode === 'ttft' ? 'TTFT (ms, lower is better)' : 'Throughput per user (tok/s/user)'}
          style={{
            axisLabel: { fontSize: 12, padding: 38, fill: '#3c3f42' },
            tickLabels: { fontSize: 11.5, fill: '#3c3f42' },
            grid: { stroke: '#f0f0f0' },
          }}
        />
        <ChartAxis
          dependentAxis
          label="Throughput per GPU (tok/s/GPU)"
          tickFormat={(t: number) => fmtThousands(t)}
          style={{
            axisLabel: { fontSize: 12, padding: 56, fill: '#3c3f42' },
            tickLabels: { fontSize: 11.5, fill: '#3c3f42' },
            grid: { stroke: '#f0f0f0' },
          }}
        />

        {frontierPoints.length > 1 && (
          <ChartLine
            data={frontierPoints}
            interpolation="stepAfter"
            style={{ data: { stroke: C_FRONTIER, strokeWidth: 2, opacity: 0.35 } }}
          />
        )}

        {feasibleOff.length > 0 && (
          <ChartScatter
            data={feasibleOff}
            size={6}
            style={{ data: { fill: C_FEASIBLE, stroke: '#fff', strokeWidth: 1 } }}
            events={[{ target: 'data', eventHandlers: { onClick: handlePointClick } }]}
          />
        )}

        {frontierPoints.length > 0 && (
          <ChartScatter
            data={frontierPoints}
            size={(args: { datum?: ChartPoint }) =>
              selectedId === args.datum?.candidate.candidate_id ? 10 : 8
            }
            style={{
              data: {
                fill: (args: { datum?: ChartPoint }) =>
                  selectedId === args.datum?.candidate.candidate_id ? C_SELECTED : C_FRONTIER,
                stroke: '#fff',
                strokeWidth: 2,
              },
            }}
            events={[{ target: 'data', eventHandlers: { onClick: handlePointClick } }]}
          />
        )}
      </Chart>

      <div className={styles.legend}>
        <LegendDot color={C_FRONTIER} label="Pareto frontier" />
        <LegendDot color={C_FEASIBLE} label="Feasible" />
        <span className={styles.legendItem}>
          <span className={styles.legendSelected} style={{ background: C_SELECTED }} />
          <span>Selected</span>
        </span>
      </div>
    </div>
  );
}

interface CandidateTableProps {
  candidates: Candidate[];
  frontierIds: Set<string>;
  selectedId: string | null;
  onSelect: (id: string) => void;
  caption: string;
}

function CandidateTable({ candidates, frontierIds, selectedId, onSelect, caption }: CandidateTableProps) {
  const sorted = React.useMemo(
    () =>
      [...candidates].sort((a, b) => {
        const feasibleDiff = Number(!isFeasible(a)) - Number(!isFeasible(b));
        if (feasibleDiff !== 0) return feasibleDiff;
        const gpuDiff = (a.used_gpus ?? Number.MAX_SAFE_INTEGER) - (b.used_gpus ?? Number.MAX_SAFE_INTEGER);
        if (gpuDiff !== 0) return gpuDiff;
        return a.candidate_id.localeCompare(b.candidate_id);
      }),
    [candidates],
  );

  if (sorted.length === 0) return null;

  return (
    <div className={styles.tableScroll}>
      <table className={styles.table}>
        <caption className={styles.tableCaption}>{caption}</caption>
        <thead>
          <tr>
            <th scope="col">Candidate</th>
            <th scope="col">Status</th>
            <th scope="col" className={styles.numeric}>GPUs</th>
            <th scope="col" className={styles.numeric}>tok/s/user</th>
            <th scope="col" className={styles.numeric}>tok/s/GPU</th>
            <th scope="col" className={styles.numeric}>TTFT</th>
            <th scope="col" className={styles.numeric}>TPOT</th>
            <th scope="col">Topology</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(candidate => {
            const selected = candidate.candidate_id === selectedId;
            const onFrontier = frontierIds.has(candidate.candidate_id);
            return (
              <tr
                key={candidate.candidate_id}
                className={selected ? styles.rowSelected : undefined}
                onClick={() => onSelect(candidate.candidate_id)}
              >
                <td>
                  <button
                    type="button"
                    className={styles.rowButton}
                    aria-pressed={selected}
                    onClick={event => {
                      event.stopPropagation();
                      onSelect(candidate.candidate_id);
                    }}
                  >
                    {formatCandidateId(candidate.candidate_id)}
                    {onFrontier && <Label isCompact color="blue" className={styles.frontierLabel}>frontier</Label>}
                  </button>
                </td>
                <td>
                  {isFeasible(candidate) ? (
                    <Label isCompact color="green" icon={<CheckCircleIcon />}>feasible</Label>
                  ) : (
                    <Label isCompact color="grey" icon={<ExclamationTriangleIcon />}>{candidate.status}</Label>
                  )}
                </td>
                <td className={styles.numeric}>{candidate.used_gpus ?? '—'}</td>
                <td className={styles.numeric}>{fmt(throughputPerUser(candidate), 1)}</td>
                <td className={styles.numeric}>{fmtThousands(throughputPerGpu(candidate))}</td>
                <td className={styles.numeric}>{fmt(ttftOf(candidate), 1)}</td>
                <td className={styles.numeric}>{fmt(tpotOf(candidate), 2)}</td>
                <td>{topologySummary(candidate)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function CandidateDetail({ candidate }: { candidate: Candidate | null }) {
  if (!candidate) {
    return (
      <p className={styles.detailEmpty}>
        Select a candidate in the chart or table to see its measured metrics and topology.
      </p>
    );
  }
  return (
    <dl className={styles.detail}>
      <div>
        <dt>Candidate</dt>
        <dd>{formatCandidateId(candidate.candidate_id)}</dd>
      </div>
      <div>
        <dt>Topology</dt>
        <dd>{topologySummary(candidate)}</dd>
      </div>
      <div>
        <dt>GPUs</dt>
        <dd>{candidate.used_gpus ?? '—'}</dd>
      </div>
      <div>
        <dt>Throughput / user</dt>
        <dd>{fmt(throughputPerUser(candidate), 2)} tok/s</dd>
      </div>
      <div>
        <dt>Throughput / GPU</dt>
        <dd>{fmt(throughputPerGpu(candidate), 1)} tok/s</dd>
      </div>
      <div>
        <dt>TTFT</dt>
        <dd>{fmt(ttftOf(candidate), 1)} ms</dd>
      </div>
      <div>
        <dt>TPOT</dt>
        <dd>{fmt(tpotOf(candidate), 2)} ms</dd>
      </div>
      <div>
        <dt>Request latency</dt>
        <dd>{fmt(requestLatencyOf(candidate), 1)} ms</dd>
      </div>
      <div>
        <dt>P/D transfer</dt>
        <dd>{candidate.transfer_delay_modeled ? 'modeled' : 'not modeled'}</dd>
      </div>
    </dl>
  );
}

export function FrontierPanel({
  state,
  gpuLabel,
  onRetry,
  elapsed = 0,
  timeoutSeconds = null,
}: FrontierPanelProps) {
  const [axisMode, setAxisMode] = React.useState<AxisMode>('throughput_per_user');
  const [selectedId, setSelectedId] = React.useState<string | null>(null);
  const [showFailures, setShowFailures] = React.useState(false);

  const candidates = React.useMemo(
    () => (state.status === 'ready' ? state.candidates : []),
    [state],
  );
  const views = state.status === 'ready' ? state.views : null;
  const sweep = state.status === 'ready' ? state.sweep : null;

  const feasible = React.useMemo(() => candidates.filter(isFeasible), [candidates]);
  const failed = React.useMemo(() => candidates.filter(candidate => !isFeasible(candidate)), [candidates]);

  const hasTtft = React.useMemo(
    () => feasible.some(candidate => ttftOf(candidate) != null),
    [feasible],
  );

  // The server's Pareto view is authoritative for the simulator's own
  // objectives. It is recomputed only for the TTFT x-axis, where the dominance
  // direction differs.
  const frontierIds = React.useMemo(() => {
    if (axisMode === 'ttft') return computeTtftFrontier(feasible);
    // An empty server frontier is still the server's answer: no evaluated
    // candidate was non-dominated. Only a missing view falls back locally.
    if (views) return new Set(views.pareto_front);
    return new Set(computeFrontier(feasible).map(candidate => candidate.candidate_id));
  }, [axisMode, feasible, views]);

  React.useEffect(() => {
    if (selectedId != null && candidates.some(candidate => candidate.candidate_id === selectedId)) return;
    const next =
      [...frontierIds].find(id => feasible.some(candidate => candidate.candidate_id === id)) ??
      feasible[0]?.candidate_id ??
      null;
    setSelectedId(next);
  }, [candidates, feasible, frontierIds, selectedId]);

  const selected = candidates.find(candidate => candidate.candidate_id === selectedId) ?? null;

  const statusSummary = React.useMemo(
    () => describeSweepCoverage(sweep, candidates),
    [candidates, sweep],
  );

  if (state.status === 'idle') return null;

  const failuresDisclosure = failed.length > 0 && (
    <div className={styles.failuresBlock}>
      <Button
        variant="link"
        isInline
        onClick={() => setShowFailures(visible => !visible)}
        aria-expanded={showFailures}
      >
        {showFailures ? 'Hide failed attempts' : `Show failed attempts (${failed.length})`}
      </Button>
      {showFailures && (
        <CandidateTable
          candidates={failed}
          frontierIds={frontierIds}
          selectedId={selectedId}
          onSelect={setSelectedId}
          caption={`Failed attempts for ${gpuLabel}. Kept for transparency; they are not plotted.`}
        />
      )}
    </div>
  );

  return (
    <section className={styles.section} aria-labelledby="frontier-heading">
      <Card>
        <CardTitle>
          <div className={styles.titleRow}>
            <h2 id="frontier-heading" className={styles.title}>Pareto frontier</h2>
            <span className={styles.subtitle}>Efficiency trade-off across aggregated and disaggregated deployments</span>
          </div>
        </CardTitle>
        <CardBody>
          {state.status === 'loading' && (
            <div className={styles.loading}>
              <Spinner size="md" aria-label="Running Pareto sweep" />
              <div>
                <div className={styles.loadingTitle}>Evaluating the efficiency frontier…</div>
                <div className={styles.loadingHint}>
                  Testing a range of GPU counts across aggregated and disaggregated deployments.
                  {elapsed > 0 && <> <span className={styles.elapsed}>{elapsed}s elapsed.</span></>}
                  {timeoutSeconds != null && <> The gateway limit is {timeoutSeconds}s.</>}
                </div>
              </div>
            </div>
          )}

          {state.status === 'unavailable' && (
            <div className={styles.notice}>
              <ExclamationTriangleIcon className={styles.noticeIcon} />
              <div>
                <div className={styles.noticeTitle}>Frontier unavailable</div>
                <div className={styles.noticeBody}>{state.reason}</div>
              </div>
            </div>
          )}

          {state.status === 'error' && (
            <div className={styles.notice}>
              <ExclamationTriangleIcon className={styles.noticeIcon} />
              <div>
                <div className={styles.noticeTitle}>The Pareto sweep did not complete</div>
                <div className={styles.noticeBody}>{state.message}</div>
                <Button variant="secondary" size="sm" className={styles.retryButton} onClick={onRetry}>
                  Retry Pareto sweep
                </Button>
              </div>
            </div>
          )}

          {state.status === 'ready' && candidates.length === 0 && (
            <EmptyState>
              <EmptyStateBody>
                No candidate configurations were evaluated for this workload and GPU window.
              </EmptyStateBody>
            </EmptyState>
          )}

          {state.status === 'ready' && candidates.length > 0 && feasible.length === 0 && (
            <EmptyState>
              <EmptyStateBody>
                <div>{statusSummary}</div>
                <div style={{ marginTop: 8 }}>
                  No candidate configuration met the workload constraints. Try a different model, GPU system, or
                  workload context.
                </div>
                <Button variant="secondary" size="sm" className={styles.retryButton} onClick={onRetry}>
                  Retry Pareto sweep
                </Button>
              </EmptyStateBody>
            </EmptyState>
          )}

          {state.status === 'ready' && feasible.length > 0 && (
            <>
              <div className={styles.toolbar}>
                <div className={styles.axisControl}>
                  <span className={styles.axisLabel}>X-axis</span>
                  <ToggleGroup aria-label="Frontier x-axis">
                    <ToggleGroupItem
                      text="Tok/s per user"
                      isSelected={axisMode === 'throughput_per_user'}
                      onChange={() => setAxisMode('throughput_per_user')}
                    />
                    <ToggleGroupItem
                      text="TTFT"
                      isSelected={axisMode === 'ttft'}
                      isDisabled={!hasTtft}
                      onChange={() => setAxisMode('ttft')}
                    />
                  </ToggleGroup>
                </div>
                <div className={styles.sweepMeta}>
                  {statusSummary}
                  {sweep?.windowMinGpus != null && sweep?.windowMaxGpus != null && (
                    <> Window: {sweep.windowMinGpus}–{sweep.windowMaxGpus} GPUs.</>
                  )}
                  {sweep?.strictSla === false && (
                    <> No latency SLO was enforced, so candidates may exceed a target you would otherwise set.</>
                  )}
                </div>
              </div>

              {axisMode === 'ttft' && (
                <p className={styles.axisNote}>
                  TTFT is minimized, so this view recomputes the non-dominated set client-side. The simulator&apos;s
                  own Pareto view remains authoritative for throughput-per-user vs throughput-per-GPU.
                </p>
              )}

              <div className={styles.bodyGrid}>
                <div className={styles.chartColumn}>
                  <FrontierChart
                    candidates={feasible}
                    frontierIds={frontierIds}
                    axisMode={axisMode}
                    selectedId={selectedId}
                    onSelect={setSelectedId}
                  />
                </div>
                <div className={styles.detailColumn}>
                  <h3 className={styles.detailHeading}>Selected candidate</h3>
                  <CandidateDetail candidate={selected} />
                </div>
              </div>

              <CandidateTable
                candidates={feasible}
                frontierIds={frontierIds}
                selectedId={selectedId}
                onSelect={setSelectedId}
                caption={`Feasible candidates for ${gpuLabel}. Select a row to inspect it on the chart.`}
              />

              {failuresDisclosure}
            </>
          )}

          {state.status === 'ready' && candidates.length > 0 && feasible.length === 0 && failuresDisclosure}
        </CardBody>
      </Card>
    </section>
  );
}

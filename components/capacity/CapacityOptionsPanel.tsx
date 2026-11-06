'use client';

import * as React from 'react';
import { Card, CardBody, CardTitle, Label } from '@patternfly/react-core';
import {
  Chart,
  ChartAxis,
  ChartLine,
  ChartScatter,
  ChartVoronoiContainer,
} from '@patternfly/react-charts/victory';
import CheckCircleIcon from '@patternfly/react-icons/dist/esm/icons/check-circle-icon';
import ExclamationTriangleIcon from '@patternfly/react-icons/dist/esm/icons/exclamation-triangle-icon';
import type {
  CapacityOptionRole,
  RecommendCapacityOption,
  RecommendResult,
} from '@/lib/api/recommend';

import styles from './CapacityOptionsPanel.module.css';

interface CapacityOptionsPanelProps {
  options: RecommendCapacityOption[];
  selectedOptionId: string | null;
  gpuLabel: string;
  onSelect: (optionId: string) => void;
}

const ROLE_LABELS: Record<CapacityOptionRole, string> = {
  minimum: 'Fewest GPUs',
  same_gpus: 'Same GPU count',
  larger: 'Larger',
  disaggregated: 'Disaggregated',
  shortfall: 'Best effort',
};

const C_MINIMUM = '#0066cc';
const C_SAME = '#009596';
const C_LARGER = '#5752d1';
const C_DISAGG = '#a18fff';
const C_SHORTFALL = '#c58c00';
const C_SELECTED = '#f0ab00';

function optionColor(role: CapacityOptionRole): string {
  switch (role) {
    case 'minimum': return C_MINIMUM;
    case 'same_gpus': return C_SAME;
    case 'disaggregated': return C_DISAGG;
    case 'shortfall': return C_SHORTFALL;
    default: return C_LARGER;
  }
}

function performanceAvailable(option: RecommendCapacityOption): boolean {
  if (!option.capacity.performanceAvailable) return false;
  const { ttftLatencyMs } = option.result.performance;
  const { tokensPerSecondPerGpu } = option.result.throughput;
  return Number.isFinite(ttftLatencyMs) && ttftLatencyMs > 0 &&
    Number.isFinite(tokensPerSecondPerGpu) && tokensPerSecondPerGpu > 0;
}

function fmt(n: number | null | undefined, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return n.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtThousands(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—';
  if (n >= 10000) return `${Math.round(n / 1000)}k`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return `${Math.round(n)}`;
}

function roleParallelism(
  tp?: number | null,
  pp?: number | null,
  dp?: number | null,
  moeTp?: number | null,
  moeEp?: number | null,
): string {
  const parts = [`tp${tp ?? '?'}`, `pp${pp ?? '?'}`];
  if (dp && dp > 1) parts.push(`dp${dp}`);
  if ((moeTp && moeTp > 1) || (moeEp && moeEp > 1)) {
    parts.push(`moe[tp${moeTp ?? 1}·ep${moeEp ?? 1}]`);
  }
  return parts.join('·');
}

function topologyLabel(result: RecommendResult, capacityMode: 'agg' | 'disagg'): string {
  if (capacityMode === 'disagg') {
    const prefill = result.phases.prefill;
    const decode = result.phases.decode;
    if (!prefill || !decode) return 'disaggregated';
    const p = roleParallelism(
      prefill.tensorParallelSize,
      prefill.pipelineParallelSize,
      prefill.dataParallelSize,
      prefill.moeTensorParallelSize,
      prefill.moeExpertParallelSize,
    );
    const d = roleParallelism(
      decode.tensorParallelSize,
      decode.pipelineParallelSize,
      decode.dataParallelSize,
      decode.moeTensorParallelSize,
      decode.moeExpertParallelSize,
    );
    return `${prefill.workers}P:${decode.workers}D — ${prefill.workers}× prefill ${p} · ${decode.workers}× decode ${d}`;
  }
  const rec = result.recommendation;
  const agg = roleParallelism(
    rec.tensorParallelSize,
    rec.pipelineParallelSize,
    rec.dataParallelSize,
    rec.moeTensorParallelSize,
    rec.moeExpertParallelSize,
  );
  return `${rec.replicasNeeded}× ${agg}`;
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

interface ChartPoint {
  x: number;
  y: number;
  option: RecommendCapacityOption;
  name: string;
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className={styles.legendItem}>
      <span className={styles.legendDot} style={{ background: color }} />
      <span>{label}</span>
    </span>
  );
}

interface CapacityChartProps {
  options: RecommendCapacityOption[];
  selectedOptionId: string | null;
  onSelect: (optionId: string) => void;
}

function CapacityChart({ options, selectedOptionId, onSelect }: CapacityChartProps) {
  const [wrapRef, width] = useContainerWidth();
  const points = options
    .filter(performanceAvailable)
    .map(option => ({
      x: option.result.performance.ttftLatencyMs,
      y: option.result.throughput.tokensPerSecondPerGpu,
      option,
      name: `${ROLE_LABELS[option.role]} · ${option.result.recommendation.gpusNeeded} GPUs · ${fmt(option.result.performance.ttftLatencyMs, 0)} ms TTFT · ${fmtThousands(option.result.throughput.tokensPerSecondPerGpu)} tok/s/GPU`,
    }));

  if (points.length === 0) {
    return (
      <p className={styles.emptyChart}>
        No option has performance data, so there is nothing to plot. Options remain selectable in the table.
      </p>
    );
  }

  // Lower TTFT and higher per-GPU throughput dominate. Draw the non-dominated
  // set as the comparison curve; it is presentation only, not a sizing input.
  const efficient = points.filter(point =>
    !points.some(other =>
      other !== point &&
      other.x <= point.x &&
      other.y >= point.y &&
      (other.x < point.x || other.y > point.y),
    ),
  ).sort((a, b) => a.x - b.x);

  const xs = points.map(point => point.x);
  const ys = points.map(point => point.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const xPad = Math.max((xMax - xMin) * 0.1, 5);
  const yPad = Math.max((yMax - yMin) * 0.1, 1);
  const domain = {
    x: [Math.max(0, xMin - xPad), xMax + xPad] as [number, number],
    y: [Math.max(0, yMin - yPad), yMax + yPad] as [number, number],
  };

  const handlePointClick = (_: unknown, props: { datum?: ChartPoint }): string[] => {
    if (props.datum) onSelect(props.datum.option.optionId);
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
        padding={{ top: 20, bottom: 56, left: 84, right: 24 }}
        width={width}
      >
        <ChartAxis
          label="TTFT (ms, lower is better)"
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
            axisLabel: { fontSize: 12, padding: 62, fill: '#3c3f42' },
            tickLabels: { fontSize: 11.5, fill: '#3c3f42' },
            grid: { stroke: '#f0f0f0' },
          }}
        />
        {efficient.length > 1 && (
          <ChartLine
            data={efficient}
            interpolation="stepAfter"
            style={{ data: { stroke: C_MINIMUM, strokeWidth: 2, opacity: 0.3 } }}
          />
        )}
        <ChartScatter
          data={points}
          size={(args: { datum?: ChartPoint }) =>
            selectedOptionId === args.datum?.option.optionId ? 11 : 8
          }
          style={{
            data: {
              fill: (args: { datum?: ChartPoint }) =>
                selectedOptionId === args.datum?.option.optionId
                  ? C_SELECTED
                  : optionColor(args.datum?.option.role ?? 'larger'),
              stroke: '#fff',
              strokeWidth: 2,
            },
          }}
          events={[{ target: 'data', eventHandlers: { onClick: handlePointClick } }]}
        />
      </Chart>
      <div className={styles.legend}>
        <LegendDot color={C_MINIMUM} label="Fewest GPUs" />
        <LegendDot color={C_SAME} label="Same GPU count" />
        <LegendDot color={C_LARGER} label="Larger" />
        <LegendDot color={C_DISAGG} label="Disaggregated" />
        <LegendDot color={C_SHORTFALL} label="Best effort" />
        <span className={styles.legendItem}>
          <span className={styles.legendSelected} style={{ background: C_SELECTED }} />
          <span>Selected</span>
        </span>
      </div>
    </div>
  );
}

interface DetailProps {
  option: RecommendCapacityOption | null;
}

function OptionDetail({ option }: DetailProps) {
  if (!option) return null;
  return (
    <dl className={styles.detail}>
      <div>
        <dt>Option</dt>
        <dd>{ROLE_LABELS[option.role]} · rank {option.rank}</dd>
      </div>
      <div>
        <dt>Topology</dt>
        <dd>{topologyLabel(option.result, option.capacity.mode)}</dd>
      </div>
      <div>
        <dt>GPUs</dt>
        <dd>
          {option.result.recommendation.gpusNeeded}
          {option.gpusOverMinimum > 0 ? ` (+${option.gpusOverMinimum} vs fewest)` : ' (fewest)'}
        </dd>
      </div>
      <div>
        <dt>Capacity</dt>
        <dd>
          {option.capacity.capacitySequences} sequences
          {option.capacity.fits ? ' · holds target' : ' · short of target'}
        </dd>
      </div>
      <div>
        <dt>TTFT</dt>
        <dd>{performanceAvailable(option) ? `${fmt(option.result.performance.ttftLatencyMs, 1)} ms` : 'unavailable'}</dd>
      </div>
      <div>
        <dt>Throughput / GPU</dt>
        <dd>{performanceAvailable(option) ? `${fmt(option.result.throughput.tokensPerSecondPerGpu, 1)} tok/s` : 'unavailable'}</dd>
      </div>
      <div>
        <dt>Throughput / user</dt>
        <dd>{performanceAvailable(option) ? `${fmt(option.result.throughput.tokensPerSecondPerUser, 1)} tok/s` : 'unavailable'}</dd>
      </div>
    </dl>
  );
}

export function CapacityOptionsPanel({
  options,
  selectedOptionId,
  gpuLabel,
  onSelect,
}: CapacityOptionsPanelProps) {
  if (options.length === 0) return null;

  const selected = options.find(option => option.optionId === selectedOptionId) ?? options[0];

  return (
    <section className={styles.section} aria-labelledby="capacity-options-heading">
      <Card>
        <CardTitle>
          <div className={styles.titleRow}>
            <h2 id="capacity-options-heading" className={styles.title}>Capacity options</h2>
            <span className={styles.subtitle}>
              Select any option to update the result above. Fewest GPUs is ranked first.
            </span>
          </div>
        </CardTitle>
        <CardBody>
          <div className={styles.bodyGrid}>
            <div className={styles.chartColumn}>
              <CapacityChart
                options={options}
                selectedOptionId={selected.optionId}
                onSelect={onSelect}
              />
            </div>
            <div className={styles.detailColumn}>
              <h3 className={styles.detailHeading}>Selected option</h3>
              <OptionDetail option={selected} />
            </div>
          </div>

          <div className={styles.tableScroll}>
            <table className={styles.table}>
              <caption className={styles.tableCaption}>
                Capacity options for {gpuLabel}. Select a row to promote it to the result above.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Option</th>
                  <th scope="col">Role</th>
                  <th scope="col" className={styles.numeric}>GPUs</th>
                  <th scope="col" className={styles.numeric}>Over minimum</th>
                  <th scope="col" className={styles.numeric}>Capacity</th>
                  <th scope="col" className={styles.numeric}>TTFT</th>
                  <th scope="col" className={styles.numeric}>tok/s/GPU</th>
                  <th scope="col">Topology</th>
                </tr>
              </thead>
              <tbody>
                {options.map(option => {
                  const isSelected = option.optionId === selected.optionId;
                  const available = performanceAvailable(option);
                  return (
                    <tr
                      key={option.optionId}
                      className={isSelected ? styles.rowSelected : undefined}
                      onClick={() => onSelect(option.optionId)}
                    >
                      <td>
                        <button
                          type="button"
                          className={styles.rowButton}
                          aria-pressed={isSelected}
                          onClick={event => {
                            event.stopPropagation();
                            onSelect(option.optionId);
                          }}
                        >
                          {option.rank === 1 ? 'Primary' : `Option ${option.rank}`}
                          {isSelected && <Label isCompact color="blue" className={styles.selectedLabel}>selected</Label>}
                        </button>
                      </td>
                      <td>
                        <Label isCompact color={option.role === 'disaggregated' ? 'purple' : 'grey'}>
                          {ROLE_LABELS[option.role]}
                        </Label>
                      </td>
                      <td className={styles.numeric}>{option.result.recommendation.gpusNeeded}</td>
                      <td className={styles.numeric}>
                        {option.gpusOverMinimum > 0 ? `+${option.gpusOverMinimum}` : '—'}
                      </td>
                      <td className={styles.numeric}>
                        {option.capacity.capacitySequences}
                        {option.capacity.fits ? (
                          <CheckCircleIcon className={styles.okIcon} aria-label="Holds target" />
                        ) : (
                          <ExclamationTriangleIcon className={styles.warnIcon} aria-label="Short of target" />
                        )}
                      </td>
                      <td className={styles.numeric}>
                        {available ? `${fmt(option.result.performance.ttftLatencyMs, 1)} ms` : '—'}
                      </td>
                      <td className={styles.numeric}>
                        {available ? fmtThousands(option.result.throughput.tokensPerSecondPerGpu) : '—'}
                      </td>
                      <td>{topologyLabel(option.result, option.capacity.mode)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardBody>
      </Card>
    </section>
  );
}

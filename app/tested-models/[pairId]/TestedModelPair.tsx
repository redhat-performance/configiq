'use client'

import * as React from 'react'
import Link from 'next/link'
import { Alert, FormSelect, FormSelectOption, Label, Spinner } from '@patternfly/react-core'
import {
  Chart,
  ChartAxis,
  ChartLine,
  ChartScatter,
  ChartTooltip,
  ChartVoronoiContainer,
} from '@patternfly/react-charts/victory'
import { CopyLinkButton } from '@/components/help/CopyLinkButton'
import { useCatalog } from '@/lib/hooks/useCatalog'
import { defineAccessors } from '@/lib/deep-links/shared-view'
import { useSharedView } from '@/lib/deep-links/useSharedView'
import { testedModelLink, type TestedModelLinkValues } from '@/lib/deep-links/tested-models'
import {
  gpuSystemLabel,
  gpuSystemSpecs,
  modelFacetChips,
  shortRevision,
} from '@/lib/tested-models/display'
import type {
  PublicRegistryPair,
  RegistryDatasetProvenance,
  ValidationGateThresholds,
} from '@/lib/tested-models/registry-types'
import { testedModelsFilterHref } from '@/lib/tested-models/links'
import {
  comparePredictionToMeasured,
  isSameWorkload,
  listWorkloads,
  matchPredictionWorkload,
  selectDefaultWorkload,
  selectSeries,
  workloadKey,
  workloadLabel,
  type PredictionComparison,
  type PredictionSnapshot,
  type ThroughputCurve,
  type Workload,
} from '@/lib/tested-models/throughput-series'
import { formatThroughput } from '@/lib/utils/format'
import { TestedModelsShell } from '../TestedModelsShell'
import styles from '../TestedModels.module.css'

interface Dashboard {
  modelId: string
  systemId: string
  records: number
  hasEnvelope?: boolean
  trainingStatus?: string
  labelDistribution: { within: number; outside: number }
  validation?: { roc_auc?: number | null; precision?: number; recall?: number; f1?: number }
  test?: { roc_auc?: number | null; precision?: number; recall?: number; f1?: number }
  throughputCurves: ThroughputCurve[]
  kneePoints: Array<Record<string, number | null>>
  featureImportance: Array<{ feature: string; importance: number }>
}

interface TestedModelPairProps {
  pair: PublicRegistryPair
  gate: ValidationGateThresholds | null
  dataset?: RegistryDatasetProvenance | null
}

/** Distinguishable, high-contrast line colours, assigned in TP order. */
const TP_COLORS = ['#0066cc', '#3e8635', '#c9190b', '#8a4fbf', '#009596', '#a15c00']

const PREDICTION_COLORS: Record<string, string> = {
  within_validated_range: '#3e8635',
  near_validated_boundary: '#f0ab00',
  outside_validated_range: '#c9190b',
  measured_no_envelope: '#0066cc',
  no_measurements: '#8a8d90',
  validation_unavailable: '#8a8d90',
}

function metric(value: number | null | undefined): string {
  return typeof value === 'number' ? value.toFixed(3) : '—'
}

function testMetrics(pair: PublicRegistryPair): { roc_auc?: number | null; f1?: number; precision?: number; recall?: number } {
  return (pair.metrics as { test?: { roc_auc?: number | null; f1?: number; precision?: number; recall?: number } } | undefined)?.test ?? {}
}

function percent(value: number): string {
  const rounded = Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(1)
  return `${value > 0 ? '+' : ''}${rounded}%`
}

/** Track the rendered chart width so Victory matches the card. */
function useElementWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = React.useRef<T>(null)
  const [width, setWidth] = React.useState(720)
  React.useEffect(() => {
    const element = ref.current
    if (!element) return
    const update = () => setWidth(Math.max(Math.round(element.clientWidth), 240))
    update()
    if (typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(update)
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  return [ref, width]
}

type PredictionDraft = Partial<PredictionSnapshot>

function snapshotFromDraft(draft: PredictionDraft): PredictionSnapshot | null {
  const { isl, osl, concurrency, tp, throughput } = draft
  if ([isl, osl, concurrency, tp, throughput].some(value => value === undefined)) return null
  return {
    isl: isl as number,
    osl: osl as number,
    concurrency: concurrency as number,
    tp: tp as number,
    throughput: throughput as number,
    replicas: draft.replicas ?? null,
    pp: draft.pp ?? null,
    status: draft.status ?? null,
    confidence: draft.confidence ?? null,
    backend: draft.backend ?? null,
    backendVersion: draft.backendVersion ?? null,
    servingMode: draft.servingMode ?? null,
  }
}

/**
 * Say why the prediction does not line up with what was measured.
 *
 * The order is deliberate: a prediction beyond the measured concurrency range
 * is the strongest explanation, because the curve cannot show the queueing that
 * caps throughput there. Only when the point is inside the range does the size
 * of the gap become the explanation.
 */
function outsideExplanation(
  comparison: PredictionComparison,
  prediction: PredictionSnapshot,
): string {
  const at = comparison.at
  if (!at) return ''
  if (at.clamped === 'above') {
    return `Your concurrency (${prediction.concurrency.toLocaleString()}) is above the highest measured level (${at.above.concurrency.toLocaleString()}), so throughput at your concurrency was not measured. Treat the top measured point as a reference, not an upper bound: throughput can keep rising past it.`
  }
  if (at.clamped === 'below') {
    return 'Your concurrency is below the lowest measured level, so the curve has no measured point that small. The lowest measured point is a conservative reference.'
  }
  const tokenCheck = comparison.checks.find(check => check.id === 'token-budget')
  if (tokenCheck && !tokenCheck.ok) {
    return `Your predicted workload moves ${tokenCheck.detail} That is outside the token budget the curve was measured over, so the measured shape may not apply.`
  }
  if (comparison.deltaPct != null && Math.abs(comparison.deltaPct) > 15) {
    return comparison.deltaPct > 0
      ? 'The simulator predicts more throughput than the benchmark observed at this concurrency. The measurement is the safer figure: the serving stack, scheduling, and measurement conditions all sit between the model and the hardware limit.'
      : 'The simulator predicts less throughput than the benchmark observed at this concurrency. Check the model configuration and backend version before sizing against the lower number.'
  }
  if (comparison.deltaPct == null) {
    return 'No percentage comparison is available: the measured throughput at this concurrency is zero, or the rates are not on the same basis.'
  }
  return 'The prediction agrees with the measured curve within the usual tolerance for this pair.'
}

export function TestedModelPair({ pair, gate, dataset = null }: TestedModelPairProps) {
  const { gpuOptions: catalogGpus, modelSpecs } = useCatalog()
  const [dashboard, setDashboard] = React.useState<Dashboard | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [unavailable, setUnavailable] = React.useState<'missing' | 'unsupported' | null>(null)
  const [mounted, setMounted] = React.useState(false)
  const [chartRef, chartWidth] = useElementWidth<HTMLDivElement>()

  const [selectedWorkload, setSelectedWorkload] = React.useState<Workload | null>(null)
  const [focusedTp, setFocusedTp] = React.useState<number | null>(null)
  const [predictionDraft, setPredictionDraft] = React.useState<PredictionDraft>({})

  React.useEffect(() => { setMounted(true) }, [])

  React.useEffect(() => {
    fetch(`/api/tested-models/dashboard/${encodeURIComponent(pair.id)}`, { cache: 'no-store' })
      .then(response => {
        if (response.status === 409) throw new Error('unsupported')
        if (!response.ok) throw new Error('unavailable')
        return response.json() as Promise<Dashboard>
      })
      .then(setDashboard)
      .catch((error: unknown) => {
        setDashboard(null)
        setUnavailable(error instanceof Error && error.message === 'unsupported' ? 'unsupported' : 'missing')
      })
      .finally(() => setLoading(false))
  }, [pair.id])

  const curves = React.useMemo(() => dashboard?.throughputCurves ?? [], [dashboard])
  const workloads = React.useMemo(() => listWorkloads(curves), [curves])
  const defaultWorkload = React.useMemo(() => selectDefaultWorkload(curves), [curves])
  const prediction = React.useMemo(() => snapshotFromDraft(predictionDraft), [predictionDraft])
  const predictionWorkload = React.useMemo(
    () => (prediction ? matchPredictionWorkload(curves, prediction) : null),
    [curves, prediction],
  )

  // The link is applied once, after the payload settles. Until the reader
  // chooses a workload, the prediction's own workload wins over the page
  // default, so an opened prediction link lands on the comparable curve.
  const workload = selectedWorkload ?? predictionWorkload ?? defaultWorkload
  const series = React.useMemo(
    () => (workload ? selectSeries(curves, workload) : { lines: [], primarySweepId: null, mixedConfigurations: [] }),
    [curves, workload],
  )

  const comparisonSeries = React.useMemo(
    () => (predictionWorkload ? selectSeries(curves, predictionWorkload) : null),
    [curves, predictionWorkload],
  )
  const comparison = React.useMemo(
    () => (prediction && comparisonSeries ? comparePredictionToMeasured(comparisonSeries, prediction) : null),
    [comparisonSeries, prediction],
  )
  const overlayVisible = Boolean(
    prediction && predictionWorkload && workload && isSameWorkload(workload, predictionWorkload)
      && comparison?.replicaComparable !== false,
  )

  const writeWorkload = React.useCallback((patch: Partial<Workload>) => {
    setSelectedWorkload(current => {
      const base = current ?? workload
      return {
        isl: patch.isl ?? base?.isl ?? 0,
        osl: patch.osl ?? base?.osl ?? 0,
        turns: patch.turns ?? base?.turns ?? 1,
      }
    })
  }, [workload])

  const lineData = series.lines.map((line, index) => ({
    tp: line.tp,
    curve: line.curve,
    color: TP_COLORS[index % TP_COLORS.length],
    points: line.curve.points.map(point => ({
      x: point.concurrency,
      y: point.throughput,
      name: `TP ${line.tp}`,
    })),
  }))
  // A focus on a TP the new workload does not measure must not dim every line.
  const effectiveFocusedTp = lineData.some(line => line.tp === focusedTp) ? focusedTp : null

  // A shared prediction is one snapshot. Applying a link replaces the draft
  // with only the prediction fields that link carries, so a field it omits
  // cannot retain a value from a prediction previously open on this page.
  const replacePrediction = React.useCallback((values: TestedModelLinkValues) => {
    setPredictionDraft({
      isl: values.predIsl,
      osl: values.predOsl,
      concurrency: values.predConcurrency,
      tp: values.predTp,
      throughput: values.predThroughput,
      replicas: values.predReplicas,
      pp: values.predPp,
      status: values.predStatus,
      confidence: values.predConfidence,
      backend: values.predBackend,
      backendVersion: values.predBackendVersion,
      servingMode: values.predServingMode,
    })
  }, [])

  const shared = useSharedView(testedModelLink, {
    fields: defineAccessors(testedModelLink, {
      isl: { read: () => workload?.isl, write: value => writeWorkload({ isl: value }) },
      osl: { read: () => workload?.osl, write: value => writeWorkload({ osl: value }) },
      turns: { read: () => workload?.turns, write: value => writeWorkload({ turns: value }) },
      tp: { read: () => effectiveFocusedTp ?? undefined, write: value => setFocusedTp(value) },
      predIsl: { read: () => predictionDraft.isl, write: value => setPredictionDraft(draft => ({ ...draft, isl: value })) },
      predOsl: { read: () => predictionDraft.osl, write: value => setPredictionDraft(draft => ({ ...draft, osl: value })) },
      predConcurrency: { read: () => predictionDraft.concurrency, write: value => setPredictionDraft(draft => ({ ...draft, concurrency: value })) },
      predTp: { read: () => predictionDraft.tp, write: value => setPredictionDraft(draft => ({ ...draft, tp: value })) },
      predThroughput: { read: () => predictionDraft.throughput, write: value => setPredictionDraft(draft => ({ ...draft, throughput: value })) },
      predReplicas: { read: () => predictionDraft.replicas ?? undefined, write: value => setPredictionDraft(draft => ({ ...draft, replicas: value })) },
      predPp: { read: () => predictionDraft.pp ?? undefined, write: value => setPredictionDraft(draft => ({ ...draft, pp: value })) },
      predStatus: { read: () => (predictionDraft.status ?? undefined) as TestedModelLinkValues['predStatus'], write: value => setPredictionDraft(draft => ({ ...draft, status: value })) },
      predConfidence: { read: () => predictionDraft.confidence ?? undefined, write: value => setPredictionDraft(draft => ({ ...draft, confidence: value })) },
      predBackend: { read: () => (predictionDraft.backend ?? undefined) as TestedModelLinkValues['predBackend'], write: value => setPredictionDraft(draft => ({ ...draft, backend: value })) },
      predBackendVersion: { read: () => predictionDraft.backendVersion ?? undefined, write: value => setPredictionDraft(draft => ({ ...draft, backendVersion: value })) },
      predServingMode: { read: () => (predictionDraft.servingMode ?? undefined) as TestedModelLinkValues['predServingMode'], write: value => setPredictionDraft(draft => ({ ...draft, servingMode: value })) },
    }),
    ready: mounted && !loading,
    onApplied: replacePrediction,
  })

  const measuredPoints = series.lines.flatMap(line => line.curve.points)
  const measuredX = measuredPoints.map(point => point.concurrency)
  const measuredY = measuredPoints.map(point => point.throughput)
  const overlayX = overlayVisible && prediction ? prediction.concurrency : null
  const overlayY = overlayVisible && prediction ? prediction.throughput : null

  const xCandidates = [...measuredX, ...(overlayX != null ? [overlayX] : [])]
  const yCandidates = [...measuredY, ...(overlayY != null ? [overlayY] : [])]
  const xMin = xCandidates.length > 0 ? Math.min(...xCandidates) : 0
  const xMaxRaw = xCandidates.length > 0 ? Math.max(...xCandidates) : 1
  const xMax = xMin === xMaxRaw ? xMaxRaw + 1 : xMaxRaw
  const maxY = Math.max(...yCandidates, 1) * 1.12

  const predictionColor = PREDICTION_COLORS[prediction?.status ?? ''] ?? '#54585c'
  const predictionLabel = prediction?.status
    ? prediction.status.replaceAll('_', ' ')
    : 'from your prediction'

  const maxImportance = Math.max(...(dashboard?.featureImportance.map(item => item.importance) ?? [1]), 1)
  const coverage = pair.coverage
  const validated = pair.status === 'validated'
  const tests = testMetrics(pair)
  const gpuLabel = gpuSystemLabel(pair.systemId, catalogGpus)
  const gpuSpecs = gpuSystemSpecs(pair.systemId, catalogGpus)
  const modelChips = modelFacetChips(pair.modelId, modelSpecs.get(pair.modelId))

  const predictionWorkloadNote = predictionWorkload && workload
    && !isSameWorkload(workload, predictionWorkload)
    && comparison?.replicaComparable !== false ? (
    <p className={styles.panelNote}>
      The chart is showing {workloadLabel(workload)}.{' '}
      <button
        type="button"
        className={styles.inlineButton}
        onClick={() => setSelectedWorkload(predictionWorkload)}
      >
        Show {workloadLabel(predictionWorkload)}
      </button>{' '}
      to plot this prediction.
    </p>
  ) : null

  return (
    <TestedModelsShell>
      <div className={styles.header}>
        <div>
          <p className={styles.eyebrow}>
            <Link href={testedModelsFilterHref({ modelId: pair.modelId })} className={styles.backLink}>Tested performance envelopes</Link>
          </p>
          <h1 className={styles.pageTitle}>{pair.modelId}</h1>
          {modelChips.length > 0 && (
            <div className={styles.chipRow}>
              {modelChips.map(chip => <span key={chip} className={styles.chip}>{chip}</span>)}
            </div>
          )}
          <p className={styles.subtitle}>
            {gpuLabel} · measured performance envelope and classifier evidence.
            {gpuSpecs ? ` ${gpuSpecs}.` : ''}
          </p>
          {dataset && (
            <p className={styles.provenance}>
              Dataset{' '}
              <a
                href={`https://huggingface.co/datasets/${dataset.repo}/tree/${dataset.revision}`}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.inlineLink}
              >
                {dataset.repo}@{shortRevision(dataset.revision)}
              </a>
            </p>
          )}
        </div>
        <div className={styles.headerActions}>
          {validated
            ? <Label color="green" isCompact>Validated envelope</Label>
            : <Label color="blue" isCompact>Measured, no validated envelope</Label>}
          <CopyLinkButton getUrl={shared.build} />
          <Link href="/predict" className={styles.backLink}>Back to performance prediction</Link>
        </div>
      </div>

      {!validated && (
        <Alert
          isInline
          variant="info"
          title="Measurements only — no classifier verdict"
          className={styles.pairAlert}
        >
          This pair has published measurements, but no classifier passes the validation gate. The graph below is
          still available; the sizing pages will not show a verdict for this pair.
          {pair.diagnostics?.reasons.length ? (
            <ul className={styles.reasonList}>
              {pair.diagnostics.reasons.map(reason => <li key={reason}>{reason}</li>)}
            </ul>
          ) : null}
        </Alert>
      )}

      {pair.diagnostics?.warnings?.length ? (
        <Alert
          isInline
          variant="info"
          title="Data notes"
          className={styles.pairAlert}
        >
          These caveats describe the measured data, not the classifier verdict.
          <ul className={styles.reasonList}>
            {pair.diagnostics.warnings.map(warning => <li key={warning}>{warning}</li>)}
          </ul>
          {pair.diagnostics.selectionBasis === 'train_fallback' ? (
            <p className={styles.panelNote}>
              Hyperparameters were selected on the training split because the validation split contained a single class.
            </p>
          ) : null}
        </Alert>
      ) : null}

      {loading && <Spinner aria-label="Loading pair coverage" />}
      {unavailable && (
        <div className={styles.notice} role="status">
          {unavailable === 'unsupported'
            ? 'Dashboard data uses a schema this build does not support. Rebuild the tested-model bundle so the reader and payloads agree.'
            : 'Dashboard data is unavailable for this pair. Coverage metadata from the registry is shown below.'}
        </div>
      )}

      <div className={styles.summaryGrid} id="coverage">
        <section className={styles.card}>
          <span className={styles.cardLabel}>Records</span>
          <strong className={styles.cardValue}>{(coverage?.records ?? dashboard?.records ?? 0).toLocaleString()}</strong>
          <span className={styles.cardSub}>Aggregate measurements</span>
        </section>
        <section className={styles.card}>
          <span className={styles.cardLabel}>Concurrency levels</span>
          <strong className={styles.cardValue}>{coverage?.concurrencyLevelCount ?? '—'}</strong>
          <span className={styles.cardSub}>Distinct measured levels</span>
        </section>
        <section className={styles.card}>
          <span className={styles.cardLabel}>Within range</span>
          <strong className={styles.cardValue}>{dashboard ? dashboard.labelDistribution.within.toLocaleString() : '—'}</strong>
          <span className={styles.cardSub}>Labelled inside the envelope</span>
        </section>
        <section className={styles.card}>
          <span className={styles.cardLabel}>Test ROC-AUC</span>
          <strong className={styles.cardValue}>{metric(tests.roc_auc ?? dashboard?.test?.roc_auc)}</strong>
          <span className={styles.cardSub}>Held-out test split</span>
        </section>
      </div>

      <section className={styles.panel} aria-label="Validation gate">
        <div className={styles.panelHeader}>
          <h2 className={styles.panelTitle}>Validation gate</h2>
          <span className={styles.panelHint}>
            {validated ? <Label color="green" isCompact>Passed</Label> : <Label color="blue" isCompact>Not passed</Label>}
          </span>
        </div>
        <p className={styles.panelNote}>
          {gate
            ? `At least ${gate.minRecords} records, ${gate.minConcurrencyLevels} concurrency levels, test AUC ≥ ${gate.minTestAuc.toFixed(2)}, and test F1 ≥ ${gate.minTestF1.toFixed(2)}.`
            : 'This pair comes from a legacy registry. Only recorded test AUC and F1 were checked.'}
        </p>
        {coverage?.backendVersions.length ? (
          <p className={styles.panelNote}>Backend versions: {coverage.backendVersions.join(', ')}</p>
        ) : null}
      </section>

      <section className={`${styles.panel} ${styles.chartPanel}`} aria-label="Throughput by concurrency">
        <div className={styles.panelHeader}>
          <h2 className={styles.panelTitle}>Throughput by concurrency</h2>
          <span className={styles.panelHint}>
            {series.lines.length > 0 ? `${series.lines.length} tensor-parallel width${series.lines.length === 1 ? '' : 's'}` : 'Measured points'}
          </span>
        </div>

        {workloads.length > 0 && (
          <div className={styles.chartControls}>
            <label className={styles.chartControl} htmlFor="tested-workload">
              <span className={styles.chartControlLabel}>Workload</span>
              <FormSelect
                id="tested-workload"
                value={workload ? workloadKey(workload) : ''}
                onChange={(_event, value) => {
                  const next = workloads.find(option => option.key === value)
                  if (next) {
                    setSelectedWorkload({ isl: next.isl, osl: next.osl, turns: next.turns })
                  }
                }}
                aria-label="Measured workload"
              >
                {workloads.map(option => (
                  <FormSelectOption key={option.key} value={option.key} label={option.label} />
                ))}
              </FormSelect>
            </label>
            <span className={styles.chartControlNote}>
              Each line is one tensor-parallel width from the run configuration that covers the most widths.
            </span>
          </div>
        )}

        <div ref={chartRef} className={styles.chartContainer}>
          {lineData.length > 0 ? (
            <Chart
              ariaDesc="Measured output throughput as concurrency increases, one line per tensor-parallel width, with a prediction overlay when one is shared"
              ariaTitle="Throughput by concurrency"
              height={320}
              width={chartWidth}
              padding={{ top: 20, right: 30, bottom: 62, left: 74 }}
              domain={{ x: [xMin, xMax], y: [0, maxY] }}
              containerComponent={(
                <ChartVoronoiContainer
                  constrainToVisibleArea
                  labels={({ datum }: { datum: { x?: number; y?: number; name?: string } }) => {
                    if (!datum.name) return ''
                    if (datum.name === 'Your prediction') {
                      return `Your prediction\nConcurrency ${Number(datum.x).toLocaleString()}\n${formatThroughput(Number(datum.y))}`
                    }
                    return `${datum.name}\nConcurrency ${Number(datum.x).toLocaleString()}\n${formatThroughput(Number(datum.y))}`
                  }}
                  labelComponent={<ChartTooltip style={{ fontFamily: 'var(--gc-font-sans)', fontSize: 12 }} />}
                />
              )}
            >
              <ChartAxis
                label="Concurrency"
                tickFormat={(tick: number) => Number(tick).toLocaleString()}
                style={{
                  axisLabel: { padding: 40, fontSize: 12, fontFamily: 'var(--gc-font-sans)' },
                  tickLabels: { fontSize: 11.5, fontFamily: 'var(--gc-font-mono)' },
                }}
              />
              <ChartAxis
                dependentAxis
                label="Output tokens / second"
                tickFormat={(tick: number) => tick >= 1_000 ? `${Number((tick / 1_000).toFixed(1))}k` : `${Math.round(tick)}`}
                style={{
                  axisLabel: { padding: 52, fontSize: 12, fontFamily: 'var(--gc-font-sans)' },
                  grid: { stroke: '#d2d2d2', strokeDasharray: '3,4' },
                  tickLabels: { fontSize: 11.5, fontFamily: 'var(--gc-font-mono)' },
                }}
              />

              {/* Dashed guides mark the prediction's concurrency and the measured
                  throughput at that concurrency. A transparent single-tick axis
                  draws them without adding Voronoi tooltip targets. */}
              {overlayVisible && prediction && comparison?.at ? (
                <>
                  <ChartAxis
                    tickValues={[prediction.concurrency]}
                    style={{
                      axis: { stroke: 'transparent' },
                      ticks: { stroke: 'transparent' },
                      tickLabels: { fill: 'transparent' },
                      grid: { stroke: '#54585c', strokeDasharray: '4,4', strokeWidth: 1 },
                    }}
                  />
                  <ChartAxis
                    dependentAxis
                    tickValues={[comparison.at.measured]}
                    style={{
                      axis: { stroke: 'transparent' },
                      ticks: { stroke: 'transparent' },
                      tickLabels: { fill: 'transparent' },
                      grid: { stroke: '#54585c', strokeDasharray: '4,4', strokeWidth: 1 },
                    }}
                  />
                </>
              ) : null}

              {lineData.map(line => (
                <ChartLine
                  key={line.tp}
                  data={line.points}
                  interpolation="linear"
                  style={{
                    data: {
                      stroke: line.color,
                      strokeWidth: effectiveFocusedTp == null ? 2.5 : effectiveFocusedTp === line.tp ? 4 : 1.5,
                      opacity: effectiveFocusedTp == null || effectiveFocusedTp === line.tp ? 1 : 0.45,
                    },
                  }}
                />
              ))}

              {overlayVisible && prediction ? (
                <ChartScatter
                  data={[{
                    x: prediction.concurrency,
                    y: prediction.throughput,
                    name: 'Your prediction',
                  }]}
                  style={{
                    data: {
                      fill: predictionColor,
                      stroke: '#151515',
                      strokeWidth: 1.5,
                    },
                  }}
                />
              ) : null}
            </Chart>
          ) : (
            <p className={styles.panelNote}>
              {workloads.length > 0
                ? 'No measured curve exists for the selected workload.'
                : 'No throughput curve is available yet.'}
            </p>
          )}
        </div>

        {lineData.length > 0 && (
          <div className={styles.legend} aria-label="Chart legend">
            {lineData.map(line => (
              <button
                type="button"
                key={line.tp}
                className={`${styles.legendItem} ${effectiveFocusedTp === line.tp ? styles.legendItemActive : ''}`}
                aria-pressed={effectiveFocusedTp === line.tp}
                onClick={() => setFocusedTp(current => current === line.tp ? null : line.tp)}
              >
                <span className={styles.legendSwatch} style={{ background: line.color }} aria-hidden />
                <span className={styles.legendName}>TP {line.tp}</span>
                <span className={styles.legendMeta}>
                  {line.curve.backendVersions.join(', ') || 'unlabelled build'} · {line.curve.points.length} levels
                </span>
              </button>
            ))}
            {overlayVisible && prediction ? (
              <span className={`${styles.legendItem} ${styles.legendStatic}`}>
                <span className={styles.legendDiamond} style={{ color: predictionColor }} aria-hidden>◆</span>
                <span className={styles.legendName}>Your prediction</span>
                <span className={styles.legendMeta}>{formatThroughput(prediction.throughput)} at concurrency {prediction.concurrency.toLocaleString()}</span>
              </span>
            ) : null}
          </div>
        )}

        {series.mixedConfigurations.length > 0 && (
          <p className={styles.panelNote}>
            {series.mixedConfigurations.map(tp => `TP ${tp}`).join(', ')} {series.mixedConfigurations.length === 1 ? 'is' : 'are'} shown
            from a different run configuration because the reference sweep did not include {series.mixedConfigurations.length === 1 ? 'it' : 'them'}.
          </p>
        )}
        <p className={styles.panelNote}>
          Points are per-level medians of benchmark-level measurements. Multi-turn per-request windows are excluded, and
          each line comes from one run configuration, so a line never mixes workloads, units, or versions.
        </p>
      </section>

      {prediction && (
        <section className={styles.panel} id="prediction" aria-label="Prediction versus measurement">
          <div className={styles.panelHeader}>
            <h2 className={styles.panelTitle}>Prediction vs measurement</h2>
            <span className={styles.panelHint}>{predictionLabel}</span>
          </div>
          <p className={styles.panelNote}>
            This snapshot came from your prediction, not from this page. Opening or editing the link changes the point
            here; it does not re-run the simulator.
          </p>

          {comparison && comparison.tpMeasured && comparison.at && comparison.curve ? (
            <>
              <div className={styles.compareGrid}>
                <div className={styles.compareStat}>
                  <span className={styles.cardLabel}>Predicted</span>
                  <strong className={styles.compareValue}>{formatThroughput(prediction.throughput)}</strong>
                  <span className={styles.cardSub}>TP {prediction.tp} · concurrency {prediction.concurrency.toLocaleString()}</span>
                </div>
                <div className={styles.compareStat}>
                  <span className={styles.cardLabel}>Measured at that concurrency</span>
                  <strong className={styles.compareValue}>{formatThroughput(comparison.at.measured)}</strong>
                  <span className={styles.cardSub}>
                    {comparison.at.clamped
                      ? 'Nearest measured level'
                      : comparison.at.exact
                        ? 'Measured level'
                        : `Interpolated between ${comparison.at.below.concurrency.toLocaleString()} and ${comparison.at.above.concurrency.toLocaleString()}`}
                  </span>
                </div>
                <div className={styles.compareStat}>
                  <span className={styles.cardLabel}>Difference</span>
                  <strong className={styles.compareValue}>
                    {comparison.deltaPct == null ? '—' : percent(comparison.deltaPct)}
                  </strong>
                  <span className={styles.cardSub}>Prediction versus measured</span>
                </div>
                <div className={styles.compareStat}>
                  <span className={styles.cardLabel}>Observed range</span>
                  <strong className={styles.compareValue}>
                    {formatThroughput(comparison.observedMin ?? 0)} – {formatThroughput(comparison.observedMax ?? 0)}
                  </strong>
                  <span className={styles.cardSub}>This workload at TP {prediction.tp}</span>
                </div>
              </div>

              <ul className={styles.checkList}>
                {comparison.checks.map(check => (
                  <li key={check.id} className={check.ok ? styles.checkPass : styles.checkWarn}>
                    <span className={styles.checkMark} aria-hidden>{check.ok ? '✓' : '!'}</span>
                    <span><strong>{check.label}.</strong> {check.detail}</span>
                  </li>
                ))}
              </ul>

              <p className={styles.panelNote}>{outsideExplanation(comparison, prediction)}</p>
            </>
          ) : (
            <>
              <Alert
                isInline
                variant="info"
                title="No comparable measured line"
              >
                {comparison?.checks[0]?.detail
                  ?? 'This prediction cannot be matched to a measured workload for this pair.'}
              </Alert>
            </>
          )}
          {predictionWorkloadNote}
        </section>
      )}

      <div className={styles.twoColumn}>
        <section className={styles.panel} aria-label="Feature importance">
          <div className={styles.panelHeader}>
            <h2 className={styles.panelTitle}>Feature importance</h2>
            <span className={styles.panelHint}>Classifier inputs</span>
          </div>
          {dashboard?.featureImportance.length ? dashboard.featureImportance.map(item => (
            <div className={styles.barRow} key={item.feature}>
              <span className={styles.barLabel}>{item.feature}</span>
              <div className={styles.barTrack}><div className={styles.bar} style={{ width: `${(item.importance / maxImportance) * 100}%` }} /></div>
              <span className={styles.barValue}>{item.importance.toFixed(3)}</span>
            </div>
          )) : <p className={styles.panelNote}>No classifier was trained for this pair.</p>}
        </section>
        <section className={styles.panel} id="boundary" aria-label="Detected boundary points">
          <div className={styles.panelHeader}>
            <h2 className={styles.panelTitle}>Detected boundary points</h2>
            <span className={styles.panelHint}>{dashboard?.kneePoints.length ?? 0} points</span>
          </div>
          {dashboard && dashboard.kneePoints.length > 0 ? dashboard.kneePoints.slice(0, 12).map((point, index) => (
            <div className={styles.kneeRow} key={`${index}-${point.concurrency ?? point.isl ?? point.osl}`}>
              <Label color="orange" isCompact>boundary</Label>
              <span className={styles.kneeDetail}>
                {Object.entries(point).map(([key, value]) => `${key}=${value}`).join(' · ')}
              </span>
            </div>
          )) : <p className={styles.panelNote}>No stable boundary points were detected.</p>}
        </section>
      </div>

      <section className={styles.panel} id="metrics" aria-label="Classifier metrics">
        <div className={styles.panelHeader}>
          <h2 className={styles.panelTitle}>Classifier metrics</h2>
          <span className={styles.panelHint}>{validated ? 'Held-out test' : 'Not available'}</span>
        </div>
        {validated ? (
          <dl className={styles.metricList}>
            <div className={styles.metricItem}><dt className={styles.metricLabel}>Test ROC-AUC</dt><dd className={styles.metricValue}>{metric(tests.roc_auc)}</dd></div>
            <div className={styles.metricItem}><dt className={styles.metricLabel}>Test F1</dt><dd className={styles.metricValue}>{metric(tests.f1)}</dd></div>
            <div className={styles.metricItem}><dt className={styles.metricLabel}>Test precision</dt><dd className={styles.metricValue}>{metric(tests.precision)}</dd></div>
            <div className={styles.metricItem}><dt className={styles.metricLabel}>Test recall</dt><dd className={styles.metricValue}>{metric(tests.recall)}</dd></div>
          </dl>
        ) : (
          <p className={styles.panelNote}>
            No metrics are published because this pair does not have a validated classifier. Training status: {pair.diagnostics?.trainingStatus ?? 'unknown'}.
          </p>
        )}
      </section>
    </TestedModelsShell>
  )
}

'use client'

import * as React from 'react'
import Link from 'next/link'
import { Alert, Label, Spinner } from '@patternfly/react-core'
import {
  Chart,
  ChartAxis,
  ChartLine,
  ChartTooltip,
  ChartVoronoiContainer,
} from '@patternfly/react-charts/victory'
import { useCatalog } from '@/lib/hooks/useCatalog'
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
  throughputLatencyByConcurrency: Array<{ concurrency: number; throughput: number | null; latency: number | null; points: number }>
  kneePoints: Array<Record<string, number>>
  featureImportance: Array<{ feature: string; importance: number }>
}

interface TestedModelPairProps {
  pair: PublicRegistryPair
  gate: ValidationGateThresholds | null
  dataset?: RegistryDatasetProvenance | null
}

function metric(value: number | null | undefined): string {
  return typeof value === 'number' ? value.toFixed(3) : '—'
}

function testMetrics(pair: PublicRegistryPair): { roc_auc?: number | null; f1?: number; precision?: number; recall?: number } {
  return (pair.metrics as { test?: { roc_auc?: number | null; f1?: number; precision?: number; recall?: number } } | undefined)?.test ?? {}
}

/** Track the rendered chart width so Victory matches the card. */
function useElementWidth<T extends HTMLElement>(): [React.RefObject<T>, number] {
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

export function TestedModelPair({ pair, gate, dataset = null }: TestedModelPairProps) {
  const { gpuOptions: catalogGpus, modelSpecs } = useCatalog()
  const [dashboard, setDashboard] = React.useState<Dashboard | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [unavailable, setUnavailable] = React.useState(false)
  const [chartRef, chartWidth] = useElementWidth<HTMLDivElement>()

  React.useEffect(() => {
    fetch(`/api/tested-models/dashboard/${encodeURIComponent(pair.id)}`, { cache: 'no-store' })
      .then(response => {
        if (!response.ok) throw new Error('unavailable')
        return response.json() as Promise<Dashboard>
      })
      .then(setDashboard)
      .catch(() => {
        setDashboard(null)
        setUnavailable(true)
      })
      .finally(() => setLoading(false))
  }, [pair.id])

  const curve = dashboard?.throughputLatencyByConcurrency.filter(point => point.throughput != null) ?? []
  const throughputData = curve.map(point => ({
    x: point.concurrency,
    y: point.throughput as number,
    name: 'throughput',
  }))
  const xValues = throughputData.map(point => point.x)
  const yValues = throughputData.map(point => point.y)
  const minX = xValues.length > 0 ? Math.min(...xValues) : 0
  const maxX = xValues.length > 0 ? Math.max(...xValues) : 1
  const maxY = Math.max(...yValues, 1) * 1.1
  const maxImportance = Math.max(...(dashboard?.featureImportance.map(item => item.importance) ?? [1]), 1)
  const coverage = pair.coverage
  const validated = pair.status === 'validated'
  const tests = testMetrics(pair)
  const gpuLabel = gpuSystemLabel(pair.systemId, catalogGpus)
  const gpuSpecs = gpuSystemSpecs(pair.systemId, catalogGpus)
  const modelChips = modelFacetChips(pair.modelId, modelSpecs.get(pair.modelId))

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
          Dashboard data is unavailable for this pair. Coverage metadata from the registry is shown below.
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
          <span className={styles.panelHint}>Measured points</span>
        </div>
        <div ref={chartRef} className={styles.chartContainer}>
          {throughputData.length > 0 ? (
            <Chart
              ariaDesc="Measured output throughput as concurrency increases"
              ariaTitle="Throughput by concurrency"
              height={280}
              width={chartWidth}
              padding={{ top: 20, right: 30, bottom: 62, left: 74 }}
              domain={{ x: [minX === maxX ? minX - 1 : minX, minX === maxX ? maxX + 1 : maxX], y: [0, maxY] }}
              containerComponent={(
                <ChartVoronoiContainer
                  constrainToVisibleArea
                  labels={({ datum }: { datum: { x?: number; y?: number } }) =>
                    `Concurrency ${Number(datum.x).toLocaleString()}\n${formatThroughput(Number(datum.y))}`}
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
              <ChartLine
                data={throughputData}
                style={{ data: { stroke: '#0066cc', strokeWidth: 3 } }}
              />
            </Chart>
          ) : (
            <p className={styles.panelNote}>No throughput curve is available yet.</p>
          )}
        </div>
      </section>

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

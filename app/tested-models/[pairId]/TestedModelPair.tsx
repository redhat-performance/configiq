'use client'

import * as React from 'react'
import Link from 'next/link'
import { Alert, Label, Spinner } from '@patternfly/react-core'
import type { RegistryPair, ValidationGateThresholds } from '@/lib/tested-models/registry-types'
import { testedModelsFilterHref } from '@/lib/tested-models/links'
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
  pair: RegistryPair
  gate: ValidationGateThresholds | null
}

function linePath(points: Array<{ x: number; y: number }>, width: number, height: number): string {
  if (points.length < 2) return ''
  const minX = Math.min(...points.map(point => point.x))
  const maxX = Math.max(...points.map(point => point.x))
  const minY = Math.min(...points.map(point => point.y))
  const maxY = Math.max(...points.map(point => point.y))
  return points.map((point, index) => {
    const x = ((point.x - minX) / Math.max(maxX - minX, 1)) * (width - 24) + 12
    const y = height - (((point.y - minY) / Math.max(maxY - minY, 1)) * (height - 24) + 12)
    return `${index === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ')
}

function metric(value: number | null | undefined): string {
  return typeof value === 'number' ? value.toFixed(3) : '—'
}

export function TestedModelPair({ pair, gate }: TestedModelPairProps) {
  const [dashboard, setDashboard] = React.useState<Dashboard | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [unavailable, setUnavailable] = React.useState(false)

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
  const path = linePath(curve.map(point => ({ x: point.concurrency, y: point.throughput! })), 640, 220)
  const maxImportance = Math.max(...(dashboard?.featureImportance.map(item => item.importance) ?? [1]), 1)
  const coverage = pair.coverage
  const validated = pair.status === 'validated'

  return (
    <main className={styles.page}>
      <div className={styles.header}>
        <div>
          <p className={styles.eyebrow}>
            <Link href={testedModelsFilterHref({ modelId: pair.modelId })} className={styles.backLink}>Tested performance envelopes</Link>
          </p>
          <h1>{pair.modelId}</h1>
          <p>{pair.systemId} · measured performance envelope and classifier evidence.</p>
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

      {loading && <Spinner aria-label="Loading pair coverage" />}
      {unavailable && (
        <div className={styles.notice} role="status">
          Dashboard data is unavailable for this pair. Coverage metadata from the registry is shown below.
        </div>
      )}

      <div className={styles.summaryGrid} id="coverage">
        <section className={styles.card}>
          <span>Records</span>
          <strong>{(coverage?.records ?? dashboard?.records ?? 0).toLocaleString()}</strong>
        </section>
        <section className={styles.card}>
          <span>Concurrency levels</span>
          <strong>{coverage?.concurrencyLevelCount ?? '—'}</strong>
        </section>
        <section className={styles.card}>
          <span>Within range</span>
          <strong>{dashboard ? dashboard.labelDistribution.within.toLocaleString() : '—'}</strong>
        </section>
        <section className={styles.card}>
          <span>Test ROC-AUC</span>
          <strong>{metric(pair.metrics ? (pair.metrics as { test?: { roc_auc?: number | null } }).test?.roc_auc : dashboard?.test?.roc_auc)}</strong>
        </section>
      </div>

      <section className={styles.panel} aria-label="Validation gate">
        <div className={styles.panelHeader}>
          <h2>Validation gate</h2>
          <span>{validated ? 'Passed' : 'Not passed'}</span>
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

      <section className={styles.panel} aria-label="Throughput by concurrency">
        <div className={styles.panelHeader}><h2>Throughput by concurrency</h2><span>Measured points</span></div>
        {curve.length > 0 ? (
          <svg viewBox="0 0 640 220" className={styles.chart} role="img" aria-label="Throughput by concurrency">
            <path d={path} fill="none" stroke="#0066cc" strokeWidth="3" />
          </svg>
        ) : (
          <p className={styles.panelNote}>No throughput curve is available yet.</p>
        )}
      </section>

      <div className={styles.twoColumn}>
        <section className={styles.panel} aria-label="Feature importance">
          <div className={styles.panelHeader}><h2>Feature importance</h2><span>Classifier inputs</span></div>
          {dashboard?.featureImportance.length ? dashboard.featureImportance.map(item => (
            <div className={styles.barRow} key={item.feature}>
              <span>{item.feature}</span>
              <div className={styles.barTrack}><div className={styles.bar} style={{ width: `${(item.importance / maxImportance) * 100}%` }} /></div>
            </div>
          )) : <p className={styles.panelNote}>No classifier was trained for this pair.</p>}
        </section>
        <section className={styles.panel} id="boundary" aria-label="Detected boundary points">
          <div className={styles.panelHeader}><h2>Detected boundary points</h2><span>{dashboard?.kneePoints.length ?? 0} points</span></div>
          {dashboard && dashboard.kneePoints.length > 0 ? dashboard.kneePoints.slice(0, 12).map((point, index) => (
            <div className={styles.kneeRow} key={`${index}-${point.concurrency ?? point.isl ?? point.osl}`}>
              <Label color="orange" isCompact>boundary</Label>
              <span>{Object.entries(point).map(([key, value]) => `${key}=${value}`).join(' · ')}</span>
            </div>
          )) : <p className={styles.panelNote}>No stable boundary points were detected.</p>}
        </section>
      </div>

      <section className={styles.panel} id="metrics" aria-label="Classifier metrics">
        <div className={styles.panelHeader}><h2>Classifier metrics</h2><span>{validated ? 'Held-out test' : 'Not available'}</span></div>
        {validated ? (
          <dl className={styles.metricList}>
            <div><dt>Test ROC-AUC</dt><dd>{metric((pair.metrics as { test?: { roc_auc?: number | null } } | undefined)?.test?.roc_auc)}</dd></div>
            <div><dt>Test F1</dt><dd>{metric((pair.metrics as { test?: { f1?: number } } | undefined)?.test?.f1)}</dd></div>
            <div><dt>Test precision</dt><dd>{metric((pair.metrics as { test?: { precision?: number } } | undefined)?.test?.precision)}</dd></div>
            <div><dt>Test recall</dt><dd>{metric((pair.metrics as { test?: { recall?: number } } | undefined)?.test?.recall)}</dd></div>
          </dl>
        ) : (
          <p className={styles.panelNote}>
            No metrics are published because this pair does not have a validated classifier. Training status: {pair.diagnostics?.trainingStatus ?? 'unknown'}.
          </p>
        )}
      </section>
    </main>
  )
}

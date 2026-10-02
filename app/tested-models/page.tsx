'use client'

import * as React from 'react'
import Link from 'next/link'
import { Label, Spinner } from '@patternfly/react-core'
import styles from './TestedModels.module.css'

interface Pair {
  id: string
  modelId: string
  systemId: string
  dashboardPath?: string
}

interface Dashboard {
  modelId: string
  systemId: string
  records: number
  labelDistribution: { within: number; outside: number }
  validation?: { roc_auc?: number | null; precision?: number; recall?: number }
  test?: { roc_auc?: number | null; precision?: number; recall?: number }
  throughputLatencyByConcurrency: Array<{ concurrency: number; throughput: number | null; latency: number | null; points: number }>
  kneePoints: Array<Record<string, number>>
  featureImportance: Array<{ feature: string; importance: number }>
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

export default function TestedModelsPage() {
  const [pairs, setPairs] = React.useState<Pair[]>([])
  const [selectedId, setSelectedId] = React.useState('')
  const [dashboard, setDashboard] = React.useState<Dashboard | null>(null)
  const [loading, setLoading] = React.useState(true)

  React.useEffect(() => {
    fetch('/api/tested-models', { cache: 'no-store' })
      .then(response => response.json() as Promise<{ pairs?: Pair[] }>)
      .then(data => {
        const nextPairs = data.pairs ?? []
        const queryId = new URLSearchParams(globalThis.location.search).get('pair')
        setPairs(nextPairs)
        setSelectedId(nextPairs.some(pair => pair.id === queryId) ? queryId! : nextPairs[0]?.id ?? '')
      })
      .finally(() => setLoading(false))
  }, [])

  React.useEffect(() => {
    if (!selectedId) return
    fetch(`/api/tested-models/dashboard/${encodeURIComponent(selectedId)}`, { cache: 'no-store' })
      .then(response => response.json() as Promise<Dashboard>)
      .then(setDashboard)
      .catch(() => setDashboard(null))
  }, [selectedId])

  if (loading) return <main className={styles.page}><Spinner aria-label="Loading tested models" /></main>
  if (pairs.length === 0) return <main className={styles.page}><h1>Tested performance envelopes</h1><p>Tested-model data is currently unavailable.</p></main>

  const curve = dashboard?.throughputLatencyByConcurrency.filter(point => point.throughput != null) ?? []
  const path = linePath(curve.map(point => ({ x: point.concurrency, y: point.throughput! })), 640, 220)
  const maxImportance = Math.max(...(dashboard?.featureImportance.map(item => item.importance) ?? [1]), 1)

  return (
    <main className={styles.page}>
      <div className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Measured operating envelopes</p>
          <h1>Tested performance envelopes</h1>
          <p>Select a model and GPU system to inspect the benchmark coverage and classifier evidence.</p>
        </div>
        <Link href="/predict" className={styles.backLink}>Back to performance prediction</Link>
      </div>

      <label className={styles.selectorLabel} htmlFor="tested-pair">Model and GPU system</label>
      <select id="tested-pair" value={selectedId} onChange={event => setSelectedId(event.target.value)} className={styles.selector}>
        {pairs.map(pair => <option key={pair.id} value={pair.id}>{pair.modelId} · {pair.systemId}</option>)}
      </select>

      {dashboard && (
        <>
          <div className={styles.summaryGrid}>
            <section className={styles.card}><span>Records</span><strong>{dashboard.records.toLocaleString()}</strong></section>
            <section className={styles.card}><span>Within range</span><strong>{dashboard.labelDistribution.within.toLocaleString()}</strong></section>
            <section className={styles.card}><span>Outside range</span><strong>{dashboard.labelDistribution.outside.toLocaleString()}</strong></section>
            <section className={styles.card}><span>Test ROC-AUC</span><strong>{dashboard.test?.roc_auc?.toFixed(3) ?? '—'}</strong></section>
          </div>

          <section className={styles.panel}>
            <div className={styles.panelHeader}><h2>Throughput by concurrency</h2><span>Measured points</span></div>
            <svg viewBox="0 0 640 220" className={styles.chart} role="img" aria-label="Throughput by concurrency">
              <path d={path} fill="none" stroke="#0066cc" strokeWidth="3" />
            </svg>
          </section>

          <div className={styles.twoColumn}>
            <section className={styles.panel}>
              <div className={styles.panelHeader}><h2>Feature importance</h2><span>Classifier inputs</span></div>
              {dashboard.featureImportance.map(item => (
                <div className={styles.barRow} key={item.feature}>
                  <span>{item.feature}</span>
                  <div className={styles.barTrack}><div className={styles.bar} style={{ width: `${(item.importance / maxImportance) * 100}%` }} /></div>
                </div>
              ))}
            </section>
            <section className={styles.panel}>
              <div className={styles.panelHeader}><h2>Detected boundary points</h2><span>{dashboard.kneePoints.length} points</span></div>
              {dashboard.kneePoints.length === 0 ? <p>No stable knee points were detected.</p> : dashboard.kneePoints.slice(0, 12).map((point, index) => (
                <div className={styles.kneeRow} key={`${index}-${point.concurrency ?? point.isl ?? point.osl}`}>
                  <Label color="orange" isCompact>boundary</Label>
                  <span>{Object.entries(point).map(([key, value]) => `${key}=${value}`).join(' · ')}</span>
                </div>
              ))}
            </section>
          </div>
        </>
      )}
    </main>
  )
}

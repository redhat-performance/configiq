'use client'

import * as React from 'react'
import Link from 'next/link'
import { Label, Spinner } from '@patternfly/react-core'
import { useTestedModels } from '@/lib/hooks/useTestedModels'
import { testedModelsFilterHref, testedPairHref } from '@/lib/tested-models/links'
import type { RegistryPair } from '@/lib/tested-models/registry-types'
import styles from './TestedModels.module.css'

interface TestedModelsIndexProps {
  initialModel?: string
  initialSystem?: string
}

function statusLabel(pair: RegistryPair) {
  return pair.status === 'validated'
    ? <Label color="green" isCompact>Validated envelope</Label>
    : <Label color="blue" isCompact>Measured, no validated envelope</Label>
}

function pairSort(a: RegistryPair, b: RegistryPair): number {
  if (a.status !== b.status) return a.status === 'validated' ? -1 : 1
  return `${a.modelId} ${a.systemId}`.localeCompare(`${b.modelId} ${b.systemId}`)
}

export function TestedModelsIndex({ initialModel = '', initialSystem = '' }: TestedModelsIndexProps) {
  const { pairs, approvedModels, gate, schemaVersion, isLoading, isAvailable } = useTestedModels()
  const [modelFilter, setModelFilter] = React.useState(initialModel)
  const [systemFilter, setSystemFilter] = React.useState(initialSystem)

  const systems = React.useMemo(
    () => [...new Set(pairs.map(pair => pair.systemId))].sort(),
    [pairs],
  )
  const filtered = React.useMemo(
    () => pairs
      .filter(pair => !modelFilter || pair.modelId.toLowerCase().includes(modelFilter.toLowerCase()))
      .filter(pair => !systemFilter || pair.systemId === systemFilter)
      .toSorted(pairSort),
    [modelFilter, pairs, systemFilter],
  )
  const approvedWithoutData = React.useMemo(
    () => approvedModels.filter(model => !pairs.some(pair => pair.modelId === model)),
    [approvedModels, pairs],
  )
  const validatedCount = pairs.filter(pair => pair.status === 'validated').length

  if (isLoading) {
    return <main className={styles.page}><Spinner aria-label="Loading tested models" /></main>
  }

  return (
    <main className={styles.page}>
      <div className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Measured operating envelopes</p>
          <h1>Tested performance envelopes</h1>
          <p>
            Measurements and classifier evidence for approved models. A validated envelope can answer
            classification questions; a measured pair without a validated envelope still exposes its graph.
          </p>
        </div>
        <Link href="/predict" className={styles.backLink}>Back to performance prediction</Link>
      </div>

      {!isAvailable && (
        <div className={styles.notice} role="status">
          The tested-model registry is unavailable. Tested status is <strong>unknown</strong>, not untested.
        </div>
      )}

      {isAvailable && schemaVersion === 1 && (
        <div className={styles.notice} role="status">
          This deployment is serving a legacy registry. Coverage and the full validation gate become
          available after the registry is upgraded to schema v2.
        </div>
      )}

      {isAvailable && (
        <section className={styles.gate} aria-label="Validation gate">
          <span className={styles.eyebrow}>Validation gate</span>
          <p>
            {gate
              ? `At least ${gate.minRecords} records, ${gate.minConcurrencyLevels} concurrency levels, test AUC ≥ ${gate.minTestAuc.toFixed(2)}, and test F1 ≥ ${gate.minTestF1.toFixed(2)}.`
              : 'Legacy registry: only recorded test AUC and F1 are available.'}
          </p>
        </section>
      )}

      <section className={styles.panel} aria-label="Tested-model filters">
        <div className={styles.filterGrid}>
          <label className={styles.selectorLabel} htmlFor="tested-model-filter">
            Model
            <input
              id="tested-model-filter"
              className={styles.selector}
              value={modelFilter}
              onChange={event => setModelFilter(event.target.value)}
              placeholder="Filter by model ID"
            />
          </label>
          <label className={styles.selectorLabel} htmlFor="tested-system-filter">
            GPU system
            <select
              id="tested-system-filter"
              className={styles.selector}
              value={systemFilter}
              onChange={event => setSystemFilter(event.target.value)}
            >
              <option value="">All GPU systems</option>
              {systems.map(system => <option key={system} value={system}>{system}</option>)}
            </select>
          </label>
        </div>
        <p className={styles.resultCount}>
          {filtered.length} measured pair{filtered.length === 1 ? '' : 's'} · {validatedCount} with a validated envelope
        </p>
      </section>

      {filtered.length === 0 && isAvailable && (
        <div className={styles.empty}>
          <h2>No measured pairs match these filters</h2>
          <p>
            {approvedWithoutData.length > 0
              ? 'Approved models without measurements are listed below. No classifier verdict is available for them.'
              : 'Try a different model or GPU system.'}
          </p>
        </div>
      )}

      <div className={styles.pairList}>
        {filtered.map(pair => (
          <Link key={pair.id} href={testedPairHref(pair.id)} className={styles.pairRow}>
            <div>
              <div className={styles.pairTitle}>{pair.modelId}</div>
              <div className={styles.pairMeta}>
                {pair.systemId}
                {pair.coverage && <> · {pair.coverage.records.toLocaleString()} records · {pair.coverage.concurrencyLevelCount} concurrency levels</>}
              </div>
            </div>
            <div className={styles.pairStatus}>
              {statusLabel(pair)}
              <span className={styles.openLink}>View graph →</span>
            </div>
          </Link>
        ))}
      </div>

      {approvedWithoutData.length > 0 && (
        <section className={styles.panel} aria-label="Approved models without measurements">
          <div className={styles.panelHeader}>
            <h2>Approved models without measurements</h2>
            <span>{approvedWithoutData.length} models</span>
          </div>
          <p className={styles.panelNote}>
            These models are approved for measurement but have no published data. They are not classifier-tested.
          </p>
          <div className={styles.approvedList}>
            {approvedWithoutData.map(model => (
              <Link key={model} href={testedModelsFilterHref({ modelId: model })} className={styles.approvedLink}>
                {model}
              </Link>
            ))}
          </div>
        </section>
      )}
    </main>
  )
}

'use client'

import * as React from 'react'
import Link from 'next/link'
import { Label, Spinner } from '@patternfly/react-core'
import { ComboBox } from '@/components/ModelComboBox/ModelComboBox'
import { GpuSystemInput } from '@/components/ui/GpuSystemInput'
import { useCatalog, type ModelSpec } from '@/lib/hooks/useCatalog'
import { useTestedModels } from '@/lib/hooks/useTestedModels'
import {
  gpuOptionsForSystems,
  gpuSystemLabel,
  modelFacetChips,
  registryModelItems,
  shortRevision,
} from '@/lib/tested-models/display'
import { testedModelsFilterHref, testedPairHref } from '@/lib/tested-models/links'
import type { RegistryPair } from '@/lib/tested-models/registry-types'
import { TestedModelsShell } from './TestedModelsShell'
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

function ModelChips({ modelId, spec }: { modelId: string; spec?: ModelSpec }) {
  const chips = modelFacetChips(modelId, spec)
  if (chips.length === 0) return null
  return (
    <div className={styles.chipRow}>
      {chips.map(chip => <span key={chip} className={styles.chip}>{chip}</span>)}
    </div>
  )
}

export function TestedModelsIndex({ initialModel = '', initialSystem = '' }: TestedModelsIndexProps) {
  const { pairs, approvedModels, gate, schemaVersion, dataset, validatedModelIds, isLoading, isAvailable } = useTestedModels()
  const { gpuOptions: catalogGpus, modelOptions, modelSpecs } = useCatalog()
  const [modelFilter, setModelFilter] = React.useState(initialModel)
  const [systemFilter, setSystemFilter] = React.useState(initialSystem)

  const systems = React.useMemo(
    () => [...new Set(pairs.map(pair => pair.systemId))].sort(),
    [pairs],
  )
  const modelIds = React.useMemo(
    () => [...new Set([...pairs.map(pair => pair.modelId), ...approvedModels])],
    [approvedModels, pairs],
  )
  const modelItems = React.useMemo(
    () => registryModelItems(modelIds, modelSpecs, validatedModelIds, modelOptions),
    [modelIds, modelOptions, modelSpecs, validatedModelIds],
  )
  const gpuFilterOptions = React.useMemo(
    () => gpuOptionsForSystems(systems, catalogGpus),
    [catalogGpus, systems],
  )
  const filtered = React.useMemo(
    () => pairs
      .filter(pair => !modelFilter || pair.modelId === modelFilter)
      .filter(pair => !systemFilter || pair.systemId === systemFilter)
      .toSorted(pairSort),
    [modelFilter, pairs, systemFilter],
  )
  const approvedWithoutData = React.useMemo(
    () => approvedModels.filter(model => !pairs.some(pair => pair.modelId === model)),
    [approvedModels, pairs],
  )
  const validatedCount = pairs.filter(pair => pair.status === 'validated').length
  const filteredValidated = filtered.filter(pair => pair.status === 'validated').length
  const measuredModelCount = new Set(pairs.map(pair => pair.modelId)).size

  if (isLoading) {
    return (
      <TestedModelsShell>
        <Spinner aria-label="Loading tested models" />
      </TestedModelsShell>
    )
  }

  return (
    <TestedModelsShell>
      <div className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Measured operating envelopes</p>
          <h1 className={styles.pageTitle}>Tested performance envelopes</h1>
          <p className={styles.subtitle}>
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
        <div className={styles.summaryGrid}>
          <section className={styles.card}>
            <span className={styles.cardLabel}>Measured pairs</span>
            <strong className={styles.cardValue}>{pairs.length}</strong>
            <span className={styles.cardSub}>{measuredModelCount} model{measuredModelCount === 1 ? '' : 's'} with data</span>
          </section>
          <section className={styles.card}>
            <span className={styles.cardLabel}>Validated envelopes</span>
            <strong className={styles.cardValue}>{validatedCount}</strong>
            <span className={styles.cardSub}>Pass the held-out quality gate</span>
          </section>
          <section className={styles.card}>
            <span className={styles.cardLabel}>Measured only</span>
            <strong className={styles.cardValue}>{pairs.length - validatedCount}</strong>
            <span className={styles.cardSub}>Graphs without a classifier verdict</span>
          </section>
          <section className={styles.card}>
            <span className={styles.cardLabel}>Approved, no data</span>
            <strong className={styles.cardValue}>{approvedWithoutData.length}</strong>
            <span className={styles.cardSub}>Not tested yet</span>
          </section>
        </div>
      )}

      {isAvailable && (
        <section className={styles.panel} aria-label="Validation gate">
          <div className={styles.panelHeader}>
            <h2 className={styles.panelTitle}>Validation gate</h2>
            {dataset && (
              <span className={styles.panelHint}>
                Dataset{' '}
                <a
                  href={`https://huggingface.co/datasets/${dataset.repo}/tree/${dataset.revision}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={styles.inlineLink}
                >
                  {dataset.repo}@{shortRevision(dataset.revision)}
                </a>
              </span>
            )}
          </div>
          <p className={styles.panelNote}>
            {gate
              ? `At least ${gate.minRecords} records, ${gate.minConcurrencyLevels} concurrency levels, test AUC ≥ ${gate.minTestAuc.toFixed(2)}, and test F1 ≥ ${gate.minTestF1.toFixed(2)}.`
              : 'Legacy registry: only recorded test AUC and F1 are available.'}
          </p>
        </section>
      )}

      <section className={`${styles.card} ${styles.filterCard}`} aria-label="Tested-model filters">
        <div className={styles.filterGrid}>
          <ComboBox
            id="tested-model-filter"
            value={modelFilter}
            onChange={setModelFilter}
            items={modelItems}
            label="Model"
            allLabel="All models"
            placeholder="Search by name, size, type or precision…"
          />
          <GpuSystemInput
            id="tested-system-filter"
            value={systemFilter}
            onChange={setSystemFilter}
            gpuOptions={gpuFilterOptions}
            testedSystemIds={systems}
            allLabel="All GPU systems"
          />
        </div>
        <p className={styles.resultCount}>
          {filtered.length} measured pair{filtered.length === 1 ? '' : 's'} · {filteredValidated} with a validated envelope
        </p>
      </section>

      {filtered.length === 0 && isAvailable && (
        <div className={styles.empty}>
          <h2 className={styles.panelTitle}>No measured pairs match these filters</h2>
          <p className={styles.panelNote}>
            {approvedWithoutData.length > 0
              ? 'Approved models without measurements are listed below. No classifier verdict is available for them.'
              : 'Try a different model or GPU system.'}
          </p>
        </div>
      )}

      <div className={styles.pairList}>
        {filtered.map(pair => (
          <Link key={pair.id} href={testedPairHref(pair.id)} className={styles.pairRow}>
            <div className={styles.pairMain}>
              <div className={styles.pairTitleRow}>
                <span className={styles.pairTitle}>{pair.modelId}</span>
                <span className={styles.gpuLabel}>{gpuSystemLabel(pair.systemId, catalogGpus)}</span>
              </div>
              <ModelChips modelId={pair.modelId} spec={modelSpecs.get(pair.modelId)} />
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
            <h2 className={styles.panelTitle}>Approved models without measurements</h2>
            <span className={styles.panelHint}>{approvedWithoutData.length} models</span>
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
    </TestedModelsShell>
  )
}

'use client'

import * as React from 'react'
import { Alert, Button, ExpandableSection, FormGroup, TextInput } from '@patternfly/react-core'
import { useCostAssumptions } from '@/contexts/CostAssumptionsContext'
import { COST_ASSUMPTION_GROUPS, isValidCostAssumption, type CostAssumptionField } from '@/lib/costing-assumptions'
import styles from './Sources.module.css'

function AssumptionInput({ field }: { field: CostAssumptionField }) {
  const { assumptions, overrides, updateAssumption } = useCostAssumptions()
  const [draft, setDraft] = React.useState<string | null>(null)
  const invalid = draft !== null && (draft.trim() === '' || !isValidCostAssumption(field.key, Number(draft)))
  const id = `cost-assumption-${field.key}`
  const commit = () => {
    if (draft === null || invalid) return
    updateAssumption(field.key, Number(draft))
    setDraft(null)
  }
  return (
    <FormGroup label={field.label} fieldId={id}>
      <TextInput
        id={id} type="number" min={field.min ?? 0} max={field.max} step={field.step ?? 'any'}
        value={draft ?? assumptions[field.key]} validated={invalid ? 'error' : 'default'}
        aria-invalid={invalid} aria-describedby={invalid ? `${id}-error` : undefined}
        onChange={(_event, value) => setDraft(value)} onBlur={commit}
        onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); event.currentTarget.blur() } }}
      />
      {invalid ? <p id={`${id}-error`} className={styles.assumptionError} role="alert">
        Enter a finite number {field.max === undefined ? `of at least ${field.min ?? 0}` : `between ${field.min ?? 0} and ${field.max}`}. The previous value is still in use.
      </p> : overrides[field.key] !== undefined && <p className={styles.assumptionSaved}>Saved override</p>}
    </FormGroup>
  )
}

function AssumptionGroup({ group }: { group: typeof COST_ASSUMPTION_GROUPS[number] }) {
  const [expanded, setExpanded] = React.useState(false)
  return <ExpandableSection isExpanded={expanded} onToggle={(_event, open) => setExpanded(open)} toggleText={group.title}>
    <p className={styles.sectionDesc}>{group.description}</p>
    <div className={styles.assumptionGrid}>
      {group.fields.map(field => <AssumptionInput key={field.key} field={field} />)}
    </div>
  </ExpandableSection>
}

export default function CostAssumptionsEditor() {
  const { assumptions, hydrated, storageError, resetAssumptions } = useCostAssumptions()
  const [revision, setRevision] = React.useState(0)
  return (
    <section className={styles.section} id="cost-assumptions" aria-labelledby="cost-assumptions-heading">
      <div className={styles.sectionHead}>
        <div>
          <h2 className={styles.sectionTitle} id="cost-assumptions-heading">Cost planning assumptions</h2>
          <p className={styles.sectionDesc}>Saved in this browser and reused by compatible costings tools. Valid changes save when you leave a field.</p>
        </div>
        <Button variant="link" isInline isDisabled={!hydrated} onClick={() => {
          resetAssumptions(); setRevision(previous => previous + 1)
        }}>Reset assumptions</Button>
      </div>
      <div className={styles.assumptionsBody}>
        {storageError && <Alert variant="warning" isInline title="Browser storage unavailable or unreadable">
          Your current changes work for this session, but may not survive a refresh.
        </Alert>}
        <p className={styles.assumptionsBasis}>
          Current Hybrid Savings basis: {assumptions.planningCapacityUsePct}% planning capacity · {assumptions.hardwareLifeYears}-year hardware life · ${assumptions.loadedMonthlyCostPerFte.toLocaleString('en-US')}/FTE-month.
        </p>
        <p className={styles.sectionDesc}>Routing Economics currently models usage and GPU rental rates, not these full-TCO assumptions. On-prem cost profiles below are separate customer-specific records.</p>
        {!hydrated ? <p role="status">Loading saved assumptions…</p> : (
          <div key={revision} className={styles.assumptionGroups}>
            {COST_ASSUMPTION_GROUPS.map(group => (
              <AssumptionGroup key={group.title} group={group} />
            ))}
          </div>
        )}
      </div>
    </section>
  )
}

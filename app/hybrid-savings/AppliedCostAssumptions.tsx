import type { ReactNode } from 'react'
import { COST_ASSUMPTION_GROUPS, type CostAssumptionKey } from '@/lib/costing-assumptions'
import type { CostAssumptions, CostOption, HybridWorkload } from '@/lib/hybrid-savings/calc'
import styles from './hybrid-savings.module.css'

const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 5 })
const money = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 5 })
const fields = new Map(COST_ASSUMPTION_GROUPS.flatMap(group => group.fields).map(field => [field.key, field]))
const shortLabels: Partial<Record<CostAssumptionKey, string>> = {
  planningCapacityUsePct: 'Planning capacity', loadedMonthlyCostPerFte: 'Staffing rate / FTE',
  rentedDirectInfrastructureMonthly: 'Supporting infrastructure', cloudRuntimeBufferPct: 'Runtime buffer',
  ownedBaseSystemPowerWattsPerServer: 'Base power / server', ownedInstallationPerServer: 'Additional installation / server',
  ownedFacilityMonthlyPerServer: 'Facilities / server', ownedDirectInfrastructureMonthly: 'Shared infrastructure',
}

function inputValue(key: CostAssumptionKey, value: number): string {
  if (key.endsWith('Pct')) return `${number.format(value)}%`
  if (key.endsWith('OperationsFte')) return `${number.format(value)} FTE`
  if (key === 'hardwareLifeYears') return `${number.format(value)} years`
  if (key === 'analysisMonths') return `${number.format(value)} months`
  if (key === 'pue') return `${number.format(value)}×`
  if (key === 'ownedBaseSystemPowerWattsPerServer') return `${number.format(value)} W`
  if (key === 'electricityPerKwh') return `${money.format(value)} / kWh`
  const monthly = key.endsWith('Monthly') || key === 'loadedMonthlyCostPerFte' || key === 'ownedFacilityMonthlyPerServer'
  return `${money.format(value)}${monthly ? ' / month' : ''}`
}

function Row({ label, children, inputKey }: { label: string; children: ReactNode; inputKey?: CostAssumptionKey }) {
  return <div data-assumption={inputKey}><dt>{label}</dt><dd>{children}</dd></div>
}
function Group({ title, children }: { title: string; children: ReactNode }) {
  return <div className={styles.assumptionSubgroup}><h5>{title}</h5><dl className={styles.calculationList}>{children}</dl></div>
}

/** Presentation only: keep shared inputs complete, without duplicating card prices or cost formulas. */
export function AppliedCostAssumptions({ assumptions, workload, rented, owned }: {
  assumptions: CostAssumptions
  workload: HybridWorkload
  rented?: CostOption
  owned?: CostOption
}) {
  const cloud = rented?.candidate
  const server = owned?.candidate
  const effectiveMode = cloud?.cloudRateKind === 'capacity_block' ? 'always-on' : assumptions.cloudBillingMode
  const billingLabel = effectiveMode === 'scale-to-zero' ? 'Scale to zero' : effectiveMode === 'active-window' ? 'Active window' : 'Always on'
  const infrastructure = cloud?.cloudDirectInfrastructureMonthly ?? assumptions.rentedDirectInfrastructureMonthly
  const inputs = (keys: CostAssumptionKey[]) => keys.map(key => <Row key={key} inputKey={key}
    label={shortLabels[key] ?? fields.get(key)!.label.replace(/\s*\([^)]*\)/g, '')}>
    {inputValue(key, key === 'rentedDirectInfrastructureMonthly' ? infrastructure : assumptions[key])}
    {key === 'rentedDirectInfrastructureMonthly' && <span className={styles.appliedValueNote}>{cloud?.cloudDirectInfrastructureMonthly != null
      ? 'Catalogue allowance — replaces the Sources fallback.'
      : cloud ? 'Sources fallback.' : 'Fallback; confirmed after sizing.'}</span>}
    {key === 'cloudRuntimeBufferPct' && effectiveMode !== 'scale-to-zero' && <span className={styles.appliedValueNote}>Not used in this billing mode.</span>}
  </Row>)

  return <>
    <section className={styles.comparisonBasis} aria-labelledby="hybrid-comparison-basis-heading">
      <h3 id="hybrid-comparison-basis-heading">Comparison basis</h3>
      <dl className={styles.basisList}>
        <Row label="Cost view">{assumptions.costLens === 'marginal' ? 'Marginal cost' : 'Full TCO'}</Row>
        {inputs(['planningCapacityUsePct', 'analysisMonths', 'loadedMonthlyCostPerFte'])}
      </dl>
    </section>
    <section className={styles.costAssumptionsSection} aria-labelledby="hybrid-cost-assumptions-heading">
      <h3 id="hybrid-cost-assumptions-heading">Cost assumptions</h3>
      <div className={styles.assumptionColumns}>
        <section className={styles.calculationGroup} aria-labelledby="hybrid-hosted-assumptions-heading">
          <h4 id="hybrid-hosted-assumptions-heading">Hosted API</h4>
          <Group title="Fees">{inputs(['hostedFixedMonthly'])}</Group>
          <Group title="People and setup">{inputs(['hostedOperationsFte', 'hostedImplementation'])}</Group>
        </section>
        <section className={styles.calculationGroup} aria-labelledby="hybrid-rented-assumptions-heading">
          <h4 id="hybrid-rented-assumptions-heading">Rented infrastructure</h4>
          <Group title="Billing and infrastructure">
            <Row label="Billing mode">{billingLabel}{cloud?.cloudRateKind === 'capacity_block' && <span className={styles.appliedValueNote}>Required by capacity block.</span>}</Row>
            <Row label="Processing window">{number.format(workload.activeHoursPerMonth)} hours/month</Row>
            <Row label="Always-on month">{number.format(assumptions.hoursPerMonth)} hours</Row>
            {inputs(['cloudRuntimeBufferPct', 'rentedDirectInfrastructureMonthly', 'rentedFixedMonthly'])}
          </Group>
          <Group title="People and setup">{inputs(['rentedOperationsFte', 'rentedImplementation'])}</Group>
        </section>
        <section className={styles.calculationGroup} aria-labelledby="hybrid-owned-assumptions-heading">
          <h4 id="hybrid-owned-assumptions-heading">Purchased hardware</h4>
          <Group title="Hardware">
            {inputs(['hardwareLifeYears', 'hardwareResidualPct', 'annualCostOfCapitalPct', 'annualMaintenancePct'])}
          </Group>
          <Group title="Power and facilities">
            <Row label="Power basis">Full TDP · {number.format(assumptions.hoursPerMonth)} hours/month</Row>
            <Row label="GPU power / GPU">{server ? `${number.format(server.tdpWattsPerGpu ?? 0)} W${server.tdpWattsPerGpu == null ? ' · not supplied; treated as zero' : ''}` : 'Available after sizing'}</Row>
            {inputs(['ownedBaseSystemPowerWattsPerServer', 'electricityPerKwh', 'pue', 'ownedFacilityMonthlyPerServer', 'ownedDirectInfrastructureMonthly'])}
          </Group>
          <Group title="People and setup">
            <Row label="Catalogue installation / server">{server ? money.format(server.purchaseInstallationPerReplica ?? 0) : 'Available after sizing'}</Row>
            {inputs(['ownedInstallationPerServer', 'ownedOperationsFte', 'ownedImplementation', 'ownedFixedMonthly'])}
          </Group>
        </section>
      </div>
      <p className={styles.calculationNote}>FTE is the share of a full-time person. Implementation is spread over the analysis period; additional installation is added to the catalogue allowance.{assumptions.costLens === 'marginal' && ' Marginal costs exclude acquisition, capital, installation, operations and initial implementation.'} Selected prices and cost breakdowns are on the flip cards.</p>
    </section>
  </>
}

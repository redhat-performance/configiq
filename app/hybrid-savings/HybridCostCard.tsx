'use client'

import * as React from 'react'
import { Button, Card, CardBody, Label } from '@patternfly/react-core'
import { JargonPopover } from '@/components/ui/JargonPopover'
import type { CostLens, CostOption } from '@/lib/hybrid-savings/calc'
import { formatCostPerMillionOutputTokens, formatHybridUsd, formatPlanningCapacityUse } from '@/lib/hybrid-savings/token-economics'
import styles from './hybrid-savings.module.css'

const whole = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 })
const hours = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 })
// Short display labels only; every included amount still comes from the cost engine.
const costLabels: Record<string, string> = {
  'Input token usage': 'Input tokens',
  'Output token usage': 'Output tokens',
  'Fixed hosted fees': 'Provider fees',
  'Operations and vendor management': 'Operations',
  'Cloud GPU compute': 'GPU rental',
  'Infrastructure and observability': 'Infrastructure',
  'Other rented-platform costs': 'Other platform costs',
  'Other owned-platform costs': 'Other platform costs',
  'Platform and model operations': 'Operations',
  'Implementation amortization': 'Setup (amortized)',
  'Hardware depreciation net of residual': 'Hardware depreciation',
  'Installation and commissioning': 'Installation',
  'Maintenance and spares': 'Maintenance',
  'Power including PUE': 'Power (incl. PUE)',
  'Rack, facilities and infrastructure': 'Facilities',
}
/** Presentation only: all prices, counts and breakdowns come from the cost engine. */
export function HybridCostCard({
  title, option, cheapest, lens, selector, children, unavailable, requestedOutputTokens,
}: {
  title: string
  option?: CostOption
  cheapest: boolean
  lens: CostLens
  selector?: React.ReactNode
  children?: React.ReactNode
  unavailable?: string
  /** Requested monthly output tokens; drives the primary unit metric explanation. */
  requestedOutputTokens?: number
}) {
  const candidate = option?.candidate
  const deployment = option?.deployment ?? null
  const concurrency = candidate?.referenceTargetConcurrency ?? 0
  const [flipped, setFlipped] = React.useState(false)
  const isFlipped = flipped && !!option
  const detailsId = React.useId()
  const frontRef = React.useRef<HTMLDivElement>(null)
  const backRef = React.useRef<HTMLDivElement>(null)
  const flipControlRef = React.useRef<HTMLButtonElement>(null)
  const priceSource = option?.key === 'rented' ? candidate?.cloudPriceSource : candidate?.purchasePriceSource
  const priceSourceUrl = option?.key === 'rented' ? candidate?.cloudPriceSourceUrl : candidate?.purchasePriceSourceUrl
  const priceDate = option?.key === 'rented' ? candidate?.cloudPriceSourceDate : candidate?.purchasePriceSourceDate
  const collectionKind = option?.key === 'rented' ? candidate?.cloudPriceCollectionKind : candidate?.purchasePriceCollectionKind
  const priceStale = option?.key === 'rented' ? candidate?.cloudPriceStale : candidate?.purchasePriceStale
  const unitCost = option?.costPerMillionOutputTokens ?? null

  React.useLayoutEffect(() => {
    if (frontRef.current) frontRef.current.inert = isFlipped
    if (backRef.current) backRef.current.inert = !isFlipped
  }, [isFlipped])

  return (
    <Card className={styles.resultCard}>
      <CardBody>
        <div className={`${styles.costFlip} ${isFlipped ? styles.costFlipped : ''}`}
          onKeyDown={event => {
            if (isFlipped && event.key === 'Escape') {
              event.preventDefault()
              setFlipped(false)
              flipControlRef.current?.focus({ preventScroll: true })
            }
          }}>
          {option && <Button variant="plain" className={styles.costFlipControl} innerRef={flipControlRef}
            aria-label={isFlipped ? `Back to ${title} summary` : `View ${title} details`}
            aria-controls={detailsId} aria-expanded={isFlipped}
            onClick={() => setFlipped(value => !value)} />}
          <div ref={frontRef} className={`${styles.costFace} ${styles.costFront}${option ? '' : ` ${styles.unpricedFront}`}`} aria-hidden={isFlipped}>
            <div className={styles.resultHeading}>
              <h3>{title}</h3>
              {cheapest && option && <Label color="green" isCompact>Cheapest</Label>}
            </div>
            {!option ? <p className={styles.unavailable} role="status">{unavailable}</p> : unitCost === null ? (
              <>
                <div className={styles.costValue}>{formatHybridUsd(option.monthlyCost)}<span> / month</span></div>
                <p className={styles.costPeriod}>{requestedOutputTokens !== undefined && requestedOutputTokens <= 0
                  ? 'Enter a positive monthly output volume to compare cost per million output tokens.'
                  : 'Cost per million output tokens is not available.'}</p>
              </>
            ) : (
              <>
                <div className={styles.costValue}>{formatCostPerMillionOutputTokens(unitCost)}</div>
                <div className={styles.costUnit}>
                  <span>per 1M output tokens</span>
                  <span className={styles.costUnitHelp}>
                    <JargonPopover term="Cost per 1M output tokens"
                      explanation="Selected monthly cost — including input processing and every other item in the selected cost view — divided by the requested monthly output tokens. The same denominator is applied to every option. It is not the provider's output-token tariff."
                      appendTo={() => document.body} />
                  </span>
                </div>
                <p className={styles.costPeriod}>{formatHybridUsd(option.monthlyCost)} / month</p>
              </>
            )}
            {selector && <div className={styles.costSelector} data-no-flip>{selector}</div>}
            {option && children}
            {deployment && <div className={styles.costCapacity}>
              <p>{whole.format(deployment.wholeUnits)} {option?.key === 'rented' ? 'instance' : 'server'}{deployment.wholeUnits === 1 ? '' : 's'} · {whole.format(deployment.billedGpus)} GPU{deployment.billedGpus === 1 ? '' : 's'}</p>
              {deployment.planningOutputCapacity > 0 && <>
                <p className={styles.costCapacityLabel}>Modeled output capacity</p>
                <p>{compact.format(deployment.planningOutputCapacity)} tokens/month
                  {deployment.planningCapacityUsedPct !== null && ` · ${formatPlanningCapacityUse(deployment.planningCapacityUsedPct)} used`}</p>
              </>}
              {concurrency > 0 && <p className={styles.costCapacityLabel}>
                Includes sizing for {whole.format(concurrency)} concurrent {concurrency === 1 ? 'request' : 'requests'}.
              </p>}
            </div>}
            {option && <span className={styles.costFlipHint} aria-hidden="true">↻ see details</span>}
          </div>
          {option && <div ref={backRef} id={detailsId} className={`${styles.costFace} ${styles.costBack}`} aria-hidden={!isFlipped}>
            <div className={styles.costBackHeading}>
              <h3>{title}</h3>
              <p>Monthly cost breakdown · {lens === 'fully-loaded' ? 'Full TCO' : 'Direct running costs'}</p>
            </div>
            <ul className={styles.costCompactBreakdown}>
              {option.breakdown.filter(item => item.monthlyCost !== 0 && (lens === 'fully-loaded' || item.includedInMarginal)).map(item => (
                <li key={item.label}><span>{costLabels[item.label] ?? item.label}</span><strong>{formatHybridUsd(item.monthlyCost)}</strong></li>
              ))}
            </ul>
            <div className={styles.costBackTotal}><span>Total / month</span><strong>{formatHybridUsd(option.monthlyCost)}</strong></div>
            {unitCost !== null && !!requestedOutputTokens && <p className={styles.costPeriod}>
              {formatCostPerMillionOutputTokens(unitCost)} per 1M output tokens at {whole.format(requestedOutputTokens)} output tokens/month.
            </p>}
            {candidate && <dl className={styles.costKeyFacts}>
              <div><dt>{option.key === 'rented' ? 'Instance rate' : 'Server price'}</dt><dd>{option.key === 'rented'
                ? (candidate.cloudHourlyCostPerInstance == null ? 'Not supplied' : `${formatHybridUsd(candidate.cloudHourlyCostPerInstance)} / hour`)
                : `${formatHybridUsd(candidate.purchasePricePerReplica ?? 0)}${candidate.purchasePriceIndicative ? ' · indicative' : ''}`}</dd></div>
              {deployment?.billableInstanceHours != null && <div><dt>Billable instance-hours</dt><dd>
                {deployment.billableInstanceHours > 0 && deployment.billableInstanceHours < 0.01 ? '<0.01' : hours.format(deployment.billableInstanceHours)} / month
              </dd></div>}
            </dl>}
            {candidate && <p className={styles.costPriceSource}>
              {priceSourceUrl ? <a href={priceSourceUrl} target="_blank" rel="noreferrer" aria-label={priceSource || 'Price source'}>Price source</a> : <span>Source link unavailable</span>}
              <span> · {collectionKind === 'automated' ? 'Automatically collected' : 'Curated estimate'}{priceStale ? ' · stale' : ''}</span>
              {priceDate && <span> · price dated {priceDate}</span>}
            </p>}
            <span className={styles.costFlipHint} aria-hidden="true">↻ flip back</span>
          </div>}
        </div>
      </CardBody>
    </Card>
  )
}

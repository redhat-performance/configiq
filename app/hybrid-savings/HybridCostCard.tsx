'use client'

import * as React from 'react'
import { Button, Card, CardBody, Label } from '@patternfly/react-core'
import type { CostLens, CostOption } from '@/lib/hybrid-savings/calc'
import styles from './hybrid-savings.module.css'

const money = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })
const rate = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 })
const requestRate = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 6 })
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
  title, option, cheapest, lens, selector, children, unavailable, hostedOfferId,
}: {
  title: string
  option?: CostOption
  cheapest: boolean
  lens: CostLens
  selector?: React.ReactNode
  children?: React.ReactNode
  unavailable?: string
  hostedOfferId?: string
}) {
  const candidate = option?.candidate
  const [flipped, setFlipped] = React.useState(false)
  const isFlipped = flipped && !!option
  const detailsId = React.useId()
  const frontRef = React.useRef<HTMLDivElement>(null)
  const backRef = React.useRef<HTMLDivElement>(null)
  const flipControlRef = React.useRef<HTMLButtonElement>(null)
  const priceSource = option?.key === 'rented' ? candidate?.cloudPriceSource : candidate?.purchasePriceSource
  const priceSourceUrl = option?.key === 'rented' ? candidate?.cloudPriceSourceUrl : candidate?.purchasePriceSourceUrl
  const priceDate = option?.key === 'rented' ? candidate?.cloudPriceSourceDate : candidate?.purchasePriceSourceDate

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
          <div ref={frontRef} className={`${styles.costFace} ${styles.costFront}`} aria-hidden={isFlipped}>
            <div className={styles.resultHeading}>
              <h3>{title}</h3>
              {cheapest && option && <Label color="green" isCompact>Cheapest</Label>}
            </div>
            {!option ? <p className={styles.unavailable}>{unavailable}</p> : <>
              <div className={styles.costValue}>{money.format(option.monthlyCost)}<span> / month</span></div>
              <p className={styles.costPeriod}>{rate.format(option.costPerMillionTokens)} / 1M tokens</p>
              <div className={styles.costSelector}>{selector}</div>
              {children}
              <span className={styles.costFlipHint} aria-hidden="true">↻ see details</span>
            </>}
          </div>
          {option && <div ref={backRef} id={detailsId} className={`${styles.costFace} ${styles.costBack}`} aria-hidden={!isFlipped}>
            <div className={styles.costBackHeading}>
              <h3>{title}</h3>
              <p>Monthly cost breakdown · {lens === 'fully-loaded' ? 'Full TCO' : 'Marginal'}</p>
            </div>
            <ul className={styles.costCompactBreakdown}>
              {option.breakdown.filter(item => item.monthlyCost !== 0 && (lens === 'fully-loaded' || item.includedInMarginal)).map(item => (
                <li key={item.label}><span>{costLabels[item.label] ?? item.label}</span><strong>{money.format(item.monthlyCost)}</strong></li>
              ))}
            </ul>
            <div className={styles.costBackTotal}><span>Total / month</span><strong>{money.format(option.monthlyCost)}</strong></div>
            <dl className={styles.costKeyFacts}>
              {hostedOfferId && <div><dt>Offer / checkpoint</dt><dd>{hostedOfferId}</dd></div>}
              {candidate ? <>
                <div><dt>GPUs</dt><dd>{option.gpuCount} required · {option.billedGpuCount} {option.key === 'rented' ? 'rented' : 'purchased'}</dd></div>
                <div><dt>{option.key === 'rented' ? 'Instance rate' : 'Server price'}</dt><dd>{option.key === 'rented'
                  ? (candidate.cloudHourlyCostPerInstance == null ? 'Not supplied' : `${rate.format(candidate.cloudHourlyCostPerInstance)} / hour`)
                  : `${money.format(candidate.purchasePricePerReplica ?? 0)}${candidate.purchasePriceIndicative ? ' · indicative' : ''}`}</dd></div>
              </> : <div><dt>Average request</dt><dd>{requestRate.format(option.costPerRequest)}</dd></div>}
            </dl>
            {candidate && <p className={styles.costPriceSource}>
              {priceSourceUrl ? <a href={priceSourceUrl} target="_blank" rel="noreferrer" aria-label={priceSource || 'Price source'}>Price source</a> : <span>Source link unavailable</span>}
              <span> · {priceDate ?? 'Date not supplied'}</span>
            </p>}
            <span className={styles.costFlipHint} aria-hidden="true">↻ flip back</span>
          </div>}
        </div>
      </CardBody>
    </Card>
  )
}

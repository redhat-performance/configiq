import { List, ListItem, ListVariant } from '@patternfly/react-core'
import { JargonPopover } from '@/components/ui/JargonPopover'
import type { InfrastructurePath } from '@/lib/hybrid-savings/calc'
import styles from './hybrid-savings.module.css'

/** Keep the units and series key next to the plot in both chart views. */
export function ChartPlotHeader({ path }: { path?: InfrastructurePath }) {
  return (
    <div className={styles.chartPlotHeader}>
      <div className={styles.chartAxisTitle}>
        Cost per 1M output tokens (USD)
        <JargonPopover term="Cost per 1M output tokens"
          explanation="All costs in the selected cost view, including input processing, divided by requested output tokens in millions. Every option uses the same denominator. This is not the provider's output-token tariff. Lower is better." />
      </div>
      <List isPlain variant={ListVariant.inline} className={styles.chartLegend} aria-label="Chart legend">
        <ListItem icon={<i className={styles.hostedSwatch} aria-hidden="true" />}>Hosted API</ListItem>
        {path !== 'owned' && <ListItem icon={<i className={styles.rentedSwatch} aria-hidden="true" />}>{path ? 'Rented (fixed)' : 'Rented'}</ListItem>}
        {path !== 'rented' && <ListItem icon={<i className={styles.ownedSwatch} aria-hidden="true" />}>{path ? 'Purchased (fixed)' : 'Purchased'}</ListItem>}
      </List>
    </div>
  )
}

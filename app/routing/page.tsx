'use client'

import { PageSection } from '@patternfly/react-core'
import RoutingEconomics from './RoutingEconomics'

export default function Page() {
  return (
    <PageSection
      padding={{ default: 'noPadding' }}
      style={{
        backgroundColor: 'var(--gc-bg-2, #f5f5f5)',
        minHeight: '100vh',
        padding: 0,
      }}
    >
      <RoutingEconomics />
    </PageSection>
  )
}

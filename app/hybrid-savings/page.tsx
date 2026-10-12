'use client'

import { PageSection } from '@patternfly/react-core'
import HybridSavings from './HybridSavings'

export default function HybridSavingsPage() {
  return (
    <PageSection
      padding={{ default: 'noPadding' }}
      style={{ backgroundColor: '#f5f5f5', padding: 0 }}
    >
      <HybridSavings />
    </PageSection>
  )
}

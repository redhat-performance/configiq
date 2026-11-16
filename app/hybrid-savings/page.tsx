'use client'

import { Suspense } from 'react'
import { PageSection } from '@patternfly/react-core'
import HybridSavings from './HybridSavings'

export default function HybridSavingsPage() {
  return (
    <PageSection
      padding={{ default: 'noPadding' }}
      style={{ backgroundColor: '#f5f5f5', padding: 0 }}
    >
      {/* This page reads its shared-view link with `useSearchParams`. */}
      <Suspense fallback={null}>
        <HybridSavings />
      </Suspense>
    </PageSection>
  )
}

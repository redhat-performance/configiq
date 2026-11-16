'use client'

import { Suspense } from 'react'
import { PageSection } from '@patternfly/react-core'
import KvCacheCalc from './KvCacheCalc'

export default function Page() {
  return (
    <PageSection
      padding={{ default: 'noPadding' }}
      style={{ backgroundColor: 'var(--gc-bg-2, #f5f5f5)', minHeight: '100vh', padding: 0 }}
    >
      {/* This page reads its shared-view link with `useSearchParams`. */}
      <Suspense fallback={null}>
        <KvCacheCalc />
      </Suspense>
    </PageSection>
  )
}

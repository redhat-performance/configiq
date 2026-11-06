'use client'

import * as React from 'react'
import { PageSection } from '@patternfly/react-core'
import styles from './TestedModels.module.css'

/**
 * Shared grey shell for the tested-model pages. Kept as a client component
 * because PatternFly's PageSection uses context; the page routes themselves
 * stay server components that read the registry at request time.
 */
export function TestedModelsShell({ children }: { children: React.ReactNode }) {
  return (
    <PageSection
      padding={{ default: 'noPadding' }}
      style={{ backgroundColor: 'var(--gc-bg-2, #f5f5f5)', minHeight: '100vh', padding: 0 }}
    >
      <main className={styles.page}>{children}</main>
    </PageSection>
  )
}

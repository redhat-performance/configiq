'use client';

import { Suspense } from 'react';
import { PageSection } from '@patternfly/react-core';
import Performance from './Performance';

export default function Page() {
  return (
    <PageSection
      padding={{ default: 'noPadding' }}
      style={{
        backgroundColor: 'var(--gc-bg-2, #f5f5f5)',
        minHeight: '100vh',
        padding: 0,
        paddingLeft: 0,
        paddingRight: 0
      }}
    >
      {/*
        The page reads its shared-view link with `useSearchParams`, which Next
        requires inside a Suspense boundary on a prerendered route. The shell
        above stays static; only the tool is client-rendered.
      */}
      <Suspense fallback={null}>
        <Performance />
      </Suspense>
    </PageSection>
  );
}

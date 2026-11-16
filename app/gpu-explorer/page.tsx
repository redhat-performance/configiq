'use client';

import { Suspense } from 'react';
import GpuExplorer from './GpuExplorer';

export default function Page() {
  return (
    // The explorer reads its shared-view link with `useSearchParams`.
    <Suspense fallback={null}>
      <GpuExplorer />
    </Suspense>
  );
}

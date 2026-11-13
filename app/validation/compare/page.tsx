'use client';

import { Suspense } from 'react';
import BenchmarkComparison from './Compare';

export default function Page() {
  return (
    // The comparison reads its shared-view link with `useSearchParams`.
    <Suspense fallback={null}>
      <BenchmarkComparison />
    </Suspense>
  );
}
